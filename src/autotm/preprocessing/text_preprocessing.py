import os
import re
import pickle
from tqdm import tqdm
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import pymystem3
from nltk.corpus import stopwords, wordnet
import multiprocessing as mp
from nltk.stem.snowball import SnowballStemmer

import matplotlib
from matplotlib import pyplot as plt
import seaborn

import plotly.offline as offline
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly
from urllib.parse import urlparse
import pickle
import numpy as np
import html
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector

from nltk.stem import WordNetLemmatizer

stop = stopwords.words("russian") + [' '] + stopwords.words("english")

r_html = re.compile(r'(\<[^>]*\>)')
r_punct = re.compile(r'[."\[\]/,()!?;:*#|\\%^$&{}~_`=-@]')
r_vk_ids = re.compile(r'(id{1}[0-9]*)')
r_num = re.compile(r'([0-9]+)')
r_white_space = re.compile(r'\s{2,}')
r_words = re.compile(r'\W+')
r_pat = re.compile(r'[aA-zZ]')

# check url pattern work
url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

m = pymystem3.Mystem()
en_lemmatizer = WordNetLemmatizer()


# class preprocessor:
#     def __init__(self, lang):
#         pass


@Language.factory('language_detector')
def language_detector(nlp, name):
    return LanguageDetector()


nlp_model = spacy.load("en_core_web_sm")
nlp_model.add_pipe('language_detector', last=True)


def new_html(text: str) -> str:
    text = r_html.sub("", text)
    return text


def process_punkt(text: str) -> str:
    text = r_punct.sub(" ", text)
    text = r_vk_ids.sub(" ", text)
    text = r_num.sub(" ", text)
    text = r_white_space.sub(" ", text)
    return text.strip()


def get_lemma(word: str) -> str:
    lemma = wordnet.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


def tokens_num(text):
    return len(text.split(' '))


def lemmatize_text(text: str, language: str = 'ru') -> str:
    try:
        text = new_html(text)
    except:
        return ''
    text = text.lower()
    text = process_punkt(text)
    #     if language == 'ru':
    #         text = re.findall(r_rus, text)
    #         text = ' '.join(text)
    try:
        tokens = r_words.split(text)
    except:
        return ''
    tokens = (x for x in tokens if len(x) >= 3 and not x.isdigit())
    if language == 'ru':
        text = ' '.join(tokens)
        tokens = m.lemmatize(text)
    else:
        tokens = [get_lemma(token) for token in tokens]
    tokens = (x for x in tokens if len(x) > 3)
    tokens = (x for x in tokens if x not in stop)
    tokens = (x for x in tokens if x.isalpha())
    text = ' '.join(tokens)
    return text


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].lower()
    tag_dict = {"a": wordnet.ADJ,
                "n": wordnet.NOUN,
                "v": wordnet.VERB,
                "r": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_text_en(text):
    stop = stopwords.words('english')
    doc = nlp_model(text)
    detect_language = doc._.language
    if detect_language['language'] != 'en':
        return ' '
    text = text.lower()  # added
    text = process_punkt(text)
    try:
        # TODO: check this
        text_token = CountVectorizer().build_tokenizer()(text)
    except:
        return ' '
    tokens = text.split()
    tokens = (x for x in tokens if len(x) >= 3 and r_pat.match(x) and not x.isdigit())
    text_rmstop = [i for i in tokens if i not in stop]

    text_stem = [en_lemmatizer.lemmatize(w, pos=get_wordnet_pos(w)) for w in text_rmstop]
    return ' '.join(text_stem)


def process_dataset(fname: str, col_to_process: str, save_path, lang='ru'):
    data = pd.read_csv(fname)
    if lang == 'ru':
        data['processed_text'] = data[col_to_process].progress_apply(lemmatize_text)
    else:
        data['processed_text'] = data[col_to_process].progress_apply(lemmatize_text_en)
    data['tokens_len'] = data['processed_text'].apply(tokens_num)
    data = data[data['tokens_len'] > 3]
    data.to_csv(save_path, index=None)
    print('Saved to {}'.format(save_path))
