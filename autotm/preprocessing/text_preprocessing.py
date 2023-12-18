import os
from typing import Union, cast

import pandas as pd
import pymystem3
from nltk.corpus import stopwords, wordnet

from autotm.preprocessing import PREPOCESSED_DATASET_FILENAME
from autotm.utils import parallelize_dataframe
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer

import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from nltk.stem import WordNetLemmatizer

PROCESSED_TEXT_COLUMN = "processed_text"

# TODO: make transformer class and prep function to download all files

nltk.download("stopwords")
nltk.download("wordnet")

stop = stopwords.words("russian") + [" "] + stopwords.words("english")

r_html = re.compile(r"(\<[^>]*\>)")
r_punct = re.compile(r'[."\[\]/,()!?;:*#|\\%^$&{}~_`=-@]')
r_vk_ids = re.compile(r"(id{1}[0-9]*)")
r_num = re.compile(r"([0-9]+)")
r_white_space = re.compile(r"\s{2,}")
r_words = re.compile(r"\W+")
r_pat = re.compile(r"[aA-zZ]")

# check url pattern work
url_pattern = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)

m = pymystem3.Mystem()
en_lemmatizer = WordNetLemmatizer()


@Language.factory("language_detector")
def language_detector(nlp, name):
    return LanguageDetector()


nlp_model = spacy.load("en_core_web_sm")
nlp_model.add_pipe("language_detector", last=True)


def remove_html(text: str) -> str:
    """html removal function"""
    text = r_html.sub("", text)
    return text


def process_punkt(text: str) -> str:
    text = r_punct.sub(" ", text)
    text = r_vk_ids.sub(" ", text)
    text = r_num.sub(" ", text)
    text = r_white_space.sub(" ", text)
    return text.strip()


def get_lemma(word: str) -> str:
    """function to extract lemma from english words"""
    synsets = wordnet.synsets(word)
    if synsets is None or len(synsets) == 0:
        return word
    else:
        return synsets[0].lemmas()[0].name()


def tokens_num(text):
    return len(text.split(" "))


def lemmatize_text_ru(text: str, leave_service_info=False) -> str:
    try:
        text = remove_html(text)
    except:
        return ""
    text = text.lower()
    text = process_punkt(text)
    try:
        tokens = r_words.split(text)
    except:
        return ""
    tokens = (x for x in tokens if len(x) >= 3 and not x.isdigit())
    text = " ".join(tokens)
    tokens = m.lemmatize(text)
    tokens = (x for x in tokens if len(x) >= 3)
    tokens = (x for x in tokens if x not in stop)
    tokens = (x for x in tokens if x.isalpha())
    text = " ".join(tokens)
    return text


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].lower()
    tag_dict = {
        "a": wordnet.ADJ,
        "n": wordnet.NOUN,
        "v": wordnet.VERB,
        "r": wordnet.ADV,
    }
    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_text_en(text):
    stop = stopwords.words("english")
    doc = nlp_model(text)
    detect_language = doc._.language
    if detect_language["language"] != "en":
        return " "
    text = text.lower()  # added
    text = process_punkt(text)
    try:
        # TODO: check this
        text_token = CountVectorizer().build_tokenizer()(text)
    except:
        return " "
    tokens = text.split()
    tokens = (x for x in tokens if len(x) >= 3 and r_pat.match(x) and not x.isdigit())
    text_rmstop = [i for i in tokens if i not in stop]

    text_stem = [
        en_lemmatizer.lemmatize(w, pos=get_wordnet_pos(w)) for w in text_rmstop
    ]
    return " ".join(text_stem)


def lemmatize_text(df, **kwargs):
    # print(kwargs)
    lang = kwargs["lang"]
    col_to_process = kwargs["col_to_process"]
    if lang == "ru":
        df["processed_text"] = df[col_to_process].apply(lemmatize_text_ru)
    elif lang == "en":
        df["processed_text"] = df[col_to_process].apply(lemmatize_text_en)
    else:
        print(f"The language {lang} is not known")
        raise NameError
    return df


def process_dataset(
        fname: Union[pd.DataFrame, str],
        col_to_process: str,
        save_path: str,
        lang: str = "ru",
        min_tokens_count: int = 3,
        n_cores: int = -1,
):
    """

    :param fname: Path to the dataset to process.
    :param col_to_process: The name of text column to be processed.
    :param save_path: Path where to store all the artifacts.
    :param lang: Language of the data (ru/en).
    :param min_tokens_count: Minimal amount of tokens to consider the further processing and topic modeling (3 by default).
    :param n_cores: Amount of cores for parallelization.
    :return:
    """
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, PREPOCESSED_DATASET_FILENAME)
    data = pd.read_csv(fname) if isinstance(fname, str) else cast(pd.DataFrame, fname)
    data = parallelize_dataframe(
        data, lemmatize_text, n_cores, lang=lang, col_to_process=col_to_process
    )
    data["tokens_len"] = data[PROCESSED_TEXT_COLUMN].apply(tokens_num)
    data = data[data["tokens_len"] > min_tokens_count]
    data.to_csv(save_path, index=None)
    print("Saved to {}".format(save_path))
