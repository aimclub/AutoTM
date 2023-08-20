import os
from typing import Union, cast, Optional

import pandas as pd
import pymystem3
from nltk.corpus import stopwords, wordnet
from autotm.utils import parallelize_dataframe
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer

import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from nltk.stem import WordNetLemmatizer



class DataTransformer():

    def __init__(self, 
                 col_to_process: str,
                 save_path: Optional[str] = None,
                 lang: str = 'ru',
                 min_tokens_count: int = 3,
                 n_cores: int = -1):
        

        self.col_to_process = col_to_process
        self.save_path = save_path
        self.lang = lang
        self.min_tokens_count = min_tokens_count
        self.n_cores = n_cores

        # prepare necessary tools
        self._prepare()

    
    def _prepare(self):

        @Language.factory("language_detector")
        def language_detector(nlp, name):
            return LanguageDetector

        # prepare stopwords
        nltk.download("stopwords")
        nltk.download("wordnet")

        self.stop = stopwords.words("russian") + [" "] + stopwords.words("english")

        # prepare spacy
        self.nlp_model = spacy.load("en_core_web_sm")
        self.nlp_model.add_pipe("language_detector", last=True)

        # prepare lemmatizer
        if self.lang == "ru":
            self.lemmatizer = pymystem3.Mystem()
        elif self.lang == "en":
            self.lemmatizer = WordNetLemmatizer()
        else:
            raise ValueError("Inferred language is not supported yet. Current supported language: [\"ru\", \"en\"].")
        
        # prepare regexes
        self.r_html = re.compile(r"(\<[^>]*\>)")
        self.r_punct = re.compile(r'[."\[\]/,()!?;:*#|\\%^$&{}~_`=-@]')
        self.r_vk_ids = re.compile(r"(id{1}[0-9]*)")
        self.r_num = re.compile(r"([0-9]+)")
        self.r_white_space = re.compile(r"\s{2,}")
        self.r_words = re.compile(r"\W+")
        self.r_pat = re.compile(r"[aA-zZ]")

        # check url pattern work
        self.url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )

    
    def _clean_pipeline(self, data, **kwargs):

        def remove_html(text: str) -> str:
            """html removal function"""
            text = self.r_html.sub("", text)
            return text


        def process_punkt(text: str) -> str:
            text = self.r_punct.sub(" ", text)
            text = self.r_vk_ids.sub(" ", text)
            text = self.r_num.sub(" ", text)
            text = self.r_white_space.sub(" ", text)
            return text.strip()


        def get_lemma(word: str) -> str:
            """function to extract lemma from english words"""
            synsets = wordnet.synsets(word)
            if synsets is None or len(synsets) == 0:
                return word
            else:
                return synsets[0].lemmas()[0].name()


        def lemmatize_text_ru(text: str, leave_service_info=False) -> str:
            try:
                text = remove_html(text)
            except:
                return ""
            text = text.lower()
            text = process_punkt(text)
            try:
                tokens = self.r_words.split(text)
            except:
                return ""
            tokens = (x for x in tokens if len(x) >= 3 and not x.isdigit())
            text = " ".join(tokens)
            tokens = self.lemmatizer.lemmatize(text)
            tokens = (x for x in tokens if len(x) >= 3)
            tokens = (x for x in tokens if x not in self.stop)
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
            doc = self.nlp_model(text)
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
            tokens = (x for x in tokens if len(x) >= 3 and self.r_pat.match(x) and not x.isdigit())
            text_rmstop = [i for i in tokens if i not in stop]

            text_stem = [
                self.lemmatizer.lemmatize(w, pos=get_wordnet_pos(w)) for w in text_rmstop
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

        return lemmatize_text(data, **kwargs)
    
    def tokens_num(self, text):
        return len(text.split(" "))


    def fit_transform(self, fname: Union[pd.DataFrame, str]):

        self.fit(fname)

        self.transform()

    
    def fit(self, fname: Union[pd.DataFrame, str]):

        # if save_path is set, then prepare directory for it
        # else, then save processed data in buffer
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)            
            self.save_path = os.path.join(self.save_path, "ppp.csv")

        data = pd.read_csv(fname) if isinstance(fname, str) else cast(pd.DataFrame, fname)
        self.data = parallelize_dataframe(
            data, self._clean_pipeline, self.n_cores, lang=self.lang, col_to_process=self.col_to_process
        )

    def transform(self):
        
        try:
            self.data["tokens_len"] = self.data["processed_text"].apply(self.tokens_num)
            self.data = self.data[self.data["tokens_len"] > self.min_tokens_count]
            
            # if save_path is set, then prepare directory for it
            # else, then save processed data in buffer
            if self.save_path is not None:
                self.data.to_csv(self.save_path, index=None)
                print("Saved to {}".format(self.save_path))

        except:
            # print(self.data == None)
            raise RuntimeError("Data is not initialized. Please call .fit() first before calling .transform().")
