"""Data loading and preprocessing helpers for LDA baselines."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import (
    preprocess_string,
    remove_stopwords,
    strip_multiple_whitespaces,
    strip_non_alphanum,
    strip_numeric,
    strip_punctuation,
    strip_short,
)
from sklearn.datasets import fetch_20newsgroups


DEFAULT_FILTERS = [
    strip_non_alphanum,
    lambda s: s.lower(),
    strip_punctuation,
    strip_numeric,
    remove_stopwords,
    strip_multiple_whitespaces,
    strip_short,
]


def tokenize_gensim(docs: List[str]) -> List[List[str]]:
    """Tokenize documents using the default gensim preprocessing pipeline."""

    tokens = [preprocess_string(document, DEFAULT_FILTERS) for document in docs]
    return [token_list for token_list in tokens if len(token_list) > 0]


def _ensure_nltk():
    import nltk as _nltk

    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ]

    for resource_path, resource_name in resources:
        try:
            _nltk.data.find(resource_path)
        except LookupError:
            _nltk.download(resource_name, quiet=True)


def _get_wordnet_pos(tag: str):
    from nltk.corpus import wordnet as _wn

    if tag.startswith("J"):
        return _wn.ADJ
    if tag.startswith("V"):
        return _wn.VERB
    if tag.startswith("N"):
        return _wn.NOUN
    if tag.startswith("R"):
        return _wn.ADV
    return _wn.NOUN


def tokenize_autotm_en(docs: List[str]) -> List[List[str]]:
    import re as _re
    import nltk as _nltk
    from nltk.corpus import stopwords as _stopwords
    from nltk.stem import WordNetLemmatizer as _WordNetLemmatizer

    _ensure_nltk()

    html_pattern = _re.compile(r"(<[^>]*>)")
    punct_pattern = _re.compile(r"[.\"\[\]/,()!?:*#|\\%^$&{}~_`=-@]")
    num_pattern = _re.compile(r"([0-9]+)")
    whitespace_pattern = _re.compile(r"\s{2,}")
    wordsplit_pattern = _re.compile(r"\W+")

    stop_words = set(_stopwords.words("english"))
    lemmatizer = _WordNetLemmatizer()

    processed: List[List[str]] = []
    for doc in docs:
        if not isinstance(doc, str):
            processed.append([])
            continue

        text = html_pattern.sub(" ", doc)
        text = text.lower()
        text = punct_pattern.sub(" ", text)
        text = num_pattern.sub(" ", text)
        text = whitespace_pattern.sub(" ", text).strip()

        raw_tokens = [token for token in wordsplit_pattern.split(text) if token]
        raw_tokens = [token for token in raw_tokens if len(token) >= 3 and token.isalpha() and token not in stop_words]

        if not raw_tokens:
            processed.append([])
            continue

        tags = _nltk.pos_tag(raw_tokens)
        lemmas = [lemmatizer.lemmatize(word, pos=_get_wordnet_pos(tag)) for word, tag in tags]
        tokens = [token for token in lemmas if len(token) >= 3 and token.isalpha() and token not in stop_words]
        processed.append(tokens)

    return [token_list for token_list in processed if len(token_list) > 0]


def tokenize_autotm_ru(docs: List[str]) -> List[List[str]]:
    import re as _re
    from nltk.corpus import stopwords as _stopwords

    try:
        import pymystem3 as _pymystem3  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "pymystem3 is required for Russian AutoTM preprocessing. Install it via 'pip install pymystem3'."
        ) from exc

    _ensure_nltk()

    html_pattern = _re.compile(r"(<[^>]*>)")
    punct_pattern = _re.compile(r"[.\"\[\]/,()!?:*#|\\%^$&{}~_`=-@]")
    num_pattern = _re.compile(r"([0-9]+)")
    whitespace_pattern = _re.compile(r"\s{2,}")
    wordsplit_pattern = _re.compile(r"\W+")

    stop_words = set(_stopwords.words("russian"))
    mystem = _pymystem3.Mystem()

    processed: List[List[str]] = []
    for doc in docs:
        if not isinstance(doc, str):
            processed.append([])
            continue

        try:
            text = html_pattern.sub(" ", doc)
        except Exception:
            processed.append([])
            continue

        text = text.lower()
        text = punct_pattern.sub(" ", text)
        text = num_pattern.sub(" ", text)
        text = whitespace_pattern.sub(" ", text).strip()

        try:
            rough_tokens = [token for token in wordsplit_pattern.split(text) if token]
        except Exception:
            processed.append([])
            continue

        rough_tokens = [token for token in rough_tokens if len(token) >= 3 and not token.isdigit()]
        normalized_text = " ".join(rough_tokens)

        lemmas = [token for token in mystem.lemmatize(normalized_text) if len(token) >= 3]
        lemmas = [token for token in lemmas if token.isalpha() and token not in stop_words]
        processed.append(lemmas)

    return [token_list for token_list in processed if len(token_list) > 0]


def dictionary_corpus(
    tokenized: List[List[str]],
    *,
    keep_n: int,
    no_below: int,
    no_above: float,
):
    """Build gensim dictionary and corpus with standard filtering."""

    dictionary = Dictionary(tokenized)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    dictionary.compactify()
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]
    return dictionary, corpus


def load_dataset(dataset: str, data_path: Optional[str], text_col: str) -> Tuple[List[str], Optional[np.ndarray], str]:
    dataset = dataset.lower()
    if dataset in {"20ng", "20newsgroups"}:
        data = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
        docs = list(data.data)
        labels = np.array(data.target, dtype=np.int64)
        return docs, labels, "20 Newsgroups (train, headers/footers/quotes removed)"

    if not data_path:
        raise ValueError("data-path is required for datasets other than 20ng")

    df = pd.read_csv(data_path)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in CSV. Available: {list(df.columns)}")

    docs = df[text_col].astype(str).tolist()
    dataset_names = {
        "amazon_food": "Amazon Fine Food Reviews",
        "hotel_reviews": "Datainfini Hotel Reviews",
        "lenta_ru": "Lenta.ru",
    }
    pretty_name = dataset_names.get(dataset, dataset)
    return docs, None, pretty_name


__all__ = [
    "DEFAULT_FILTERS",
    "dictionary_corpus",
    "load_dataset",
    "tokenize_autotm_en",
    "tokenize_autotm_ru",
    "tokenize_gensim",
]


