# -*- coding: utf-8 -*-
"""
Gensim LDA baseline (AutoTM metrics-compatible)
-------------------------------------
- Time-bounded training (default 180s)
- Reproducible seed
- Computes metrics comparable to AutoTM: fitness, Coherence 25, NPMI 25, SwitchP
- Supports datasets: 20 Newsgroups, Amazon Fine Food Reviews, Datainfini Hotel Reviews, Lenta.ru

Required Libraries:
    pip install gensim>=4.1.2
    pip install scikit-learn>=1.1.1
    pip install numpy
    pip install scipy<=1.10.1
    pip install pandas

Note: This script uses standard library modules (argparse, json, time, random, dataclasses, typing)
which are included with Python 3.9+.

CLI examples:
    # 20 Newsgroups
    python gensim_lda.py --dataset 20ng --topics 10 --budget 180 --seed 42

    # Custom CSVs (Amazon/Hotel/Lenta) with column 'text' (change with --text-col)
    python gensim_lda.py --dataset amazon_food --data-path /path/to/amazon.csv --text-col text
    python gensim_lda.py --dataset hotel_reviews --data-path /path/to/hotel.csv --text-col text
    python gensim_lda.py --dataset lenta_ru --data-path /path/to/lenta.csv --text-col text

Python:
    from gensim_lda import run_gensim_lda
    res = run_gensim_lda(dataset="20ng", num_topics=10, time_budget_s=180, random_seed=42)
"""

import argparse
import json
import time
import random
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from gensim.parsing.preprocessing import (
    preprocess_string,
    strip_punctuation,
    strip_multiple_whitespaces,
    strip_numeric,
    strip_short,
    remove_stopwords,
    strip_non_alphanum
)

try:
    from autotm.fitness.external_scores import switchp
except Exception:
    # Fallback: local implementation matching autotm.fitness.external_scores.switchp
    def switchp(phi, texts):
        words = phi.index.to_list()
        max_topic_word_dist = np.argmax(phi.to_numpy(), axis=1)
        max_topic_word_dist = dict(zip(words, max_topic_word_dist))
        switchp_scores = []
        for text in texts:
            mapped_text = [
                max_topic_word_dist[word]
                for word in text.split()
                if word in max_topic_word_dist
            ]
            if len(mapped_text) <= 1:
                switchp_scores.append(0.0)
                continue
            switches = (np.diff(mapped_text) != 0).sum()
            switchp_scores.append(switches / (len(mapped_text) - 1))

        return switchp_scores

DEFAULT_FILTERS = [
    strip_non_alphanum,
    lambda s: s.lower(),  # IMPORTANT: lowercase BEFORE remove_stopwords!
    strip_punctuation,
    strip_numeric,
    remove_stopwords,  # Must come AFTER lowercasing
    strip_multiple_whitespaces,
    strip_short
]

# AutoTM-like English preprocessing helpers
import re as _re
import nltk as _nltk
from nltk.corpus import stopwords as _stopwords
from nltk.corpus import wordnet as _wn
from nltk.stem import WordNetLemmatizer as _WordNetLemmatizer
from nltk import pos_tag as _pos_tag
import pymystem3 as _pymystem3

_R_HTML = _re.compile(r"(<[^>]*>)")
_R_PUNCT = _re.compile(r"[.\"\[\]/,()!?;:*#|\\%^$&{}~_`=-@]")
_R_NUM = _re.compile(r"([0-9]+)")
_R_WHITE = _re.compile(r"\s{2,}")
_R_WORDSPLIT = _re.compile(r"\W+")


def _ensure_nltk():
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ]
    for res_path, res_name in resources:
        try:
            _nltk.data.find(res_path)
        except LookupError:
            _nltk.download(res_name, quiet=True)


def _get_wordnet_pos(tag: str):
    if tag.startswith("J"):
        return _wn.ADJ
    if tag.startswith("V"):
        return _wn.VERB
    if tag.startswith("N"):
        return _wn.NOUN
    if tag.startswith("R"):
        return _wn.ADV
    return _wn.NOUN


def _tokenize_autotm_en(docs: List[str]) -> List[List[str]]:
    _ensure_nltk()
    sw = set(_stopwords.words("english"))
    lemm = _WordNetLemmatizer()

    tokenized: List[List[str]] = []
    for d in docs:
        if not isinstance(d, str):
            tokenized.append([])
            continue
        # basic cleanup similar to AutoTM
        txt = d
        txt = _R_HTML.sub(" ", txt)
        txt = txt.lower()
        txt = _R_PUNCT.sub(" ", txt)
        txt = _R_NUM.sub(" ", txt)
        txt = _R_WHITE.sub(" ", txt).strip()
        raw_tokens = [t for t in _R_WORDSPLIT.split(txt) if t]
        raw_tokens = [t for t in raw_tokens if len(t) >= 3 and t.isalpha() and t not in sw]
        if not raw_tokens:
            tokenized.append([])
            continue
        tags = _pos_tag(raw_tokens)
        lemmas = [lemm.lemmatize(w, pos=_get_wordnet_pos(t)) for w, t in tags]
        toks = [t for t in lemmas if len(t) >= 3 and t.isalpha() and t not in sw]
        tokenized.append(toks)
    return [t for t in tokenized if len(t) > 0]


def _tokenize_autotm_ru(docs: List[str]) -> List[List[str]]:
    # Mirrors autotm.preprocessing.text_preprocessing.lemmatize_text_ru
    _ensure_nltk()
    ru_sw = set(_stopwords.words("russian"))
    mystem = _pymystem3.Mystem()

    tokenized: List[List[str]] = []
    for d in docs:
        if not isinstance(d, str):
            tokenized.append([])
            continue
        try:
            txt = _R_HTML.sub(" ", d)
        except Exception:
            tokenized.append([])
            continue
        txt = txt.lower()
        txt = _R_PUNCT.sub(" ", txt)
        txt = _R_NUM.sub(" ", txt)
        txt = _R_WHITE.sub(" ", txt).strip()
        try:
            rough = [t for t in _R_WORDSPLIT.split(txt) if t]
        except Exception:
            tokenized.append([])
            continue
        rough = [t for t in rough if len(t) >= 3 and not t.isdigit()]
        txt2 = " ".join(rough)
        lemmas = [t for t in mystem.lemmatize(txt2) if len(t) >= 3]
        lemmas = [t for t in lemmas if t.isalpha() and t not in ru_sw]
        tokenized.append(lemmas)
    return [t for t in tokenized if len(t) > 0]


@dataclass
class Config:
    num_topics: int = 10
    time_budget_s: int = 180
    random_seed: int = 42
    keep_n: int = 50000
    no_below: int = 5
    no_above: float = 0.5
    chunksize: int = 2000
    passes_per_round: int = 1
    iterations_per_round: int = 20
    eval_every: int = 0
    alpha: str = "auto"
    eta: str = "auto"
    coherence_texts_sample: int = 5000
    knn_k: int = 5
    n_splits_f1: int = 5


@dataclass
class Result:
    dataset: str
    num_docs: int
    vocab_size: int
    num_topics: int
    wall_clock_s: float
    fitness: float
    coherence_25: float
    npmi_25: float
    switchP: Optional[float]
    notes: str


def _tokenize(docs: List[str]) -> List[List[str]]:
    toks = [preprocess_string(d, DEFAULT_FILTERS) for d in docs]
    return [t for t in toks if len(t) > 0]


def _dictionary_corpus(tokenized: List[List[str]], cfg: Config):
    dct = Dictionary(tokenized)
    dct.filter_extremes(no_below=cfg.no_below, no_above=cfg.no_above, keep_n=cfg.keep_n)
    dct.compactify()
    corpus = [dct.doc2bow(t) for t in tokenized]
    return dct, corpus


def _coherence(model: LdaModel, texts: List[List[str]], dct: Dictionary, cfg: Config):
    if cfg.coherence_texts_sample and len(texts) > cfg.coherence_texts_sample:
        rng = np.random.default_rng(cfg.random_seed)
        idx = rng.choice(len(texts), size=cfg.coherence_texts_sample, replace=False)
        texts = [texts[i] for i in idx]
    cm_cv = CoherenceModel(model=model, texts=texts, dictionary=dct, coherence='c_v').get_coherence()
    cm_npmi = CoherenceModel(model=model, texts=texts, dictionary=dct, coherence='c_npmi').get_coherence()
    return float(cm_cv), float(cm_npmi)


def _doc_topic_matrix(model: LdaModel, corpus):
    n_topics = model.num_topics
    mat = np.zeros((len(corpus), n_topics), dtype=np.float32)
    for i, bow in enumerate(corpus):
        for t_id, prob in model.get_document_topics(bow, minimum_probability=0.0):
            mat[i, t_id] = prob
    return mat


def _knn_f1(doc_topic: np.ndarray, labels: np.ndarray, cfg: Config) -> float:
    skf = StratifiedKFold(n_splits=cfg.n_splits_f1, shuffle=True, random_state=cfg.random_seed)
    scores = []
    for tr, te in skf.split(doc_topic, labels):
        clf = KNeighborsClassifier(n_neighbors=cfg.knn_k, weights='distance')
        clf.fit(doc_topic[tr], labels[tr])
        yhat = clf.predict(doc_topic[te])
        scores.append(f1_score(labels[te], yhat, average='weighted'))
    return float(np.mean(scores))


def _topics_top_words(model: LdaModel, topn: int) -> List[List[str]]:
    return [[w for w, _ in model.show_topic(t, topn=topn)] for t in range(model.num_topics)]


def _compute_metrics(model: LdaModel, tokenized: List[List[str]], dct: Dictionary, topn: int = 25) -> Tuple[float, float, float]:
    topics_words = _topics_top_words(model, topn)

    # Per-topic c_v using the filtered dictionary
    cm_cv = CoherenceModel(topics=topics_words, texts=tokenized, dictionary=dct, coherence='c_v')
    cv_per_topic = np.asarray(cm_cv.get_coherence_per_topic(), dtype=np.float64)

    # Per-topic c_npmi
    # Match AutoTM: build a fresh unfiltered dictionary from tokenized texts
    # This is important because AutoTM does: id2word = corpora.Dictionary([text.split() for text in texts])
    # where texts are already tokenized. We have tokenized texts directly as List[List[str]].
    dct_npmi = Dictionary(tokenized)
    # AutoTM does NOT filter this dictionary
    
    cm_npmi = CoherenceModel(topics=topics_words, texts=tokenized, dictionary=dct_npmi, coherence='c_npmi')
    npmi_per_topic = np.asarray(cm_npmi.get_coherence_per_topic(), dtype=np.float64)

    # Fitness analogous to AutoTM avg_coherence_score: mean + min over topics
    fitness = float(np.mean(cv_per_topic) + np.min(cv_per_topic))

    coherence_25 = float(np.mean(cv_per_topic))
    npmi_25 = float(np.mean(npmi_per_topic))
    return fitness, coherence_25, npmi_25


def _compute_switchp(model: LdaModel, dct: Dictionary, tokenized: List[List[str]]) -> Optional[float]:
    try:
        # gensim get_topics: shape (num_topics, vocab_size)
        topics = model.get_topics()  # topics x vocab
        words = [dct[i] for i in range(len(dct))]
        import pandas as _pd
        phi_df = _pd.DataFrame(topics.T, index=words, columns=[f"topic_{i}" for i in range(model.num_topics)])
        texts_raw = [" ".join(toks) for toks in tokenized]
        sp_scores = switchp(phi_df, texts_raw)
        # Remove potential NaNs/inf and empty cases
        sp_arr = np.asarray([s for s in sp_scores if s is not None and np.isfinite(s)], dtype=np.float64)
        return float(np.mean(sp_arr)) if sp_arr.size > 0 else None
    except Exception:
        return None


def _load_dataset(dataset: str, data_path: Optional[str], text_col: str) -> Tuple[List[str], Optional[np.ndarray], str]:
    dataset = dataset.lower()
    if dataset in {"20ng", "20newsgroups"}:
        data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        docs = list(data.data)
        labels = np.array(data.target, dtype=np.int64)
        return docs, labels, "20 Newsgroups (train, headers/footers/quotes removed)"
    else:
        if not data_path:
            raise ValueError("data-path is required for datasets other than 20ng")
        df = pd.read_csv(data_path)
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in CSV. Available: {list(df.columns)}")
        docs = df[text_col].astype(str).tolist()
        # labels unknown for these; return None
        pretty = {
            "amazon_food": "Amazon Fine Food Reviews",
            "hotel_reviews": "Datainfini Hotel Reviews",
            "lenta_ru": "Lenta.ru",
        }.get(dataset, dataset)
        return docs, None, pretty


def run_gensim_lda(dataset: str = "20ng",
                   data_path: Optional[str] = None,
                   text_col: str = "text",
                   num_topics: int = 10,
                   time_budget_s: int = 180,
                   random_seed: int = 42,
                   preproc: str = "auto") -> Result:
    cfg = Config(num_topics=num_topics, time_budget_s=time_budget_s, random_seed=random_seed)

    # Reproducibility
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    docs, labels, dataset_name = _load_dataset(dataset, data_path, text_col)

    # Choose preprocessing
    if preproc == "auto":
        if dataset in {"amazon_food", "hotel_reviews"}:
            preproc = "autotm_en"
        elif dataset in {"lenta_ru"}:
            preproc = "autotm_ru"
        else:
            preproc = "gensim"

    if preproc == "autotm_en":
        tokenized = _tokenize_autotm_en(docs)
    elif preproc == "autotm_ru":
        tokenized = _tokenize_autotm_ru(docs)
    else:
        tokenized = _tokenize(docs)
    if len(tokenized) == 0:
        raise RuntimeError("Tokenization produced no documents. Check preprocessing.")

    dct, corpus = _dictionary_corpus(tokenized, cfg)

    model = LdaModel(
        corpus=None,
        id2word=dct,
        num_topics=cfg.num_topics,
        alpha=cfg.alpha,
        eta=cfg.eta,
        random_state=cfg.random_seed,
        chunksize=cfg.chunksize,
        passes=1,
        iterations=1,
        eval_every=cfg.eval_every,
        minimum_probability=0.0
    )

    start = time.time()
    rounds = 0
    while True:
        elapsed = time.time() - start
        if elapsed >= cfg.time_budget_s:
            break
        model.update(corpus, passes=cfg.passes_per_round, iterations=cfg.iterations_per_round)
        rounds += 1
    wall = time.time() - start

    fitness, coherence_25, npmi_25 = _compute_metrics(model, tokenized, dct, topn=25)
    switchp_avg = _compute_switchp(model, dct, tokenized)

    return Result(
        dataset=dataset_name,
        num_docs=len(corpus),
        vocab_size=len(dct),
        num_topics=cfg.num_topics,
        wall_clock_s=float(wall),
        fitness=fitness,
        coherence_25=coherence_25,
        npmi_25=npmi_25,
        switchP=switchp_avg,
        notes=f"Time-bounded incremental training; rounds={rounds}; passes/iter={cfg.passes_per_round}/{cfg.iterations_per_round}"
    )


def main():
    ap = argparse.ArgumentParser(description="Gensim LDA baseline with AutoTM-like metrics")
    ap.add_argument("--dataset", type=str, default="20ng", choices=["20ng", "20newsgroups", "amazon_food", "hotel_reviews", "lenta_ru"], help="Dataset selector")
    ap.add_argument("--data-path", type=str, default="", help="Path to CSV with texts (required for non-20ng)")
    ap.add_argument("--text-col", type=str, default="text", help="Text column name in CSV (non-20ng)")
    ap.add_argument("--topics", type=int, default=10, help="Number of topics")
    ap.add_argument("--budget", type=int, default=180, help="Time budget in seconds (default 180 = 3 minutes)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--preproc", type=str, default="auto", choices=["auto", "gensim", "autotm_en", "autotm_ru"], help="Preprocessing pipeline")
    ap.add_argument("--out-json", type=str, default="", help="Optional path to save JSON results")
    args = ap.parse_args()

    res = run_gensim_lda(
        dataset=args.dataset,
        data_path=args.data_path or None,
        text_col=args.text_col,
        num_topics=args.topics,
        time_budget_s=args.budget,
        random_seed=args.seed,
        preproc=args.preproc,
    )
    payload = asdict(res)
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj

    payload = convert_numpy_types(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
