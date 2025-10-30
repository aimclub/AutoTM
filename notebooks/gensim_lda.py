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
import random
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from gensim.models import LdaModel

from gensim_data import (
    dictionary_corpus,
    load_dataset,
    tokenize_autotm_en,
    tokenize_autotm_ru,
    tokenize_gensim,
)
from gensim_metrics import compute_metrics, compute_switchp


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


def run_gensim_lda(
    dataset: str = "20ng",
    data_path: Optional[str] = None,
    text_col: str = "text",
    num_topics: int = 10,
    time_budget_s: int = 180,
    random_seed: int = 42,
    preproc: str = "auto",
) -> Result:
    cfg = Config(num_topics=num_topics, time_budget_s=time_budget_s, random_seed=random_seed)

    # Reproducibility
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    docs, _labels, dataset_name = load_dataset(dataset, data_path, text_col)

    # Choose preprocessing
    if preproc == "auto":
        if dataset in {"amazon_food", "hotel_reviews"}:
            preproc = "autotm_en"
        elif dataset in {"lenta_ru"}:
            preproc = "autotm_ru"
        else:
            preproc = "gensim"

    if preproc == "autotm_en":
        tokenized = tokenize_autotm_en(docs)
    elif preproc == "autotm_ru":
        tokenized = tokenize_autotm_ru(docs)
    else:
        tokenized = tokenize_gensim(docs)

    if len(tokenized) == 0:
        raise RuntimeError("Tokenization produced no documents. Check preprocessing.")

    dct, corpus = dictionary_corpus(
        tokenized,
        keep_n=cfg.keep_n,
        no_below=cfg.no_below,
        no_above=cfg.no_above,
    )

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
        minimum_probability=0.0,
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

    fitness, coherence_25, npmi_25 = compute_metrics(model, tokenized, dct, topn=25)
    switchp_avg = compute_switchp(model, dct, tokenized)

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
        notes=f"Time-bounded incremental training; rounds={rounds}; passes/iter={cfg.passes_per_round}/{cfg.iterations_per_round}",
    )


def main():
    ap = argparse.ArgumentParser(description="Gensim LDA baseline with AutoTM-like metrics")
    ap.add_argument(
        "--dataset",
        type=str,
        default="20ng",
        choices=["20ng", "20newsgroups", "amazon_food", "hotel_reviews", "lenta_ru"],
        help="Dataset selector",
    )
    ap.add_argument("--data-path", type=str, default="", help="Path to CSV with texts (required for non-20ng)")
    ap.add_argument("--text-col", type=str, default="text", help="Text column name in CSV (non-20ng)")
    ap.add_argument("--topics", type=int, default=10, help="Number of topics")
    ap.add_argument("--budget", type=int, default=180, help="Time budget in seconds (default 180 = 3 minutes)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument(
        "--preproc",
        type=str,
        default="auto",
        choices=["auto", "gensim", "autotm_en", "autotm_ru"],
        help="Preprocessing pipeline",
    )
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
        if isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        if hasattr(obj, "item"):
            return obj.item()
        return obj

    payload = convert_numpy_types(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
