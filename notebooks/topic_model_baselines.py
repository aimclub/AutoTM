# -*- coding: utf-8 -*-
"""
Topic modeling baselines (AutoTM metrics-compatible)
----------------------------------------------------
- Gensim LDA: incremental updates bounded by a wall-clock budget
- MALLET LDA: wrapper around the MALLET Java implementation
- Shared preprocessing/tokenization pipeline and metric computation
- Supports datasets: 20 Newsgroups, Amazon Fine Food Reviews, Datainfini Hotel Reviews, Lenta.ru

Required Libraries:
    pip install gensim>=4.1.2
    pip install scikit-learn>=1.1.1
    pip install numpy
    pip install scipy
    pip install pandas
    pip install nltk  # for AutoTM-like preprocessing (downloads resources on demand)

MALLET requirements:
    Download MALLET and provide the binary path via --mallet-path when using --engine mallet.

CLI examples:
    # Gensim engine (baseline)
    python topic_model_baselines.py --engine gensim --dataset 20ng --topics 10 --budget 180 --seed 42

    # MALLET engine (specify MALLET binary)
    python topic_model_baselines.py --engine mallet --mallet-path /path/to/mallet --dataset 20ng --topics 20 --mallet-iterations 1000

Python API:
    from topic_model_baselines import run_lda
    res = run_lda(engine="gensim", dataset="20ng", num_topics=10, time_budget_s=180, random_seed=42)
"""

import argparse
import json
import random
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

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
class PreprocessConfig:
    keep_n: int = 50000
    no_below: int = 5
    no_above: float = 0.5


@dataclass
class GensimConfig(PreprocessConfig):
    num_topics: int = 10
    time_budget_s: int = 180
    random_seed: int = 42
    chunksize: int = 2000
    passes_per_round: int = 1
    iterations_per_round: int = 20
    eval_every: int = 0
    alpha: str = "auto"
    eta: str = "auto"


@dataclass
class MalletConfig(PreprocessConfig):
    num_topics: int = 10
    random_seed: int = 42
    iterations: int = 1000
    workers: int = 4
    optimize_interval: int = 10
    optimize_burnin: int = 200
    prefix: Optional[str] = None


@dataclass
class Result:
    engine: str
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


def _resolve_preproc(dataset: str, preproc: str) -> str:
    dataset = dataset.lower()
    if preproc == "auto":
        if dataset in {"amazon_food", "hotel_reviews"}:
            return "autotm_en"
        if dataset in {"lenta_ru"}:
            return "autotm_ru"
        return "gensim"
    return preproc


def _tokenize_documents(docs: List[str], strategy: str) -> List[List[str]]:
    if strategy == "autotm_en":
        return tokenize_autotm_en(docs)
    if strategy == "autotm_ru":
        return tokenize_autotm_ru(docs)
    return tokenize_gensim(docs)


def _prepare_corpus(
    dataset: str,
    data_path: Optional[str],
    text_col: str,
    preproc: str,
    keep_n: int,
    no_below: int,
    no_above: float,
) -> Tuple[str, List[List[str]], Tuple]:
    docs, _labels, dataset_name = load_dataset(dataset, data_path, text_col)
    strategy = _resolve_preproc(dataset, preproc)
    tokenized = _tokenize_documents(docs, strategy)
    if not tokenized:
        raise RuntimeError("Tokenization produced no documents. Check preprocessing.")
    dct, corpus = dictionary_corpus(
        tokenized,
        keep_n=keep_n,
        no_below=no_below,
        no_above=no_above,
    )
    return dataset_name, tokenized, (dct, corpus)


def run_gensim_lda(
    dataset: str = "20ng",
    data_path: Optional[str] = None,
    text_col: str = "text",
    num_topics: int = 10,
    time_budget_s: int = 180,
    random_seed: int = 42,
    preproc: str = "auto",
    config: Optional[GensimConfig] = None,
) -> Result:
    cfg = config or GensimConfig(num_topics=num_topics, time_budget_s=time_budget_s, random_seed=random_seed)
    cfg.num_topics = num_topics
    cfg.time_budget_s = time_budget_s
    cfg.random_seed = random_seed

    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    dataset_name, tokenized, (dct, corpus) = _prepare_corpus(
        dataset=dataset,
        data_path=data_path,
        text_col=text_col,
        preproc=preproc,
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
        engine="gensim",
        dataset=dataset_name,
        num_docs=len(corpus),
        vocab_size=len(dct),
        num_topics=cfg.num_topics,
        wall_clock_s=float(wall),
        fitness=fitness,
        coherence_25=coherence_25,
        npmi_25=npmi_25,
        switchP=switchp_avg,
        notes=(
            f"Time-bounded incremental training; rounds={rounds}; "
            f"passes/iter={cfg.passes_per_round}/{cfg.iterations_per_round}"
        ),
    )


def run_mallet_lda(
    dataset: str,
    mallet_path: str,
    data_path: Optional[str] = None,
    text_col: str = "text",
    num_topics: int = 10,
    random_seed: int = 42,
    preproc: str = "auto",
    config: Optional[MalletConfig] = None,
) -> Result:
    if not mallet_path:
        raise ValueError("mallet_path is required when engine='mallet'.")
    mallet_executable = Path(mallet_path).expanduser()
    if not mallet_executable.exists():
        raise FileNotFoundError(f"MALLET binary not found at '{mallet_executable}'")

    cfg = config or MalletConfig(num_topics=num_topics, random_seed=random_seed)
    cfg.num_topics = num_topics
    cfg.random_seed = random_seed

    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    dataset_name, tokenized, (dct, corpus) = _prepare_corpus(
        dataset=dataset,
        data_path=data_path,
        text_col=text_col,
        preproc=preproc,
        keep_n=cfg.keep_n,
        no_below=cfg.no_below,
        no_above=cfg.no_above,
    )

    with tempfile.TemporaryDirectory(prefix="mallet_lda_") as workdir:
        workdir_path = Path(workdir)
        raw_input_path = workdir_path / "input.txt"
        with raw_input_path.open("w", encoding="utf-8") as fout:
            for doc_id, tokens in enumerate(tokenized):
                if not tokens:
                    continue
                text = " ".join(tokens)
                fout.write(f"doc{doc_id}\tdoc{doc_id}\t{text}\n")

        corpus_path = workdir_path / "corpus.mallet"
        import_cmd = [
            str(mallet_executable),
            "import-file",
            "--input",
            str(raw_input_path),
            "--output",
            str(corpus_path),
            "--keep-sequence",
            "--token-regex",
            r"\S+",
            "--encoding",
            "utf-8",
        ]
        subprocess.run(import_cmd, check=True)

        topic_word_path = workdir_path / "topic-word-weights.txt"
        doc_topics_path = workdir_path / "doc-topics.txt"
        state_path = workdir_path / "topic-state.gz"
        keys_path = workdir_path / "topic-keys.txt"

        train_cmd = [
            str(mallet_executable),
            "train-topics",
            "--input",
            str(corpus_path),
            "--num-topics",
            str(cfg.num_topics),
            "--random-seed",
            str(cfg.random_seed),
            "--num-iterations",
            str(cfg.iterations),
            "--optimize-interval",
            str(cfg.optimize_interval),
            "--optimize-burn-in",
            str(cfg.optimize_burnin),
            "--output-topic-keys",
            str(keys_path),
            "--topic-word-weights-file",
            str(topic_word_path),
            "--output-doc-topics",
            str(doc_topics_path),
            "--doc-topics-threshold",
            "0.0",
            "--output-state",
            str(state_path),
        ]
        if cfg.workers and cfg.workers > 1:
            train_cmd.extend(["--num-threads", str(cfg.workers)])

        start = time.time()
        subprocess.run(train_cmd, check=True)
        wall = time.time() - start

        topic_word = np.zeros((cfg.num_topics, len(dct)), dtype=np.float64)
        with topic_word_path.open("r", encoding="utf-8") as f_tw:
            for line in f_tw:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                topic_idx = int(parts[0])
                word = parts[1]
                weight = float(parts[2])
                word_id = dct.token2id.get(word)
                if word_id is None:
                    continue
                topic_word[topic_idx, word_id] = weight

        topic_word = np.where(topic_word < 0, 0.0, topic_word)
        topic_word_sum = topic_word.sum(axis=1, keepdims=True)
        topic_word_sum[topic_word_sum == 0] = 1.0
        topic_word_prob = topic_word / topic_word_sum

        doc_topic = np.zeros((len(tokenized), cfg.num_topics), dtype=np.float64)
        with doc_topics_path.open("r", encoding="utf-8") as f_dt:
            for line in f_dt:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                try:
                    doc_idx = int(parts[0])
                except ValueError:
                    continue
                entries = parts[2:]
                for j in range(0, len(entries) - 1, 2):
                    topic_token, prob_token = entries[j], entries[j + 1]
                    try:
                        topic_idx = int(float(topic_token))
                        prob = float(prob_token)
                    except ValueError:
                        continue
                    if 0 <= topic_idx < cfg.num_topics:
                        doc_topic[doc_idx, topic_idx] = prob

    class _MalletModelProxy:
        def __init__(self, probs: np.ndarray, dictionary):
            self._probs = probs
            self.id2word = dictionary
            self.num_topics = probs.shape[0]

        def get_topics(self):
            return self._probs

        def show_topic(self, topicid: int, topn: int = 10):
            row = self._probs[topicid]
            best = np.argsort(row)[::-1][:topn]
            return [(self.id2word[idx], float(row[idx])) for idx in best]

    proxy_model = _MalletModelProxy(topic_word_prob, dct)
    fitness, coherence_25, npmi_25 = compute_metrics(proxy_model, tokenized, dct, topn=25)
    switchp_avg = compute_switchp(proxy_model, dct, tokenized)

    return Result(
        engine="mallet",
        dataset=dataset_name,
        num_docs=len(corpus),
        vocab_size=len(dct),
        num_topics=cfg.num_topics,
        wall_clock_s=float(wall),
        fitness=fitness,
        coherence_25=coherence_25,
        npmi_25=npmi_25,
        switchP=switchp_avg,
        notes=(
            f"MALLET iterations={cfg.iterations}; optimize_interval={cfg.optimize_interval}; "
            f"optimize_burnin={cfg.optimize_burnin}; workers={cfg.workers}"
        ),
    )


def run_lda(
    engine: str = "gensim",
    dataset: str = "20ng",
    data_path: Optional[str] = None,
    text_col: str = "text",
    num_topics: int = 10,
    time_budget_s: int = 180,
    random_seed: int = 42,
    preproc: str = "auto",
    mallet_path: Optional[str] = None,
    gensim_config: Optional[GensimConfig] = None,
    mallet_config: Optional[MalletConfig] = None,
) -> Result:
    engine = engine.lower()
    if engine not in {"gensim", "mallet"}:
        raise ValueError("engine must be either 'gensim' or 'mallet'")

    if engine == "gensim":
        return run_gensim_lda(
            dataset=dataset,
            data_path=data_path,
            text_col=text_col,
            num_topics=num_topics,
            time_budget_s=time_budget_s,
            random_seed=random_seed,
            preproc=preproc,
            config=gensim_config,
        )

    return run_mallet_lda(
        dataset=dataset,
        mallet_path=mallet_path or "",
        data_path=data_path,
        text_col=text_col,
        num_topics=num_topics,
        random_seed=random_seed,
        preproc=preproc,
        config=mallet_config,
    )


def main():
    ap = argparse.ArgumentParser(description="Topic modeling baselines with shared AutoTM metrics")
    ap.add_argument(
        "--engine",
        type=str,
        default="gensim",
        choices=["gensim", "mallet"],
        help="Select the LDA engine",
    )
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
    ap.add_argument("--budget", type=int, default=180, help="Time budget in seconds (gensim engine)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument(
        "--preproc",
        type=str,
        default="auto",
        choices=["auto", "gensim", "autotm_en", "autotm_ru"],
        help="Preprocessing pipeline",
    )

    # MALLET-specific options
    ap.add_argument("--mallet-path", type=str, default="", help="Path to MALLET binary (required for mallet engine)")
    ap.add_argument("--mallet-iterations", type=int, default=1000, help="Number of MALLET training iterations")
    ap.add_argument("--mallet-workers", type=int, default=4, help="Parallel workers for MALLET (if supported)")
    ap.add_argument(
        "--mallet-optimize-interval",
        type=int,
        default=10,
        help="Interval for MALLET hyperparameter optimisation",
    )
    ap.add_argument(
        "--mallet-optimize-burnin",
        type=int,
        default=200,
        help="Burn-in iterations before MALLET optimisation",
    )
    ap.add_argument(
        "--mallet-prefix",
        type=str,
        default="",
        help="Optional MALLET working directory prefix",
    )

    ap.add_argument("--out-json", type=str, default="", help="Optional path to save JSON results")
    args = ap.parse_args()

    mallet_cfg = MalletConfig(
        num_topics=args.topics,
        random_seed=args.seed,
        iterations=args.mallet_iterations,
        workers=args.mallet_workers,
        optimize_interval=args.mallet_optimize_interval,
        optimize_burnin=args.mallet_optimize_burnin,
        prefix=args.mallet_prefix or None,
    )

    gensim_cfg = GensimConfig(num_topics=args.topics, time_budget_s=args.budget, random_seed=args.seed)

    res = run_lda(
        engine=args.engine,
        dataset=args.dataset,
        data_path=args.data_path or None,
        text_col=args.text_col,
        num_topics=args.topics,
        time_budget_s=args.budget,
        random_seed=args.seed,
        preproc=args.preproc,
        mallet_path=args.mallet_path or None,
        gensim_config=gensim_cfg,
        mallet_config=mallet_cfg,
    )

    payload = asdict(res)

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
