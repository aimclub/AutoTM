#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run BERTopic baselines (repeatable) on one or multiple datasets and export:
- per-run metrics (coherence, diversity, runtime, #topics, outlier rate)
- per-topic top words + sizes
- topic assignments

Expected input: CSV/JSONL/Parquet with a text column (default: "text").

Install (example):
  pip install "bertopic>=0.16.0" "sentence-transformers>=3.0.0" "umap-learn>=0.5.6" "hdbscan>=0.8.38" \
              "gensim>=4.3.3" "scikit-learn>=1.4.0" "pandas>=2.2.0" "pyarrow>=15.0.0" "tqdm>=4.66.0"

Example usage:
  python run_bertopic_experiments.py \
    --datasets "20ng:/data/20ng.csv:text,amazon:/data/amazon.csv:review,banners:/data/banners.csv:text,hotel:/data/hotel.csv:text,lenta:/data/lenta.csv:text" \
    --language_map "20ng:en,amazon:en,banners:en,hotel:en,lenta:ru" \
    --embedding_model "sentence-transformers/all-MiniLM-L6-v2" \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --grid preset_small \
    --out_dir results/bertopic_baseline \
    --cache_dir cache/bertopic

Notes:
- For fair repeats, embeddings are cached per dataset+embedding_model.
- Randomness is controlled via UMAP random_state + global seeds; set --n_jobs 1 for best determinism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer

import umap
import hdbscan
from sentence_transformers import SentenceTransformer

from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel

# Try to load dotenv for .env file support
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Try to import OpenAI client
try:
    from openai import OpenAI, AsyncOpenAI
    import asyncio
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

# -----------------------------
# Environment variables for LLM evaluation
# -----------------------------
ENV_LLM_API_KEY = "AUTOTM_LLM_API_KEY"
ENV_LLM_BASE_URL = "AUTOTM_LLM_BASE_URL"
ENV_LLM_MODEL_NAME = "AUTOTM_LLM_MODEL_NAME"
ENV_LLM_MAX_ESTIMATED_TOPICS = "AUTOTM_LLM_MAX_ESTIMATED_TOPICS"
ENV_LLM_ESTIMATIONS_PER_TOPIC = "AUTOTM_LLM_ESTIMATIONS_PER_TOPIC"

# System prompt for LLM topic evaluation
SYSTEM_PROMPT_TOPICS_EVAL = """You rate topic coherence. Given a list of words from a topic model, rate how semantically related they are on a scale of 1-4:
1 = unrelated words
2 = weakly related  
3 = related
4 = strongly related

IMPORTANT: Reply with ONLY a single digit (1, 2, 3, or 4). No explanation needed."""


# -----------------------------
# Basic preprocessing (no downloads)
# -----------------------------

RU_STOPWORDS_MIN = {
    # minimal set; you may extend via --stopwords_file
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
    "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот"
}

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_]+")


def load_stopwords(language: str, stopwords_file: Optional[str] = None) -> set:
    sw = set()
    if language.lower().startswith("en"):
        sw |= set(ENGLISH_STOP_WORDS)
    elif language.lower().startswith("ru"):
        sw |= set(RU_STOPWORDS_MIN)

    if stopwords_file:
        with open(stopwords_file, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w:
                    sw.add(w.lower())
    return sw


def simple_tokenize(text: str, stopwords: set, min_len: int = 3) -> List[str]:
    toks = [t.lower() for t in TOKEN_RE.findall(str(text))]
    toks = [t for t in toks if len(t) >= min_len and t not in stopwords]
    return toks


# -----------------------------
# IO
# -----------------------------

def read_texts(path: str, text_column: str = "text") -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if p.suffix.lower() in [".csv"]:
        df = pd.read_csv(p)
    elif p.suffix.lower() in [".jsonl", ".json"]:
        df = pd.read_json(p, lines=(p.suffix.lower() == ".jsonl"))
    elif p.suffix.lower() in [".parquet"]:
        df = pd.read_parquet(p)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}. Use CSV/JSONL/Parquet.")

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in {path}. Available: {list(df.columns)[:30]}")
    texts = df[text_column].astype(str).tolist()
    return texts


def parse_kv_list(s: str) -> Dict[str, str]:
    """
    Parse "a:x,b:y" into {"a":"x","b":"y"}.
    """
    out: Dict[str, str] = {}
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for part in parts:
        name, val = part.split(":", 1)
        out[name.strip()] = val.strip()
    return out


def parse_datasets(s: str) -> Dict[str, Tuple[str, str]]:
    """
    Parse:
      "20ng:/path/a.csv:text,amazon:/path/b.csv:review"
    -> {"20ng": ("/path/a.csv","text"), "amazon": ("/path/b.csv","review")}
    """
    out: Dict[str, Tuple[str, str]] = {}
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for part in parts:
        seg = part.split(":")
        if len(seg) < 2:
            raise ValueError(f"Bad dataset spec: '{part}' (expected name:path[:column])")
        name = seg[0].strip()
        path = seg[1].strip()
        col = seg[2].strip() if len(seg) >= 3 else "text"
        out[name] = (path, col)
    return out


def stable_hash(items: Sequence[str]) -> str:
    h = hashlib.sha256()
    for it in items:
        h.update(it.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


# -----------------------------
# Metrics
# -----------------------------

def topic_diversity(topic_words: List[List[str]], top_n: int = 10) -> float:
    flat = [w for t in topic_words for w in t[:top_n]]
    if not flat:
        return 0.0
    return len(set(flat)) / float(len(flat))


def compute_coherence(
    tokenized_docs: List[List[str]],
    topics_words: List[List[str]],
    coherence: str = "c_v",
) -> float:
    """
    coherence in {"c_v","c_npmi","u_mass"} etc (gensim).
    """
    if len(tokenized_docs) == 0 or len(topics_words) == 0:
        return float("nan")
    dictionary = Dictionary(tokenized_docs)
    # Filter extremes lightly (optional; keep minimal to avoid altering results too much)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    # Remove empty topics after filtering
    filtered_topics = []
    for tw in topics_words:
        tw2 = [w for w in tw if w in dictionary.token2id]
        if tw2:
            filtered_topics.append(tw2)
    if not filtered_topics:
        return float("nan")

    cm = CoherenceModel(
        topics=filtered_topics,
        texts=tokenized_docs,
        corpus=corpus,
        dictionary=dictionary,
        coherence=coherence,
        processes=1,  # Avoid multiprocessing deadlocks
    )
    return float(cm.get_coherence())


def _parse_llm_score(score_text: str) -> int:
    """Parse LLM response to extract score (1-4)."""
    # Handle chain-of-thought responses (e.g., Qwen with <think> tags)
    if '</think>' in score_text:
        clean_text = score_text.split('</think>')[-1].strip()
    else:
        clean_text = re.sub(r'<think>.*', '', score_text, flags=re.DOTALL).strip()
    
    if not clean_text:
        clean_text = score_text
    
    # Parse score - look for standalone digits 1-4
    score_match = re.search(r'\b([1-4])\b\s*$', clean_text)
    if not score_match:
        score_match = re.search(r'\b([1-4])\b', clean_text)
    if not score_match:
        score_match = re.search(r'[1-4]', clean_text)
    
    if score_match:
        return int(score_match.group(1) if score_match.lastindex else score_match.group())
    
    return 2  # Default neutral score


async def _evaluate_single_topic_async(
    topic_id: str,
    words: List[str],
    client: "AsyncOpenAI",
    model_name: str,
    estimations_per_topic: int,
    semaphore: asyncio.Semaphore,
) -> Tuple[str, float]:
    """Evaluate a single topic asynchronously."""
    user_prompt = ", ".join(words)
    scores = []
    
    async with semaphore:
        for _ in range(estimations_per_topic):
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_TOPICS_EVAL},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                )
                score_text = response.choices[0].message.content.strip()
                scores.append(_parse_llm_score(score_text))
            except Exception as e:
                logger.warning(f"LLM API error for topic {topic_id}: {e}")
                continue
    
    return topic_id, mean(scores) if scores else 2.0


async def _estimate_topics_with_llm_async(
    topics: Dict[str, List[str]],
    api_key: str,
    base_url: Optional[str],
    model_name: str,
    num_top_words: int,
    max_estimated_topics: Optional[int],
    estimations_per_topic: int,
    max_concurrent: int,
    seed: int,
    agg_func: Callable[[Dict[str, float]], float],
) -> float:
    """Async implementation of LLM topic evaluation with parallelization."""
    # Filter topics
    main_topics = {
        tid: words[:num_top_words] 
        for tid, words in topics.items() 
        if not str(tid).startswith("back") and tid != -1
    }
    
    if not main_topics:
        logger.warning("No main topics found for LLM evaluation.")
        return float("nan")
    
    all_main_topics_count = len(main_topics)
    
    # Sample topics if needed
    if max_estimated_topics and len(main_topics) > max_estimated_topics:
        keys = sorted(main_topics.keys(), key=str)
        keys = random.Random(x=seed).sample(keys, k=max_estimated_topics)
        main_topics = {k: main_topics[k] for k in keys}
    
    logger.info(
        f"Evaluating {len(main_topics)}/{all_main_topics_count} topics with LLM "
        f"({estimations_per_topic} estimations each, {max_concurrent} concurrent)"
    )
    
    # Initialize async client
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = AsyncOpenAI(**client_kwargs)
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create tasks for all topics
    tasks = [
        _evaluate_single_topic_async(
            str(topic_id), words, client, model_name, estimations_per_topic, semaphore
        )
        for topic_id, words in main_topics.items()
    ]
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Collect scores
    topics_scores = {tid: score for tid, score in results}
    
    if not topics_scores:
        logger.warning("No topics were successfully evaluated by LLM.")
        return float("nan")
    
    fitness = agg_func(topics_scores)
    total_calls = len(main_topics) * estimations_per_topic
    logger.info(f"LLM evaluation complete: {fitness:.3f} (from {total_calls} API calls)")
    
    return fitness


def estimate_topics_with_llm(
    topics: Dict[str, List[str]],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: str = "gpt-4o",
    num_top_words: int = 10,
    max_estimated_topics: Optional[int] = None,
    estimations_per_topic: int = 3,
    max_concurrent: int = 10,
    seed: int = 42,
    agg_func: Union[str, Callable[[Dict[str, float]], float]] = 'mean',
) -> float:
    """
    Evaluate topic coherence using LLM (parallelized).
    
    Args:
        topics: Dict mapping topic_id -> list of top words
        api_key: OpenAI API key (or from env AUTOTM_LLM_API_KEY)
        base_url: OpenAI API base URL (or from env AUTOTM_LLM_BASE_URL)
        model_name: Model to use (default: gpt-4o, or from env AUTOTM_LLM_MODEL_NAME)
        num_top_words: Number of words to evaluate per topic
        max_estimated_topics: Max topics to evaluate (None = all)
        estimations_per_topic: Number of LLM calls per topic (averaged)
        max_concurrent: Maximum concurrent API requests (default: 10)
        seed: Random seed for topic sampling
        agg_func: Aggregation function ('mean', 'min', 'max') or callable
        
    Returns:
        Average score across topics (1-4 scale)
    """
    if not OPENAI_AVAILABLE:
        logger.warning("OpenAI package not installed. Skipping LLM evaluation.")
        return float("nan")
    
    # Get credentials from environment if not provided
    if api_key is None:
        api_key = os.environ.get(ENV_LLM_API_KEY)
    if base_url is None:
        base_url = os.environ.get(ENV_LLM_BASE_URL)
    if model_name == "gpt-4o":  # Default, check env
        model_name = os.environ.get(ENV_LLM_MODEL_NAME, model_name)
    
    if not api_key:
        logger.warning(f"No API key provided and {ENV_LLM_API_KEY} not set. Skipping LLM evaluation.")
        return float("nan")
    
    # Setup aggregation function
    agg_funcs = {
        'mean': lambda x: mean(x.values()) if len(x) > 0 else 0.0,
        'min': lambda x: min(x.values()) if len(x) > 0 else 0.0,
        'max': lambda x: max(x.values()) if len(x) > 0 else 0.0
    }
    if isinstance(agg_func, str):
        if agg_func not in agg_funcs:
            raise ValueError(f"Unknown agg_func: {agg_func}. Use: {list(agg_funcs.keys())}")
        agg_func = agg_funcs[agg_func]
    
    # Run async evaluation
    return asyncio.run(_estimate_topics_with_llm_async(
        topics=topics,
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        num_top_words=num_top_words,
        max_estimated_topics=max_estimated_topics,
        estimations_per_topic=estimations_per_topic,
        max_concurrent=max_concurrent,
        seed=seed,
        agg_func=agg_func,
    ))


# -----------------------------
# Experiment config
# -----------------------------

@dataclass(frozen=True)
class BERTopicConfig:
    # Vectorizer
    ngram_min: int = 1
    ngram_max: int = 1
    max_features: Optional[int] = 20000
    min_df: int = 5

    # UMAP
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"

    # HDBSCAN
    hdbscan_min_cluster_size: int = 15
    hdbscan_min_samples: Optional[int] = None

    # BERTopic
    nr_topics: Optional[int] = None   # e.g. 50, 100, or None
    top_n_words: int = 10
    calculate_probabilities: bool = False


def preset_grid(name: str) -> List[BERTopicConfig]:
    """
    Grids for BERTopic experiments.
    
    Note: nr_topics=None (automatic) is MUCH faster than nr_topics=50/100.
    Topic reduction with specific nr_topics can take 10-100x longer!
    
    Presets:
    - preset_fast: Only automatic topic discovery (fastest, ~13s/run)
    - preset_tiny: 2 configs for quick tests
    - preset_small: 12 configs with topic reduction (slow!)
    - preset_medium: 27 configs (very slow!)
    """
    if name == "preset_fast":
        # FASTEST: Only nr_topics=None (automatic), varies clustering params
        cfgs = []
        for mcs in [10, 15, 20, 30]:
            for nn in [10, 15, 30]:
                cfgs.append(BERTopicConfig(nr_topics=None, hdbscan_min_cluster_size=mcs, umap_n_neighbors=nn))
        return cfgs  # 12 configs, ~13s each = ~3 min per dataset
    
    if name == "preset_tiny":
        return [
            BERTopicConfig(nr_topics=None, hdbscan_min_cluster_size=15, umap_n_neighbors=15),
            BERTopicConfig(nr_topics=None, hdbscan_min_cluster_size=10, umap_n_neighbors=10),
        ]
    
    if name == "preset_small":
        # WARNING: nr_topics=50/100 triggers slow topic reduction!
        cfgs = []
        for nr_topics in [None, 50, 100]:
            for mcs in [10, 20]:
                for nn in [10, 15]:
                    cfgs.append(BERTopicConfig(nr_topics=nr_topics, hdbscan_min_cluster_size=mcs, umap_n_neighbors=nn))
        return cfgs
    
    if name == "preset_medium":
        cfgs = []
        for nr_topics in [None, 50, 100]:
            for mcs in [10, 20, 30]:
                for nn in [10, 15, 30]:
                    cfgs.append(BERTopicConfig(nr_topics=nr_topics, hdbscan_min_cluster_size=mcs, umap_n_neighbors=nn))
        return cfgs
    
    raise ValueError(f"Unknown grid preset: {name}. Use: preset_fast, preset_tiny, preset_small, preset_medium")


# -----------------------------
# Core run
# -----------------------------

def set_global_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)


def build_model(
    cfg: BERTopicConfig,
    language: str,
    stopwords: set,
    seed: int,
    n_jobs: int,
) -> BERTopic:
    vectorizer = CountVectorizer(
        ngram_range=(cfg.ngram_min, cfg.ngram_max),
        max_features=cfg.max_features,
        min_df=cfg.min_df,
        stop_words=list(stopwords) if stopwords else None,
    )

    umap_model = umap.UMAP(
        n_neighbors=cfg.umap_n_neighbors,
        n_components=cfg.umap_n_components,
        min_dist=cfg.umap_min_dist,
        metric=cfg.umap_metric,
        random_state=seed,
        low_memory=True,
    )

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=cfg.hdbscan_min_cluster_size,
        min_samples=cfg.hdbscan_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False,
        core_dist_n_jobs=n_jobs,
    )

    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=False)

    model = BERTopic(
        language="multilingual",  # robust default; you can set "english" if you want
        vectorizer_model=vectorizer,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        ctfidf_model=ctfidf_model,
        nr_topics=cfg.nr_topics,
        top_n_words=cfg.top_n_words,
        calculate_probabilities=cfg.calculate_probabilities,
        verbose=False,
    )
    return model


def extract_topic_words(model: BERTopic) -> List[List[str]]:
    words: List[List[str]] = []
    info = model.get_topic_info()
    topic_ids = info["Topic"].tolist()
    for tid in topic_ids:
        if tid == -1:
            continue
        t = model.get_topic(tid) or []
        words.append([w for (w, _score) in t])
    return words


def cache_embeddings_path(cache_dir: Path, dataset_name: str, embedding_model: str, texts_hash: str) -> Path:
    key = f"{dataset_name}::{embedding_model}::{texts_hash}"
    fname = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24] + ".npy"
    return cache_dir / "embeddings" / fname


def compute_or_load_embeddings(
    cache_dir: Path,
    dataset_name: str,
    embedding_model_name: str,
    texts: List[str],
    batch_size: int = 64,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "embeddings").mkdir(parents=True, exist_ok=True)

    th = stable_hash(texts[:1000] + [str(len(texts))])  # stable + cheap
    emb_path = cache_embeddings_path(cache_dir, dataset_name, embedding_model_name, th)

    if emb_path.exists():
        return np.load(emb_path)

    encoder = SentenceTransformer(embedding_model_name)
    embs = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    np.save(emb_path, embs)
    return embs


def run_one(
    dataset_name: str,
    texts: List[str],
    language: str,
    embedding_model_name: str,
    embeddings: np.ndarray,
    cfg: BERTopicConfig,
    seed: int,
    stopwords: set,
    n_jobs: int,
    topn_for_metrics: int,
    use_llm: bool = False,
    llm_max_topics: Optional[int] = None,
    llm_estimations_per_topic: int = 3,
) -> Dict[str, Any]:
    set_global_seed(seed)

    t0 = time.time()
    model = build_model(cfg, language=language, stopwords=stopwords, seed=seed, n_jobs=n_jobs)
    topics, _probs = model.fit_transform(texts, embeddings)
    runtime_sec = time.time() - t0

    # Topic stats
    topics_arr = np.asarray(topics)
    outlier_pct = float(np.mean(topics_arr == -1)) * 100.0
    n_topics = int(len(set(topics_arr.tolist())) - (1 if -1 in topics_arr else 0))

    # Topic words (list of lists for coherence calc)
    topic_words = extract_topic_words(model)
    div = topic_diversity(topic_words, top_n=topn_for_metrics)

    # Coherence (tokenize once per run; cheap enough)
    tokenized = [simple_tokenize(t, stopwords=stopwords) for t in texts]
    coh_cv = compute_coherence(tokenized, topic_words, coherence="c_v")
    coh_npmi = compute_coherence(tokenized, topic_words, coherence="c_npmi")

    # LLM-based evaluation (optional)
    llm_score = float("nan")
    if use_llm:
        # Build topics dict for LLM evaluation: topic_id -> list of words
        topics_for_llm = {}
        for tid in model.get_topic_info()["Topic"].tolist():
            if tid == -1:
                continue
            t = model.get_topic(tid) or []
            topics_for_llm[tid] = [w for (w, _score) in t]
        
        if topics_for_llm:
            llm_score = estimate_topics_with_llm(
                topics=topics_for_llm,
                num_top_words=topn_for_metrics,
                max_estimated_topics=llm_max_topics,
                estimations_per_topic=llm_estimations_per_topic,
                seed=seed,
            )

    res = {
        "dataset": dataset_name,
        "language": language,
        "embedding_model": embedding_model_name,
        "seed": seed,
        "runtime_sec": runtime_sec,
        "n_docs": len(texts),
        "n_topics": n_topics,
        "outlier_pct": outlier_pct,
        "topic_diversity_topn": topn_for_metrics,
        "topic_diversity": div,
        "coherence_c_v": coh_cv,
        "coherence_c_npmi": coh_npmi,
        "llm_score": llm_score,
        **{f"cfg_{k}": v for k, v in asdict(cfg).items()},
    }

    # Export per-topic details
    topic_info = model.get_topic_info().to_dict(orient="records")
    topics_export = {
        "topic_info": topic_info,
        "topics": topics_arr.tolist(),
        "top_words": {str(t): model.get_topic(t) for t in model.get_topic_info()["Topic"].tolist() if t != -1},
    }
    return res, topics_export


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=str, required=True,
                    help='Comma-separated: "name:/path/file.csv:textcol,..."')
    ap.add_argument("--language_map", type=str, default="",
                    help='Comma-separated: "name:en,name2:ru,..." (defaults to "en")')
    ap.add_argument("--embedding_model", type=str, required=True,
                    help='SentenceTransformer model name, e.g. "sentence-transformers/all-MiniLM-L6-v2"')
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4,5,6,7,8,9])
    ap.add_argument("--grid", type=str, default="preset_fast",
                    choices=["preset_fast", "preset_tiny", "preset_small", "preset_medium"],
                    help="Grid preset. preset_fast (~13s/run) recommended. preset_small/medium include topic reduction which is VERY slow.")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--cache_dir", type=str, default="cache_bertopic")
    ap.add_argument("--stopwords_file", type=str, default=None,
                    help="Optional file with one stopword per line (added on top of built-ins).")
    ap.add_argument("--max_docs", type=int, default=0,
                    help="If >0, subsample first N docs for quick tests.")
    ap.add_argument("--n_jobs", type=int, default=1,
                    help="Set 1 for better determinism; >1 for speed.")
    ap.add_argument("--topn_for_metrics", type=int, default=10)
    
    # LLM evaluation arguments
    ap.add_argument("--use_llm", action="store_true",
                    help="Enable LLM-based topic evaluation (requires AUTOTM_LLM_API_KEY env var)")
    ap.add_argument("--llm_max_topics", type=int, default=None,
                    help="Max topics to evaluate with LLM (for cost/speed). Default: all topics.")
    ap.add_argument("--llm_estimations", type=int, default=3,
                    help="Number of LLM estimations per topic (default: 3)")

    args = ap.parse_args()

    datasets = parse_datasets(args.datasets)
    lang_map = parse_kv_list(args.language_map)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cfgs = preset_grid(args.grid)
    
    # Print experiment summary
    total_experiments = len(datasets) * len(cfgs) * len(args.seeds)
    print(f"\n{'='*80}", flush=True)
    print(f"BERTopic Baseline Experiments", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Datasets: {list(datasets.keys())}", flush=True)
    print(f"Grid: {args.grid} ({len(cfgs)} configurations)", flush=True)
    print(f"Seeds: {args.seeds} ({len(args.seeds)} seeds)", flush=True)
    print(f"Total experiments: {total_experiments}", flush=True)
    print(f"Output: {out_dir}", flush=True)
    if args.use_llm:
        print(f"LLM Evaluation: ENABLED", flush=True)
        print(f"  - Max topics: {args.llm_max_topics or 'all'}", flush=True)
        print(f"  - Estimations per topic: {args.llm_estimations}", flush=True)
    else:
        print(f"LLM Evaluation: disabled (use --use_llm to enable)", flush=True)
    print(f"{'='*80}\n", flush=True)

    # Run
    all_rows: List[Dict[str, Any]] = []
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_idx = 0
    for dname, (dpath, dcol) in datasets.items():
        dataset_idx += 1
        print(f"\n{'='*80}", flush=True)
        print(f"[DATASET {dataset_idx}/{len(datasets)}] Processing: {dname}", flush=True)
        print(f"  Path: {dpath}", flush=True)
        print(f"  Column: {dcol}", flush=True)
        print(f"{'='*80}\n", flush=True)
        language = lang_map.get(dname, "en")
        stopwords = load_stopwords(language=language, stopwords_file=args.stopwords_file)

        texts = read_texts(dpath, text_column=dcol)
        if args.max_docs and args.max_docs > 0:
            texts = texts[: args.max_docs]

        embeddings = compute_or_load_embeddings(
            cache_dir=cache_dir,
            dataset_name=dname,
            embedding_model_name=args.embedding_model,
            texts=texts,
            batch_size=64,
        )

        total_runs = len(cfgs) * len(args.seeds)
        completed_in_dataset = 0
        skipped_in_dataset = 0
        
        for cfg_i, cfg in enumerate(cfgs):
            for seed in args.seeds:
                run_id = f"{dname}__cfg{cfg_i:03d}__seed{seed}"
                config_file = runs_dir / f"{run_id}.config.json"
                
                # Skip if already completed (for resume capability)
                if config_file.exists():
                    skipped_in_dataset += 1
                    print(f"[SKIP] {run_id} (already completed)", flush=True)
                    # Load existing result for summary
                    with open(config_file, "r", encoding="utf-8") as f:
                        res = json.load(f)
                    all_rows.append(res)
                    continue
                
                completed_in_dataset += 1
                progress = f"[{completed_in_dataset + skipped_in_dataset}/{total_runs}]"
                print(f"{progress} [RUN] {run_id}", flush=True)

                res, topics_export = run_one(
                    dataset_name=dname,
                    texts=texts,
                    language=language,
                    embedding_model_name=args.embedding_model,
                    embeddings=embeddings,
                    cfg=cfg,
                    seed=seed,
                    stopwords=stopwords,
                    n_jobs=args.n_jobs,
                    topn_for_metrics=args.topn_for_metrics,
                    use_llm=args.use_llm,
                    llm_max_topics=args.llm_max_topics,
                    llm_estimations_per_topic=args.llm_estimations,
                )
                res["run_id"] = run_id
                all_rows.append(res)

                # Print metrics immediately
                llm_str = f", llm={res['llm_score']:.2f}" if args.use_llm and not np.isnan(res['llm_score']) else ""
                print(f"        -> coherence={res['coherence_c_v']:.4f}, diversity={res['topic_diversity']:.4f}, "
                      f"topics={res['n_topics']}, time={res['runtime_sec']:.1f}s{llm_str}", flush=True)

                # Save per-run artifacts
                with open(runs_dir / f"{run_id}.topics.json", "w", encoding="utf-8") as f:
                    json.dump(topics_export, f, ensure_ascii=False)

                with open(runs_dir / f"{run_id}.config.json", "w", encoding="utf-8") as f:
                    json.dump(res, f, ensure_ascii=False, indent=2)

                pd.DataFrame({"topic": topics_export["topics"]}).to_csv(
                    runs_dir / f"{run_id}.assignments.csv",
                    index=False,
                )
        
        print(f"\n[DATASET {dname}] Completed: {completed_in_dataset}, Skipped: {skipped_in_dataset}\n", flush=True)

        # Save intermediate results per dataset (safe checkpoints)
        pd.DataFrame(all_rows).to_csv(out_dir / "results_partial.csv", index=False)

    # Final export
    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "results.csv", index=False)

    # Summary table (mean±std across seeds per dataset+cfg)
    group_cols = ["dataset"] + [c for c in df.columns if c.startswith("cfg_")]
    
    # Base aggregations
    agg_dict = {
        "coherence_c_v_mean": ("coherence_c_v", "mean"),
        "coherence_c_v_std": ("coherence_c_v", "std"),
        "coherence_c_npmi_mean": ("coherence_c_npmi", "mean"),
        "coherence_c_npmi_std": ("coherence_c_npmi", "std"),
        "topic_diversity_mean": ("topic_diversity", "mean"),
        "topic_diversity_std": ("topic_diversity", "std"),
        "runtime_sec_mean": ("runtime_sec", "mean"),
        "runtime_sec_std": ("runtime_sec", "std"),
        "n_topics_mean": ("n_topics", "mean"),
        "outlier_pct_mean": ("outlier_pct", "mean"),
        "runs": ("run_id", "count"),
    }
    
    # Add LLM score if present and not all NaN
    if "llm_score" in df.columns and not df["llm_score"].isna().all():
        agg_dict["llm_score_mean"] = ("llm_score", "mean")
        agg_dict["llm_score_std"] = ("llm_score", "std")
    
    summary = df.groupby(group_cols).agg(**agg_dict).reset_index()
    summary.to_csv(out_dir / "summary.csv", index=False)

    print(f"\nDone. Wrote:\n  {out_dir / 'results.csv'}\n  {out_dir / 'summary.csv'}\n  {runs_dir}\n")


if __name__ == "__main__":
    main()
