#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-based Topic Evaluation Script (Parallelized)

Evaluates topic quality using an LLM (OpenAI-compatible API).
Can be run as a post-processing step after topic modeling experiments.

Usage:
    # Evaluate BERTopic results
    python llm_evaluate.py --results_dir results/bertopic_hotel_full --framework bertopic
    
    # Evaluate Gensim LDA results
    python llm_evaluate.py --results_dir results/gensim_hotel --framework gensim
    
    # Custom settings with parallelization
    python llm_evaluate.py --results_dir results/bertopic_hotel_full \
        --framework bertopic \
        --max_topics 10 \
        --estimations 3 \
        --max_concurrent 10 \
        --output results/bertopic_hotel_full/llm_scores.csv

Environment variables (or use .env file):
    AUTOTM_LLM_API_KEY - API key for LLM service
    AUTOTM_LLM_BASE_URL - Base URL for OpenAI-compatible API
    AUTOTM_LLM_MODEL_NAME - Model name to use
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
from tqdm import tqdm

# Try to load dotenv for .env file support
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Try to import OpenAI client
try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI package not installed. Install with: pip install openai")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment variables
ENV_LLM_API_KEY = "AUTOTM_LLM_API_KEY"
ENV_LLM_BASE_URL = "AUTOTM_LLM_BASE_URL"
ENV_LLM_MODEL_NAME = "AUTOTM_LLM_MODEL_NAME"

# System prompt for topic evaluation
SYSTEM_PROMPT = """You rate topic coherence. Given a list of words from a topic model, rate how semantically related they are on a scale of 1-4:
1 = unrelated words
2 = weakly related  
3 = related
4 = strongly related

IMPORTANT: Reply with ONLY a single digit (1, 2, 3, or 4). No explanation needed."""


def parse_llm_score(score_text: str) -> int:
    """Parse LLM response to extract score (1-4)."""
    # Handle chain-of-thought responses (e.g., Qwen with <think> tags)
    if '</think>' in score_text:
        clean_text = score_text.split('</think>')[-1].strip()
    else:
        clean_text = re.sub(r'<think>.*', '', score_text, flags=re.DOTALL).strip()
    
    if not clean_text:
        clean_text = score_text
    
    # Parse score
    score_match = re.search(r'\b([1-4])\b', clean_text)
    if not score_match:
        score_match = re.search(r'[1-4]', clean_text)
    
    if score_match:
        return int(score_match.group(1) if score_match.lastindex else score_match.group())
    
    logger.debug(f"Could not parse score from: {score_text[:50]}...")
    return 2  # Default neutral


async def evaluate_single_topic_async(
    topic_id: str,
    words: List[str],
    client: AsyncOpenAI,
    model_name: str,
    num_estimations: int,
    semaphore: asyncio.Semaphore,
) -> Tuple[str, float]:
    """
    Evaluate a single topic asynchronously.
    
    Returns:
        Tuple of (topic_id, average_score)
    """
    user_prompt = ", ".join(words[:10])
    scores = []
    
    async with semaphore:
        for _ in range(num_estimations):
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                )
                
                score_text = response.choices[0].message.content.strip()
                scores.append(parse_llm_score(score_text))
                
            except Exception as e:
                logger.warning(f"LLM API error for topic {topic_id}: {e}")
                continue
    
    return topic_id, mean(scores) if scores else 2.0


async def evaluate_topics_async(
    topics: Dict[str, List[str]],
    client: AsyncOpenAI,
    model_name: str,
    max_topics: Optional[int] = None,
    num_estimations: int = 3,
    max_concurrent: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate multiple topics using LLM with parallelization.
    
    Args:
        topics: Dict of topic_id -> list of words
        client: AsyncOpenAI client
        model_name: Model name
        max_topics: Maximum topics to evaluate
        num_estimations: Estimations per topic
        max_concurrent: Maximum concurrent API calls
        seed: Random seed for sampling
        
    Returns:
        Dict with scores and metadata
    """
    # Filter out background/outlier topics
    main_topics = {
        k: v for k, v in topics.items()
        if not str(k).startswith("back") and str(k) != "-1"
    }
    
    if not main_topics:
        return {"llm_score": float("nan"), "topics_evaluated": 0}
    
    # Sample if needed
    topic_ids = list(main_topics.keys())
    if max_topics and len(topic_ids) > max_topics:
        random.seed(seed)
        topic_ids = random.sample(topic_ids, max_topics)
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create tasks for all topics
    tasks = [
        evaluate_single_topic_async(
            tid, main_topics[tid], client, model_name, num_estimations, semaphore
        )
        for tid in topic_ids
    ]
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Collect scores
    topic_scores = {tid: score for tid, score in results}
    
    return {
        "llm_score": mean(topic_scores.values()) if topic_scores else float("nan"),
        "llm_score_min": min(topic_scores.values()) if topic_scores else float("nan"),
        "llm_score_max": max(topic_scores.values()) if topic_scores else float("nan"),
        "topics_evaluated": len(topic_scores),
        "total_topics": len(main_topics),
        "per_topic_scores": topic_scores,
    }


async def evaluate_run_async(
    run_info: Dict[str, Any],
    client: AsyncOpenAI,
    model_name: str,
    max_topics: int,
    num_estimations: int,
    max_concurrent: int,
    runs_dir: Path,
) -> Dict[str, Any]:
    """Evaluate a single run asynchronously."""
    run_id = run_info.get("run_id", "unknown")
    config = run_info.copy()
    
    # Load topics
    topics_file = runs_dir / f"{run_id}.topics.json"
    if not topics_file.exists():
        logger.warning(f"No topics found for {run_id}")
        config["llm_score"] = float("nan")
        return config
    
    with open(topics_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract top words per topic
    topics = {}
    if "top_words" in data:
        for tid, words_scores in data["top_words"].items():
            if tid != "-1":
                topics[tid] = [w for w, _ in words_scores]
    
    if not topics:
        config["llm_score"] = float("nan")
        return config
    
    # Evaluate
    seed = config.get("seed", 42)
    eval_result = await evaluate_topics_async(
        topics, client, model_name, max_topics, num_estimations, max_concurrent, seed
    )
    
    config.update({
        "llm_score": eval_result["llm_score"],
        "llm_score_min": eval_result["llm_score_min"],
        "llm_score_max": eval_result["llm_score_max"],
        "llm_topics_evaluated": eval_result["topics_evaluated"],
    })
    
    # Save updated config
    config_file = runs_dir / f"{run_id}.config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    return config


async def evaluate_bertopic_results_async(
    results_dir: Path,
    client: AsyncOpenAI,
    model_name: str,
    max_topics: int,
    num_estimations: int,
    max_concurrent: int,
) -> pd.DataFrame:
    """Evaluate all BERTopic results in a directory with parallelization."""
    runs_dir = results_dir / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    
    # Find all config files
    config_files = list(runs_dir.glob("*.config.json"))
    if not config_files:
        raise FileNotFoundError(f"No config files found in {runs_dir}")
    
    logger.info(f"Found {len(config_files)} runs to evaluate")
    
    # Load all configs
    run_infos = []
    for config_file in config_files:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
            config["run_id"] = config_file.stem.replace(".config", "")
            run_infos.append(config)
    
    # Process runs with progress bar (runs are sequential, but topics within run are parallel)
    results = []
    for run_info in tqdm(run_infos, desc="Evaluating runs"):
        result = await evaluate_run_async(
            run_info, client, model_name, max_topics, num_estimations, max_concurrent, runs_dir
        )
        results.append(result)
    
    return pd.DataFrame(results)


def load_gensim_topics(results_file: Path) -> Optional[List[Dict[str, Any]]]:
    """Load topics from Gensim LDA results (JSONL format)."""
    if not results_file.exists():
        return None
    
    runs = []
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    return runs if runs else None


async def evaluate_gensim_results_async(
    results_file: Path,
    client: AsyncOpenAI,
    model_name: str,
    max_topics: int,
    num_estimations: int,
    max_concurrent: int,
) -> pd.DataFrame:
    """Evaluate Gensim LDA results from JSONL file with parallelization."""
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    # Load all results
    results = load_gensim_topics(results_file)
    
    if not results:
        raise ValueError(f"No results found in {results_file}")
    
    logger.info(f"Found {len(results)} runs to evaluate")
    
    evaluated = []
    for run in tqdm(results, desc="Evaluating runs"):
        if "topics" not in run:
            run["llm_score"] = float("nan")
            evaluated.append(run)
            continue
        
        # Convert topics to dict format
        topics = {}
        for tid, words in enumerate(run["topics"]):
            topics[str(tid)] = words if isinstance(words, list) else words.split()
        
        # Evaluate with parallelization
        seed = run.get("seed", 42)
        eval_result = await evaluate_topics_async(
            topics, client, model_name, max_topics, num_estimations, max_concurrent, seed
        )
        
        run.update({
            "llm_score": eval_result["llm_score"],
            "llm_score_min": eval_result["llm_score_min"],
            "llm_score_max": eval_result["llm_score_max"],
            "llm_topics_evaluated": eval_result["topics_evaluated"],
        })
        evaluated.append(run)
    
    # Save updated results
    output_file = results_file.parent / f"{results_file.stem}_with_llm.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for run in evaluated:
            f.write(json.dumps(run, ensure_ascii=False) + "\n")
    logger.info(f"Saved results with LLM scores to {output_file}")
    
    return pd.DataFrame(evaluated)


async def main_async(args):
    """Async main function."""
    if not OPENAI_AVAILABLE:
        print("Error: OpenAI package not installed. Install with: pip install openai")
        sys.exit(1)
    
    # Get API credentials
    api_key = os.environ.get(ENV_LLM_API_KEY)
    base_url = os.environ.get(ENV_LLM_BASE_URL)
    model_name = os.environ.get(ENV_LLM_MODEL_NAME, "gpt-4o")
    
    if not api_key:
        print(f"Error: {ENV_LLM_API_KEY} environment variable not set")
        print("Set it in .env file or export it in your shell")
        sys.exit(1)
    
    # Initialize async client
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = AsyncOpenAI(**client_kwargs)
    
    results_dir = Path(args.results_dir)
    
    print(f"\n{'='*60}")
    print("LLM Topic Evaluation (Parallelized)")
    print(f"{'='*60}")
    print(f"Results directory: {results_dir}")
    print(f"Framework: {args.framework}")
    print(f"Max topics per run: {args.max_topics}")
    print(f"Estimations per topic: {args.estimations}")
    print(f"Max concurrent requests: {args.max_concurrent}")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")
    
    try:
        if args.framework == "bertopic":
            df = await evaluate_bertopic_results_async(
                results_dir, client, model_name,
                args.max_topics, args.estimations, args.max_concurrent
            )
        else:  # gensim
            results_file = Path(args.results_file) if args.results_file else None
            if not results_file:
                # Try to find results file
                candidates = list(results_dir.glob("*_all_results.jsonl"))
                if candidates:
                    results_file = candidates[0]
                else:
                    print(f"Error: No results file found. Specify with --results_file")
                    sys.exit(1)
            
            df = await evaluate_gensim_results_async(
                results_file, client, model_name,
                args.max_topics, args.estimations, args.max_concurrent
            )
        
        # Save combined results
        output_file = Path(args.output) if args.output else results_dir / "llm_scores.csv"
        df.to_csv(output_file, index=False)
        
        # Print summary
        print(f"\n{'='*60}")
        print("Evaluation Complete!")
        print(f"{'='*60}")
        print(f"Runs evaluated: {len(df)}")
        if "llm_score" in df.columns:
            valid_scores = df["llm_score"].dropna()
            if len(valid_scores) > 0:
                print(f"LLM Score: {valid_scores.mean():.3f} ± {valid_scores.std():.3f}")
                print(f"  Min: {valid_scores.min():.3f}")
                print(f"  Max: {valid_scores.max():.3f}")
        print(f"\nResults saved to: {output_file}")
        
        # Update main results.csv if it exists
        main_results = results_dir / "results.csv"
        if main_results.exists() and args.framework == "bertopic":
            main_df = pd.read_csv(main_results)
            if "llm_score" not in main_df.columns or main_df["llm_score"].isna().all():
                # Merge LLM scores
                if "run_id" in df.columns:
                    llm_df = df[["run_id", "llm_score", "llm_score_min", "llm_score_max"]].copy()
                    main_df = main_df.merge(llm_df, on="run_id", how="left", suffixes=("", "_new"))
                    # Update columns
                    for col in ["llm_score", "llm_score_min", "llm_score_max"]:
                        if f"{col}_new" in main_df.columns:
                            main_df[col] = main_df[f"{col}_new"]
                            main_df.drop(f"{col}_new", axis=1, inplace=True)
                    main_df.to_csv(main_results, index=False)
                    print(f"Updated: {main_results}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate topic model results using LLM (parallelized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing experiment results")
    parser.add_argument("--framework", type=str, required=True,
                        choices=["bertopic", "gensim"],
                        help="Topic modeling framework used")
    parser.add_argument("--max_topics", type=int, default=10,
                        help="Maximum topics to evaluate per run (default: 10)")
    parser.add_argument("--estimations", type=int, default=3,
                        help="LLM estimations per topic (default: 3)")
    parser.add_argument("--max_concurrent", type=int, default=10,
                        help="Maximum concurrent LLM requests (default: 10)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV file (default: {results_dir}/llm_scores.csv)")
    parser.add_argument("--results_file", type=str, default=None,
                        help="For gensim: specific results JSONL file")
    
    args = parser.parse_args()
    
    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
