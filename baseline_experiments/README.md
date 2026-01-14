# Baseline Experiments

This folder contains baseline topic modeling experiments with various frameworks for comparison with AutoTM.

## Structure

### Core Experiment Scripts

- **`run_experiments.py`** - Unified experiment runner that supports multiple topic modeling frameworks
  - Supports Gensim LDA and BERTopic baselines
  - Handles multiple seeds, result aggregation, and summary statistics
  - Optional LLM-based topic evaluation
  - See usage examples below

- **`llm_evaluate.py`** - Standalone LLM topic evaluation script
  - Can evaluate saved topic model results using OpenAI-compatible API
  - Works with vLLM, Qwen, GPT-4, and other compatible models

### Framework-Specific Scripts

- **`gensim_lda.py`** - Gensim LDA baseline implementation
- **`run_bertopic.py`** - BERTopic baseline implementation with grid search
- **`topic_model_baselines.py`** - Additional baseline implementations

### Data & Metrics

- **`gensim_data.py`** - Data loading utilities for Gensim experiments
- **`gensim_metrics.py`** - Metrics computation for Gensim models

### Analysis

- **`analyze_results.py`** - Result analysis utilities
- **`autotm_results_analysis.py`** - AutoTM-specific analysis tools

## Usage

### Running Gensim LDA Experiments

```bash
# Run Gensim LDA with multiple seeds
python baseline_experiments/run_experiments.py \
  --model gensim_lda \
  --dataset 20ng \
  --topics 10 \
  --budget 180 \
  --seeds 42-51 \
  --output-dir results/gensim_lda \
  --preproc auto

# Run on custom dataset
python baseline_experiments/run_experiments.py \
  --model gensim_lda \
  --dataset hotel_reviews \
  --data-path data/hotel_reviews/Datafiniti_Hotel_Reviews.csv \
  --text-col text \
  --topics 20 \
  --budget 300 \
  --seeds 0,1,2,3,4 \
  --output-dir results/hotel_gensim
```

### Running BERTopic Experiments

```bash
# Run BERTopic with grid search
python baseline_experiments/run_experiments.py \
  --model bertopic \
  --datasets "20ng:data/20ng.csv:text,hotel:data/hotel_reviews/Datafiniti_Hotel_Reviews.csv:text" \
  --language-map "20ng:en,hotel:en" \
  --embedding-model "sentence-transformers/all-MiniLM-L6-v2" \
  --seeds 0-9 \
  --grid preset_small \
  --output-dir results/bertopic_baseline \
  --cache-dir cache/bertopic \
  --n-jobs 4

# Quick test with tiny grid
python baseline_experiments/run_experiments.py \
  --model bertopic \
  --datasets "hotel:data/hotel_reviews/Datafiniti_Hotel_Reviews.csv:text" \
  --language-map "hotel:en" \
  --embedding-model "sentence-transformers/all-MiniLM-L6-v2" \
  --seeds 0,1,2 \
  --grid preset_tiny \
  --output-dir results/bertopic_test
```

### Direct Framework Calls

You can also run the framework-specific scripts directly for more control:

```bash
# Direct Gensim LDA call
python baseline_experiments/gensim_lda.py \
  --dataset 20ng \
  --topics 10 \
  --budget 180 \
  --seed 42

# Direct BERTopic call
python baseline_experiments/run_bertopic.py \
  --datasets "hotel:data/hotel.csv:text" \
  --language_map "hotel:en" \
  --embedding_model "sentence-transformers/all-MiniLM-L6-v2" \
  --seeds 0 1 2 3 4 \
  --grid preset_small \
  --out_dir results/bertopic \
  --cache_dir cache/bertopic
```

## Output Structure

### Gensim LDA Outputs

```
results/
  {experiment_name}_all_results.jsonl  # One result per line (for each seed)
  {experiment_name}_summary.json        # Aggregated statistics (mean, std, etc.)
```

### BERTopic Outputs

```
results/
  results.csv              # All runs with metrics
  summary.csv              # Aggregated statistics per config
  runs/                    # Per-run artifacts
    {run_id}.topics.json       # Topic information
    {run_id}.config.json       # Configuration
    {run_id}.assignments.csv   # Document-topic assignments
```

## Grid Presets (BERTopic)

- **`preset_tiny`** - 2 configurations (quick test)
- **`preset_small`** - 12 configurations (moderate coverage)
- **`preset_medium`** - 27 configurations (extensive search)

Each preset varies:
- `nr_topics`: None (automatic), 50, 100
- `hdbscan_min_cluster_size`: 10, 20, 30
- `umap_n_neighbors`: 10, 15, 30

## Metrics

Both frameworks compute:
- **Coherence** (C_V, NPMI) - semantic coherence of topics
- **Topic Diversity** - uniqueness of top words across topics
- **Runtime** - execution time
- **Number of Topics** - discovered/specified topics
- **Outlier Rate** (BERTopic) - percentage of documents not assigned to any topic
- **LLM Score** (optional) - LLM-based topic quality rating (1-4 scale)

## LLM Evaluation

Evaluate topic quality using an LLM (OpenAI-compatible API, including vLLM with Qwen).

### Setup

Create a `.env` file with your API credentials:

```bash
# For vLLM with Qwen
AUTOTM_LLM_API_KEY=your-api-key-here
AUTOTM_LLM_BASE_URL=http://your-server:8041/v1
AUTOTM_LLM_MODEL_NAME=/model

# For OpenAI
AUTOTM_LLM_API_KEY=sk-your-openai-key
# AUTOTM_LLM_BASE_URL=  # Not needed for OpenAI
AUTOTM_LLM_MODEL_NAME=gpt-4o
```

### Run with Experiments

```bash
# BERTopic with LLM evaluation
python baseline_experiments/run_experiments.py \
  --model bertopic \
  --datasets "hotel:data/hotel.csv:text" \
  --language-map "hotel:en" \
  --seeds 0-4 \
  --grid preset_fast \
  --output-dir results/bertopic_hotel \
  --llm-evaluate \
  --llm-max-topics 10 \
  --llm-estimations 3
```

### Post-Processing Evaluation

Evaluate already completed experiments:

```bash
# BERTopic results
python baseline_experiments/llm_evaluate.py \
  --results_dir results/bertopic_hotel_full \
  --framework bertopic \
  --max_topics 10 \
  --estimations 3

# Gensim LDA results
python baseline_experiments/llm_evaluate.py \
  --results_dir results/gensim_hotel \
  --framework gensim \
  --results_file results/gensim_hotel/hotel_all_results.jsonl
```

### LLM Score Interpretation

- **1** = Unrelated words (poor topic)
- **2** = Weakly related (marginal topic)
- **3** = Related (good topic)
- **4** = Strongly related (excellent topic)

## Dependencies

Install required packages:

```bash
# For Gensim LDA
pip install gensim scikit-learn

# For BERTopic
pip install "bertopic>=0.16.0" "sentence-transformers>=3.0.0" "umap-learn>=0.5.6" \
            "hdbscan>=0.8.38" "gensim>=4.3.3" "scikit-learn>=1.4.0" \
            "pandas>=2.2.0" "pyarrow>=15.0.0" "tqdm>=4.66.0"
```

## Notes

- BERTopic caches embeddings per dataset+model for reproducibility
- Set `--n-jobs 1` for maximum determinism in BERTopic
- For fair comparisons, use the same preprocessing and metrics across frameworks
