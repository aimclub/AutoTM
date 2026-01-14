# Baseline Experiments Quick Start

## Prerequisites

Make sure you have Poetry installed and the environment set up:

```bash
# Install dependencies (if not already done)
cd /Users/ngc436/Documents/projects/AutoTM
poetry install

# Install BERTopic dependencies
poetry add "bertopic>=0.16.0" "sentence-transformers>=3.0.0" "umap-learn>=0.5.6" "hdbscan>=0.8.38"

# For LLM evaluation
poetry add openai python-dotenv
```

## LLM Evaluation Setup

Create a `.env` file in the project root:

```bash
# For vLLM with Qwen (or similar)
AUTOTM_LLM_API_KEY=your-api-key-here
AUTOTM_LLM_BASE_URL=http://your-server:8041/v1
AUTOTM_LLM_MODEL_NAME=/model
```

## Running Experiments

### Quick Test (Single Dataset, Few Seeds)

```bash
# Test BERTopic on hotel reviews dataset
poetry run python3 baseline_experiments/run_experiments.py \
  --model bertopic \
  --datasets "hotel:data/hotel_reviews/Datafiniti_Hotel_Reviews.csv:reviews.text" \
  --language-map "hotel:en" \
  --embedding-model "sentence-transformers/all-MiniLM-L6-v2" \
  --seeds 0-2 \
  --grid preset_tiny \
  --output-dir results/bertopic_test \
  --cache-dir cache/bertopic \
  --n-jobs 2
```

### Full Baseline Experiments (All Datasets)

```bash
# Run BERTopic on all three datasets with comprehensive grid
poetry run python3 baseline_experiments/run_experiments.py \
  --model bertopic \
  --datasets "hotel:data/hotel_reviews/Datafiniti_Hotel_Reviews.csv:reviews.text,amazon:data/amazon_food/Reviews.csv:Text,lenta:data/lenta_ru/lenta-ru-news.csv:text" \
  --language-map "hotel:en,amazon:en,lenta:ru" \
  --embedding-model "sentence-transformers/all-MiniLM-L6-v2" \
  --seeds 0-4 \
  --grid preset_small \
  --output-dir results/bertopic_all_datasets \
  --cache-dir cache/bertopic \
  --n-jobs 4 \
  --name bertopic_baseline_all
```

**Note:** This will run ~180 experiments (3 datasets × 5 seeds × 12 configs) and may take several hours.

### Monitoring Progress

While experiments are running, you can monitor progress:

```bash
# Option 1: Use the monitoring script
poetry run python3 baseline_experiments/monitor_experiments.py \
  --results-dir results/bertopic_all_datasets \
  --interval 30

# Option 2: Check the log file
tail -f bertopic_experiments.log

# Option 3: Check partial results
ls -lh results/bertopic_all_datasets/
cat results/bertopic_all_datasets/results_partial.csv | wc -l
```

### Gensim LDA Experiments

For comparison, you can also run Gensim LDA baselines:

```bash
# Single dataset
poetry run python3 baseline_experiments/run_experiments.py \
  --model gensim_lda \
  --dataset hotel_reviews \
  --data-path data/hotel_reviews/Datafiniti_Hotel_Reviews.csv \
  --text-col reviews.text \
  --topics 20 \
  --budget 300 \
  --seeds 0-4 \
  --output-dir results/gensim_hotel \
  --preproc auto
```

## Understanding Results

### BERTopic Output Structure

```
results/bertopic_all_datasets/
├── results.csv              # All runs with full metrics
├── summary.csv              # Aggregated stats (mean±std per config)
└── runs/                    # Per-run artifacts
    ├── hotel__cfg000__seed0.topics.json
    ├── hotel__cfg000__seed0.config.json
    ├── hotel__cfg000__seed0.assignments.csv
    └── ...
```

### Key Metrics

- **Coherence (C_V)**: Semantic coherence of topics (higher is better)
- **Coherence (C_NPMI)**: Normalized pointwise mutual information (higher is better)
- **Topic Diversity**: Uniqueness of top words across topics (higher is better)
- **Runtime**: Execution time in seconds
- **N Topics**: Number of discovered topics
- **Outlier Rate**: % of documents not assigned to any topic (lower is better)

### Analyzing Results

```python
import pandas as pd

# Load results
df = pd.read_csv('results/bertopic_all_datasets/results.csv')

# Best configuration per dataset
for dataset in df['dataset'].unique():
    best = df[df['dataset'] == dataset].nlargest(1, 'coherence_c_v')
    print(f"\n{dataset} - Best Config:")
    print(f"  Coherence: {best['coherence_c_v'].values[0]:.4f}")
    print(f"  Diversity: {best['topic_diversity'].values[0]:.4f}")
    print(f"  Topics: {best['n_topics'].values[0]}")

# Summary statistics
summary = pd.read_csv('results/bertopic_all_datasets/summary.csv')
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(summary)
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`, make sure you're using `poetry run`:

```bash
# ✗ Wrong
python3 baseline_experiments/run_experiments.py ...

# ✓ Correct
poetry run python3 baseline_experiments/run_experiments.py ...
```

### Memory Issues

Large datasets may require more memory. Reduce batch size or use fewer parallel jobs:

```bash
--n-jobs 1  # More deterministic, less memory
```

### Slow Embeddings

Embeddings are cached after first computation. First run per dataset will be slower.

## Next Steps

After running baselines:

1. **Compare with AutoTM**: Use `analyze_results.py` to compare baseline results with AutoTM
2. **Visualize**: Create plots showing coherence/diversity trade-offs
3. **Analyze Topics**: Examine per-topic output to understand discovered themes
4. **Write Up**: Document findings in your research paper/report

## Tips

- Start with `preset_tiny` for quick tests
- Use `preset_small` for balanced exploration
- Reserve `preset_medium` for final comprehensive experiments
- Always use the same seeds across frameworks for fair comparison
- Cache embeddings to save time on repeated runs
