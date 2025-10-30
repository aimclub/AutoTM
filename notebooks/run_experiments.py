#!/usr/bin/env python3
"""
Enhanced experiment runner that saves all iterations, not just the summary.
"""
import json
import subprocess
import sys
import statistics
import time
import os
from pathlib import Path
from typing import List, Dict, Any
import argparse


def extract_json(output: str) -> dict:
    """Extract JSON from script output"""
    i = output.find('{')
    j = output.rfind('}')
    if i == -1 or j == -1 or j <= i:
        raise ValueError('No JSON found in output')
    return json.loads(output[i:j+1])


def run_single_experiment(
    dataset: str,
    data_path: str = None,
    text_col: str = "text",
    topics: int = 10,
    budget: int = 180,
    seed: int = 42,
    preproc: str = "auto"
) -> Dict[str, Any]:
    """Run a single experiment and return results"""
    cmd = [
        'python', 'notebooks/gensim_lda.py',
        '--dataset', dataset,
        '--topics', str(topics),
        '--budget', str(budget),
        '--seed', str(seed),
        '--preproc', preproc
    ]
    
    if data_path:
        cmd.extend(['--data-path', data_path])
    if text_col:
        cmd.extend(['--text-col', text_col])
    
    print(f"Running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running experiment:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Experiment failed with exit code {result.returncode}")
    
    return extract_json(result.stdout)


def run_multiple_experiments(
    dataset: str,
    seeds: List[int],
    output_dir: str = "notebooks",
    data_path: str = None,
    text_col: str = "text",
    topics: int = 10,
    budget: int = 180,
    preproc: str = "auto",
    experiment_name: str = None
) -> Dict[str, Any]:
    """
    Run multiple experiments with different seeds and save all results.
    
    Returns:
        Dictionary with summary statistics and paths to saved files
    """
    if experiment_name is None:
        experiment_name = f"{dataset}_{budget}s"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # File paths
    all_results_path = output_dir / f"{experiment_name}_all_results.jsonl"
    summary_path = output_dir / f"{experiment_name}_summary.json"
    partial_path = output_dir / f"{experiment_name}_partial.json"
    
    # Initialize
    all_results = []
    metrics = {'fitness': [], 'coherence_25': [], 'npmi_25': [], 'switchP': []}
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"Starting experiment: {experiment_name}")
    print(f"Dataset: {dataset}, Budget: {budget}s, Seeds: {seeds}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")
    
    # Run experiments
    for idx, seed in enumerate(seeds, 1):
        print(f"\n[{idx}/{len(seeds)}] Running with seed {seed}...", file=sys.stderr)
        
        try:
            result = run_single_experiment(
                dataset=dataset,
                data_path=data_path,
                text_col=text_col,
                topics=topics,
                budget=budget,
                seed=seed,
                preproc=preproc
            )
            
            # Add metadata
            result['seed'] = seed
            result['run_index'] = idx
            result['timestamp'] = time.time()
            
            # Save to JSONL file (one result per line)
            with open(all_results_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            all_results.append(result)
            
            # Collect metrics
            for key in metrics.keys():
                value = result.get(key)
                if value is not None:
                    metrics[key].append(float(value))
            
            # Update partial results
            elapsed = time.time() - start_time
            partial = {
                'experiment_name': experiment_name,
                'dataset': dataset,
                'topics': topics,
                'budget_s': budget,
                'seeds': seeds[:idx],
                'completed': idx,
                'total': len(seeds),
                'elapsed_s': elapsed,
                'estimated_remaining_s': (elapsed / idx) * (len(seeds) - idx) if idx > 0 else 0,
                'last_seed': seed,
                'last_metrics': {k: metrics[k][-1] if metrics[k] else None for k in metrics},
                'current_mean': {
                    k: float(statistics.fmean(v)) if len(v) > 0 else None
                    for k, v in metrics.items()
                },
                'all_results_file': str(all_results_path),
            }
            
            with open(partial_path, 'w', encoding='utf-8') as f:
                json.dump(partial, f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ Completed: fitness={result.get('fitness', 'N/A'):.4f}, "
                  f"coherence_25={result.get('coherence_25', 'N/A'):.4f}, "
                  f"npmi_25={result.get('npmi_25', 'N/A'):.4f}", file=sys.stderr)
            
        except Exception as e:
            print(f"  ✗ Failed: {e}", file=sys.stderr)
            # Continue with other seeds
            continue
    
    # Calculate final summary statistics
    summary = {
        'experiment_name': experiment_name,
        'dataset': dataset,
        'topics': topics,
        'budget_s': budget,
        'runs': len(all_results),
        'seeds': seeds,
        'completed_seeds': [r['seed'] for r in all_results],
        'total_elapsed_s': time.time() - start_time,
        'summary': {},
        'all_results_file': str(all_results_path),
    }
    
    # Compute statistics for each metric
    for key, values in metrics.items():
        if len(values) == 0:
            summary['summary'][key] = {'mean': None, 'std': None, 'min': None, 'max': None}
        elif len(values) == 1:
            summary['summary'][key] = {
                'mean': values[0],
                'std': 0.0,
                'min': values[0],
                'max': values[0]
            }
        else:
            summary['summary'][key] = {
                'mean': float(statistics.fmean(values)),
                'std': float(statistics.pstdev(values)),  # population std
                'min': float(min(values)),
                'max': float(max(values))
            }
    
    # Save final summary
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Clean up partial file
    if partial_path.exists():
        partial_path.unlink()
    
    print(f"\n{'='*80}")
    print(f"Experiment completed!")
    print(f"Total runs: {len(all_results)}/{len(seeds)}")
    print(f"Total time: {summary['total_elapsed_s']:.1f}s")
    print(f"\nResults saved to:")
    print(f"  - All results: {all_results_path}")
    print(f"  - Summary: {summary_path}")
    print(f"\nSummary statistics:")
    for metric, stats in summary['summary'].items():
        if stats['mean'] is not None:
            print(f"  {metric:15s}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                  f"(min={stats['min']:.4f}, max={stats['max']:.4f})")
    print(f"{'='*80}\n")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run multiple Gensim LDA experiments")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['20ng', '20newsgroups', 'amazon_food', 'hotel_reviews', 'lenta_ru'],
                        help='Dataset to use')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to CSV file (required for non-20ng datasets)')
    parser.add_argument('--text-col', type=str, default='text',
                        help='Text column name in CSV')
    parser.add_argument('--topics', type=int, default=10,
                        help='Number of topics')
    parser.add_argument('--budget', type=int, default=180,
                        help='Time budget in seconds')
    parser.add_argument('--seeds', type=str, default='42-51',
                        help='Seed range (e.g., "42-51") or comma-separated list (e.g., "42,43,44")')
    parser.add_argument('--output-dir', type=str, default='notebooks',
                        help='Output directory for results')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name (default: {dataset}_{budget}s)')
    parser.add_argument('--preproc', type=str, default='auto',
                        choices=['auto', 'gensim', 'autotm_en', 'autotm_ru'],
                        help='Preprocessing pipeline')
    
    args = parser.parse_args()
    
    # Parse seeds
    if '-' in args.seeds:
        start, end = map(int, args.seeds.split('-'))
        seeds = list(range(start, end + 1))
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
    
    # Run experiments
    summary = run_multiple_experiments(
        dataset=args.dataset,
        seeds=seeds,
        output_dir=args.output_dir,
        data_path=args.data_path,
        text_col=args.text_col,
        topics=args.topics,
        budget=args.budget,
        preproc=args.preproc,
        experiment_name=args.name
    )
    
    # Print summary as JSON to stdout
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

