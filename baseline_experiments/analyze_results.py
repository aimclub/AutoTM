#!/usr/bin/env python3
"""
Analyze experiment results from JSONL file.
"""
import json
import argparse
from pathlib import Path
import statistics
from typing import List, Dict, Any


def load_results(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load all results from JSONL file"""
    results = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze results and compute statistics"""
    if not results:
        return {}
    
    metrics = ['fitness', 'coherence_25', 'npmi_25', 'switchP', 'wall_clock_s']
    analysis = {
        'num_runs': len(results),
        'seeds': [r['seed'] for r in results],
        'metrics': {}
    }
    
    for metric in metrics:
        values = [r[metric] for r in results if metric in r]
        if values:
            analysis['metrics'][metric] = {
                'mean': statistics.fmean(values),
                'std': statistics.pstdev(values) if len(values) > 1 else 0.0,
                'min': min(values),
                'max': max(values),
                'median': statistics.median(values),
                'values': values
            }
    
    return analysis


def print_analysis(analysis: Dict[str, Any], verbose: bool = False):
    """Print analysis results"""
    print(f"\nAnalysis of {analysis['num_runs']} runs")
    print(f"Seeds: {analysis['seeds']}")
    print(f"\n{'Metric':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Median':<10}")
    print("=" * 70)
    
    for metric, stats in analysis['metrics'].items():
        print(f"{metric:<20} {stats['mean']:<10.4f} {stats['std']:<10.4f} "
              f"{stats['min']:<10.4f} {stats['max']:<10.4f} {stats['median']:<10.4f}")
    
    if verbose:
        print("\n\nDetailed values:")
        for metric, stats in analysis['metrics'].items():
            print(f"\n{metric}:")
            for i, (seed, value) in enumerate(zip(analysis['seeds'], stats['values'])):
                print(f"  Run {i+1} (seed={seed}): {value:.4f}")


def compare_experiments(exp1_path: str, exp2_path: str, exp1_name: str = "Exp 1", exp2_name: str = "Exp 2"):
    """Compare two experiments"""
    results1 = load_results(exp1_path)
    results2 = load_results(exp2_path)
    
    analysis1 = analyze_results(results1)
    analysis2 = analyze_results(results2)
    
    print(f"\nComparison: {exp1_name} vs {exp2_name}")
    print(f"\n{'Metric':<20} {exp1_name:<15} {exp2_name:<15} {'Diff':<10} {'% Change':<10}")
    print("=" * 75)
    
    for metric in analysis1['metrics'].keys():
        if metric in analysis2['metrics']:
            val1 = analysis1['metrics'][metric]['mean']
            val2 = analysis2['metrics'][metric]['mean']
            diff = val2 - val1
            pct_change = (diff / val1 * 100) if val1 != 0 else 0
            
            print(f"{metric:<20} {val1:<15.4f} {val2:<15.4f} {diff:<+10.4f} {pct_change:<+10.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument('results_file', type=str, help='Path to JSONL results file')
    parser.add_argument('--compare', type=str, default=None,
                        help='Path to second JSONL file for comparison')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed values')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Save analysis to JSON file')
    
    args = parser.parse_args()
    
    results = load_results(args.results_file)
    analysis = analyze_results(results)
    
    print_analysis(analysis, verbose=args.verbose)
    
    if args.compare:
        exp1_name = Path(args.results_file).stem.replace('_all_results', '')
        exp2_name = Path(args.compare).stem.replace('_all_results', '')
        compare_experiments(args.results_file, args.compare, exp1_name, exp2_name)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"\nAnalysis saved to {args.output}")


if __name__ == '__main__':
    main()

