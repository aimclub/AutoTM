#!/usr/bin/env python3
"""
Monitor progress of BERTopic experiments in real-time.
"""
import argparse
import time
from pathlib import Path
import pandas as pd


def monitor_progress(results_dir: str, check_interval: int = 30):
    """
    Monitor experiment progress by checking results files.
    
    Args:
        results_dir: Directory containing experiment results
        check_interval: Seconds between checks
    """
    results_path = Path(results_dir)
    runs_dir = results_path / "runs"
    partial_csv = results_path / "results_partial.csv"
    final_csv = results_path / "results.csv"
    
    print(f"Monitoring experiments in: {results_dir}")
    print(f"Checking every {check_interval} seconds...")
    print(f"Press Ctrl+C to stop monitoring\n")
    
    last_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Check for results
            if final_csv.exists():
                df = pd.read_csv(final_csv)
                print(f"\n{'='*80}")
                print(f"✓ EXPERIMENTS COMPLETED!")
                print(f"{'='*80}")
                print(f"Total runs: {len(df)}")
                print(f"Total time: {time.time() - start_time:.1f}s")
                print(f"\nResults saved to: {final_csv}")
                print(f"Summary saved to: {results_path / 'summary.csv'}")
                
                # Show summary by dataset
                if 'dataset' in df.columns:
                    print(f"\n{'='*80}")
                    print("SUMMARY BY DATASET:")
                    print(f"{'='*80}")
                    for dataset in df['dataset'].unique():
                        ds_df = df[df['dataset'] == dataset]
                        print(f"\n{dataset}:")
                        print(f"  Runs: {len(ds_df)}")
                        if 'coherence_c_v' in ds_df.columns:
                            print(f"  Coherence (C_V): {ds_df['coherence_c_v'].mean():.4f} ± {ds_df['coherence_c_v'].std():.4f}")
                        if 'topic_diversity' in ds_df.columns:
                            print(f"  Diversity: {ds_df['topic_diversity'].mean():.4f} ± {ds_df['topic_diversity'].std():.4f}")
                        if 'n_topics' in ds_df.columns:
                            print(f"  Topics: {ds_df['n_topics'].mean():.1f} ± {ds_df['n_topics'].std():.1f}")
                
                break
            
            elif partial_csv.exists():
                df = pd.read_csv(partial_csv)
                current_count = len(df)
                
                if current_count != last_count:
                    elapsed = time.time() - start_time
                    runs_per_sec = current_count / elapsed if elapsed > 0 else 0
                    
                    print(f"\n[{time.strftime('%H:%M:%S')}] Progress Update:")
                    print(f"  Completed: {current_count} runs")
                    print(f"  Elapsed: {elapsed:.1f}s")
                    print(f"  Rate: {runs_per_sec:.2f} runs/sec")
                    
                    if 'dataset' in df.columns:
                        print(f"  By dataset:")
                        for dataset in df['dataset'].unique():
                            ds_count = len(df[df['dataset'] == dataset])
                            print(f"    {dataset}: {ds_count} runs")
                    
                    # Latest metrics
                    if current_count > 0:
                        latest = df.iloc[-1]
                        print(f"  Latest run ({latest.get('run_id', 'N/A')}):")
                        if 'coherence_c_v' in latest:
                            print(f"    Coherence: {latest['coherence_c_v']:.4f}")
                        if 'topic_diversity' in latest:
                            print(f"    Diversity: {latest['topic_diversity']:.4f}")
                        if 'runtime_sec' in latest:
                            print(f"    Runtime: {latest['runtime_sec']:.1f}s")
                    
                    last_count = current_count
            
            elif runs_dir.exists():
                # Count files in runs directory
                run_files = list(runs_dir.glob("*.config.json"))
                current_count = len(run_files)
                
                if current_count != last_count:
                    print(f"\n[{time.strftime('%H:%M:%S')}] {current_count} runs completed...")
                    last_count = current_count
            
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Waiting for experiments to start...")
            
            time.sleep(check_interval)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        if last_count > 0:
            print(f"Last count: {last_count} runs completed")


def main():
    parser = argparse.ArgumentParser(description="Monitor BERTopic experiment progress")
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Results directory to monitor')
    parser.add_argument('--interval', type=int, default=30,
                        help='Check interval in seconds (default: 30)')
    
    args = parser.parse_args()
    monitor_progress(args.results_dir, args.interval)


if __name__ == '__main__':
    main()
