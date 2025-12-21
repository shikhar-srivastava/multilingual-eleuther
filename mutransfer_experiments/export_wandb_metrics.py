#!/usr/bin/env python3
"""
Export wandb Metrics to CSV for muTransfer Plotting

This script downloads training metrics from wandb runs and saves them as CSV files
in the expected directory structure for plot_mutransfer_multilingual.py.

Usage:
    python export_wandb_metrics.py --project YOUR_PROJECT --filter "muP_9M-base_*"
    python export_wandb_metrics.py --project YOUR_PROJECT --filter "muP_9M-base_width-scaling_eng_latn_*"

The script will:
1. Query wandb for runs matching the filter
2. Download loss metrics from each run
3. Save to CSV in the correct directory structure

Requirements:
    pip install wandb pandas
"""

import os
import re
import argparse
import pandas as pd

try:
    import wandb
except ImportError:
    print("Error: wandb not installed. Run: pip install wandb")
    exit(1)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_run_name(run_name):
    """
    Parse run name to extract experiment parameters.
    
    Expected format: muP_9M-base_<scale_type>-scaling_<lang>_w<W>_d<D>_lr<LR>_s<seed>_<param>
    
    Examples:
        muP_9M-base_width-scaling_eng_latn_w256_d4_lr0.001_s1_sp
        muP_9M-base_depth-scaling_tha_thai_w128_d8_lr0.01_s2_completep
    """
    pattern = r'muP_\d+M-base_(\w+)-scaling_(\w+)_w(\d+)_d(\d+)_lr([\d.]+)_s(\d+)_(\w+)'
    match = re.match(pattern, run_name)
    
    if not match:
        return None
    
    return {
        'scale_type': match.group(1),  # width or depth
        'language': match.group(2),
        'width': int(match.group(3)),
        'depth': int(match.group(4)),
        'lr': float(match.group(5)),
        'seed': int(match.group(6)),
        'parameterization': match.group(7),  # sp or completep
    }


def export_run_to_csv(run, base_dir):
    """Export a single run's metrics to CSV."""
    
    run_name = run.name
    params = parse_run_name(run_name)
    
    if params is None:
        print(f"  [SKIP] Cannot parse run name: {run_name}")
        return False
    
    # Determine output path
    param_dir = params['parameterization']
    lang = params['language']
    width = params['width']
    depth = params['depth']
    seed = params['seed']
    lr = params['lr']
    
    out_dir = os.path.join(
        base_dir, param_dir, 'out', lang,
        f'width{width}_depth{depth}_seed{seed}_lr{lr}'
    )
    os.makedirs(out_dir, exist_ok=True)
    
    csv_path = os.path.join(out_dir, 'log.csv')
    
    # Check if already exported
    if os.path.exists(csv_path):
        print(f"  [SKIP] Already exists: {csv_path}")
        return True
    
    # Download metrics
    try:
        history = run.history(keys=['loss', 'eval_loss', 'lr', '_step'])
        
        if history.empty:
            print(f"  [WARN] No history data for: {run_name}")
            return False
        
        # Rename columns to match expected format
        df = history.rename(columns={
            'loss': 'train/loss',
            'eval_loss': 'val/loss',
            '_step': 'iter'
        })
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"  [OK] Exported: {csv_path}")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Failed to export {run_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Export wandb metrics to CSV')
    parser.add_argument('--project', type=str, required=True,
                        help='wandb project name (e.g., "username/project")')
    parser.add_argument('--filter', type=str, default='muP_9M-base_*',
                        help='Filter pattern for run names (supports * wildcard)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory (default: script directory)')
    parser.add_argument('--entity', type=str, default=None,
                        help='wandb entity (default: from project string)')
    args = parser.parse_args()
    
    base_dir = args.out_dir or SCRIPT_DIR
    
    # Parse project/entity
    if '/' in args.project:
        entity, project = args.project.split('/', 1)
    else:
        entity = args.entity
        project = args.project
    
    print("=" * 70)
    print("Exporting wandb Metrics to CSV")
    print("=" * 70)
    print(f"Project: {entity}/{project}" if entity else f"Project: {project}")
    print(f"Filter:  {args.filter}")
    print(f"Output:  {base_dir}")
    print()
    
    # Initialize wandb API
    api = wandb.Api()
    
    # Convert filter to regex
    filter_regex = args.filter.replace('*', '.*')
    
    # Query runs
    print("Querying runs...")
    try:
        if entity:
            runs = api.runs(f"{entity}/{project}")
        else:
            runs = api.runs(project)
    except Exception as e:
        print(f"Error querying wandb: {e}")
        print("Make sure you're logged in: wandb login")
        return
    
    # Filter runs
    matching_runs = []
    for run in runs:
        if re.match(filter_regex, run.name):
            matching_runs.append(run)
    
    print(f"Found {len(matching_runs)} matching runs")
    print()
    
    if not matching_runs:
        print("No runs found matching the filter.")
        print("Check your project name and filter pattern.")
        return
    
    # Export each run
    success = 0
    failed = 0
    skipped = 0
    
    for run in matching_runs:
        result = export_run_to_csv(run, base_dir)
        if result is True:
            success += 1
        elif result is False:
            failed += 1
        else:
            skipped += 1
    
    print()
    print("=" * 70)
    print(f"Export complete: {success} exported, {skipped} skipped, {failed} failed")
    print("=" * 70)
    print()
    print("To generate plots, run:")
    print(f"  python {os.path.join(SCRIPT_DIR, 'plot_mutransfer_multilingual.py')} --out_dir out")


if __name__ == "__main__":
    main()

