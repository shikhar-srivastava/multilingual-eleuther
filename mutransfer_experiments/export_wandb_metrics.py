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


def export_run_to_csv(run, base_dir, skip_existing=True):
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
    
    # Check if already exported (if skip_existing is enabled)
    if skip_existing and os.path.exists(csv_path):
        # Verify file is not empty
        if os.path.getsize(csv_path) > 0:
            print(f"  [SKIP] Already exists: {csv_path}")
            return 'skipped'
    
    # Download metrics with error handling
    try:
        # Use samples parameter to limit data size if needed
        # Download in chunks if the run is very large
        try:
            history = run.history(keys=['loss', 'eval_loss', 'lr', '_step'], samples=50000)
        except Exception:
            # Fallback: try without samples limit
            history = run.history(keys=['loss', 'eval_loss', 'lr', '_step'])
        
        if history is None or history.empty:
            print(f"  [WARN] No history data for: {run_name}")
            return False
        
        # Rename columns to match expected format
        # Handle missing columns gracefully
        rename_map = {}
        if 'loss' in history.columns:
            rename_map['loss'] = 'train/loss'
        if 'eval_loss' in history.columns:
            rename_map['eval_loss'] = 'val/loss'
        if '_step' in history.columns:
            rename_map['_step'] = 'iter'
        
        if rename_map:
            df = history.rename(columns=rename_map)
        else:
            df = history.copy()
        
        # Ensure we have at least 'iter' column
        if 'iter' not in df.columns and '_step' in history.columns:
            df['iter'] = history['_step']
        elif 'iter' not in df.columns:
            # Create iter from index if _step not available
            df['iter'] = df.index
        
        # Reorder columns: iter, train/loss, val/loss, lr
        cols = ['iter']
        if 'train/loss' in df.columns:
            cols.append('train/loss')
        if 'val/loss' in df.columns:
            cols.append('val/loss')
        if 'lr' in df.columns:
            cols.append('lr')
        
        df = df[cols]
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"  [OK] Exported {len(df)} rows: {csv_path}")
        return True
        
    except MemoryError as e:
        print(f"  [ERROR] Memory error exporting {run_name}: {e}")
        print(f"  [HINT] Run may be too large. Try exporting individual runs.")
        return False
    except Exception as e:
        print(f"  [ERROR] Failed to export {run_name}: {e}")
        import traceback
        traceback.print_exc()
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
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of runs to process (for testing)')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='Skip runs that already have log.csv files (default: True)')
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
    try:
        api = wandb.Api(timeout=60)  # Increase timeout for large queries
    except Exception as e:
        print(f"Error initializing wandb API: {e}")
        print("Make sure you're logged in: wandb login")
        return
    
    # Convert filter to regex
    filter_regex = args.filter.replace('*', '.*')
    
    # Query runs with pagination to avoid memory issues
    print("Querying runs...")
    matching_runs = []
    try:
        if entity:
            project_path = f"{entity}/{project}"
        else:
            project_path = project
        
        # Use pagination to avoid loading all runs at once
        # Process in batches to prevent memory issues
        page_size = 50
        offset = 0
        
        while True:
            try:
                runs = api.runs(project_path, per_page=page_size, offset=offset)
                batch = list(runs)
                
                if not batch:
                    break
                
                # Filter runs in this batch
                for run in batch:
                    try:
                        if re.match(filter_regex, run.name):
                            matching_runs.append(run.id)  # Store ID instead of full run object
                    except Exception as e:
                        print(f"  [WARN] Error processing run {run.name}: {e}")
                        continue
                
                print(f"  Processed {offset + len(batch)} runs, found {len(matching_runs)} matching...")
                
                if len(batch) < page_size:
                    break
                
                offset += page_size
                
            except Exception as e:
                print(f"  [WARN] Error fetching batch at offset {offset}: {e}")
                break
        
    except Exception as e:
        print(f"Error querying wandb: {e}")
        print("Make sure you're logged in: wandb login")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\nFound {len(matching_runs)} matching runs")
    print()
    
    if not matching_runs:
        print("No runs found matching the filter.")
        print("Check your project name and filter pattern.")
        return
    
    # Apply limit if specified
    if args.limit:
        matching_runs = matching_runs[:args.limit]
        print(f"Limited to {len(matching_runs)} runs (--limit={args.limit})")
        print()
    
    # Export each run (load one at a time to avoid memory issues)
    success = 0
    failed = 0
    skipped = 0
    
    for idx, run_id in enumerate(matching_runs, 1):
        try:
            # Load run one at a time
            run = api.run(f"{project_path}/{run_id}")
            print(f"[{idx}/{len(matching_runs)}] Processing: {run.name}")
            
            result = export_run_to_csv(run, base_dir, skip_existing=args.skip_existing)
            if result is True:
                success += 1
            elif result == 'skipped':
                skipped += 1
            else:
                failed += 1
            
            # Force garbage collection periodically
            if idx % 10 == 0:
                import gc
                gc.collect()
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"  [ERROR] Failed to process run {run_id}: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
            continue
    
    print()
    print("=" * 70)
    print(f"Export complete: {success} exported, {skipped} skipped, {failed} failed")
    print("=" * 70)
    print()
    print("To generate plots, run:")
    print(f"  python {os.path.join(SCRIPT_DIR, 'plot_mutransfer_multilingual.py')} --out_dir out")


if __name__ == "__main__":
    main()

