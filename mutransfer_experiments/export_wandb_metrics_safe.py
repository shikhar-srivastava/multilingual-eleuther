#!/usr/bin/env python3
"""
Safe version of export_wandb_metrics.py with better error handling and memory management.

This version processes runs one at a time and handles errors gracefully.
"""

import os
import sys
import re
import argparse
import gc

try:
    import pandas as pd
except ImportError:
    print("Error: pandas not installed. Run: pip install pandas")
    sys.exit(1)

try:
    import wandb
except ImportError:
    print("Error: wandb not installed. Run: pip install wandb")
    sys.exit(1)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_run_name(run_name):
    """Parse run name to extract experiment parameters."""
    pattern = r'muP_\d+M-base_(\w+)-scaling_(\w+)_w(\d+)_d(\d+)_lr([\d.]+)_s(\d+)_(\w+)'
    match = re.match(pattern, run_name)
    
    if not match:
        return None
    
    return {
        'scale_type': match.group(1),
        'language': match.group(2),
        'width': int(match.group(3)),
        'depth': int(match.group(4)),
        'lr': float(match.group(5)),
        'seed': int(match.group(6)),
        'parameterization': match.group(7),
    }


def export_single_run(api, project_path, run_id, base_dir, skip_existing=True):
    """Export a single run's metrics to CSV with extensive error handling."""
    
    try:
        # Load run
        run = api.run(f"{project_path}/{run_id}")
        run_name = run.name
        
        params = parse_run_name(run_name)
        if params is None:
            return {'status': 'skipped', 'reason': 'cannot_parse'}
        
        # Build output path
        out_dir = os.path.join(
            base_dir, params['parameterization'], 'out', params['language'],
            f"width{params['width']}_depth{params['depth']}_seed{params['seed']}_lr{params['lr']}"
        )
        os.makedirs(out_dir, exist_ok=True)
        
        csv_path = os.path.join(out_dir, 'log.csv')
        
        # Check if already exists
        if skip_existing and os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            return {'status': 'skipped', 'reason': 'exists'}
        
        # Download history with error handling
        try:
            history = run.history(keys=['loss', 'eval_loss', 'lr', '_step'], samples=50000)
        except Exception as e1:
            try:
                # Fallback without samples limit
                history = run.history(keys=['loss', 'eval_loss', 'lr', '_step'])
            except Exception as e2:
                return {'status': 'failed', 'reason': f'history_error: {e2}'}
        
        if history is None or history.empty:
            return {'status': 'failed', 'reason': 'empty_history'}
        
        # Prepare DataFrame
        rename_map = {}
        if 'loss' in history.columns:
            rename_map['loss'] = 'train/loss'
        if 'eval_loss' in history.columns:
            rename_map['eval_loss'] = 'val/loss'
        if '_step' in history.columns:
            rename_map['_step'] = 'iter'
        
        df = history.rename(columns=rename_map) if rename_map else history.copy()
        
        # Ensure iter column exists
        if 'iter' not in df.columns:
            if '_step' in history.columns:
                df['iter'] = history['_step']
            else:
                df['iter'] = df.index
        
        # Reorder columns
        cols = ['iter']
        for col in ['train/loss', 'val/loss', 'lr']:
            if col in df.columns:
                cols.append(col)
        df = df[cols]
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        
        # Clean up
        del history, df
        gc.collect()
        
        return {'status': 'success', 'rows': len(df)}
        
    except MemoryError:
        gc.collect()
        return {'status': 'failed', 'reason': 'memory_error'}
    except Exception as e:
        return {'status': 'failed', 'reason': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Export wandb metrics to CSV (safe version)')
    parser.add_argument('--project', type=str, required=True,
                        help='wandb project (e.g., "entity/project" or "project")')
    parser.add_argument('--filter', type=str, default='muP_9M-base_*',
                        help='Filter pattern for run names')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory (default: script directory)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of runs to process')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Number of runs to fetch per API call')
    args = parser.parse_args()
    
    base_dir = args.out_dir or SCRIPT_DIR
    
    # Parse project/entity
    if '/' in args.project:
        entity, project = args.project.split('/', 1)
        project_path = f"{entity}/{project}"
    else:
        entity = None
        project = args.project
        project_path = project
    
    print("=" * 70)
    print("Exporting wandb Metrics to CSV (Safe Version)")
    print("=" * 70)
    print(f"Project: {project_path}")
    print(f"Filter:  {args.filter}")
    print(f"Output:  {base_dir}")
    print()
    
    # Initialize API
    try:
        api = wandb.Api(timeout=120)
    except Exception as e:
        print(f"Error initializing wandb API: {e}")
        print("Make sure you're logged in: wandb login")
        sys.exit(1)
    
    # Query runs with pagination
    print("Querying runs...")
    filter_regex = args.filter.replace('*', '.*')
    matching_run_ids = []
    
    try:
        offset = 0
        while True:
            try:
                runs = api.runs(project_path, per_page=args.batch_size, offset=offset)
                batch = list(runs)
                
                if not batch:
                    break
                
                for run in batch:
                    try:
                        if re.match(filter_regex, run.name):
                            matching_run_ids.append(run.id)
                    except:
                        continue
                
                print(f"  Processed {offset + len(batch)} runs, found {len(matching_run_ids)} matching...")
                
                if len(batch) < args.batch_size:
                    break
                
                offset += args.batch_size
                
                # Memory management
                del runs, batch
                gc.collect()
                
            except Exception as e:
                print(f"  [WARN] Error at offset {offset}: {e}")
                break
                
    except Exception as e:
        print(f"Error querying wandb: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\nFound {len(matching_run_ids)} matching runs")
    
    if args.limit:
        matching_run_ids = matching_run_ids[:args.limit]
        print(f"Limited to {len(matching_run_ids)} runs")
    
    print()
    
    # Process runs one at a time
    stats = {'success': 0, 'failed': 0, 'skipped': 0}
    
    for idx, run_id in enumerate(matching_run_ids, 1):
        try:
            result = export_single_run(api, project_path, run_id, base_dir)
            
            if result['status'] == 'success':
                stats['success'] += 1
                print(f"[{idx}/{len(matching_run_ids)}] ✓ Exported {result.get('rows', 0)} rows")
            elif result['status'] == 'skipped':
                stats['skipped'] += 1
                if result['reason'] == 'exists':
                    print(f"[{idx}/{len(matching_run_ids)}] ⊘ Skipped (exists)")
                else:
                    print(f"[{idx}/{len(matching_run_ids)}] ⊘ Skipped ({result['reason']})")
            else:
                stats['failed'] += 1
                print(f"[{idx}/{len(matching_run_ids)}] ✗ Failed: {result.get('reason', 'unknown')}")
            
            # Periodic cleanup
            if idx % 10 == 0:
                gc.collect()
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            break
        except Exception as e:
            stats['failed'] += 1
            print(f"[{idx}/{len(matching_run_ids)}] ✗ Exception: {e}")
            gc.collect()
            continue
    
    print()
    print("=" * 70)
    print(f"Complete: {stats['success']} exported, {stats['skipped']} skipped, {stats['failed']} failed")
    print("=" * 70)


if __name__ == "__main__":
    main()

