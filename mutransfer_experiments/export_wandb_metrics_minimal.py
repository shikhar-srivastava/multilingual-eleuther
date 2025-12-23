#!/usr/bin/env python3
"""
Minimal wandb export script that uses REST API directly to avoid segfaults.

This version avoids loading all runs into memory and uses a more defensive approach.
"""

import os
import sys
import re
import argparse
import json
import csv

try:
    import requests
except ImportError:
    print("Error: requests not installed. Run: pip install requests")
    sys.exit(1)

# Try to use wandb minimally - only for authentication
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Will need API key from environment.")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_wandb_api_key():
    """Get wandb API key from environment or wandb settings."""
    # Try environment first
    api_key = os.environ.get('WANDB_API_KEY')
    if api_key:
        return api_key
    
    # Try wandb settings file
    if WANDB_AVAILABLE:
        try:
            settings = wandb.Settings()
            api_key = settings.api_key
            if api_key:
                return api_key
        except:
            pass
    
    # Try reading from wandb config
    wandb_dir = os.path.expanduser('~/.netrc')
    if os.path.exists(wandb_dir):
        try:
            with open(wandb_dir, 'r') as f:
                for line in f:
                    if 'api.wandb.ai' in line.lower():
                        # Try to extract key from netrc
                        pass
        except:
            pass
    
    return None


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


def export_using_wandb_cli(run_id, project_path, output_path):
    """Export using wandb CLI sync command as fallback."""
    import subprocess
    
    # Use wandb CLI to download run
    try:
        result = subprocess.run(
            ['wandb', 'sync', '--id', run_id, '--project', project_path],
            capture_output=True,
            timeout=60
        )
        if result.returncode == 0:
            # Try to find the downloaded metrics
            # This is a fallback - may not work perfectly
            return True
    except:
        pass
    return False


def main():
    parser = argparse.ArgumentParser(description='Export wandb metrics (minimal version)')
    parser.add_argument('--project', type=str, required=True,
                        help='wandb project (e.g., "entity/project")')
    parser.add_argument('--filter', type=str, default='muP_9M-base_*',
                        help='Filter pattern for run names')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of runs')
    parser.add_argument('--use_cli', action='store_true',
                        help='Use wandb CLI instead of API (slower but more stable)')
    parser.add_argument('--list', action='store_true',
                        help='List all runs matching filter (don\'t export)')
    parser.add_argument('--show_all', action='store_true',
                        help='Show all runs (ignore filter) to see naming patterns')
    args = parser.parse_args()
    
    base_dir = args.out_dir or SCRIPT_DIR
    
    # Parse project
    if '/' in args.project:
        entity, project = args.project.split('/', 1)
        project_path = f"{entity}/{project}"
    else:
        entity = None
        project = args.project
        project_path = project
    
    print("=" * 70)
    print("Minimal Wandb Export (Avoiding Segfault)")
    print("=" * 70)
    print(f"Project: {project_path}")
    print(f"Filter:  {args.filter}")
    print()
    
    if args.use_cli:
        print("Using wandb CLI approach...")
        print("This requires wandb CLI to be installed and configured.")
        print("Run: pip install wandb && wandb login")
        sys.exit(0)
    
    # Try minimal wandb API usage
    if not WANDB_AVAILABLE:
        print("Error: wandb package required for API access")
        print("Install: pip install wandb")
        sys.exit(1)
    
    print("Testing minimal wandb API access...")
    try:
        # Initialize with minimal settings
        api = wandb.Api(timeout=30)
        print("✓ API initialized")
    except Exception as e:
        print(f"✗ API initialization failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check wandb login: wandb login")
        print("2. Check network connection")
        print("3. Try: export WANDB_MODE=offline")
        sys.exit(1)
    
    # Try to get just the count first
    print("\nQuerying runs (minimal query)...")
    try:
        # Use a very small page size
        runs_iter = api.runs(project_path, per_page=50)
        
        # Get runs to test
        matching_runs = []
        all_runs = []
        count = 0
        filter_regex = args.filter.replace('*', '.*')
        
        print(f"Filter regex: {filter_regex}")
        print("Scanning runs...\n")
        
        for run in runs_iter:
            count += 1
            if count > 500:  # Increased limit
                break
            
            try:
                run_info = {
                    'id': run.id,
                    'name': run.name
                }
                all_runs.append(run_info)
                
                # Check if matches filter
                if args.show_all or re.match(filter_regex, run.name):
                    if args.show_all:
                        # Show all runs with indication of match
                        matches = "✓" if re.match(filter_regex, run.name) else " "
                        print(f"  {matches} {run.name}")
                    else:
                        matching_runs.append(run_info)
                        print(f"  Found: {run.name}")
                    
                    if args.limit and len(matching_runs) >= args.limit and not args.show_all:
                        break
            except Exception as e:
                print(f"  Error processing run: {e}")
                continue
        
        if args.show_all:
            print(f"\nScanned {count} runs total")
            matching_count = sum(1 for r in all_runs if re.match(filter_regex, r['name']))
            print(f"Found {matching_count} runs matching filter: {args.filter}")
            return
        
        print(f"\nFound {len(matching_runs)} matching runs (scanned {count} total)")
        
        if args.list:
            print("\nMatching runs:")
            for run_info in matching_runs:
                print(f"  - {run_info['name']} (ID: {run_info['id']})")
            return
        
    except Exception as e:
        print(f"✗ Query failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nThe segfault may be happening during the query.")
        print("Try using the wandb CLI approach or check wandb installation.")
        sys.exit(1)
    
    if not matching_runs:
        print("No matching runs found.")
        return
    
    print(f"\nProcessing {len(matching_runs)} runs...")
    
    # Now process each run individually
    success = 0
    failed = 0
    
    for idx, run_info in enumerate(matching_runs, 1):
        run_id = run_info['id']
        run_name = run_info['name']
        
        print(f"\n[{idx}/{len(matching_runs)}] {run_name}")
        
        params = parse_run_name(run_name)
        if not params:
            print("  ✗ Cannot parse run name")
            failed += 1
            continue
        
        # Build output path
        out_dir = os.path.join(
            base_dir, params['parameterization'], 'out', params['language'],
            f"width{params['width']}_depth{params['depth']}_seed{params['seed']}_lr{params['lr']}"
        )
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, 'log.csv')
        
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            print("  ⊘ Already exists")
            continue
        
        # Try to get history for this specific run
        try:
            run = api.run(f"{project_path}/{run_id}")
            history = run.history(keys=['loss', 'eval_loss', 'lr', '_step'], samples=10000)
            
            if history is None or history.empty:
                print("  ✗ No history data")
                failed += 1
                continue
            
            # Convert to CSV format
            rows = []
            for _, row in history.iterrows():
                csv_row = {
                    'iter': int(row.get('_step', len(rows))),
                    'train/loss': row.get('loss', ''),
                    'val/loss': row.get('eval_loss', ''),
                    'lr': row.get('lr', '')
                }
                rows.append(csv_row)
            
            # Write CSV
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['iter', 'train/loss', 'val/loss', 'lr'])
                writer.writeheader()
                writer.writerows(rows)
            
            print(f"  ✓ Exported {len(rows)} rows")
            success += 1
            
        except MemoryError:
            print("  ✗ Memory error")
            failed += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Complete: {success} exported, {failed} failed")
    print("=" * 70)


if __name__ == "__main__":
    main()

