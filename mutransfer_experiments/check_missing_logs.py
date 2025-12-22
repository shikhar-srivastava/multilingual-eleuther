#!/usr/bin/env python3
"""
Check which experiment directories are missing log.csv files.

This helps identify what needs to be exported without using wandb API.
"""

import os
import sys
import argparse
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_missing_logs(base_dir, languages=None):
    """Find all experiment directories missing log.csv files."""
    
    if languages is None:
        languages = ['eng_latn', 'tha_thai', 'urd_arab', 'amh_ethi', 'vie_latn']
    
    missing = []
    existing = []
    
    for param in ['sp', 'completep']:
        param_dir = os.path.join(base_dir, param, 'out')
        if not os.path.exists(param_dir):
            continue
        
        for lang in languages:
            lang_dir = os.path.join(param_dir, lang)
            if not os.path.exists(lang_dir):
                continue
            
            # Find all experiment directories
            exp_dirs = glob.glob(os.path.join(lang_dir, 'width*_depth*_seed*_lr*'))
            
            for exp_dir in exp_dirs:
                csv_path = os.path.join(exp_dir, 'log.csv')
                if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                    existing.append(exp_dir)
                else:
                    missing.append(exp_dir)
    
    return existing, missing


def main():
    parser = argparse.ArgumentParser(description='Check for missing log.csv files')
    parser.add_argument('--base_dir', type=str, default=SCRIPT_DIR,
                        help='Base directory containing sp/ and completep/')
    parser.add_argument('--languages', type=str, nargs='+',
                        default=['eng_latn', 'tha_thai', 'urd_arab', 'amh_ethi'],
                        help='Languages to check')
    parser.add_argument('--export_list', type=str, default=None,
                        help='Export list of missing runs to file (for manual processing)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Checking for missing log.csv files")
    print("=" * 70)
    print(f"Base directory: {args.base_dir}")
    print(f"Languages: {', '.join(args.languages)}")
    print()
    
    existing, missing = find_missing_logs(args.base_dir, args.languages)
    
    print(f"Found {len(existing)} directories WITH log.csv")
    print(f"Found {len(missing)} directories MISSING log.csv")
    print()
    
    if missing:
        print("Missing log.csv files:")
        print("-" * 70)
        for exp_dir in sorted(missing):
            rel_path = os.path.relpath(exp_dir, args.base_dir)
            print(f"  {rel_path}")
        
        if args.export_list:
            with open(args.export_list, 'w') as f:
                for exp_dir in sorted(missing):
                    f.write(f"{exp_dir}\n")
            print(f"\nExported list to: {args.export_list}")
    else:
        print("✓ All directories have log.csv files!")
    
    print()
    print("=" * 70)
    print(f"Summary: {len(existing)} existing, {len(missing)} missing")
    print("=" * 70)
    
    if missing:
        print("\nTo export missing files, try:")
        print("1. Use wandb CLI: wandb sync <run_id>")
        print("2. Use wandb web interface to download metrics")
        print("3. Fix wandb API issues and use export script")


if __name__ == "__main__":
    main()

