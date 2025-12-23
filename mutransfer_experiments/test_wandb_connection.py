#!/usr/bin/env python3
"""Test wandb connection to diagnose segfault issues."""

import sys

print("Testing wandb import...")
try:
    import wandb
    print("✓ wandb imported successfully")
except Exception as e:
    print(f"✗ Failed to import wandb: {e}")
    sys.exit(1)

print("\nTesting wandb API initialization...")
try:
    api = wandb.Api(timeout=30)
    print("✓ wandb API initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize API: {e}")
    sys.exit(1)

print("\nTesting project access...")
project = "klab-shikhar/cod"
try:
    print(f"  Attempting to access: {project}")
    runs = api.runs(project, per_page=1)
    print("✓ Successfully queried runs")
    
    # Try to get first run
    try:
        first_run = next(iter(runs))
        print(f"✓ Got first run: {first_run.name}")
    except StopIteration:
        print("  (No runs found, but query succeeded)")
    except Exception as e:
        print(f"✗ Error getting run: {e}")
        
except Exception as e:
    print(f"✗ Failed to query runs: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed!")

