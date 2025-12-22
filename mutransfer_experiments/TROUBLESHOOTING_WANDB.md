# Troubleshooting Wandb Export Segfault

If you're experiencing segmentation faults when trying to export wandb metrics, try these solutions:

## Quick Diagnosis

1. **Test wandb connection:**
```bash
python mutransfer_experiments/test_wandb_connection.py
```

2. **Check which files are missing:**
```bash
python mutransfer_experiments/check_missing_logs.py
```

## Solutions (in order of preference)

### Solution 1: Clear wandb cache and reinstall

```bash
# Clear wandb cache
rm -rf ~/.wandb

# Reinstall wandb
pip uninstall wandb -y
pip install wandb --upgrade

# Re-login
wandb login
```

### Solution 2: Use minimal export script

```bash
python mutransfer_experiments/export_wandb_metrics_minimal.py \
    --project "klab-shikhar/cod" \
    --filter "muP_9M-base_*" \
    --limit 5  # Test with 5 first
```

### Solution 3: Export by language/experiment type

Instead of exporting everything at once, export in smaller batches:

```bash
# Depth scaling only
python mutransfer_experiments/export_wandb_metrics_minimal.py \
    --project "klab-shikhar/cod" \
    --filter "muP_9M-base_depth-scaling_*" \
    --limit 20

# Width scaling only  
python mutransfer_experiments/export_wandb_metrics_minimal.py \
    --project "klab-shikhar/cod" \
    --filter "muP_9M-base_width-scaling_*" \
    --limit 20
```

### Solution 4: Use wandb CLI (manual but stable)

For each missing run, you can use wandb CLI:

```bash
# First, get list of missing runs
python mutransfer_experiments/check_missing_logs.py --export_list missing_runs.txt

# Then for each run, use wandb CLI to download
wandb sync <run_id> --project klab-shikhar/cod
```

### Solution 5: Export from wandb web interface

1. Go to https://wandb.ai/klab-shikhar/cod
2. For each run, click "Download" → "Metrics (CSV)"
3. Save to the appropriate directory structure

### Solution 6: Use Python with environment variables

Sometimes setting environment variables helps:

```bash
export WANDB_MODE=online
export WANDB_CACHE_DIR=/tmp/wandb_cache
python mutransfer_experiments/export_wandb_metrics_minimal.py \
    --project "klab-shikhar/cod" \
    --filter "muP_9M-base_*" \
    --limit 10
```

## Alternative: Generate plots from existing files

If you have some log.csv files already, you can generate plots for those:

```bash
python mutransfer_experiments/plot_mutransfer_multilingual.py \
    --out_dir out \
    --experiment both \
    --verbose
```

The plotting script will skip directories without log.csv files.

## Check wandb installation

```bash
python -c "import wandb; print(wandb.__version__)"
wandb --version
```

If versions don't match or there are errors, reinstall wandb.

## Memory issues

If the segfault is due to memory:

```bash
# Use ulimit to check memory limits
ulimit -v

# Increase if needed (if you have root)
ulimit -v unlimited
```

## Contact wandb support

If none of these work, the issue might be with wandb itself. Check:
- wandb GitHub issues
- wandb community forum
- wandb support

