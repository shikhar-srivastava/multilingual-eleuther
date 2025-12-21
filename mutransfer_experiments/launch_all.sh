#!/bin/bash
# Multilingual muTransfer Width/Depth Scaling Experiments
#
# This script launches the complete muTransfer experiment suite across:
# - Multiple languages (eng, tha, urd, amh, vie)
# - Both parameterizations (SP vs CompleteP)
# - Width and depth scaling experiments
#
# All runs are tracked in wandb and pushed to HuggingFace Hub.
# Run naming convention: muP_9M-base_<scale>-scaling_<lang>_w<W>_d<D>_lr<LR>_s<seed>_<param>
#
# Usage:
#   bash launch_all.sh [EXPERIMENT_TYPE] [PARAMETERIZATION] [OUT_DIR]
#
# Examples:
#   bash launch_all.sh width sp out          # SP width scaling, all languages
#   bash launch_all.sh depth completep out   # CompleteP depth scaling
#   bash launch_all.sh both both out         # All experiments

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parameters
EXPERIMENT_TYPE=${1:-both}     # width | depth | both
PARAMETERIZATION=${2:-both}    # sp | completep | both
OUT_DIR=${3:-out}

# All available languages
LANGUAGES="eng_latn tha_thai urd_arab amh_ethi vie_latn"

echo "========================================================================"
echo "Multilingual muTransfer Scaling Experiments"
echo "========================================================================"
echo ""
echo "This experiment demonstrates the key property of CompleteP:"
echo "  - SP: Optimal LR shifts with model width/depth"
echo "  - CompleteP: Optimal LR is CONSTANT across width/depth"
echo ""
echo "Configuration:"
echo "  Base Model:        9M (width=128, depth=4)"
echo "  Vocabulary Size:   8192"
echo "  Languages:         $LANGUAGES"
echo "  Experiments:       $EXPERIMENT_TYPE"
echo "  Parameterization:  $PARAMETERIZATION"
echo "  Output:            $SCRIPT_DIR/{sp,completep}/$OUT_DIR/"
echo ""
echo "Naming convention: muP_9M-base_<scale>-scaling_<lang>_w<W>_d<D>_lr<LR>_s<seed>_<param>"
echo ""
echo "All runs tracked in wandb and pushed to HuggingFace Hub."
echo "========================================================================"
echo ""

# Track overall progress
total_experiments=0
completed_experiments=0

# Count total experiments
for lang in $LANGUAGES; do
    if [[ "$PARAMETERIZATION" == "sp" || "$PARAMETERIZATION" == "both" ]]; then
        ((total_experiments++))
    fi
    if [[ "$PARAMETERIZATION" == "completep" || "$PARAMETERIZATION" == "both" ]]; then
        ((total_experiments++))
    fi
done

echo "Total experiment batches: $total_experiments"
echo ""

# Run experiments for each language
for lang in $LANGUAGES; do
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║ Language: $lang"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    
    # SP experiments
    if [[ "$PARAMETERIZATION" == "sp" || "$PARAMETERIZATION" == "both" ]]; then
        echo ""
        echo "──────────────────────────────────────────────────────────────────────"
        echo "Running SP experiments for $lang..."
        echo "──────────────────────────────────────────────────────────────────────"
        bash "$SCRIPT_DIR/run_sp.sh" "$lang" "$EXPERIMENT_TYPE" "$OUT_DIR"
        ((completed_experiments++))
        echo "Progress: $completed_experiments / $total_experiments"
    fi
    
    # CompleteP experiments
    if [[ "$PARAMETERIZATION" == "completep" || "$PARAMETERIZATION" == "both" ]]; then
        echo ""
        echo "──────────────────────────────────────────────────────────────────────"
        echo "Running CompleteP experiments for $lang..."
        echo "──────────────────────────────────────────────────────────────────────"
        bash "$SCRIPT_DIR/run_completep.sh" "$lang" "$EXPERIMENT_TYPE" "$OUT_DIR"
        ((completed_experiments++))
        echo "Progress: $completed_experiments / $total_experiments"
    fi
done

echo ""
echo "========================================================================"
echo "All experiments complete!"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo "  - SP:        $SCRIPT_DIR/sp/$OUT_DIR/"
echo "  - CompleteP: $SCRIPT_DIR/completep/$OUT_DIR/"
echo ""
echo "wandb runs: Filter by 'muP_9M-base_*' to see all runs"
echo ""
echo "To export wandb metrics and generate plots:"
echo "  python $SCRIPT_DIR/export_wandb_metrics.py --project YOUR_PROJECT --filter 'muP_9M-base_*'"
echo "  python $SCRIPT_DIR/plot_mutransfer_multilingual.py"
echo ""
