#!/bin/bash
# Launch muTransfer Experiments for a Single Language
#
# This is a convenience script to run both SP and CompleteP experiments
# for a single language with all experiment types.
#
# All runs are tracked in wandb and pushed to HuggingFace Hub.
#
# Usage:
#   bash launch_single_language.sh <LANGUAGE> [EXPERIMENT_TYPE] [OUT_DIR]
#
# LANGUAGE: eng_latn | tha_thai | urd_arab | amh_ethi | vie_latn
# EXPERIMENT_TYPE: width | depth | both (default: both)
# OUT_DIR: out | out_test (default: out)
#
# Examples:
#   bash launch_single_language.sh eng_latn           # Full experiments for English
#   bash launch_single_language.sh tha_thai width     # Width scaling only for Thai
#   bash launch_single_language.sh amh_ethi both out  # Full experiments for Amharic

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
LANGUAGE=${1:?Error: Language is required. Use: eng_latn | tha_thai | urd_arab | amh_ethi | vie_latn}
EXPERIMENT_TYPE=${2:-both}
OUT_DIR=${3:-out}

# Validate language
case "$LANGUAGE" in
    eng_latn|tha_thai|urd_arab|amh_ethi|vie_latn)
        ;;
    *)
        echo "Error: Invalid language '$LANGUAGE'"
        echo "Valid options: eng_latn | tha_thai | urd_arab | amh_ethi | vie_latn"
        exit 1
        ;;
esac

# Display friendly language names
declare -A LANG_NAMES=(
    ["eng_latn"]="English (Latin)"
    ["tha_thai"]="Thai (Thai)"
    ["urd_arab"]="Urdu (Arabic)"
    ["amh_ethi"]="Amharic (Ethiopic)"
    ["vie_latn"]="Vietnamese (Latin)"
)

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  muTransfer Experiments: ${LANG_NAMES[$LANGUAGE]}"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Language:    $LANGUAGE"
echo "  Experiment:  $EXPERIMENT_TYPE"
echo "  Output:      $OUT_DIR"
echo "  Base Model:  9M (width=128, depth=4)"
echo "  Vocab Size:  8192"
echo ""
echo "Run naming: muP_9M-base_<scale>-scaling_${LANGUAGE}_w<W>_d<D>_lr<LR>_s<seed>_<param>"
echo ""

# Run SP experiments
echo ""
echo "┌─────────────────────────────────────────────────────────────────────┐"
echo "│ Running Standard Parameterization (SP) experiments...              │"
echo "└─────────────────────────────────────────────────────────────────────┘"
bash "$SCRIPT_DIR/run_sp.sh" "$LANGUAGE" "$EXPERIMENT_TYPE" "$OUT_DIR"

# Run CompleteP experiments
echo ""
echo "┌─────────────────────────────────────────────────────────────────────┐"
echo "│ Running CompleteP (μP + Depth Scaling) experiments...              │"
echo "└─────────────────────────────────────────────────────────────────────┘"
bash "$SCRIPT_DIR/run_completep.sh" "$LANGUAGE" "$EXPERIMENT_TYPE" "$OUT_DIR"

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║ Experiments Complete: ${LANG_NAMES[$LANGUAGE]}"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to:"
echo "  - SP:        $SCRIPT_DIR/sp/$OUT_DIR/$LANGUAGE/"
echo "  - CompleteP: $SCRIPT_DIR/completep/$OUT_DIR/$LANGUAGE/"
echo ""
echo "wandb runs: Filter by 'muP_9M-base_*_${LANGUAGE}_*'"
echo ""
echo "To export metrics and generate plots:"
echo "  python $SCRIPT_DIR/export_wandb_metrics.py --project YOUR_PROJECT --filter 'muP_9M-base_*_${LANGUAGE}_*'"
echo "  python $SCRIPT_DIR/plot_mutransfer_multilingual.py --out_dir $OUT_DIR --languages $LANGUAGE"
echo ""
