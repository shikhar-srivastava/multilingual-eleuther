#!/bin/bash
# Launch CompleteP muTransfer experiments for English (eng_latn)
# GPU Assignment: 0
#
# This script runs both width and depth scaling experiments for English.
# Can be run in parallel with other language scripts (on different GPUs).
#
# Usage:
#   bash launch_completep_eng.sh [EXPERIMENT_TYPE] [OUT_DIR]
#
# Examples:
#   bash launch_completep_eng.sh both out     # Run both width and depth scaling
#   bash launch_completep_eng.sh width out    # Run only width scaling
#   nohup bash launch_completep_eng.sh both out > logs/eng.log 2>&1 &  # Background

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LANGUAGE="eng_latn"
GPU_ID=0
EXPERIMENT_TYPE=${1:-both}
OUT_DIR=${2:-out}

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  CompleteP muTransfer: English (eng_latn)                          ║"
echo "║  GPU: $GPU_ID                                                           ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Start time: $(date)"
echo ""

# Run CompleteP experiments
bash "$SCRIPT_DIR/run_completep.sh" "$LANGUAGE" "$EXPERIMENT_TYPE" "$OUT_DIR" "$GPU_ID"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "English (eng_latn) CompleteP experiments complete!"
echo "End time: $(date)"
echo "═══════════════════════════════════════════════════════════════════════"

