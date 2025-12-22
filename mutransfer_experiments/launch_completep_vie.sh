#!/bin/bash
# Launch CompleteP muTransfer experiments for Vietnamese (vie_latn)
# GPU Assignment: 0 (shares with English - run after English completes, or change GPU_ID)
#
# This script runs both width and depth scaling experiments for Vietnamese.
# NOTE: Uses GPU 0 which is shared with English. Either:
#   1. Run after launch_completep_eng.sh completes, OR
#   2. Change GPU_ID below to an available GPU
#
# Usage:
#   bash launch_completep_vie.sh [EXPERIMENT_TYPE] [OUT_DIR] [GPU_ID] [SINGLE_VALUE]
#
# Examples:
#   bash launch_completep_vie.sh both out           # Run on GPU 0 (default)
#   bash launch_completep_vie.sh both out 1         # Run on GPU 1 instead
#   bash launch_completep_vie.sh depth out 0 4      # Run depth=4 only on GPU 0
#   bash launch_completep_vie.sh depth out 1 8      # Run depth=8 only on GPU 1
#   nohup bash launch_completep_vie.sh depth out 0 4 > logs/vie_d4.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LANGUAGE="vie_latn"
EXPERIMENT_TYPE=${1:-both}
OUT_DIR=${2:-out}
GPU_ID=${3:-0}
SINGLE_VALUE=${4:-}

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  CompleteP muTransfer: Vietnamese (vie_latn)                       ║"
echo "║  GPU: $GPU_ID                                                           ║"
if [ -n "$SINGLE_VALUE" ]; then
echo "║  Single Value: $SINGLE_VALUE                                              ║"
fi
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Start time: $(date)"
echo ""

# Run CompleteP experiments
bash "$SCRIPT_DIR/run_completep.sh" "$LANGUAGE" "$EXPERIMENT_TYPE" "$OUT_DIR" "$GPU_ID" "$SINGLE_VALUE"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "Vietnamese (vie_latn) CompleteP experiments complete!"
echo "End time: $(date)"
echo "═══════════════════════════════════════════════════════════════════════"

