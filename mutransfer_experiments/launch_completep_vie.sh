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
#   bash launch_completep_vie.sh [EXPERIMENT_TYPE] [OUT_DIR] [GPU_ID]
#
# Examples:
#   bash launch_completep_vie.sh both out       # Run on GPU 0 (default)
#   bash launch_completep_vie.sh both out 1     # Run on GPU 1 instead
#   nohup bash launch_completep_vie.sh both out > logs/vie.log 2>&1 &  # Background

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LANGUAGE="vie_latn"
EXPERIMENT_TYPE=${1:-both}
OUT_DIR=${2:-out}
GPU_ID=${3:-0}  # Default GPU 0, but can be overridden

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  CompleteP muTransfer: Vietnamese (vie_latn)                       ║"
echo "║  GPU: $GPU_ID                                                           ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Start time: $(date)"
echo ""

# Run CompleteP experiments
bash "$SCRIPT_DIR/run_completep.sh" "$LANGUAGE" "$EXPERIMENT_TYPE" "$OUT_DIR" "$GPU_ID"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "Vietnamese (vie_latn) CompleteP experiments complete!"
echo "End time: $(date)"
echo "═══════════════════════════════════════════════════════════════════════"

