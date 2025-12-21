#!/bin/bash
# Launch CompleteP muTransfer experiments for Thai (tha_thai)
# GPU Assignment: 1
#
# This script runs both width and depth scaling experiments for Thai.
# Can be run in parallel with other language scripts (on different GPUs).
#
# Usage:
#   bash launch_completep_tha.sh [EXPERIMENT_TYPE] [OUT_DIR]
#
# Examples:
#   bash launch_completep_tha.sh both out     # Run both width and depth scaling
#   bash launch_completep_tha.sh width out    # Run only width scaling
#   nohup bash launch_completep_tha.sh both out > logs/tha.log 2>&1 &  # Background

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LANGUAGE="tha_thai"
GPU_ID=1
EXPERIMENT_TYPE=${1:-both}
OUT_DIR=${2:-out}

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  CompleteP muTransfer: Thai (tha_thai)                             ║"
echo "║  GPU: $GPU_ID                                                           ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Start time: $(date)"
echo ""

# Run CompleteP experiments
bash "$SCRIPT_DIR/run_completep.sh" "$LANGUAGE" "$EXPERIMENT_TYPE" "$OUT_DIR" "$GPU_ID"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "Thai (tha_thai) CompleteP experiments complete!"
echo "End time: $(date)"
echo "═══════════════════════════════════════════════════════════════════════"

