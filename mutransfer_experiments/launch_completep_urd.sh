#!/bin/bash
# Launch CompleteP muTransfer experiments for Urdu (urd_arab)
# GPU Assignment: 2
#
# This script runs both width and depth scaling experiments for Urdu.
# Can be run in parallel with other language scripts (on different GPUs).
#
# Usage:
#   bash launch_completep_urd.sh [EXPERIMENT_TYPE] [OUT_DIR] [GPU_ID] [SINGLE_VALUE]
#
# Examples:
#   bash launch_completep_urd.sh both out           # Run both width and depth scaling
#   bash launch_completep_urd.sh width out          # Run only width scaling
#   bash launch_completep_urd.sh depth out 0 4      # Run depth=4 only on GPU 0
#   bash launch_completep_urd.sh depth out 1 8      # Run depth=8 only on GPU 1
#   nohup bash launch_completep_urd.sh depth out 0 4 > logs/urd_d4.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LANGUAGE="urd_arab"
EXPERIMENT_TYPE=${1:-both}
OUT_DIR=${2:-out}
GPU_ID=${3:-2}
SINGLE_VALUE=${4:-}

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  CompleteP muTransfer: Urdu (urd_arab)                             ║"
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
echo "Urdu (urd_arab) CompleteP experiments complete!"
echo "End time: $(date)"
echo "═══════════════════════════════════════════════════════════════════════"

