#!/bin/bash
# Launch CompleteP muTransfer experiments for Urdu (urd_arab)
# GPU Assignment: 2
#
# This script runs both width and depth scaling experiments for Urdu.
# Can be run in parallel with other language scripts (on different GPUs).
#
# Usage:
#   bash launch_completep_urd.sh [EXPERIMENT_TYPE] [OUT_DIR]
#
# Examples:
#   bash launch_completep_urd.sh both out     # Run both width and depth scaling
#   bash launch_completep_urd.sh width out    # Run only width scaling
#   nohup bash launch_completep_urd.sh both out > logs/urd.log 2>&1 &  # Background

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LANGUAGE="urd_arab"
GPU_ID=2
EXPERIMENT_TYPE=${1:-both}
OUT_DIR=${2:-out}

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  CompleteP muTransfer: Urdu (urd_arab)                             ║"
echo "║  GPU: $GPU_ID                                                           ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Start time: $(date)"
echo ""

# Run CompleteP experiments
bash "$SCRIPT_DIR/run_completep.sh" "$LANGUAGE" "$EXPERIMENT_TYPE" "$OUT_DIR" "$GPU_ID"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "Urdu (urd_arab) CompleteP experiments complete!"
echo "End time: $(date)"
echo "═══════════════════════════════════════════════════════════════════════"

