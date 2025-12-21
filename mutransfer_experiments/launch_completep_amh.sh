#!/bin/bash
# Launch CompleteP muTransfer experiments for Amharic (amh_ethi)
# GPU Assignment: 3
#
# This script runs both width and depth scaling experiments for Amharic.
# Can be run in parallel with other language scripts (on different GPUs).
#
# Usage:
#   bash launch_completep_amh.sh [EXPERIMENT_TYPE] [OUT_DIR]
#
# Examples:
#   bash launch_completep_amh.sh both out     # Run both width and depth scaling
#   bash launch_completep_amh.sh width out    # Run only width scaling
#   nohup bash launch_completep_amh.sh both out > logs/amh.log 2>&1 &  # Background

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LANGUAGE="amh_ethi"
GPU_ID=3
EXPERIMENT_TYPE=${1:-both}
OUT_DIR=${2:-out}

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  CompleteP muTransfer: Amharic (amh_ethi)                          ║"
echo "║  GPU: $GPU_ID                                                           ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Start time: $(date)"
echo ""

# Run CompleteP experiments
bash "$SCRIPT_DIR/run_completep.sh" "$LANGUAGE" "$EXPERIMENT_TYPE" "$OUT_DIR" "$GPU_ID"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "Amharic (amh_ethi) CompleteP experiments complete!"
echo "End time: $(date)"
echo "═══════════════════════════════════════════════════════════════════════"

