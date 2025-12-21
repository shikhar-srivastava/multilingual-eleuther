#!/bin/bash
# Launch ALL CompleteP muTransfer experiments in parallel across 4 GPUs
#
# GPU Assignments:
#   GPU 0: English (eng_latn)
#   GPU 1: Thai (tha_thai)
#   GPU 2: Urdu (urd_arab)
#   GPU 3: Amharic (amh_ethi)
#   GPU 0: Vietnamese (vie_latn) - runs AFTER English completes
#
# Usage:
#   bash launch_all_completep_parallel.sh [EXPERIMENT_TYPE] [OUT_DIR]
#
# Examples:
#   bash launch_all_completep_parallel.sh both out     # Run all experiments
#   bash launch_all_completep_parallel.sh width out    # Width scaling only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPERIMENT_TYPE=${1:-both}
OUT_DIR=${2:-out}

# Create logs directory
LOGS_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGS_DIR"

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  Launching ALL CompleteP muTransfer Experiments in Parallel        ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Experiment Type: $EXPERIMENT_TYPE"
echo "Output Dir:      $OUT_DIR"
echo "Logs Dir:        $LOGS_DIR"
echo ""
echo "GPU Assignments:"
echo "  GPU 0: English (eng_latn) → then Vietnamese (vie_latn)"
echo "  GPU 1: Thai (tha_thai)"
echo "  GPU 2: Urdu (urd_arab)"
echo "  GPU 3: Amharic (amh_ethi)"
echo ""
echo "Start time: $(date)"
echo ""

# Launch first 4 languages in parallel (one per GPU)
echo "Launching 4 languages in parallel..."

# GPU 0: English
nohup bash "$SCRIPT_DIR/launch_completep_eng.sh" "$EXPERIMENT_TYPE" "$OUT_DIR" \
    > "$LOGS_DIR/eng_latn.log" 2>&1 &
PID_ENG=$!
echo "  [GPU 0] English (eng_latn) started - PID: $PID_ENG"

# GPU 1: Thai
nohup bash "$SCRIPT_DIR/launch_completep_tha.sh" "$EXPERIMENT_TYPE" "$OUT_DIR" \
    > "$LOGS_DIR/tha_thai.log" 2>&1 &
PID_THA=$!
echo "  [GPU 1] Thai (tha_thai) started - PID: $PID_THA"

# GPU 2: Urdu
nohup bash "$SCRIPT_DIR/launch_completep_urd.sh" "$EXPERIMENT_TYPE" "$OUT_DIR" \
    > "$LOGS_DIR/urd_arab.log" 2>&1 &
PID_URD=$!
echo "  [GPU 2] Urdu (urd_arab) started - PID: $PID_URD"

# GPU 3: Amharic
nohup bash "$SCRIPT_DIR/launch_completep_amh.sh" "$EXPERIMENT_TYPE" "$OUT_DIR" \
    > "$LOGS_DIR/amh_ethi.log" 2>&1 &
PID_AMH=$!
echo "  [GPU 3] Amharic (amh_ethi) started - PID: $PID_AMH"

echo ""
echo "Waiting for English to complete before starting Vietnamese on GPU 0..."

# Wait for English to complete
wait $PID_ENG
echo "  [GPU 0] English complete! Starting Vietnamese..."

# GPU 0: Vietnamese (after English)
nohup bash "$SCRIPT_DIR/launch_completep_vie.sh" "$EXPERIMENT_TYPE" "$OUT_DIR" 0 \
    > "$LOGS_DIR/vie_latn.log" 2>&1 &
PID_VIE=$!
echo "  [GPU 0] Vietnamese (vie_latn) started - PID: $PID_VIE"

echo ""
echo "Waiting for all remaining experiments to complete..."

# Wait for all remaining processes
wait $PID_THA
echo "  [GPU 1] Thai complete!"
wait $PID_URD
echo "  [GPU 2] Urdu complete!"
wait $PID_AMH
echo "  [GPU 3] Amharic complete!"
wait $PID_VIE
echo "  [GPU 0] Vietnamese complete!"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "All CompleteP experiments complete!"
echo "End time: $(date)"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "Logs saved to: $LOGS_DIR/"
echo ""
echo "To generate plots:"
echo "  python $SCRIPT_DIR/plot_mutransfer_multilingual.py --out_dir $OUT_DIR"

