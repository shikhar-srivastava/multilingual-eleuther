#!/bin/bash
# Run the complete muTransfer learning rate sweep experiment
#
# This script:
# 1. Runs SP (Standard Parameterization) experiments
# 2. Runs μP (muTransfer) experiments
# 3. Generates the comparison plot

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "muTransfer Learning Rate Sweep Experiment"
echo "========================================"
echo ""
echo "This experiment demonstrates the key property of μP:"
echo "  - SP: Different model widths have different optimal LRs"
echo "  - μP: All model widths share the SAME optimal LR"
echo ""

# Run SP experiments
echo "Step 1/3: Running SP experiments..."
echo "========================================="
cd "$SCRIPT_DIR"
bash sp/run.sh

# Run μP experiments
echo ""
echo "Step 2/3: Running μP experiments..."
echo "========================================="
bash mup/run.sh

# Generate plot
echo ""
echo "Step 3/3: Generating comparison plot..."
echo "========================================="
python plot.py

echo ""
echo "========================================"
echo "Experiment complete!"
echo "========================================"
echo "Results saved to:"
echo "  - SP results: $SCRIPT_DIR/sp/out/"
echo "  - μP results: $SCRIPT_DIR/mup/out/"
echo "  - Plot: $SCRIPT_DIR/mutransfer_lr_shakespeare.png"

