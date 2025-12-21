#!/usr/bin/env bash
# Run All Coordinate Check Tests
#
# This script runs all coordinate check experiments to validate muP and CompleteP
# implementations. Run this to verify the implementation is working correctly.
#
# Expected outcomes:
# 1. SP+muP depth check: Activations should DIVERGE as depth increases
# 2. CompleteP depth check: Activations should remain STABLE across depths
# 3. SP width check: Activations should CHANGE with width
# 4. muP width check: Activations should remain STABLE across widths

set -euo pipefail

cd /localdisk/ssrivas9/multilingual-eleuther

# Delete old plots if they exist
echo "Cleaning old plots in coord_check_experiments/..."
rm -f coord_check_experiments/width_coord_*.png
rm -f coord_check_experiments/depth_coord_*.png

echo "=========================================="
echo "Coordinate Check Test Suite"
echo "=========================================="

# First, prepare Shakespeare data if needed
echo ""
echo "[1/5] Preparing Shakespeare data..."
if [ ! -f "data/shakespeare_char/train.bin" ]; then
    echo "Downloading and preparing Shakespeare data..."
    python coord_check_train.py --dataset shakespeare_char --max_iters 0 2>/dev/null || true
fi
echo "Data ready."

# Run depth checks
echo ""
echo "[2/5] Running SP+muP depth check (expect divergence)..."
echo "  This tests muP WITHOUT CompleteP depth scaling"
bash coord_check_experiments/sp_and_mup/run.sh 2>&1 | tail -20
echo ""
echo "[3/5] Running CompleteP depth check (expect stability)..."
echo "  This tests muP WITH CompleteP depth scaling (alpha=1.0)"
bash coord_check_experiments/completep/run.sh 2>&1 | tail -20

echo ""
echo "[4/5] Running CompleteP alpha=0.5 depth check..."
bash coord_check_experiments/depth_alpha_05/run.sh 2>&1 | tail -20

# Run width checks
echo ""
echo "[5/5] Running width scaling checks..."
echo "  SP (expect change with width):"
bash coord_check_experiments/width_scaling/run_sp.sh 2>&1 | tail -10
echo "  muP (expect stability across width):"
bash coord_check_experiments/width_scaling/run_mup.sh 2>&1 | tail -10
echo "  CompleteP (expect stability across width):"
bash coord_check_experiments/width_scaling/run_completep.sh 2>&1 | tail -10

echo ""
echo "=========================================="
echo "All coordinate checks complete!"
echo ""
echo "Generating enhanced plots..."
cd coord_check_experiments
python plot_coord_checks_enhanced.py
cd ..
echo ""
echo "Results saved in:"
echo "  - CSV data: coord_check_experiments/*/out/"
echo "  - Enhanced plots: coord_check_experiments/width_coord_*.png"
echo "  - Enhanced plots: coord_check_experiments/depth_coord_*.png"
echo "=========================================="

