#!/bin/bash
#
# Master script to run all depth ablation experiments for CompleteP
#
# This script tests which architectural components affect depth scaling behavior

set -euo pipefail

echo "================================================================================"
echo "CompleteP Depth Ablation Study"
echo "================================================================================"
echo ""
echo "This will run 5 depth experiments to test architectural components:"
echo ""
echo "  1. Baseline:      RMSNorm + SwiGLU + RoPE (LLAMA standard)"
echo "  2. LayerNorm:     LayerNorm + SwiGLU + RoPE"
echo "  3. GELU:          RMSNorm + GELU + RoPE"
echo "  4. Learned Pos:   RMSNorm + SwiGLU + Learned"
echo "  5. GPT-2-like:    LayerNorm + GELU + Learned"
echo ""
echo "Each experiment tests 6 depths (2, 4, 8, 16, 32, 64) with 3 seeds each."
echo "Total runs: 5 experiments × 6 depths × 3 seeds = 90 runs"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# Change to the project root
cd /localdisk/ssrivas9/multilingual-eleuther

# Experiment 1: Baseline (RMSNorm + SwiGLU + RoPE)
echo "================================================================================"
echo "[1/5] Running Baseline (RMSNorm + SwiGLU + RoPE)..."
echo "================================================================================"
bash coord_check_experiments/depth_scaling/run_completep_baseline.sh

# Experiment 2: LayerNorm
echo ""
echo "================================================================================"
echo "[2/5] Running LayerNorm test (LayerNorm + SwiGLU + RoPE)..."
echo "================================================================================"
bash coord_check_experiments/depth_scaling/run_completep_layernorm.sh

# Experiment 3: GELU
echo ""
echo "================================================================================"
echo "[3/5] Running GELU test (RMSNorm + GELU + RoPE)..."
echo "================================================================================"
bash coord_check_experiments/depth_scaling/run_completep_gelu.sh

# Experiment 4: Learned Position Embeddings
echo ""
echo "================================================================================"
echo "[4/5] Running Learned Pos test (RMSNorm + SwiGLU + Learned)..."
echo "================================================================================"
bash coord_check_experiments/depth_scaling/run_completep_learned_pos.sh

# Experiment 5: GPT-2-like
echo ""
echo "================================================================================"
echo "[5/5] Running GPT-2-like test (LayerNorm + GELU + Learned)..."
echo "================================================================================"
bash coord_check_experiments/depth_scaling/run_completep_gpt2like.sh

# Generate plots
echo ""
echo "================================================================================"
echo "Generating enhanced plots..."
echo "================================================================================"
cd coord_check_experiments
python plot_coord_checks_enhanced.py
cd ..

echo ""
echo "================================================================================"
echo "✅ Depth Ablation Study Complete!"
echo "================================================================================"
echo ""
echo "Results:"
echo "  • Experiments saved in: depth_scaling/completep_*/"
echo "  • Plots saved in: coord_check_experiments/"
echo ""
echo "Key plots to review:"
echo "  - depth_coord_completep_baseline.png - LLAMA standard"
echo "  - depth_coord_completep_ln.png - LayerNorm effect"
echo "  - depth_coord_completep_gelu.png - GELU effect"
echo "  - depth_coord_completep_learned.png - Position embedding effect"
echo "  - depth_coord_completep_gpt2.png - GPT-2-like (combined effects)"
echo ""
echo "Compare these plots to identify which component(s) improve depth scaling."
echo "================================================================================"

