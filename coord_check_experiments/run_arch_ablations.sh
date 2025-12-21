#!/bin/bash
#
# Master script to run all architecture ablation experiments for CompleteP
#
# This script tests which architectural components cause activation drift:
# 1. Baseline: RMSNorm + SwiGLU + RoPE (LLAMA standard)
# 2. LayerNorm: LayerNorm + SwiGLU + RoPE  
# 3. GELU: RMSNorm + GELU + RoPE
# 4. Learned Pos: RMSNorm + SwiGLU + Learned
# 5. GPT-2-like: LayerNorm + GELU + Learned (all nanoGPT components)

set -euo pipefail

echo "================================================================================"
echo "CompleteP Architecture Ablation Study"
echo "================================================================================"
echo ""
echo "This will run 5 experiments to isolate which architectural components"
echo "contribute to activation drift in coordinate checks:"
echo ""
echo "  1. Baseline (LLAMA):  RMSNorm + SwiGLU + RoPE"
echo "  2. LayerNorm test:    LayerNorm + SwiGLU + RoPE"
echo "  3. GELU test:         RMSNorm + GELU + RoPE"
echo "  4. Learned Pos test:  RMSNorm + SwiGLU + Learned"
echo "  5. GPT-2-like (all):  LayerNorm + GELU + Learned"
echo ""
echo "Each experiment tests 5 widths (128, 256, 512, 1024, 2048) with 3 seeds each."
echo "Total runs: 5 experiments × 5 widths × 3 seeds = 75 runs"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# Change to the project root
cd /localdisk/ssrivas9/multilingual-eleuther

# Experiment 1: Baseline
    echo "================================================================================"
    echo "[1/5] Running Baseline (RMSNorm + SwiGLU + RoPE)..."
    echo "================================================================================"
bash coord_check_experiments/width_scaling/run_completep.sh

# Experiment 2: LayerNorm only
echo ""
echo "================================================================================"
echo "[2/5] Running LayerNorm test (LayerNorm + SwiGLU + RoPE)..."
echo "================================================================================"
bash coord_check_experiments/width_scaling/run_completep_layernorm.sh

# Experiment 3: GELU only
echo ""
echo "================================================================================"
echo "[3/5] Running GELU test (RMSNorm + GELU + RoPE)..."
echo "================================================================================"
bash coord_check_experiments/width_scaling/run_completep_gelu.sh

# Experiment 4: Learned Position Embeddings only
echo ""
echo "================================================================================"
echo "[4/5] Running Learned Pos test (RMSNorm + SwiGLU + Learned)..."
echo "================================================================================"
bash coord_check_experiments/width_scaling/run_completep_learned_pos.sh

# Experiment 5: GPT-2-like (all changes)
echo ""
echo "================================================================================"
echo "[5/5] Running GPT-2-like test (LayerNorm + GELU + Learned)..."
echo "================================================================================"
bash coord_check_experiments/width_scaling/run_completep_gpt2like.sh

# Generate plots
echo ""
echo "================================================================================"
echo "Generating plots..."
echo "================================================================================"
cd coord_check_experiments
python plot_coord_checks_enhanced.py
python plot_arch_ablations.py
cd ..

echo ""
echo "================================================================================"
echo "✅ Architecture Ablation Study Complete!"
echo "================================================================================"
echo ""
echo "Results:"
echo "  • Experiments saved in: width_scaling/completep_*/"
echo "  • Plots saved in: coord_check_experiments/"
echo ""
echo "Key plots to review:"
echo "  1. width_coord_check_arch_ablation.png - Full comparison"
echo "  2. width_coord_check_baseline.png - LLAMA standard"
echo "  3. width_coord_check_layernorm.png - LayerNorm effect"
echo "  4. width_coord_check_gelu.png - GELU effect"
echo "  5. width_coord_check_learned_pos.png - Position embedding effect"
echo "  6. width_coord_check_gpt2like.png - GPT-2-like (combined effects)"
echo ""
echo "Compare these plots to identify which component(s) reduce activation drift."
echo "================================================================================"

