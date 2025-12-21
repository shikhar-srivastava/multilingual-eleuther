#!/usr/bin/env bash
# Coordinate check script for CompleteP
# Based on nanoGPT-mup: completep_examples/coord_check_shakespeare_char/depth_alpha_1_aka_completep/run.sh

set -euo pipefail

# Set environment for pre-norm LLAMA 1 architecture
export NORM_TYPE=pre
export OMP_NUM_THREADS=8

# Base configuration
BASE_WIDTH=256
BASE_DEPTH=2
HEAD_SIZE=64
DEPTH_ALPHA_EXP=1.0

# Coordinate check parameters
LEARNING_RATE=1e-3
MAX_ITERS=10
BATCH_SIZE=8
GRADIENT_ACCUM=1
MAX_LENGTH=256
VOCAB_SIZE=65  # Shakespeare char-level
EVAL_INTERVAL=1
EVAL_ITERS=1

# Data
DATASET="shakespeare_char"
DATA_DIR="/localdisk/ssrivas9/multilingual-eleuther/data/shakespeare_char"

echo "=============================================================================="
echo "CompleteP Coordinate Check"
echo "=============================================================================="
echo "This script runs coordinate checks to verify CompleteP is correctly configured"
echo "Base width: $BASE_WIDTH, Base depth: $BASE_DEPTH"
echo "Testing across different depths to verify stable transfer"
echo "=============================================================================="

# Function to run a single coordinate check
run_coord_check() {
    local depth=$1
    local seed=$2
    local width=$BASE_WIDTH
    local n_heads=$((width / HEAD_SIZE))
    local depth_multiplier=$(echo "scale=8; $depth/$BASE_DEPTH" | bc -l)
    local out_dir="coord_check_out/completep/depth${depth}_width${width}_seed${seed}"
    
    echo ""
    echo "------------------------------------------------------------------------------"
    echo "Running: depth=$depth, width=$width, seed=$seed"
    echo "  depth_multiplier=$depth_multiplier"
    echo "  depth_alpha_exp=$DEPTH_ALPHA_EXP"
    echo "  Output: $out_dir"
    echo "------------------------------------------------------------------------------"
    
    python coord_check_train.py \
        --out_dir=$out_dir \
        --eval_interval=$EVAL_INTERVAL \
        --log_interval=1 \
        --eval_iters=$EVAL_ITERS \
        --dataset=$DATASET \
        --data_dir=$DATA_DIR \
        --gradient_accumulation_steps=$GRADIENT_ACCUM \
        --batch_size=$BATCH_SIZE \
        --max_length=$MAX_LENGTH \
        --n_layer=$depth \
        --num_attention_heads=$n_heads \
        --hidden_size=$width \
        --vocab_size=$VOCAB_SIZE \
        --init_std=0.02 \
        --lr=$LEARNING_RATE \
        --max_iters=$MAX_ITERS \
        --weight_decay=0.1 \
        --beta1=0.9 \
        --beta2=0.95 \
        --grad_clip=1.0 \
        --decay_lr=False \
        --mup_enabled \
        --mup_width_multiplier=1.0 \
        --mup_base_width=$BASE_WIDTH \
        --mup_input_alpha=1.0 \
        --mup_output_alpha=1.0 \
        --mup_enable_coord_check_logging \
        --depth_alpha_enabled \
        --depth_multiplier=$depth_multiplier \
        --depth_base_depth=$BASE_DEPTH \
        --depth_alpha_exp=$DEPTH_ALPHA_EXP \
        --seed=$seed \
        --device=cuda \
        --dtype=float32 \
        --csv_log
    
    echo "✓ Completed: depth=$depth, seed=$seed"
}

# Run coordinate checks across different depths
echo ""
echo "Starting coordinate checks..."
echo ""

for depth in 2 4 8 16
do
    for seed in 1 2 3
    do
        run_coord_check $depth $seed
    done
done

echo ""
echo "=============================================================================="
echo "✅ All coordinate checks completed!"
echo "=============================================================================="
echo ""
echo "Results saved to: coord_check_out/completep/"
echo ""
echo "To analyze results:"
echo "  1. Check CSV logs in each output directory"
echo "  2. Plot activation magnitudes across depths"
echo "  3. Verify that activations are stable (not exploding/vanishing)"
echo ""
echo "Expected behavior with correct CompleteP:"
echo "  - Token embedding activations: stable across depths"
echo "  - Attention activations: stable across depths"
echo "  - MLP activations: stable across depths"
echo "  - LM head activations: stable across depths"
echo "=============================================================================="

