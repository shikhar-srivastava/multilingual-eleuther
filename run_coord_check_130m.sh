#!/usr/bin/env bash
# Coordinate check specifically for 130M model configuration
# This verifies CompleteP works correctly for the dimensions in configs/llama_130m.json

set -euo pipefail

export NORM_TYPE=pre
export OMP_NUM_THREADS=8

# 130M model dimensions from configs/llama_130m.json
HIDDEN_SIZE=768
NUM_HEADS=12
INTERMEDIATE_SIZE=2048
HEAD_SIZE=$((HIDDEN_SIZE / NUM_HEADS))  # 64

# CompleteP base configuration
# For 130M with 12 layers, we use base_depth=6 (so 12 is 2x base)
BASE_WIDTH=768   # Use actual 130M width as base
BASE_DEPTH=6     # Half of 130M depth as base

DEPTH_ALPHA_EXP=1.0
LEARNING_RATE=1e-3
MAX_ITERS=10
BATCH_SIZE=8
GRADIENT_ACCUM=1
MAX_LENGTH=256  # Shorter for coord check speed
VOCAB_SIZE=65   # Shakespeare char-level

DATASET="shakespeare_char"
DATA_DIR="/localdisk/ssrivas9/multilingual-eleuther/data/shakespeare_char"

echo "=============================================================================="
echo "Coordinate Check for 130M Model Configuration"
echo "=============================================================================="
echo "Testing CompleteP parameterization for:"
echo "  Hidden size: $HIDDEN_SIZE (130M actual size)"
echo "  Intermediate size: $INTERMEDIATE_SIZE"
echo "  Num heads: $NUM_HEADS (head_size=$HEAD_SIZE)"
echo ""
echo "Base configuration:"
echo "  Base width: $BASE_WIDTH"
echo "  Base depth: $BASE_DEPTH"
echo "  Depth alpha: $DEPTH_ALPHA_EXP"
echo ""
echo "Will test across depths: 6 (base), 12 (130M actual), 24 (2x 130M)"
echo "This verifies CompleteP enables stable scaling to deeper models"
echo "=============================================================================="

# Function to run a single coordinate check
run_coord_check() {
    local depth=$1
    local seed=$2
    local width=$HIDDEN_SIZE
    local n_heads=$NUM_HEADS
    local intermediate=$INTERMEDIATE_SIZE
    local depth_multiplier=$(echo "scale=8; $depth/$BASE_DEPTH" | bc -l)
    local width_multiplier=$(echo "scale=8; $width/$BASE_WIDTH" | bc -l)
    local out_dir="coord_check_out/130m_config/depth${depth}_width${width}_seed${seed}"
    
    echo ""
    echo "------------------------------------------------------------------------------"
    echo "Running: depth=$depth (${depth_multiplier}x base), width=$width (${width_multiplier}x base), seed=$seed"
    echo "  This matches 130M config: hidden=$width, heads=$n_heads, intermediate=$intermediate"
    echo "  Output: $out_dir"
    echo "------------------------------------------------------------------------------"
    
    python coord_check_train.py \
        --out_dir=$out_dir \
        --eval_interval=1 \
        --log_interval=1 \
        --eval_iters=1 \
        --dataset=$DATASET \
        --data_dir=$DATA_DIR \
        --gradient_accumulation_steps=$GRADIENT_ACCUM \
        --batch_size=$BATCH_SIZE \
        --max_length=$MAX_LENGTH \
        --n_layer=$depth \
        --num_attention_heads=$n_heads \
        --hidden_size=$width \
        --intermediate_size=$intermediate \
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
        --mup_width_multiplier=$width_multiplier \
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

echo ""
echo "Starting coordinate checks for 130M configuration..."
echo ""

# Test at base depth (6), actual 130M depth (12), and 2x 130M depth (24)
for depth in 6 12 24
do
    for seed in 1 2 3
    do
        run_coord_check $depth $seed
    done
done

echo ""
echo "=============================================================================="
echo "✅ All coordinate checks completed for 130M configuration!"
echo "=============================================================================="
echo ""
echo "Results saved to: coord_check_out/130m_config/"
echo ""
echo "Analysis:"
echo "  - depth=6:  Base configuration (reference point)"
echo "  - depth=12: Your actual 130M model (2x base depth)"
echo "  - depth=24: Deeper model (4x base depth / 2x 130M)"
echo ""
echo "Expected results with correct CompleteP:"
echo "  ✓ All depths should have similar activation magnitudes"
echo "  ✓ The 130M config (depth=12) should be stable"
echo "  ✓ This validates CompleteP will work for your actual training"
echo ""
echo "To visualize:"
echo "  cd coord_check_out/130m_config"
echo "  python ../../../plot_coord_check.py"
echo "=============================================================================="

