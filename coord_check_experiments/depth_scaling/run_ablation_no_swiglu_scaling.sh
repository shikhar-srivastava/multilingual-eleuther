#!/usr/bin/env bash
#
# Ablation: CompleteP WITHOUT sqrt(2) SwiGLU scaling (Depth)
# Tests if activation magnitudes remain stable across depths when SwiGLU scaling is disabled
#

set -euo pipefail

# Change to the root directory
cd /localdisk/ssrivas9/multilingual-eleuther

BASE_DIR="coord_check_experiments/depth_scaling/ablation_no_swiglu_scaling"

# Delete old logs if they exist
echo "Cleaning old logs in ${BASE_DIR}/out/..."
rm -rf "${BASE_DIR}/out/"

mkdir -p $BASE_DIR

# Fixed configuration
WIDTH=256
BASE_WIDTH=64
NUM_HEADS=4
INTERMEDIATE_SIZE=$((WIDTH * 4))
BASE_DEPTH=2

echo "Running Ablation (No SwiGLU Scale) depth scaling..."
echo ""

for DEPTH in 2 4 8 16 32 64
do
    echo "Testing depth=${DEPTH}..."
    
    # Calculate depth multiplier for CompleteP
    DEPTH_MULT=$(echo "scale=8; $DEPTH / $BASE_DEPTH" | bc -l)
    
    # Calculate width multiplier for muP
    WIDTH_MULT=$(echo "scale=8; $WIDTH / $BASE_WIDTH" | bc -l)
    
    for SEED in 1 2 3; do
        OUT_DIR="${BASE_DIR}/out/depth${DEPTH}_seed${SEED}"
        mkdir -p "$OUT_DIR"
        
        python coord_check_train.py \
            --out_dir="$OUT_DIR" \
            --hidden_size=$WIDTH \
            --n_layer=$DEPTH \
            --num_attention_heads=$NUM_HEADS \
            --intermediate_size=$INTERMEDIATE_SIZE \
            --max_iters=10 \
            --eval_interval=1 \
            --log_interval=1 \
            --eval_iters=1 \
            --batch_size=8 \
            --max_length=256 \
            --lr=1e-3 \
            --weight_decay=0.1 \
            --beta1=0.9 \
            --beta2=0.95 \
            --grad_clip=1.0 \
            --mup_enabled \
            --mup_width_multiplier=$WIDTH_MULT \
            --mup_base_width=$BASE_WIDTH \
            --mup_enable_coord_check_logging \
            --mup_disable_swiglu_scaling \
            --depth_alpha_enabled \
            --depth_multiplier=$DEPTH_MULT \
            --depth_base_depth=$BASE_DEPTH \
            --depth_alpha_exp=1.0 \
            --seed=$SEED \
            --csv_log \
            --dtype=float32 \
            --position_embedding_type=rope \
            --device=cuda 2>&1 | tee "${OUT_DIR}/train.log"
    done
done

echo "Ablation depth check complete! Results in ${BASE_DIR}/out/"

