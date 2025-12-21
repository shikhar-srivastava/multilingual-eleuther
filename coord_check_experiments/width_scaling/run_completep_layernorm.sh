#!/bin/bash
#
# Width scaling experiment for CompleteP with LayerNorm only
# (LayerNorm + SwiGLU + RoPE)
#
# This isolates the effect of LayerNorm vs RMSNorm on activation drift.

set -euo pipefail

# Change to the root directory
cd /localdisk/ssrivas9/multilingual-eleuther

BASE_DIR="coord_check_experiments/width_scaling/completep_layernorm"

# Delete old logs if they exist
echo "Cleaning old logs in ${BASE_DIR}/out/..."
rm -rf "${BASE_DIR}/out/"

mkdir -p $BASE_DIR

# Test widths (hidden_size) - match baseline experiments
WIDTHS=(128 256 512 1024 2048)

# Base width for muP scaling
BASE_WIDTH=64

# Fixed depth
N_LAYER=6

# Base depth for CompleteP scaling
BASE_DEPTH=2

echo "Running CompleteP width scaling with LayerNorm only..."
echo "  Norm: LayerNorm"
echo "  MLP: SwiGLU (LLAMA standard)"
echo "  Position: RoPE (LLAMA standard)"
echo ""

for WIDTH in "${WIDTHS[@]}"; do
    echo "Testing width=${WIDTH}..."
    
    # Calculate intermediate_size (4x for consistency)
    INTERMEDIATE_SIZE=$((WIDTH * 4))
    
    # Calculate attention heads (keep 64 dim per head)
    NUM_HEADS=$((WIDTH / 64))
    if [ $NUM_HEADS -lt 1 ]; then
        NUM_HEADS=1
    fi
    
    # Calculate width multiplier for muP
    WIDTH_MULT=$(echo "scale=8; $WIDTH / $BASE_WIDTH" | bc -l)
    
    # Calculate depth multiplier for CompleteP
    DEPTH_MULT=$(echo "scale=8; $N_LAYER / $BASE_DEPTH" | bc -l)
    
    for SEED in 1 2 3; do
        OUT_DIR="${BASE_DIR}/out/width${WIDTH}_seed${SEED}"
        mkdir -p "$OUT_DIR"
        
        python coord_check_train.py \
            --out_dir="$OUT_DIR" \
            --hidden_size=$WIDTH \
            --n_layer=$N_LAYER \
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
            --depth_alpha_enabled \
            --depth_multiplier=$DEPTH_MULT \
            --depth_base_depth=$BASE_DEPTH \
            --depth_alpha_exp=1.0 \
            --seed=$SEED \
            --csv_log \
            --mup_enable_coord_check_logging \
            --dtype=float32 \
            --use_layernorm \
            --position_embedding_type=rope \
            --device=cuda 2>&1 | tee "${OUT_DIR}/train.log"
    done
done

echo ""
echo "Width scaling experiment complete (LayerNorm only)!"
echo "Results saved to $BASE_DIR"

