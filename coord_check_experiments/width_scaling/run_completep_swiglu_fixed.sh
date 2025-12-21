#!/bin/bash
#
# Width scaling experiment for CompleteP with SwiGLU + muP fix
# (RMSNorm + SwiGLU with 1/√width scaling + RoPE)
#
# This tests whether the muP fix for SwiGLU's multiplicative variance growth
# resolves the activation drift issue.

set -e

# Change to the root directory
cd /localdisk/ssrivas9/multilingual-eleuther

BASE_DIR="coord_check_experiments/width_scaling/completep_swiglu_fixed"

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

echo "Running CompleteP width scaling with SwiGLU..."
echo "  Norm: RMSNorm (LLAMA standard)"
echo "  MLP: SwiGLU (muP initialization handles variance)"
echo "  Position: RoPE (LLAMA standard)"
echo ""
echo "This should show STABLE activations like the GELU variant!"
echo ""

for WIDTH in "${WIDTHS[@]}"; do
    echo "Testing width=${WIDTH}..."
    
    # Calculate intermediate_size (4x for consistency, though SwiGLU often uses 8/3x)
    INTERMEDIATE_SIZE=$((WIDTH * 4))
    
    # Calculate attention heads (keep 64 dim per head)
    NUM_HEADS=$((WIDTH / 64))
    if [ $NUM_HEADS -lt 1 ]; then
        NUM_HEADS=1
    fi
    
    # Calculate width multiplier for muP
    WIDTH_MULT=$(echo "$WIDTH / $BASE_WIDTH" | bc -l)
    
    # Calculate depth multiplier for CompleteP
    DEPTH_MULT=$(echo "$N_LAYER / $BASE_DEPTH" | bc -l)
    
    for SEED in 1 2 3; do
        OUT_DIR="${BASE_DIR}/out/width${WIDTH}_seed${SEED}"
        mkdir -p $OUT_DIR
        
        echo "  Running width=${WIDTH}, seed=${SEED}..."
        
        python coord_check_train.py \
            --out_dir="$OUT_DIR" \
            --seed=$SEED \
            --n_layer=$N_LAYER \
            --num_attention_heads=$NUM_HEADS \
            --hidden_size=$WIDTH \
            --intermediate_size=$INTERMEDIATE_SIZE \
            --max_iters=10 \
            --eval_interval=1 \
            --log_interval=1 \
            --eval_iters=1 \
            --batch_size=8 \
            --max_length=256 \
            --lr=0.001 \
            --weight_decay=0.1 \
            --beta1=0.9 \
            --beta2=0.95 \
            --grad_clip=1.0 \
            --mup_enabled \
            --mup_base_width=$BASE_WIDTH \
            --mup_width_multiplier=$WIDTH_MULT \
            --depth_alpha_enabled \
            --depth_multiplier=$DEPTH_MULT \
            --depth_alpha_exp=1.0 \
            --depth_base_depth=$BASE_DEPTH \
            --mup_enable_coord_check_logging \
            --csv_log 2>&1 | tee "${OUT_DIR}/train.log"
        
        echo "    ✓ Completed width=${WIDTH}, seed=${SEED}"
    done
done

echo ""
echo "✅ Width scaling experiment complete!"
echo ""
echo "Results saved to: ${BASE_DIR}/out/"
echo ""
echo "To plot results, run:"
echo "  cd coord_check_experiments"
echo "  python plot_arch_ablations.py"
echo ""
echo "Expected: Activation curves should now be STABLE across widths,"
echo "          similar to the GELU variant!"

