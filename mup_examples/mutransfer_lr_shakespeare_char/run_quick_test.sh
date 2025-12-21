#!/bin/bash
# Quick test to verify the muTransfer experiment setup
#
# This runs a minimal version of the experiment with:
# - Only 2 widths (256, 512)
# - Only 3 learning rates
# - Only 100 iterations
# - Only 1 seed

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT_DIR"

echo "========================================"
echo "Quick Test: muTransfer LR Sweep"
echo "========================================"

LAYERS=2
WIDTHS="256 512"
LRS="0.00390625 0.001953125 0.0009765625"  # 2^-8, 2^-9, 2^-10
SEED=1
MAX_ITERS=100
MUP_BASE_WIDTH=256

echo ""
echo "Testing SP (Standard Parameterization)..."
echo "----------------------------------------"

for width in $WIDTHS
do
    for lr in $LRS
    do
        n_heads=$((width / 64))
        out_dir="$SCRIPT_DIR/sp/out_test/width${width}_lr${lr}"
        
        echo "SP: width=$width, lr=$lr"
        
        python coord_check_train.py \
            --out_dir="$out_dir" \
            --n_layer=$LAYERS \
            --hidden_size=$width \
            --num_attention_heads=$n_heads \
            --max_length=256 \
            --max_iters=$MAX_ITERS \
            --batch_size=16 \
            --lr=$lr \
            --weight_decay=0.1 \
            --seed=$SEED \
            --dtype=bfloat16 \
            --csv_log \
            --position_embedding_type=learned \
            --eval_interval=10 \
            --log_interval=10
    done
done

echo ""
echo "Testing μP (muTransfer)..."
echo "----------------------------------------"

for width in $WIDTHS
do
    for lr in $LRS
    do
        n_heads=$((width / 64))
        mup_width_multiplier=$(echo "scale=8; $width/$MUP_BASE_WIDTH" | bc -l)
        out_dir="$SCRIPT_DIR/mup/out_test/width${width}_lr${lr}"
        
        echo "μP: width=$width, lr=$lr, mup_width_multiplier=$mup_width_multiplier"
        
        python coord_check_train.py \
            --out_dir="$out_dir" \
            --n_layer=$LAYERS \
            --hidden_size=$width \
            --num_attention_heads=$n_heads \
            --max_length=256 \
            --max_iters=$MAX_ITERS \
            --batch_size=16 \
            --lr=$lr \
            --weight_decay=0.1 \
            --seed=$SEED \
            --dtype=bfloat16 \
            --csv_log \
            --position_embedding_type=learned \
            --mup_enabled \
            --mup_width_multiplier=$mup_width_multiplier \
            --mup_base_width=$MUP_BASE_WIDTH \
            --eval_interval=10 \
            --log_interval=10
    done
done

echo ""
echo "========================================"
echo "Quick test complete!"
echo "========================================"
echo "Check results in:"
echo "  - $SCRIPT_DIR/sp/out_test/"
echo "  - $SCRIPT_DIR/mup/out_test/"

