#!/bin/bash
# muTransfer Learning Rate Sweep - μP (Maximal Update Parameterization)
#
# This experiment demonstrates that with μP (muTransfer),
# the optimal learning rate REMAINS CONSTANT as model width changes.
# This is the key property that enables hyperparameter transfer.

# Single-GPU Launching
LAUNCHER=python

# Multi-GPU Launching (single node)
#GPU=2
#LAUNCHER=torchrun --standalone --nproc_per_node=$GPU

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$ROOT_DIR"

LAYERS=2
MUP_BASE_WIDTH=256

# Widths to sweep (powers of 2 from 256 to 2048)
WIDTHS="256 512 1024 2048"

# Learning rates to sweep (powers of 2 from 2^-14 to 2^-4)
# Note: For muP, all widths should converge to the same optimal LR
LRS="0.125 0.0625 0.03125 0.015625 0.0078125 0.00390625 0.001953125 0.0009765625 0.00048828125 0.000244140625 0.0001220703125 0.00006103515625"

# Seeds for multiple runs
SEEDS="1 2 3"

echo "Running muTransfer LR sweep with μP (Maximal Update Parameterization)"
echo "====================================================================="
echo "Widths: $WIDTHS"
echo "Learning rates: $LRS"
echo "Seeds: $SEEDS"
echo "Layers: $LAYERS"
echo "Base width for muP: $MUP_BASE_WIDTH"
echo ""

for width in $WIDTHS
do
    for lr in $LRS
    do
        for seed in $SEEDS
        do
            head_size=64
            n_heads=$((width / head_size))
            # Minimum learning rate = lr / 10
            min_lr=$(awk "BEGIN {print $lr/10}")
            # Compute muP width multiplier: width / base_width
            mup_width_multiplier=$(echo "scale=8; $width/$MUP_BASE_WIDTH" | bc -l)
            out_dir="$SCRIPT_DIR/out/width${width}_depth${LAYERS}_seed${seed}_lr${lr}"
            
            echo "Running: width=$width, lr=$lr, seed=$seed, mup_width_multiplier=$mup_width_multiplier"
            
            $LAUNCHER coord_check_train.py \
                --out_dir="$out_dir" \
                --n_layer=$LAYERS \
                --hidden_size=$width \
                --num_attention_heads=$n_heads \
                --max_length=256 \
                --max_iters=1000 \
                --batch_size=32 \
                --gradient_accumulation_steps=1 \
                --lr=$lr \
                --weight_decay=0.1 \
                --beta1=0.9 \
                --beta2=0.95 \
                --adam_eps=1e-12 \
                --grad_clip=1.0 \
                --init_std=0.02 \
                --dataset=shakespeare_char \
                --eval_interval=1 \
                --log_interval=1 \
                --eval_iters=1 \
                --seed=$seed \
                --dtype=bfloat16 \
                --csv_log \
                --position_embedding_type=learned \
                --mup_enabled \
                --mup_width_multiplier=$mup_width_multiplier \
                --mup_base_width=$MUP_BASE_WIDTH \
                --mup_input_alpha=1.0 \
                --mup_output_alpha=1.0
            
            if [ $? -ne 0 ]; then
                echo "Warning: Run failed for width=$width, lr=$lr, seed=$seed"
            fi
        done
    done
done

echo ""
echo "μP experiment complete. Results saved to $SCRIPT_DIR/out/"

