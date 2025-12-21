#!/usr/bin/env bash
#
# Ablation: CompleteP WITHOUT sqrt(2) SwiGLU scaling
# Tests if activation magnitudes remain stable across widths when SwiGLU scaling is disabled
#

set -euo pipefail

# Change to the root directory
cd /localdisk/ssrivas9/multilingual-eleuther

BASE_DIR="coord_check_experiments/width_scaling/ablation_no_swiglu_scaling"

# Delete old logs if they exist
echo "Cleaning old logs in ${BASE_DIR}/out/..."
rm -rf "${BASE_DIR}/out/"

mkdir -p $BASE_DIR

# Base configuration
BASE_WIDTH=64
BASE_DEPTH=2
DEPTH=4

# CompleteP settings
DEPTH_ALPHA_EXP=1.0
DEPTH_MULTIPLIER=$(echo "scale=8; $DEPTH/$BASE_DEPTH" | bc -l)

# Iterate over widths
for width in 128 256 512 1024 2048
do
    for SEED in 1 2 3
    do
        # Calculate width multiplier
        width_multiplier=$(echo "scale=8; $width/$BASE_WIDTH" | bc -l)
        
        # Calculate head size and number of heads
        head_size=64
        n_heads=$((width / head_size))
        if [ $n_heads -lt 1 ]; then n_heads=1; fi
        
        out_dir="${BASE_DIR}/out/width${width}_depth${DEPTH}_seed${SEED}"
        mkdir -p "$out_dir"
        
        echo "Running Ablation (No SwiGLU Scale) width check: width=$width, seed=$SEED"
        
        python coord_check_train.py \
            --out_dir="$out_dir" \
            --eval_interval=1 \
            --log_interval=1 \
            --eval_iters=1 \
            --dataset=shakespeare_char \
            --gradient_accumulation_steps=1 \
            --batch_size=8 \
            --max_length=256 \
            --n_layer=$DEPTH \
            --num_attention_heads=$n_heads \
            --hidden_size=$width \
            --vocab_size=65 \
            --init_std=0.02 \
            --lr=1e-3 \
            --weight_decay=0.1 \
            --beta1=0.9 \
            --beta2=0.95 \
            --grad_clip=1.0 \
            --mup_enabled \
            --mup_width_multiplier=$width_multiplier \
            --mup_base_width=$BASE_WIDTH \
            --mup_input_alpha=1.0 \
            --mup_output_alpha=1.0 \
            --mup_enable_coord_check_logging \
            --mup_disable_swiglu_scaling \
            --depth_alpha_enabled \
            --depth_multiplier=$DEPTH_MULTIPLIER \
            --depth_base_depth=$BASE_DEPTH \
            --depth_alpha_exp=$DEPTH_ALPHA_EXP \
            --seed=$SEED \
            --device=cuda \
            --dtype=float32 \
            --csv_log 2>&1 | tee "${out_dir}/train.log"
    done
done

echo "Ablation width check complete! Results in ${BASE_DIR}/out/"

