#!/usr/bin/env bash
# Width Scaling Coordinate Check for CompleteP
# Tests if activation magnitudes remain stable across different widths with CompleteP

set -euo pipefail

# Change to the root directory
cd /localdisk/ssrivas9/multilingual-eleuther

BASE_DIR="coord_check_experiments/width_scaling/completep"

# Delete old logs if they exist
echo "Cleaning old logs in ${BASE_DIR}/out/..."
rm -rf "${BASE_DIR}/out/"

# Base configuration
BASE_WIDTH=64
BASE_DEPTH=2  # Using shallow depth for width checks to isolate width effects
DEPTH=4       # Fixed depth for all width checks

# CompleteP settings
DEPTH_ALPHA_EXP=1.0
DEPTH_MULTIPLIER=$(echo "scale=8; $DEPTH/$BASE_DEPTH" | bc -l)

# Iterate over widths: 128, 256 (base), 512, 1024, 2048
for width in 128 256 512 1024 2048
do
    for seed in 1 2 3
    do
        # Calculate width multiplier
        width_multiplier=$(echo "scale=8; $width/$BASE_WIDTH" | bc -l)
        
        # Calculate head size and number of heads (keeping head size constant at 64)
        head_size=64
        n_heads=$((width / head_size))
        
        out_dir="${BASE_DIR}/out/width${width}_seed${seed}"
        mkdir -p "$out_dir"
        
        echo "Running CompleteP width check: width=$width ($width_multiplier x base), depth=$DEPTH, seed=$seed"
        
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
            --depth_alpha_enabled \
            --depth_multiplier=$DEPTH_MULTIPLIER \
            --depth_base_depth=$BASE_DEPTH \
            --depth_alpha_exp=$DEPTH_ALPHA_EXP \
            --seed=$seed \
            --device=cuda \
            --dtype=float32 \
            --csv_log 2>&1 | tee "${out_dir}/train.log"
    done
done

