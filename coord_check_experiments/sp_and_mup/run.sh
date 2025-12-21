#!/usr/bin/env bash
# Standard Parameterization (SP) and muP Depth Coordinate Check
# This script runs depth scaling experiments with muP but WITHOUT CompleteP depth scaling
# Expects to see activations diverge as depth increases (muP alone doesn't handle depth)

set -euo pipefail

cd /localdisk/ssrivas9/multilingual-eleuther

# Delete old logs if they exist
echo "Cleaning old logs in coord_check_experiments/sp_and_mup/out/..."
rm -rf coord_check_experiments/sp_and_mup/out/

for depth in 2 4 8 16 32 64
do
    for seed in 1 2 3
    do
        width=256
        head_size=64
        n_heads=$((width / head_size))
        out_dir="coord_check_experiments/sp_and_mup/out/depth${depth}_width256_seed${seed}"
        
        echo "Running: depth=$depth, width=$width, seed=$seed"
        
        python coord_check_train.py \
            --out_dir="$out_dir" \
            --n_layer=$depth \
            --hidden_size=$width \
            --num_attention_heads=$n_heads \
            --max_length=256 \
            --batch_size=8 \
            --gradient_accumulation_steps=1 \
            --max_iters=10 \
            --lr=1e-3 \
            --weight_decay=0.1 \
            --beta1=0.9 \
            --beta2=0.95 \
            --grad_clip=1.0 \
            --init_std=0.02 \
            --mup_enabled \
            --mup_width_multiplier=1.0 \
            --mup_input_alpha=1.0 \
            --mup_output_alpha=1.0 \
            --mup_enable_coord_check_logging \
            --seed=$seed \
            --device='cuda' \
            --dtype='float32' \
            --csv_log \
            --eval_interval=1 \
            --log_interval=1 \
            --eval_iters=1
    done
done

echo "SP+muP depth coordinate check complete!"
echo "Results saved to coord_check_experiments/sp_and_mup/out/"

