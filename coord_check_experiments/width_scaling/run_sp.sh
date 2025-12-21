#!/usr/bin/env bash
# Standard Parameterization (SP) Width Coordinate Check
# Runs width scaling experiments WITHOUT muP
# Expects activations to change with width (no transfer)

set -euo pipefail

cd /localdisk/ssrivas9/multilingual-eleuther

# Delete old logs if they exist
echo "Cleaning old logs in coord_check_experiments/width_scaling/sp/out/..."
rm -rf coord_check_experiments/width_scaling/sp/out/

for width in 128 256 512 1024 2048
do
    for seed in 1 2 3
    do
        depth=4
        head_size=64
        n_heads=$((width / head_size))
        # Ensure at least 1 head
        if [ $n_heads -lt 1 ]; then
            n_heads=1
        fi
        out_dir="coord_check_experiments/width_scaling/sp/out/width${width}_depth4_seed${seed}"
        
        echo "Running SP: width=$width, depth=$depth, seed=$seed"
        
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

echo "SP width coordinate check complete!"
echo "Results saved to coord_check_experiments/width_scaling/sp/out/"

