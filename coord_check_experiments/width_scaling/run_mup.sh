#!/usr/bin/env bash
# muP Width Coordinate Check
# Runs width scaling experiments WITH muP
# Expects activations to remain stable across widths (hyperparameter transfer)

set -euo pipefail

cd /localdisk/ssrivas9/multilingual-eleuther

# Delete old logs if they exist
echo "Cleaning old logs in coord_check_experiments/width_scaling/mup/out/..."
rm -rf coord_check_experiments/width_scaling/mup/out/

mup_base_width=256

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
        mup_width_multiplier=$(echo "scale=8; $width/$mup_base_width" | bc -l)
        out_dir="coord_check_experiments/width_scaling/mup/out/width${width}_depth4_seed${seed}"
        
        echo "Running muP: width=$width, depth=$depth, width_mult=$mup_width_multiplier, seed=$seed"
        
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
            --mup_base_width=$mup_base_width \
            --mup_width_multiplier=$mup_width_multiplier \
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

echo "muP width coordinate check complete!"
echo "Results saved to coord_check_experiments/width_scaling/mup/out/"

