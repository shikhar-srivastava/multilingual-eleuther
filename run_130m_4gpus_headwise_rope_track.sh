#!/bin/bash

# Define the set of learning rates and normalization types
norm_type=$1
learning_rates=1e-3
export NORM_TYPE=$norm_type
export POST_NUM=$2
export OMP_NUM_THREADS=8
export NCCL_TIMEOUT=1800
# If $3 provided then use it as master_port otherwise it is 29510
if [ -n "$3" ]; then
    export MASTER_PORT=$3
else
    export MASTER_PORT=29510
fi

# Function to run a single training task

echo "Training with learning rate: $learning_rates, norm type: $norm_type on 4 GPUs"
echo "Post num: $POST_NUM"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port=$MASTER_PORT torchrun_main.py \
    --model_config configs/llama_130m.json \
    --lr $learning_rates \
    --batch_size 16 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer adam \
    --grad_clipping 0.0 \
    --run_name "130m_res_${norm_type}_lr${learning_rates}_headwise_rope_track" \
    --save_dir "130m_res_${norm_type}_lr${learning_rates}_headwise_rope_track" \
    --track_activations \
    --activation_track_every 100 \
    --track_weights \
    --weight_track_every 100 \
    --activation_sample_ratio 0.1 \
    --track_head_activations \
    --track_dataset_bytes \
    --log_bytes_every 100