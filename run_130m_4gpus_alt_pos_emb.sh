#!/bin/bash

# Define the set of learning rates and normalization types
norm_type=$1
position_embedding_type=$2
learning_rates=1e-3
export NORM_TYPE=$norm_type
export POST_NUM=$3
export OMP_NUM_THREADS=8
export NCCL_TIMEOUT=1800

# If $4 provided then use it as master_port otherwise it is 29510
if [ -n "$4" ]; then
    export MASTER_PORT=$4
else
    export MASTER_PORT=29510
fi

# Validate position embedding type
if [[ ! "$position_embedding_type" =~ ^(rope|learned|sinusoidal|none)$ ]]; then
    echo "Error: Invalid position_embedding_type '$position_embedding_type'. Must be one of: rope, learned, sinusoidal, none"
    echo "Usage: $0 <norm_type> <position_embedding_type> <post_num> [master_port]"
    echo "Example: $0 pre learned 1 29510"
    exit 1
fi

# Function to run a single training task
echo "Training with learning rate: $learning_rates, norm type: $norm_type, position embedding: $position_embedding_type on 4 GPUs"
echo "Post num: $POST_NUM"

# Select appropriate config file or use default with override
config_file="configs/llama_130m.json"
if [ "$position_embedding_type" = "learned" ]; then
    config_file="configs/llama_130m_learned_pos.json"
elif [ "$position_embedding_type" = "sinusoidal" ]; then
    config_file="configs/llama_130m_sinusoidal_pos.json"
elif [ "$position_embedding_type" = "none" ]; then
    config_file="configs/llama_130m_no_pos.json"
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port=$MASTER_PORT torchrun_main.py \
    --model_config $config_file \
    --position_embedding_type $position_embedding_type \
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
    --run_name "130m_res_${norm_type}_${position_embedding_type}_lr${learning_rates}_layer_scale_track_bytes" \
    --save_dir "130m_res_${norm_type}_${position_embedding_type}_lr${learning_rates}_track_bytes" \
    --track_activations \
    --activation_track_every 100 \
    --activation_sample_ratio 1.0 \
    --track_dataset_bytes \
    --log_bytes_every 100