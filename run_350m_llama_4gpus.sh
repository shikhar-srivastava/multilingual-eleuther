#!/bin/bash

# Simplified version without logging overhead

# Define the set of learning rates and normalization types
norm_type=$1
learning_rates=5e-4
export NORM_TYPE=$norm_type
export POST_NUM=$2
export OMP_NUM_THREADS=8
export NCCL_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300

# If $3 provided then use it as master_port otherwise it is 29503
if [ -n "$3" ]; then
    export MASTER_PORT=$3
else
    export MASTER_PORT=29503
fi

echo "=========================================="
echo "Training 350M model with Llama tokenizer"
echo "=========================================="
echo "Training with learning rate: $learning_rates, norm type: $norm_type on 4 GPUs"
echo "Post num: $POST_NUM"
echo "Master port: $MASTER_PORT"
echo "Started at: $(date)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port=$MASTER_PORT torchrun_main.py \
    --model_config configs/llama_350m.json \
    --lr $learning_rates \
    --batch_size 16 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer adam \
    --grad_clipping 1.0 \
    --tokenizer_name meta-llama/Llama-2-7b-hf \
    --run_name "350m_res_${norm_type}_lr${learning_rates}_llama" \
    --save_dir "350m_res_${norm_type}_lr${learning_rates}_llama"


