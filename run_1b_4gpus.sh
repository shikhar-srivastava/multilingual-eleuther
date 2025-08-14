# Define the set of learning rates and normalization types
norm_type=$1
learning_rates=5e-4
export NORM_TYPE=$norm_type
export POST_NUM=$2
export OMP_NUM_THREADS=8
export NCCL_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300
# If $3 provided then use it as master_port otherwise it is 29500
if [ -n "$3" ]; then
    export MASTER_PORT=$3
else
    export MASTER_PORT=29500
fi

# Function to run a single training task

echo "Training with learning rate: $learning_rates, norm type: $norm_type on 4 GPUs"
echo "Post num: $POST_NUM"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port=$MASTER_PORT torchrun_main.py \
    --model_config configs/llama_1b.json \
    --lr $learning_rates \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 100000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer adam \
    --grad_clipping 0.0 \
    --run_name "1b_res_${norm_type}_lr${learning_rates}" \
    --save_dir "1b_res_${norm_type}_lr${learning_rates}" \
    --track_activations \
    --activation_track_every 100 \
    --activation_sample_ratio 1.0