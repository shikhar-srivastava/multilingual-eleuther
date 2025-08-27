#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash monolingual_130m.sh <norm_type> <monolingual_dataset> <vocab_size> <tokenizer_type> [post_num] [master_port]
# Example:
#   bash monolingual_130m.sh pre eng_latn 32768 bpe_unscaled 6 29510

norm_type=${1:-pre}
monolingual_dataset=${2:-eng_latn}
vocab_size=${3:-32768}
tokenizer_type=${4:-bpe_unscaled}   # bpe_unscaled | unigram_unscaled
export POST_NUM=${5:-6}
export MASTER_PORT=${6:-29510}

learning_rates=1e-3
export NORM_TYPE=$norm_type
export OMP_NUM_THREADS=8
export NCCL_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300

run_name="mono_130m_${norm_type}_lr${learning_rates}_${monolingual_dataset}_${tokenizer_type}_${vocab_size}"
save_dir="$run_name"

echo "Training 130M with lr=$learning_rates, norm=$norm_type, dataset=$monolingual_dataset, tok=${tokenizer_type}/${vocab_size} on 4 GPUs"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port=$MASTER_PORT torchrun_main.py \
    --model_config configs/llama_130m.json \
    --lr $learning_rates \
    --batch_size 32 \
    --total_batch_size 512 \
    --num_epochs 10 \
    --warmup_steps_ratio 0.1 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 50 \
    --optimizer adam \
    --grad_clipping 0.0 \
    --max_length 1024 \
    --run_name "$run_name" \
    --save_dir "$save_dir" \
    --monolingual-dataset $monolingual_dataset \
    --tokenizer_type $tokenizer_type \
    --tokenizer_vocabulary $vocab_size \
    --tokenizer_name t5-base \
    --hf_repo_name "shikhar-srivastava/${run_name}" \
    --hf_push_final --hf_push_checkpoints