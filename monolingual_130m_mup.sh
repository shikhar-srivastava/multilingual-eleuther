#!/usr/bin/env bash
# Monolingual Training with muP/CompleteP Support
#
# This script enables hyperparameter transfer (muTransfer) and stable depth scaling
# (CompleteP) by using the Llama-9M model as a base width/depth reference.
#
# Usage:
#   bash monolingual_130m_mup.sh <norm_type> <monolingual_dataset> <vocab_size> <tokenizer_type> [post_num] [master_port] [mup_mode] [model_config]
# 
# mup_mode can be:
#   - "sp" : Standard Parameterization (no muP)
#   - "mup" : muP only (width scaling)
#   - "completep" : CompleteP (muP + depth scaling with alpha=1.0)
#   - "completep_05" : CompleteP with alpha=0.5 (more stable for very deep models)
#
# model_config:
#   - Path to model config (e.g., configs/llama_130m.json)
#
# Example:
#   bash monolingual_130m_mup.sh pre eng_latn 32768 bpe_unscaled 6 29510 completep configs/llama_130m.json

set -euo pipefail

norm_type=${1:-pre}
monolingual_dataset=${2:-eng_latn}
vocab_size=${3:-32768}
tokenizer_type=${4:-bpe_unscaled}   # bpe_unscaled | unigram_unscaled
export POST_NUM=${5:-6}
export MASTER_PORT=${6:-29510}
mup_mode=${7:-completep}           # sp | mup | completep | completep_05
model_config=${8:-configs/llama_130m.json}

# --- Base Reference (Llama-9M) ---
# We use llama_9m.json as the base for all scaling experiments
mup_base_width=128
mup_base_depth=4

# --- Extract Target Model Dimensions ---
if [ ! -f "$model_config" ]; then
    echo "Error: Config file $model_config not found."
    exit 1
fi

hidden_size=$(jq -r '.hidden_size' "$model_config")
n_layer=$(jq -r '.num_hidden_layers' "$model_config")

# --- Optimized Hyperparameters for CompleteP/muP ---
# These are based on coordinate check analysis and theoretical best practices
learning_rates=1e-3  # Optimal base LR found in 9M coord checks
adam_eps=1e-12       # Base epsilon to be scaled by CompleteP logic
beta1=0.9            # Standard Adam momentum for muP
beta2=0.95           # Standard Adam momentum for muP
weight_decay=0.1     # Scaled weight decay for hidden weights
grad_clipping=1.0    # Essential for stability in deep models (CompleteP)

export NORM_TYPE=$norm_type
export OMP_NUM_THREADS=8
export NCCL_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300

# Compute muP/CompleteP multipliers relative to 9M base
mup_width_multiplier=$(echo "scale=8; $hidden_size/$mup_base_width" | bc -l)
depth_multiplier=$(echo "scale=8; $n_layer/$mup_base_depth" | bc -l)

# Build muP/CompleteP flags based on mode
mup_flags=""
case "$mup_mode" in
    sp)
        mup_flags=""
        run_suffix="sp"
        ;;
    mup)
        mup_flags="--mup_enabled --mup_width_multiplier=$mup_width_multiplier --mup_base_width=$mup_base_width"
        run_suffix="mup"
        ;;
    completep)
        mup_flags="--mup_enabled --mup_width_multiplier=$mup_width_multiplier --mup_base_width=$mup_base_width --depth_alpha_enabled --depth_multiplier=$depth_multiplier --depth_base_depth=$mup_base_depth --depth_alpha_exp=1.0"
        run_suffix="completep"
        ;;
    completep_05)
        mup_flags="--mup_enabled --mup_width_multiplier=$mup_width_multiplier --mup_base_width=$mup_base_width --depth_alpha_enabled --depth_multiplier=$depth_multiplier --depth_base_depth=$mup_base_depth --depth_alpha_exp=0.5"
        run_suffix="completep05"
        ;;
    *)
        echo "Error: Unknown mup_mode '$mup_mode'. Use: sp, mup, completep, or completep_05"
        exit 1
        ;;
esac

model_name=$(basename "$model_config" .json)
run_name="mono_${model_name}_${norm_type}_lr${learning_rates}_${monolingual_dataset}_${run_suffix}"
save_dir="$run_name"

echo "================================================================================"
echo "muP/CompleteP Training Launch"
echo "================================================================================"
echo "Target Model:    $model_name (Width: $hidden_size, Depth: $n_layer)"
echo "Base Reference:  Llama-9M (Width: $mup_base_width, Depth: $mup_base_depth)"
echo "Scaling Mode:    $mup_mode"
echo "Dataset:         $monolingual_dataset ($tokenizer_type, $vocab_size)"
echo "Base LR:         $learning_rates"
echo "Base EPS:        $adam_eps"
echo "Width Mult:      $mup_width_multiplier"
echo "Depth Mult:      $depth_multiplier"
echo "================================================================================"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port=$MASTER_PORT torchrun_main.py \
    --model_config "$model_config" \
    --lr $learning_rates \
    --eps $adam_eps \
    --beta1 $beta1 \
    --beta2 $beta2 \
    --optimizer adam \
    --weight_decay $weight_decay \
    --grad_clipping $grad_clipping \
    --batch_size 32 \
    --total_batch_size 512 \
    --num_epochs 10 \
    --warmup_steps_ratio 0.1 \
    --dtype bfloat16 \
    --eval_every 100 \
    --max_length 1024 \
    --run_name "$run_name" \
    --save_dir "$save_dir" \
    --monolingual-dataset $monolingual_dataset \
    --tokenizer_type $tokenizer_type \
    --tokenizer_vocabulary $vocab_size \
    --tokenizer_name t5-base \
    --hf_repo_name "shikhar-srivastava/${run_name}" \
    --hf_push_final --hf_push_checkpoints \
    $mup_flags
