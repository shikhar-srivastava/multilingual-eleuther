#!/bin/bash
# muTransfer Width/Depth Scaling - Standard Parameterization (SP)
#
# Uses torchrun_main.py directly (same as main training pipeline)
# with dynamically generated model configs.
#
# Usage:
#   bash run_sp.sh [DATASET] [EXPERIMENT_TYPE] [OUT_DIR] [GPU_ID]
#
# DATASET: eng_latn | tha_thai | urd_arab | amh_ethi | vie_latn
# EXPERIMENT_TYPE: width | depth | both
# OUT_DIR: out | out_test
# GPU_ID: 0 | 1 | 2 | 3 (default: 0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

# Parameters
DATASET=${1:-eng_latn}
EXPERIMENT_TYPE=${2:-width}  # width | depth | both
OUT_BASE=${3:-out}
GPU_ID=${4:-0}

# Fixed tokenizer settings (8192 vocab for small models)
TOKENIZER_TYPE="bpe_unscaled"
VOCAB_SIZE=8192

# Base reference for scaling (matches llama_9m.json base model)
BASE_WIDTH=128
BASE_DEPTH=4

# Width sweep settings (fix depth at BASE_DEPTH)
WIDTHS="128 256 512 768"

# Depth sweep settings (fix width at BASE_WIDTH)
DEPTHS="4 8 12 16"

# Learning rates to sweep
LRS="0.1 0.05 0.02 0.01 0.005 0.002 0.001 0.0005 0.0002 0.0001"

# Seeds for multiple runs
SEEDS="1 2 3"

# Training settings (5 epochs for muTransfer experiments)
NUM_EPOCHS=5
BATCH_SIZE=32
TOTAL_BATCH_SIZE=128
MAX_LENGTH=1024
WEIGHT_DECAY=0.1
GRAD_CLIP=1.0
WARMUP_RATIO=0.1
EVAL_EVERY=100

# Config directory
CONFIG_DIR="$SCRIPT_DIR/configs"
mkdir -p "$CONFIG_DIR"

# HF username for pushing
HF_USERNAME="shikhar-srivastava"

echo "============================================================"
echo "muTransfer LR Sweep - Standard Parameterization (SP)"
echo "============================================================"
echo "Dataset:         $DATASET"
echo "Tokenizer:       $TOKENIZER_TYPE (vocab=$VOCAB_SIZE)"
echo "Experiment:      $EXPERIMENT_TYPE"
echo "GPU:             $GPU_ID"
echo "Base Model:      width=$BASE_WIDTH, depth=$BASE_DEPTH (9M proxy)"
echo "Output:          $SCRIPT_DIR/sp/$OUT_BASE/"
echo ""

# Function to create model config
create_config() {
    local width=$1
    local depth=$2
    local config_path="$CONFIG_DIR/llama_w${width}_d${depth}.json"
    
    local head_size=64
    local n_heads=$((width / head_size))
    # Ensure n_heads is at least 2
    [ $n_heads -lt 2 ] && n_heads=2
    
    # Intermediate size: approximately 2.7x hidden_size (LLAMA style)
    local intermediate_size=$((width * 8 / 3))
    
    cat > "$config_path" << EOF
{
    "architectures": ["LLaMAForCausalLM"],
    "bos_token_id": 0,
    "eos_token_id": 1,
    "hidden_act": "silu",
    "hidden_size": $width,
    "intermediate_size": $intermediate_size,
    "initializer_range": 0.02,
    "max_sequence_length": $MAX_LENGTH,
    "max_position_embeddings": $MAX_LENGTH,
    "model_type": "llama",
    "num_attention_heads": $n_heads,
    "num_hidden_layers": $depth,
    "pad_token_id": -1,
    "position_embedding_type": "rope",
    "rms_norm_eps": 1e-06,
    "transformers_version": "4.28.1",
    "use_cache": true,
    "vocab_size": $VOCAB_SIZE
}
EOF
    echo "$config_path"
}

run_experiment() {
    local width=$1
    local depth=$2
    local lr=$3
    local seed=$4
    local scale_type=$5  # "width" or "depth"
    
    # Proper naming: muP_9M-base_<scale_type>-scaling_<lang>_w<W>_d<D>_lr<LR>_s<seed>_sp
    local run_name="muP_9M-base_${scale_type}-scaling_${DATASET}_w${width}_d${depth}_lr${lr}_s${seed}_sp"
    local out_dir="$SCRIPT_DIR/sp/$OUT_BASE/${DATASET}/width${width}_depth${depth}_seed${seed}_lr${lr}"
    mkdir -p "$out_dir"
    
    # Skip if already completed
    if [ -f "$out_dir/final_model/config.json" ]; then
        echo "  [SKIP] Already completed: $run_name"
        return 0
    fi
    
    local config_path=$(create_config $width $depth)
    
    echo "  Running: $run_name [GPU $GPU_ID]"
    
    # Use single GPU for these smaller experiments (torchrun with 1 process)
    CUDA_VISIBLE_DEVICES=$GPU_ID torchrun --standalone --nproc_per_node=1 torchrun_main.py \
        --model_config="$config_path" \
        --max_length=$MAX_LENGTH \
        --num_epochs=$NUM_EPOCHS \
        --batch_size=$BATCH_SIZE \
        --total_batch_size=$TOTAL_BATCH_SIZE \
        --lr=$lr \
        --weight_decay=$WEIGHT_DECAY \
        --eps=1e-8 \
        --beta1=0.9 \
        --beta2=0.95 \
        --grad_clipping=$GRAD_CLIP \
        --warmup_steps_ratio=$WARMUP_RATIO \
        --dtype=bfloat16 \
        --eval_every=$EVAL_EVERY \
        --optimizer=adam \
        --monolingual-dataset=$DATASET \
        --tokenizer_type=$TOKENIZER_TYPE \
        --tokenizer_vocabulary=$VOCAB_SIZE \
        --tokenizer_name=t5-base \
        --run_name="$run_name" \
        --save_dir="$out_dir" \
        --seed=$seed \
        --single_gpu \
        --hf_repo_name="${HF_USERNAME}/${run_name}" \
        --hf_push_final 2>&1 | tee "$out_dir/train.log" | tail -n 5
    
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "  [WARN] Run failed: $run_name (exit: $exit_code)"
    fi
}

# Width scaling experiments
if [[ "$EXPERIMENT_TYPE" == "width" || "$EXPERIMENT_TYPE" == "both" ]]; then
    echo ""
    echo "=== Width Scaling (fixed depth=$BASE_DEPTH) ==="
    echo "Widths: $WIDTHS"
    echo "Learning rates: $LRS"
    echo ""
    
    for width in $WIDTHS; do
        for lr in $LRS; do
            for seed in $SEEDS; do
                run_experiment $width $BASE_DEPTH $lr $seed "width"
            done
        done
    done
fi

# Depth scaling experiments
if [[ "$EXPERIMENT_TYPE" == "depth" || "$EXPERIMENT_TYPE" == "both" ]]; then
    echo ""
    echo "=== Depth Scaling (fixed width=$BASE_WIDTH) ==="
    echo "Depths: $DEPTHS"
    echo "Learning rates: $LRS"
    echo ""
    
    for depth in $DEPTHS; do
        for lr in $LRS; do
            for seed in $SEEDS; do
                run_experiment $BASE_WIDTH $depth $lr $seed "depth"
            done
        done
    done
fi

echo ""
echo "============================================================"
echo "SP experiment complete!"
echo "Results saved to: $SCRIPT_DIR/sp/$OUT_BASE/${DATASET}/"
echo "wandb runs tracked with prefix: muP_9M-base_*_sp"
echo "============================================================"
