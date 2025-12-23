#!/bin/bash
# muTransfer Width/Depth Scaling - CompleteP (muP + Depth Scaling)
#
# Uses torchrun_main.py directly (same as main training pipeline)
# with dynamically generated model configs.
#
# Usage:
#   bash run_completep.sh [DATASET] [EXPERIMENT_TYPE] [OUT_DIR] [GPU_ID] [SINGLE_VALUE]
#
# DATASET: eng_latn | tha_thai | urd_arab | amh_ethi | vie_latn
# EXPERIMENT_TYPE: width | depth | both
# OUT_DIR: out | out_test
# GPU_ID: 0 | 1 | 2 | 3 (default: 0)
# SINGLE_VALUE: (optional) Run only this specific width/depth value
#               For width experiments: 128 | 256 | 512 | 768
#               For depth experiments: 4 | 8 | 12 | 16

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

# Parameters
DATASET=${1:-eng_latn}
EXPERIMENT_TYPE=${2:-width}  # width | depth | both
OUT_BASE=${3:-out}
GPU_ID=${4:-0}
SINGLE_VALUE=${5:-}
ENABLE_HF_UPLOAD=${ENABLE_HF_UPLOAD:-false}

# Fixed tokenizer settings (8192 vocab for small models)
TOKENIZER_TYPE="bpe_unscaled"
VOCAB_SIZE=8192

# Base reference for muP/CompleteP scaling (matches llama_9m.json)
BASE_WIDTH=128
BASE_DEPTH=4
DEPTH_ALPHA_EXP=1.0  # Full CompleteP depth scaling

# Width sweep settings (fix depth at BASE_DEPTH)
WIDTHS="128 256 512 768"

# Depth sweep settings (fix width at BASE_WIDTH)
DEPTHS="4 8 12 16"

# Learning rates to sweep
LRS_WIDTH="0.1 0.05 0.02 0.01 0.005 0.002 0.001 0.0005 0.0002 0.0001"
LRS_DEPTH="0.05 0.02 0.01 0.005 0.002 0.001"  # 6 LRs sufficient for depth

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
ADAM_EPS=1e-12  # CompleteP uses smaller epsilon

# Config directory
CONFIG_DIR="$SCRIPT_DIR/configs"
mkdir -p "$CONFIG_DIR"

# HF username for pushing
HF_USERNAME="shikhar-srivastava"

echo "============================================================"
echo "muTransfer LR Sweep - CompleteP (μP + Depth Scaling)"
echo "============================================================"
echo "Dataset:         $DATASET"
echo "Tokenizer:       $TOKENIZER_TYPE (vocab=$VOCAB_SIZE)"
echo "Experiment:      $EXPERIMENT_TYPE"
echo "GPU:             $GPU_ID"
if [ -n "$SINGLE_VALUE" ]; then
    echo "Single Value:    $SINGLE_VALUE (running only this width/depth)"
fi
echo "Base Model:      width=$BASE_WIDTH, depth=$BASE_DEPTH (9M proxy)"
echo "Depth Alpha:     $DEPTH_ALPHA_EXP"
echo "Output:          $SCRIPT_DIR/completep/$OUT_BASE/"
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
    
    # Compute muP/CompleteP multipliers
    local mup_width_multiplier=$(echo "scale=8; $width/$BASE_WIDTH" | bc -l)
    local depth_multiplier=$(echo "scale=8; $depth/$BASE_DEPTH" | bc -l)
    
    # Proper naming: muP_9M-base_<scale_type>-scaling_<lang>_w<W>_d<D>_lr<LR>_s<seed>_completep
    local run_name="muP_9M-base_${scale_type}-scaling_${DATASET}_w${width}_d${depth}_lr${lr}_s${seed}_completep"
    local out_dir="$SCRIPT_DIR/completep/$OUT_BASE/${DATASET}/width${width}_depth${depth}_seed${seed}_lr${lr}"
    local hf_flags=()
    if [ "${ENABLE_HF_UPLOAD}" = "true" ]; then
        hf_flags=(--hf_repo_name "${HF_USERNAME}/${run_name}" --hf_push_final)
    fi

    # Skip if already completed
    if [ -f "$out_dir/final_model/config.json" ]; then
        echo "  [SKIP] Already completed: $run_name"
        return 0
    fi

    # Clean any partial run to restart from scratch
    if [ -d "$out_dir" ]; then
        echo "  [RESET] Incomplete run found. Removing $out_dir to restart from scratch."
        rm -rf "$out_dir"
    fi
    mkdir -p "$out_dir"
    
    local config_path=$(create_config $width $depth)
    
    echo "  Running: $run_name (w_mult=$mup_width_multiplier, d_mult=$depth_multiplier) [GPU $GPU_ID]"
    
    # Use single GPU for these smaller experiments (torchrun with 1 process)
    CUDA_VISIBLE_DEVICES=$GPU_ID torchrun --standalone --nproc_per_node=1 torchrun_main.py \
        --model_config="$config_path" \
        --max_length=$MAX_LENGTH \
        --num_epochs=$NUM_EPOCHS \
        --batch_size=$BATCH_SIZE \
        --total_batch_size=$TOTAL_BATCH_SIZE \
        --lr=$lr \
        --weight_decay=$WEIGHT_DECAY \
        --eps=$ADAM_EPS \
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
        --mup_enabled \
        --mup_width_multiplier=$mup_width_multiplier \
        --mup_base_width=$BASE_WIDTH \
        --mup_input_alpha=1.0 \
        --mup_output_alpha=1.0 \
        --depth_alpha_enabled \
        --depth_multiplier=$depth_multiplier \
        --depth_base_depth=$BASE_DEPTH \
        --depth_alpha_exp=$DEPTH_ALPHA_EXP \
        --disable_hf_upload \
        "${hf_flags[@]}" 2>&1 | tee "$out_dir/train.log" | tail -n 5
    
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "  [WARN] Run failed: $run_name (exit: $exit_code)"
    fi
}

# Width scaling experiments
if [[ "$EXPERIMENT_TYPE" == "width" || "$EXPERIMENT_TYPE" == "both" ]]; then
    # If SINGLE_VALUE is set, only run that width
    if [ -n "$SINGLE_VALUE" ]; then
        RUN_WIDTHS="$SINGLE_VALUE"
    else
        RUN_WIDTHS="$WIDTHS"
    fi
    
    echo ""
    echo "=== Width Scaling (fixed depth=$BASE_DEPTH) ==="
    echo "Widths: $RUN_WIDTHS"
    echo "Learning rates: $LRS_WIDTH"
    echo ""
    
    for width in $RUN_WIDTHS; do
        for lr in $LRS_WIDTH; do
            for seed in $SEEDS; do
                run_experiment $width $BASE_DEPTH $lr $seed "width"
            done
        done
    done
fi

# Depth scaling experiments
if [[ "$EXPERIMENT_TYPE" == "depth" || "$EXPERIMENT_TYPE" == "both" ]]; then
    # If SINGLE_VALUE is set, only run that depth
    if [ -n "$SINGLE_VALUE" ]; then
        RUN_DEPTHS="$SINGLE_VALUE"
    else
        RUN_DEPTHS="$DEPTHS"
    fi
    
    echo ""
    echo "=== Depth Scaling (fixed width=$BASE_WIDTH) ==="
    echo "Depths: $RUN_DEPTHS"
    echo "Learning rates: $LRS_DEPTH"
    echo ""
    
    for depth in $RUN_DEPTHS; do
        for lr in $LRS_DEPTH; do
            for seed in $SEEDS; do
                run_experiment $BASE_WIDTH $depth $lr $seed "depth"
            done
        done
    done
fi

echo ""
echo "============================================================"
echo "CompleteP experiment complete!"
echo "Results saved to: $SCRIPT_DIR/completep/$OUT_BASE/${DATASET}/"
echo "wandb runs tracked with prefix: muP_9M-base_*_completep"
echo "============================================================"
