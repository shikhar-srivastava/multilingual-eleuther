#!/bin/bash
# Quick Test: Multilingual muTransfer Experiments
#
# Runs a minimal version of the experiment with:
# - Only 2 widths (128, 256)
# - Only 2 depths (4, 8)
# - Only 3 learning rates
# - Only 1 epoch
# - Only 1 seed
# - Only English language
#
# Usage:
#   bash run_quick_test.sh [sp|completep|both]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

PARAMETERIZATION=${1:-both}  # sp | completep | both

echo "========================================================================"
echo "Quick Test: Multilingual muTransfer Experiments"
echo "========================================================================"

# Quick test settings
DATASET="eng_latn"
TOKENIZER_TYPE="bpe_unscaled"
VOCAB_SIZE=8192
BASE_WIDTH=128
BASE_DEPTH=4
DEPTH_ALPHA_EXP=1.0

WIDTHS="128 256"
DEPTHS="4 8"
LRS="0.01 0.002 0.0005"  # 3 LRs
SEED=1
NUM_EPOCHS=1
BATCH_SIZE=16
TOTAL_BATCH_SIZE=32
MAX_LENGTH=256
EVAL_EVERY=50

# Config directory
CONFIG_DIR="$SCRIPT_DIR/configs"
mkdir -p "$CONFIG_DIR"

# HF username
HF_USERNAME="shikhar-srivastava"

echo ""
echo "Settings:"
echo "  Dataset:    $DATASET"
echo "  Widths:     $WIDTHS"
echo "  Depths:     $DEPTHS"
echo "  LRs:        $LRS"
echo "  Epochs:     $NUM_EPOCHS"
echo "  Seed:       $SEED"
echo ""

# Function to create model config
create_config() {
    local width=$1
    local depth=$2
    local config_path="$CONFIG_DIR/llama_w${width}_d${depth}.json"
    
    local head_size=64
    local n_heads=$((width / head_size))
    [ $n_heads -lt 2 ] && n_heads=2
    
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

run_sp() {
    local width=$1
    local depth=$2
    local lr=$3
    local scale_type=$4
    
    local run_name="muP_9M-base_${scale_type}-scaling_${DATASET}_w${width}_d${depth}_lr${lr}_s${SEED}_sp_test"
    local out_dir="$SCRIPT_DIR/sp/out_test/${DATASET}/width${width}_depth${depth}_seed${SEED}_lr${lr}"
    mkdir -p "$out_dir"
    
    local config_path=$(create_config $width $depth)
    
    echo "  SP: $run_name"
    
    CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 torchrun_main.py \
        --model_config="$config_path" \
        --max_length=$MAX_LENGTH \
        --num_epochs=$NUM_EPOCHS \
        --batch_size=$BATCH_SIZE \
        --total_batch_size=$TOTAL_BATCH_SIZE \
        --lr=$lr \
        --weight_decay=0.1 \
        --eps=1e-8 \
        --grad_clipping=1.0 \
        --warmup_steps_ratio=0.1 \
        --dtype=bfloat16 \
        --eval_every=$EVAL_EVERY \
        --optimizer=adam \
        --monolingual-dataset=$DATASET \
        --tokenizer_type=$TOKENIZER_TYPE \
        --tokenizer_vocabulary=$VOCAB_SIZE \
        --tokenizer_name=t5-base \
        --run_name="$run_name" \
        --save_dir="$out_dir" \
        --seed=$SEED \
        --single_gpu 2>&1 | tee "$out_dir/train.log" | tail -n 3
}

run_completep() {
    local width=$1
    local depth=$2
    local lr=$3
    local scale_type=$4
    
    local mup_width_multiplier=$(echo "scale=8; $width/$BASE_WIDTH" | bc -l)
    local depth_multiplier=$(echo "scale=8; $depth/$BASE_DEPTH" | bc -l)
    
    local run_name="muP_9M-base_${scale_type}-scaling_${DATASET}_w${width}_d${depth}_lr${lr}_s${SEED}_completep_test"
    local out_dir="$SCRIPT_DIR/completep/out_test/${DATASET}/width${width}_depth${depth}_seed${SEED}_lr${lr}"
    mkdir -p "$out_dir"
    
    local config_path=$(create_config $width $depth)
    
    echo "  CompleteP: $run_name (w_mult=$mup_width_multiplier, d_mult=$depth_multiplier)"
    
    CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 torchrun_main.py \
        --model_config="$config_path" \
        --max_length=$MAX_LENGTH \
        --num_epochs=$NUM_EPOCHS \
        --batch_size=$BATCH_SIZE \
        --total_batch_size=$TOTAL_BATCH_SIZE \
        --lr=$lr \
        --weight_decay=0.1 \
        --eps=1e-12 \
        --grad_clipping=1.0 \
        --warmup_steps_ratio=0.1 \
        --dtype=bfloat16 \
        --eval_every=$EVAL_EVERY \
        --optimizer=adam \
        --monolingual-dataset=$DATASET \
        --tokenizer_type=$TOKENIZER_TYPE \
        --tokenizer_vocabulary=$VOCAB_SIZE \
        --tokenizer_name=t5-base \
        --run_name="$run_name" \
        --save_dir="$out_dir" \
        --seed=$SEED \
        --single_gpu \
        --mup_enabled \
        --mup_width_multiplier=$mup_width_multiplier \
        --mup_base_width=$BASE_WIDTH \
        --mup_input_alpha=1.0 \
        --mup_output_alpha=1.0 \
        --depth_alpha_enabled \
        --depth_multiplier=$depth_multiplier \
        --depth_base_depth=$BASE_DEPTH \
        --depth_alpha_exp=$DEPTH_ALPHA_EXP 2>&1 | tee "$out_dir/train.log" | tail -n 3
}

# Width scaling tests
echo ""
echo "=== Width Scaling Tests (depth=$BASE_DEPTH) ==="
for width in $WIDTHS; do
    for lr in $LRS; do
        if [[ "$PARAMETERIZATION" == "sp" || "$PARAMETERIZATION" == "both" ]]; then
            run_sp $width $BASE_DEPTH $lr "width"
        fi
        if [[ "$PARAMETERIZATION" == "completep" || "$PARAMETERIZATION" == "both" ]]; then
            run_completep $width $BASE_DEPTH $lr "width"
        fi
    done
done

# Depth scaling tests
echo ""
echo "=== Depth Scaling Tests (width=$BASE_WIDTH) ==="
for depth in $DEPTHS; do
    for lr in $LRS; do
        if [[ "$PARAMETERIZATION" == "sp" || "$PARAMETERIZATION" == "both" ]]; then
            run_sp $BASE_WIDTH $depth $lr "depth"
        fi
        if [[ "$PARAMETERIZATION" == "completep" || "$PARAMETERIZATION" == "both" ]]; then
            run_completep $BASE_WIDTH $depth $lr "depth"
        fi
    done
done

echo ""
echo "========================================================================"
echo "Quick test complete!"
echo "========================================================================"
echo "Results saved to:"
echo "  - $SCRIPT_DIR/sp/out_test/$DATASET/"
echo "  - $SCRIPT_DIR/completep/out_test/$DATASET/"
echo ""
echo "To export wandb metrics and plot:"
echo "  python $SCRIPT_DIR/export_wandb_metrics.py --project YOUR_PROJECT --filter 'muP_9M-base_*_test'"
echo "  python $SCRIPT_DIR/plot_mutransfer_multilingual.py --out_dir out_test --languages $DATASET"
echo ""
