#!/usr/bin/env bash
# Compare Standard Parameterization (SP), muP, and CompleteP
# This demonstrates the difference between the three approaches

set -euo pipefail

export NORM_TYPE=pre
export OMP_NUM_THREADS=8

BASE_WIDTH=256
BASE_DEPTH=2
DEPTH=8  # Test at 4x base depth
WIDTH=256
HEAD_SIZE=64
N_HEADS=$((WIDTH / HEAD_SIZE))

LEARNING_RATE=1e-3
MAX_ITERS=10
BATCH_SIZE=8
MAX_LENGTH=256

DEPTH_MULTIPLIER=$(echo "scale=8; $DEPTH/$BASE_DEPTH" | bc -l)
WIDTH_MULTIPLIER=1.0  # Testing at base width

echo "=============================================================================="
echo "Comparing Parameterizations: SP vs muP vs CompleteP"
echo "=============================================================================="
echo "Configuration:"
echo "  Depth: $DEPTH (${DEPTH_MULTIPLIER}x base depth of $BASE_DEPTH)"
echo "  Width: $WIDTH (${WIDTH_MULTIPLIER}x base width of $BASE_WIDTH)"
echo "  This tests DEPTH scaling at a fixed width"
echo "=============================================================================="

# 1. Standard Parameterization (SP) - baseline, will have issues at depth
echo ""
echo "1. Standard Parameterization (SP)"
echo "   - No width or depth scaling"
echo "   - Expected: Activations may explode/vanish with depth"
python coord_check_train.py \
    --out_dir=coord_check_out/comparison/sp \
    --n_layer=$DEPTH \
    --hidden_size=$WIDTH \
    --num_attention_heads=$N_HEADS \
    --lr=$LEARNING_RATE \
    --max_iters=$MAX_ITERS \
    --batch_size=$BATCH_SIZE \
    --max_length=$MAX_LENGTH \
    --mup_enable_coord_check_logging \
    --csv_log \
    --seed=1

# 2. muP only - fixes width scaling but not depth
echo ""
echo "2. muP only (without CompleteP depth scaling)"
echo "   - Width scaling enabled"
echo "   - No depth scaling"
echo "   - Expected: Better than SP, but may still have depth issues"
python coord_check_train.py \
    --out_dir=coord_check_out/comparison/mup_only \
    --n_layer=$DEPTH \
    --hidden_size=$WIDTH \
    --num_attention_heads=$N_HEADS \
    --lr=$LEARNING_RATE \
    --max_iters=$MAX_ITERS \
    --batch_size=$BATCH_SIZE \
    --max_length=$MAX_LENGTH \
    --mup_enabled \
    --mup_width_multiplier=$WIDTH_MULTIPLIER \
    --mup_base_width=$BASE_WIDTH \
    --mup_enable_coord_check_logging \
    --csv_log \
    --seed=1

# 3. CompleteP - full muP + depth scaling
echo ""
echo "3. CompleteP (muP + depth scaling)"
echo "   - Width scaling enabled"
echo "   - Depth scaling enabled with alpha=1.0"
echo "   - Expected: Stable activations across both width and depth"
python coord_check_train.py \
    --out_dir=coord_check_out/comparison/completep \
    --n_layer=$DEPTH \
    --hidden_size=$WIDTH \
    --num_attention_heads=$N_HEADS \
    --lr=$LEARNING_RATE \
    --max_iters=$MAX_ITERS \
    --batch_size=$BATCH_SIZE \
    --max_length=$MAX_LENGTH \
    --mup_enabled \
    --mup_width_multiplier=$WIDTH_MULTIPLIER \
    --mup_base_width=$BASE_WIDTH \
    --depth_alpha_enabled \
    --depth_multiplier=$DEPTH_MULTIPLIER \
    --depth_base_depth=$BASE_DEPTH \
    --depth_alpha_exp=1.0 \
    --mup_enable_coord_check_logging \
    --csv_log \
    --seed=1

echo ""
echo "=============================================================================="
echo "✅ Comparison complete!"
echo "=============================================================================="
echo ""
echo "Results:"
echo "  SP:         coord_check_out/comparison/sp/csv_logs.csv"
echo "  muP only:   coord_check_out/comparison/mup_only/csv_logs.csv"
echo "  CompleteP:  coord_check_out/comparison/completep/csv_logs.csv"
echo ""
echo "Key metrics to compare:"
echo "  - token_embedding_act_abs_mean: Should be stable for muP and CompleteP"
echo "  - attn_act_abs_mean: Should be stable for CompleteP"
echo "  - mlp_act_abs_mean: Should be stable for CompleteP"
echo "  - lm_head_act_abs_mean: Should be stable for CompleteP"
echo ""
echo "Expected results:"
echo "  SP:         Activations may explode/vanish"
echo "  muP only:   Better, but still depth-dependent"
echo "  CompleteP:  Stable activations (closest to base config)"
echo "=============================================================================="

