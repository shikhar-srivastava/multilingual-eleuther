#!/usr/bin/env bash
# Launch Monolingual Training with muP/CompleteP Support
#
# This script iterates over datasets, vocabulary sizes, tokenizer types, AND muP modes
# to run comprehensive experiments comparing SP, muP, and CompleteP.
#
# Usage:
#   bash launch_monolingual_mup.sh [mup_mode]
#
# mup_mode options:
#   - "sp" : Standard Parameterization (baseline)
#   - "mup" : muP only (width scaling)
#   - "completep" : CompleteP (muP + depth scaling alpha=1.0)
#   - "completep_05" : CompleteP with alpha=0.5
#   - "all" : Run all modes (default)

set -euo pipefail

mup_mode_filter=${1:-all}

TRAIN_SCRIPT="/localdisk/ssrivas9/multilingual-eleuther/monolingual_130m_mup.sh"

echo "[Config] Using training script: $TRAIN_SCRIPT"
echo "[Config] muP mode filter: $mup_mode_filter"

# Datasets to train on
DATASETS=(eng_latn)  # Add: tha_thai urd_arab amh_ethi vie_latn

# Vocabulary sizes to test
VOCABS=(32768)  # Add: 8192 16384 49152 65536 81920 98304 114688 262144

# Tokenizer types
TOKENIZERS=(bpe_unscaled)  # Add: unigram_unscaled

# muP modes to run
if [[ "$mup_mode_filter" == "all" ]]; then
    MUP_MODES=(sp mup completep completep_05)
else
    MUP_MODES=($mup_mode_filter)
fi

MAX_SEQ_LEN=1024
GA=2

tokenize_fn() {
    local dataset=$1
    local tokenizer_type=$2
    local vocab=$3
    python /localdisk/ssrivas9/multilingual-eleuther/scripts/tokenize_and_pack.py \
        --dataset "$dataset" --tokenizer_type "$tokenizer_type" --tokenizer_vocabulary "$vocab" \
        --split train --max_seq_len $MAX_SEQ_LEN --max_segments -1 --prepend_cls True --include_sep True --shuffle True
    python /localdisk/ssrivas9/multilingual-eleuther/scripts/tokenize_and_pack.py \
        --dataset "$dataset" --tokenizer_type "$tokenizer_type" --tokenizer_vocabulary "$vocab" \
        --split eval --max_seq_len $MAX_SEQ_LEN --max_segments -1 --prepend_cls True --include_sep True --shuffle False
}

for dataset in "${DATASETS[@]}"; do
    for vocab in "${VOCABS[@]}"; do
        for tok in "${TOKENIZERS[@]}"; do
            echo "[Tokenize] dataset=$dataset, tok=$tok, vocab=$vocab"
            tokenize_fn "$dataset" "$tok" "$vocab"

            for mup_mode in "${MUP_MODES[@]}"; do
                echo "[Train] dataset=$dataset, tok=$tok, vocab=$vocab, mup_mode=$mup_mode"
                bash "$TRAIN_SCRIPT" pre "$dataset" "$vocab" "$tok" 6 29510 "$mup_mode"
            done
        done
    done
done

echo "Training complete!"

