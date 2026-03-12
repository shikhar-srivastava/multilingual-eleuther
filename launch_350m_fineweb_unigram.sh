#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local.env"
TRAIN_SCRIPT="${SCRIPT_DIR}/monolingual_350m.sh"

echo "[Config] Using training script: $TRAIN_SCRIPT"

DATASETS=(fineweb_eng)
VOCABS=(8192 32768 65536 98304)
TOKENIZERS=(unigram_unscaled)

MAX_SEQ_LEN=1024
TARGET_TOKENS=7000000000
EVAL_LINES=8000
OUTPUT_ROOT="${DATA_ROOT}/monolingual_training_data_tokenized"
INDEX_PATH="${SCRIPT_DIR}/configs/monolingual_bp_index.json"

tokenize_and_split_fn() {
  local dataset=$1
  local tokenizer_type=$2
  local vocab=$3

  # Derive tokenizer basename (fineweb_eng maps to eng_latn tokenizers)
  local tok_dataset="eng_latn"
  local prefix
  if [[ "$tokenizer_type" == "bpe_unscaled" ]]; then
    prefix="bpe"
  else
    prefix="unigram"
  fi
  local tok_basename="${prefix}_${tok_dataset}_${vocab}_300mb_unscaled"
  local tok_dir="${OUTPUT_ROOT}/${tok_basename}"
  local bp="1.0"
  local tok_file="${tok_dir}/${dataset}_${bp}_tokenized.txt"
  local full_file="${tok_dir}/${dataset}_${bp}_tokenized_full.txt"
  local eval_file="${tok_dir}/${dataset}_${bp}_eval_tokenized.txt"

  # Skip if the final train + eval splits already exist
  if [[ -f "$tok_file" && -f "$eval_file" ]]; then
    echo "  [Skip] Tokenized splits already exist: $tok_file"
  else
    # Step 1: Tokenize ALL raw text (full ~10BT) if not already done
    if [[ -f "$full_file" ]]; then
      echo "  [Step 1] Full tokenized file already exists: $full_file"
    else
      echo "  [Step 1] Tokenizing full dataset: dataset=$dataset, tok=$tokenizer_type, vocab=$vocab"
      python "${SCRIPT_DIR}/scripts/tokenize_and_pack.py" \
        --dataset "$dataset" --tokenizer_type "$tokenizer_type" --tokenizer_vocabulary "$vocab" \
        --split train --max_seq_len $MAX_SEQ_LEN --max_segments -1 \
        --prepend_cls True --include_sep True --shuffle False \
        --index_path "$INDEX_PATH" --output_root "$OUTPUT_ROOT"
      # Rename to _full so tokenize_and_pack won't see it as already done on next run
      mv "$tok_file" "$full_file"
    fi

    # Step 2: Split by 7B tokens
    echo "  [Step 2] Splitting: first ${TARGET_TOKENS} tokens for train, last ${EVAL_LINES} lines for eval"
    python "${SCRIPT_DIR}/scripts/split_tokenized.py" \
      --input "$full_file" \
      --train_output "$tok_file" \
      --eval_output "$eval_file" \
      --target_tokens $TARGET_TOKENS \
      --eval_lines $EVAL_LINES
  fi
}

for dataset in "${DATASETS[@]}"; do
  for vocab in "${VOCABS[@]}"; do
    for tok in "${TOKENIZERS[@]}"; do
      echo "[Tokenize+Split] dataset=$dataset, tok=$tok, vocab=$vocab"
      tokenize_and_split_fn "$dataset" "$tok" "$vocab"

      echo "[Train] dataset=$dataset, tok=$tok, vocab=$vocab"
      bash "$TRAIN_SCRIPT" pre "$dataset" "$vocab" "$tok" 6 29510
    done
  done
done
