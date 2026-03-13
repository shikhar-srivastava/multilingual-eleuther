#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local.env"
TRAIN_SCRIPT="${SCRIPT_DIR}/monolingual_350m_8gpu.sh"

echo "[Config] Using training script: $TRAIN_SCRIPT"

DATASETS=(fineweb2_vie)
VOCABS=(8192 98304)
TOKENIZERS=(bpe_unscaled unigram_unscaled)

MAX_SEQ_LEN=1024
OUTPUT_ROOT="${DATA_ROOT}/monolingual_training_data_tokenized"
INDEX_PATH="${SCRIPT_DIR}/configs/monolingual_bp_index.json"

# Per-language config: tokenizer dataset, byte premium, eval lines
declare -A TOK_DATASET_MAP=( [fineweb2_amh]=amh_ethi [fineweb2_tha]=tha_thai [fineweb2_urd]=urd_arab [fineweb2_vie]=vie_latn )
declare -A BYTE_PREMIUM_MAP=( [fineweb2_amh]=1.72 [fineweb2_tha]=2.74 [fineweb2_urd]=1.71 [fineweb2_vie]=1.35 )
declare -A EVAL_LINES_MAP=( [fineweb2_amh]=13760 [fineweb2_tha]=21920 [fineweb2_urd]=13680 [fineweb2_vie]=10800 )
declare -A LANG_CODE_MAP=( [fineweb2_amh]=amh [fineweb2_tha]=tha [fineweb2_urd]=urd [fineweb2_vie]=vie )

# Verify raw text files exist (run download_fineweb2.sh first)
for dataset in "${DATASETS[@]}"; do
  raw_file="${DATA_ROOT}/monolingual_training_data/${dataset}.txt"
  if [[ ! -f "$raw_file" ]]; then
    echo "[ERROR] Raw text file not found: $raw_file"
    echo "Run download_fineweb2.sh first."
    exit 1
  fi
done

tokenize_and_split_fn() {
  local dataset=$1
  local tokenizer_type=$2
  local vocab=$3

  local tok_dataset="${TOK_DATASET_MAP[$dataset]}"
  local bp="${BYTE_PREMIUM_MAP[$dataset]}"
  local eval_lines="${EVAL_LINES_MAP[$dataset]}"

  local prefix
  if [[ "$tokenizer_type" == "bpe_unscaled" ]]; then
    prefix="bpe"
  else
    prefix="unigram"
  fi
  local tok_basename="${prefix}_${tok_dataset}_${vocab}_300mb_unscaled"
  local tok_dir="${OUTPUT_ROOT}/${tok_basename}"
  local tok_file="${tok_dir}/${dataset}_${bp}_tokenized.txt"
  local full_file="${tok_dir}/${dataset}_${bp}_tokenized_full.txt"
  local eval_file="${tok_dir}/${dataset}_${bp}_eval_tokenized.txt"

  # Look up English reference bytes for this tokenizer/vocab
  local eng_tok_basename="${prefix}_eng_latn_${vocab}_300mb_unscaled"
  local eng_ref_file="${OUTPUT_ROOT}/${eng_tok_basename}/fineweb_eng_1.0_tokenized.txt"
  if [[ ! -f "$eng_ref_file" ]]; then
    echo "  [ERROR] English reference file not found: $eng_ref_file"
    echo "  Run the English FineWeb pipeline first (launch_350m_fineweb.sh)."
    return 1
  fi
  local eng_ref_bytes
  eng_ref_bytes=$(stat -c%s "$eng_ref_file")
  local target_bytes
  target_bytes=$(python3 -c "import math; print(math.floor($bp * $eng_ref_bytes))")
  echo "  [Config] eng_ref=$eng_ref_bytes bytes, BP=$bp, target_bytes=$target_bytes, eval_lines=$eval_lines"

  # Skip if the final train + eval splits already exist
  if [[ -f "$tok_file" && -f "$eval_file" ]]; then
    echo "  [Skip] Tokenized splits already exist: $tok_file"
  else
    # Step 1: Tokenize ALL raw text if not already done
    if [[ -f "$full_file" ]]; then
      echo "  [Step 1] Full tokenized file already exists: $full_file"
    else
      echo "  [Step 1] Tokenizing full dataset: dataset=$dataset, tok=$tokenizer_type, vocab=$vocab"
      python "${SCRIPT_DIR}/scripts/tokenize_and_pack.py" \
        --dataset "$dataset" --tokenizer_type "$tokenizer_type" --tokenizer_vocabulary "$vocab" \
        --split train --max_seq_len $MAX_SEQ_LEN --max_segments -1 \
        --prepend_cls True --include_sep True --shuffle False \
        --index_path "$INDEX_PATH" --output_root "$OUTPUT_ROOT"
      mv "$tok_file" "$full_file"
    fi

    # Step 2: Split by target bytes (with disjointness guarantee)
    echo "  [Step 2] Splitting: first ${target_bytes} bytes for train, last ${eval_lines} lines for eval"
    python "${SCRIPT_DIR}/scripts/split_tokenized.py" \
      --input "$full_file" \
      --train_output "$tok_file" \
      --eval_output "$eval_file" \
      --target_bytes "$target_bytes" \
      --eval_lines "$eval_lines"
  fi
}

for dataset in "${DATASETS[@]}"; do
  for vocab in "${VOCABS[@]}"; do
    for tok in "${TOKENIZERS[@]}"; do
      echo "================================================================"
      echo "[Tokenize+Split] dataset=$dataset, tok=$tok, vocab=$vocab"
      tokenize_and_split_fn "$dataset" "$tok" "$vocab"

      # Compute dynamic epoch count
      local_bp="${BYTE_PREMIUM_MAP[$dataset]}"
      local_tok_dataset="${TOK_DATASET_MAP[$dataset]}"
      if [[ "$tok" == "bpe_unscaled" ]]; then
        local_prefix="bpe"
      else
        local_prefix="unigram"
      fi
      local_tok_basename="${local_prefix}_${local_tok_dataset}_${vocab}_300mb_unscaled"
      local_tok_dir="${OUTPUT_ROOT}/${local_tok_basename}"
      local_tok_file="${local_tok_dir}/${dataset}_${local_bp}_tokenized.txt"

      eng_tok_basename="${local_prefix}_eng_latn_${vocab}_300mb_unscaled"
      eng_ref_file="${OUTPUT_ROOT}/${eng_tok_basename}/fineweb_eng_1.0_tokenized.txt"
      eng_ref_bytes=$(stat -c%s "$eng_ref_file")
      target_bytes=$(python3 -c "import math; print(math.floor($local_bp * $eng_ref_bytes))")
      actual_bytes=$(stat -c%s "$local_tok_file")

      if (( actual_bytes < target_bytes )); then
        NUM_EPOCHS=$(python3 -c "print(min(20, $target_bytes // $actual_bytes))")
      else
        NUM_EPOCHS=1
      fi
      echo "  [Epochs] target=$target_bytes actual=$actual_bytes -> epochs=$NUM_EPOCHS"

      echo "[Train] dataset=$dataset, tok=$tok, vocab=$vocab, epochs=$NUM_EPOCHS"
      bash "$TRAIN_SCRIPT" pre "$dataset" "$vocab" "$tok" 6 29510 "$NUM_EPOCHS"
    done
  done
done
