#!/usr/bin/env bash
# Tokenize + split FineWeb for BPE tokenizers.
# Each block below is independent -- copy and run any single block on its own.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local.env"
OUTPUT_ROOT="${DATA_ROOT}/monolingual_training_data_tokenized"
INDEX_PATH="${SCRIPT_DIR}/configs/monolingual_bp_index.json"

# ============================================================
# BPE vocab=8192
# ============================================================
python "${SCRIPT_DIR}/scripts/tokenize_and_pack.py" \
  --dataset fineweb_eng --tokenizer_type bpe_unscaled --tokenizer_vocabulary 8192 \
  --split train --max_seq_len 1024 --max_segments -1 \
  --prepend_cls True --include_sep True --shuffle False \
  --index_path "$INDEX_PATH" --output_root "$OUTPUT_ROOT"

mv "${OUTPUT_ROOT}/bpe_eng_latn_8192_300mb_unscaled/fineweb_eng_1.0_tokenized.txt" \
   "${OUTPUT_ROOT}/bpe_eng_latn_8192_300mb_unscaled/fineweb_eng_1.0_tokenized_full.txt"

python "${SCRIPT_DIR}/scripts/split_tokenized.py" \
  --input  "${OUTPUT_ROOT}/bpe_eng_latn_8192_300mb_unscaled/fineweb_eng_1.0_tokenized_full.txt" \
  --train_output "${OUTPUT_ROOT}/bpe_eng_latn_8192_300mb_unscaled/fineweb_eng_1.0_tokenized.txt" \
  --eval_output  "${OUTPUT_ROOT}/bpe_eng_latn_8192_300mb_unscaled/fineweb_eng_1.0_eval_tokenized.txt" \
  --target_tokens 7000000000 --eval_lines 8000

  

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local.env"
OUTPUT_ROOT="${DATA_ROOT}/monolingual_training_data_tokenized"
INDEX_PATH="${SCRIPT_DIR}/configs/monolingual_bp_index.json"


python "${SCRIPT_DIR}/scripts/tokenize_and_pack.py" \
  --dataset fineweb_eng --tokenizer_type bpe_unscaled --tokenizer_vocabulary 32768 \
  --split train --max_seq_len 1024 --max_segments -1 \
  --prepend_cls True --include_sep True --shuffle False \
  --index_path "$INDEX_PATH" --output_root "$OUTPUT_ROOT"

mv "${OUTPUT_ROOT}/bpe_eng_latn_32768_300mb_unscaled/fineweb_eng_1.0_tokenized.txt" \
   "${OUTPUT_ROOT}/bpe_eng_latn_32768_300mb_unscaled/fineweb_eng_1.0_tokenized_full.txt"

python "${SCRIPT_DIR}/scripts/split_tokenized.py" \
  --input  "${OUTPUT_ROOT}/bpe_eng_latn_32768_300mb_unscaled/fineweb_eng_1.0_tokenized_full.txt" \
  --train_output "${OUTPUT_ROOT}/bpe_eng_latn_32768_300mb_unscaled/fineweb_eng_1.0_tokenized.txt" \
  --eval_output  "${OUTPUT_ROOT}/bpe_eng_latn_32768_300mb_unscaled/fineweb_eng_1.0_eval_tokenized.txt" \
  --target_tokens 7000000000 --eval_lines 8000



SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local.env"
OUTPUT_ROOT="${DATA_ROOT}/monolingual_training_data_tokenized"
INDEX_PATH="${SCRIPT_DIR}/configs/monolingual_bp_index.json"


python "${SCRIPT_DIR}/scripts/tokenize_and_pack.py" \
  --dataset fineweb_eng --tokenizer_type bpe_unscaled --tokenizer_vocabulary 65536 \
  --split train --max_seq_len 1024 --max_segments -1 \
  --prepend_cls True --include_sep True --shuffle False \
  --index_path "$INDEX_PATH" --output_root "$OUTPUT_ROOT"

mv "${OUTPUT_ROOT}/bpe_eng_latn_65536_300mb_unscaled/fineweb_eng_1.0_tokenized.txt" \
   "${OUTPUT_ROOT}/bpe_eng_latn_65536_300mb_unscaled/fineweb_eng_1.0_tokenized_full.txt"

python "${SCRIPT_DIR}/scripts/split_tokenized.py" \
  --input  "${OUTPUT_ROOT}/bpe_eng_latn_65536_300mb_unscaled/fineweb_eng_1.0_tokenized_full.txt" \
  --train_output "${OUTPUT_ROOT}/bpe_eng_latn_65536_300mb_unscaled/fineweb_eng_1.0_tokenized.txt" \
  --eval_output  "${OUTPUT_ROOT}/bpe_eng_latn_65536_300mb_unscaled/fineweb_eng_1.0_eval_tokenized.txt" \
  --target_tokens 7000000000 --eval_lines 8000

  


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local.env"
OUTPUT_ROOT="${DATA_ROOT}/monolingual_training_data_tokenized"
INDEX_PATH="${SCRIPT_DIR}/configs/monolingual_bp_index.json"


python "${SCRIPT_DIR}/scripts/tokenize_and_pack.py" \
  --dataset fineweb_eng --tokenizer_type bpe_unscaled --tokenizer_vocabulary 98304 \
  --split train --max_seq_len 1024 --max_segments -1 \
  --prepend_cls True --include_sep True --shuffle False \
  --index_path "$INDEX_PATH" --output_root "$OUTPUT_ROOT"

mv "${OUTPUT_ROOT}/bpe_eng_latn_98304_300mb_unscaled/fineweb_eng_1.0_tokenized.txt" \
   "${OUTPUT_ROOT}/bpe_eng_latn_98304_300mb_unscaled/fineweb_eng_1.0_tokenized_full.txt"

python "${SCRIPT_DIR}/scripts/split_tokenized.py" \
  --input  "${OUTPUT_ROOT}/bpe_eng_latn_98304_300mb_unscaled/fineweb_eng_1.0_tokenized_full.txt" \
  --train_output "${OUTPUT_ROOT}/bpe_eng_latn_98304_300mb_unscaled/fineweb_eng_1.0_tokenized.txt" \
  --eval_output  "${OUTPUT_ROOT}/bpe_eng_latn_98304_300mb_unscaled/fineweb_eng_1.0_eval_tokenized.txt" \
  --target_tokens 7000000000 --eval_lines 8000


