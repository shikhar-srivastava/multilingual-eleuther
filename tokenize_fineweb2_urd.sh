#!/usr/bin/env bash
# Tokenize + split FineWeb-2 Urdu (fineweb2_urd, byte_premium=1.71, eval_lines=13680)
# for BPE and Unigram tokenizers across all vocab sizes.
# Each block below is independent -- copy and run any single block on its own.

# ============================================================
# BPE vocab=8192
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local.env"
OUTPUT_ROOT="${DATA_ROOT}/monolingual_training_data_tokenized"
INDEX_PATH="${SCRIPT_DIR}/configs/monolingual_bp_index.json"
ENG_REF="${OUTPUT_ROOT}/bpe_eng_latn_8192_300mb_unscaled/fineweb_eng_1.0_tokenized.txt"
ENG_BYTES=$(stat -c%s "$ENG_REF")
TARGET_BYTES=$(python3 -c "import math; print(math.floor(1.71 * $ENG_BYTES))")
TOK_DIR="${OUTPUT_ROOT}/bpe_urd_arab_8192_300mb_unscaled"

python "${SCRIPT_DIR}/scripts/tokenize_and_pack.py" \
  --dataset fineweb2_urd --tokenizer_type bpe_unscaled --tokenizer_vocabulary 8192 \
  --split train --max_seq_len 1024 --max_segments -1 \
  --prepend_cls True --include_sep True --shuffle False \
  --index_path "$INDEX_PATH" --output_root "$OUTPUT_ROOT"

mv "${TOK_DIR}/fineweb2_urd_1.71_tokenized.txt" \
   "${TOK_DIR}/fineweb2_urd_1.71_tokenized_full.txt"

python "${SCRIPT_DIR}/scripts/split_tokenized.py" \
  --input  "${TOK_DIR}/fineweb2_urd_1.71_tokenized_full.txt" \
  --train_output "${TOK_DIR}/fineweb2_urd_1.71_tokenized.txt" \
  --eval_output  "${TOK_DIR}/fineweb2_urd_1.71_eval_tokenized.txt" \
  --target_bytes "$TARGET_BYTES" --eval_lines 13680


# ============================================================
# BPE vocab=32768
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local.env"
OUTPUT_ROOT="${DATA_ROOT}/monolingual_training_data_tokenized"
INDEX_PATH="${SCRIPT_DIR}/configs/monolingual_bp_index.json"
ENG_REF="${OUTPUT_ROOT}/bpe_eng_latn_32768_300mb_unscaled/fineweb_eng_1.0_tokenized.txt"
ENG_BYTES=$(stat -c%s "$ENG_REF")
TARGET_BYTES=$(python3 -c "import math; print(math.floor(1.71 * $ENG_BYTES))")
TOK_DIR="${OUTPUT_ROOT}/bpe_urd_arab_32768_300mb_unscaled"

python "${SCRIPT_DIR}/scripts/tokenize_and_pack.py" \
  --dataset fineweb2_urd --tokenizer_type bpe_unscaled --tokenizer_vocabulary 32768 \
  --split train --max_seq_len 1024 --max_segments -1 \
  --prepend_cls True --include_sep True --shuffle False \
  --index_path "$INDEX_PATH" --output_root "$OUTPUT_ROOT"

mv "${TOK_DIR}/fineweb2_urd_1.71_tokenized.txt" \
   "${TOK_DIR}/fineweb2_urd_1.71_tokenized_full.txt"

python "${SCRIPT_DIR}/scripts/split_tokenized.py" \
  --input  "${TOK_DIR}/fineweb2_urd_1.71_tokenized_full.txt" \
  --train_output "${TOK_DIR}/fineweb2_urd_1.71_tokenized.txt" \
  --eval_output  "${TOK_DIR}/fineweb2_urd_1.71_eval_tokenized.txt" \
  --target_bytes "$TARGET_BYTES" --eval_lines 13680


# ============================================================
# BPE vocab=65536
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local.env"
OUTPUT_ROOT="${DATA_ROOT}/monolingual_training_data_tokenized"
INDEX_PATH="${SCRIPT_DIR}/configs/monolingual_bp_index.json"
ENG_REF="${OUTPUT_ROOT}/bpe_eng_latn_65536_300mb_unscaled/fineweb_eng_1.0_tokenized.txt"
ENG_BYTES=$(stat -c%s "$ENG_REF")
TARGET_BYTES=$(python3 -c "import math; print(math.floor(1.71 * $ENG_BYTES))")
TOK_DIR="${OUTPUT_ROOT}/bpe_urd_arab_65536_300mb_unscaled"

python "${SCRIPT_DIR}/scripts/tokenize_and_pack.py" \
  --dataset fineweb2_urd --tokenizer_type bpe_unscaled --tokenizer_vocabulary 65536 \
  --split train --max_seq_len 1024 --max_segments -1 \
  --prepend_cls True --include_sep True --shuffle False \
  --index_path "$INDEX_PATH" --output_root "$OUTPUT_ROOT"

mv "${TOK_DIR}/fineweb2_urd_1.71_tokenized.txt" \
   "${TOK_DIR}/fineweb2_urd_1.71_tokenized_full.txt"

python "${SCRIPT_DIR}/scripts/split_tokenized.py" \
  --input  "${TOK_DIR}/fineweb2_urd_1.71_tokenized_full.txt" \
  --train_output "${TOK_DIR}/fineweb2_urd_1.71_tokenized.txt" \
  --eval_output  "${TOK_DIR}/fineweb2_urd_1.71_eval_tokenized.txt" \
  --target_bytes "$TARGET_BYTES" --eval_lines 13680


# ============================================================
# BPE vocab=98304
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local.env"
OUTPUT_ROOT="${DATA_ROOT}/monolingual_training_data_tokenized"
INDEX_PATH="${SCRIPT_DIR}/configs/monolingual_bp_index.json"
ENG_REF="${OUTPUT_ROOT}/bpe_eng_latn_98304_300mb_unscaled/fineweb_eng_1.0_tokenized.txt"
ENG_BYTES=$(stat -c%s "$ENG_REF")
TARGET_BYTES=$(python3 -c "import math; print(math.floor(1.71 * $ENG_BYTES))")
TOK_DIR="${OUTPUT_ROOT}/bpe_urd_arab_98304_300mb_unscaled"

python "${SCRIPT_DIR}/scripts/tokenize_and_pack.py" \
  --dataset fineweb2_urd --tokenizer_type bpe_unscaled --tokenizer_vocabulary 98304 \
  --split train --max_seq_len 1024 --max_segments -1 \
  --prepend_cls True --include_sep True --shuffle False \
  --index_path "$INDEX_PATH" --output_root "$OUTPUT_ROOT"

mv "${TOK_DIR}/fineweb2_urd_1.71_tokenized.txt" \
   "${TOK_DIR}/fineweb2_urd_1.71_tokenized_full.txt"

python "${SCRIPT_DIR}/scripts/split_tokenized.py" \
  --input  "${TOK_DIR}/fineweb2_urd_1.71_tokenized_full.txt" \
  --train_output "${TOK_DIR}/fineweb2_urd_1.71_tokenized.txt" \
  --eval_output  "${TOK_DIR}/fineweb2_urd_1.71_eval_tokenized.txt" \
  --target_bytes "$TARGET_BYTES" --eval_lines 13680


# ============================================================
# Unigram vocab=8192
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local.env"
OUTPUT_ROOT="${DATA_ROOT}/monolingual_training_data_tokenized"
INDEX_PATH="${SCRIPT_DIR}/configs/monolingual_bp_index.json"
ENG_REF="${OUTPUT_ROOT}/unigram_eng_latn_8192_300mb_unscaled/fineweb_eng_1.0_tokenized.txt"
ENG_BYTES=$(stat -c%s "$ENG_REF")
TARGET_BYTES=$(python3 -c "import math; print(math.floor(1.71 * $ENG_BYTES))")
TOK_DIR="${OUTPUT_ROOT}/unigram_urd_arab_8192_300mb_unscaled"

python "${SCRIPT_DIR}/scripts/tokenize_and_pack.py" \
  --dataset fineweb2_urd --tokenizer_type unigram_unscaled --tokenizer_vocabulary 8192 \
  --split train --max_seq_len 1024 --max_segments -1 \
  --prepend_cls True --include_sep True --shuffle False \
  --index_path "$INDEX_PATH" --output_root "$OUTPUT_ROOT"

mv "${TOK_DIR}/fineweb2_urd_1.71_tokenized.txt" \
   "${TOK_DIR}/fineweb2_urd_1.71_tokenized_full.txt"

python "${SCRIPT_DIR}/scripts/split_tokenized.py" \
  --input  "${TOK_DIR}/fineweb2_urd_1.71_tokenized_full.txt" \
  --train_output "${TOK_DIR}/fineweb2_urd_1.71_tokenized.txt" \
  --eval_output  "${TOK_DIR}/fineweb2_urd_1.71_eval_tokenized.txt" \
  --target_bytes "$TARGET_BYTES" --eval_lines 13680


# ============================================================
# Unigram vocab=32768
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local.env"
OUTPUT_ROOT="${DATA_ROOT}/monolingual_training_data_tokenized"
INDEX_PATH="${SCRIPT_DIR}/configs/monolingual_bp_index.json"
ENG_REF="${OUTPUT_ROOT}/unigram_eng_latn_32768_300mb_unscaled/fineweb_eng_1.0_tokenized.txt"
ENG_BYTES=$(stat -c%s "$ENG_REF")
TARGET_BYTES=$(python3 -c "import math; print(math.floor(1.71 * $ENG_BYTES))")
TOK_DIR="${OUTPUT_ROOT}/unigram_urd_arab_32768_300mb_unscaled"

python "${SCRIPT_DIR}/scripts/tokenize_and_pack.py" \
  --dataset fineweb2_urd --tokenizer_type unigram_unscaled --tokenizer_vocabulary 32768 \
  --split train --max_seq_len 1024 --max_segments -1 \
  --prepend_cls True --include_sep True --shuffle False \
  --index_path "$INDEX_PATH" --output_root "$OUTPUT_ROOT"

mv "${TOK_DIR}/fineweb2_urd_1.71_tokenized.txt" \
   "${TOK_DIR}/fineweb2_urd_1.71_tokenized_full.txt"

python "${SCRIPT_DIR}/scripts/split_tokenized.py" \
  --input  "${TOK_DIR}/fineweb2_urd_1.71_tokenized_full.txt" \
  --train_output "${TOK_DIR}/fineweb2_urd_1.71_tokenized.txt" \
  --eval_output  "${TOK_DIR}/fineweb2_urd_1.71_eval_tokenized.txt" \
  --target_bytes "$TARGET_BYTES" --eval_lines 13680


# ============================================================
# Unigram vocab=65536
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local.env"
OUTPUT_ROOT="${DATA_ROOT}/monolingual_training_data_tokenized"
INDEX_PATH="${SCRIPT_DIR}/configs/monolingual_bp_index.json"
ENG_REF="${OUTPUT_ROOT}/unigram_eng_latn_65536_300mb_unscaled/fineweb_eng_1.0_tokenized.txt"
ENG_BYTES=$(stat -c%s "$ENG_REF")
TARGET_BYTES=$(python3 -c "import math; print(math.floor(1.71 * $ENG_BYTES))")
TOK_DIR="${OUTPUT_ROOT}/unigram_urd_arab_65536_300mb_unscaled"

python "${SCRIPT_DIR}/scripts/tokenize_and_pack.py" \
  --dataset fineweb2_urd --tokenizer_type unigram_unscaled --tokenizer_vocabulary 65536 \
  --split train --max_seq_len 1024 --max_segments -1 \
  --prepend_cls True --include_sep True --shuffle False \
  --index_path "$INDEX_PATH" --output_root "$OUTPUT_ROOT"

mv "${TOK_DIR}/fineweb2_urd_1.71_tokenized.txt" \
   "${TOK_DIR}/fineweb2_urd_1.71_tokenized_full.txt"

python "${SCRIPT_DIR}/scripts/split_tokenized.py" \
  --input  "${TOK_DIR}/fineweb2_urd_1.71_tokenized_full.txt" \
  --train_output "${TOK_DIR}/fineweb2_urd_1.71_tokenized.txt" \
  --eval_output  "${TOK_DIR}/fineweb2_urd_1.71_eval_tokenized.txt" \
  --target_bytes "$TARGET_BYTES" --eval_lines 13680


# ============================================================
# Unigram vocab=98304
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local.env"
OUTPUT_ROOT="${DATA_ROOT}/monolingual_training_data_tokenized"
INDEX_PATH="${SCRIPT_DIR}/configs/monolingual_bp_index.json"
ENG_REF="${OUTPUT_ROOT}/unigram_eng_latn_98304_300mb_unscaled/fineweb_eng_1.0_tokenized.txt"
ENG_BYTES=$(stat -c%s "$ENG_REF")
TARGET_BYTES=$(python3 -c "import math; print(math.floor(1.71 * $ENG_BYTES))")
TOK_DIR="${OUTPUT_ROOT}/unigram_urd_arab_98304_300mb_unscaled"

python "${SCRIPT_DIR}/scripts/tokenize_and_pack.py" \
  --dataset fineweb2_urd --tokenizer_type unigram_unscaled --tokenizer_vocabulary 98304 \
  --split train --max_seq_len 1024 --max_segments -1 \
  --prepend_cls True --include_sep True --shuffle False \
  --index_path "$INDEX_PATH" --output_root "$OUTPUT_ROOT"

mv "${TOK_DIR}/fineweb2_urd_1.71_tokenized.txt" \
   "${TOK_DIR}/fineweb2_urd_1.71_tokenized_full.txt"

python "${SCRIPT_DIR}/scripts/split_tokenized.py" \
  --input  "${TOK_DIR}/fineweb2_urd_1.71_tokenized_full.txt" \
  --train_output "${TOK_DIR}/fineweb2_urd_1.71_tokenized.txt" \
  --eval_output  "${TOK_DIR}/fineweb2_urd_1.71_eval_tokenized.txt" \
  --target_bytes "$TARGET_BYTES" --eval_lines 13680
