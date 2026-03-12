#!/usr/bin/env bash
# eng_latn, 32768 vocab — tokenize both train and eval sets

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local.env"

python "${SCRIPT_DIR}/scripts/tokenize_and_pack.py" \
  --dataset eng_latn --tokenizer_type bpe_unscaled --tokenizer_vocabulary 32768 \
  --split train --max_seq_len 1024 --max_segments -1 --prepend_cls True --include_sep True

python "${SCRIPT_DIR}/scripts/tokenize_and_pack.py" \
  --dataset eng_latn --tokenizer_type bpe_unscaled --tokenizer_vocabulary 32768 \
  --split eval --max_seq_len 1024 --max_segments -1 --prepend_cls True --include_sep True
