#!/usr/bin/env bash
set -euo pipefail
goldfish=${goldfish:-False}
TRAIN_SCRIPT="/scratch/ssrivas9/multilingual-eleuther/monolingual_130m.sh"
if [[ "$goldfish" == "True" ]]; then
  TRAIN_SCRIPT="/scratch/ssrivas9/multilingual-eleuther/monolingual_130m_gold.sh"
fi
# Iterate over all monolingual datasets, vocabulary sizes, and tokenizer types.
# 1) Tokenize train and eval splits using scripts/tokenize_and_pack.py
# 2) Train using monolingual_130m.sh

DATASETS=(amh_ethi) # tha_thai urd_arab amh_ethi vie_latn)
VOCABS=(16384 32768 49152 65536 81920)  #(8192 98304 114688 262144)
TOKENIZERS=(bpe_unscaled) # unigram_unscaled)

MAX_SEQ_LEN=1024
GA=2

tokenize_fn() {
  local dataset=$1
  local tokenizer_type=$2
  local vocab=$3
  python /scratch/ssrivas9/multilingual-eleuther/scripts/tokenize_and_pack.py \
    --dataset "$dataset" --tokenizer_type "$tokenizer_type" --tokenizer_vocabulary "$vocab" \
    --split train --max_seq_len $MAX_SEQ_LEN --max_segments -1 --prepend_cls True --include_sep True --shuffle True
  python /scratch/ssrivas9/multilingual-eleuther/scripts/tokenize_and_pack.py \
    --dataset "$dataset" --tokenizer_type "$tokenizer_type" --tokenizer_vocabulary "$vocab" \
    --split eval --max_seq_len $MAX_SEQ_LEN --max_segments -1 --prepend_cls True --include_sep True --shuffle False
}

for dataset in "${DATASETS[@]}"; do
  for vocab in "${VOCABS[@]}"; do
    for tok in "${TOKENIZERS[@]}"; do
      echo "[Tokenize] dataset=$dataset, tok=$tok, vocab=$vocab"
      tokenize_fn "$dataset" "$tok" "$vocab"

      echo "[Train] dataset=$dataset, tok=$tok, vocab=$vocab"
      bash "$TRAIN_SCRIPT" pre "$dataset" "$vocab" "$tok" 6 29510
    done
  done
done

