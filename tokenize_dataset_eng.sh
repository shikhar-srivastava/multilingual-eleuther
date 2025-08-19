# eng_latn
# 32768 vocab
# Tokenize both train and eval sets

python /localdisk/ssrivas9/multilingual-eleuther/scripts/tokenize_and_pack.py \
  --dataset eng_latn --tokenizer_type bpe_unscaled --tokenizer_vocabulary 32768 \
  --split train --max_seq_len 1024 --max_segments -1 --prepend_cls True --include_sep True

python /localdisk/ssrivas9/multilingual-eleuther/scripts/tokenize_and_pack.py \
  --dataset eng_latn --tokenizer_type bpe_unscaled --tokenizer_vocabulary 32768 \
  --split eval --max_seq_len 1024 --max_segments -1 --prepend_cls True --include_sep True