# FineWeb 350M Training Guide

## Overview

Train 350M parameter models on FineWeb 1 (sample-10BT) with BPE and Unigram tokenizers across vocab sizes (8192, 32768, 65536, 98304). Each model trains for a single epoch on 7B tokens.

## Prerequisites

- HuggingFace `datasets` library installed (`pip install datasets`)
- `tqdm` installed (`pip install tqdm`)
- Access to eng_latn tokenizers at `/localdisk/ssrivas9/catherinearnett/monolingual-tokenizers/`
- 4 GPUs available for training

## Step 1: Download FineWeb

Download the full sample-10BT subset (~14.9M documents, ~40 GB raw text).

```bash
python scripts/download_fineweb.py \
    --output /scratch/ssrivas9/catherinearnett/monolingual_training_data/fineweb_eng.txt
```

**Expected output**: `/scratch/ssrivas9/catherinearnett/monolingual_training_data/fineweb_eng.txt` (~40 GB)

**Testing with a small subset first**:
```bash
python scripts/download_fineweb.py \
    --output /tmp/fineweb_test.txt --max_docs 1000
```

## Step 2: Tokenize + Split + Train (all-in-one)

The launch scripts handle tokenization, splitting by 7B tokens, and training in sequence for each vocab size.

**BPE tokenizers**:
```bash
bash launch_350m_fineweb.sh
```

**Unigram tokenizers**:
```bash
bash launch_350m_fineweb_unigram.sh
```

Each launch script does the following per vocab size:

1. **Tokenize**: Runs `tokenize_and_pack.py` on the full raw text file using the eng_latn tokenizer for the given vocab/type. Outputs a full tokenized file (~10B tokens, ints-per-line format).
2. **Split**: Runs `split_tokenized.py` to extract the first 7B tokens as the training file and the last 8000 lines as the eval file. Prints the byte size of the training data.
3. **Train**: Runs `monolingual_350m.sh` for 1 epoch on the 7B-token training file.

## Step-by-step breakdown (manual)

If you prefer to run each stage separately (e.g., to tokenize first and train later):

### 2a. Tokenize full dataset for one tokenizer

```bash
python scripts/tokenize_and_pack.py \
    --dataset fineweb_eng \
    --tokenizer_type bpe_unscaled \
    --tokenizer_vocabulary 32768 \
    --split train \
    --max_seq_len 1024 \
    --max_segments -1 \
    --prepend_cls True \
    --include_sep True \
    --shuffle False \
    --index_path configs/monolingual_bp_index.json \
    --output_root /scratch/ssrivas9/catherinearnett/monolingual_training_data_tokenized
```

**Output**: `/scratch/ssrivas9/catherinearnett/monolingual_training_data_tokenized/bpe_eng_latn_32768_300mb_unscaled/fineweb_eng_1.0_tokenized.txt`

### 2b. Split by 7B tokens

```bash
# Rename the full tokenized file
TOK_DIR="/scratch/ssrivas9/catherinearnett/monolingual_training_data_tokenized/bpe_eng_latn_32768_300mb_unscaled"
mv "${TOK_DIR}/fineweb_eng_1.0_tokenized.txt" "${TOK_DIR}/fineweb_eng_1.0_tokenized_full.txt"

# Split: first 7B tokens for train, last 8000 lines for eval
python scripts/split_tokenized.py \
    --input  "${TOK_DIR}/fineweb_eng_1.0_tokenized_full.txt" \
    --train_output "${TOK_DIR}/fineweb_eng_1.0_tokenized.txt" \
    --eval_output  "${TOK_DIR}/fineweb_eng_1.0_eval_tokenized.txt" \
    --target_tokens 7000000000 \
    --eval_lines 8000
```

This will print:
- Number of lines and tokens in the training split
- **Byte size of the training file** (this is `ref_bytes` for future languages)
- Number of lines and tokens in the eval split

### 2c. Train

```bash
bash monolingual_350m.sh pre fineweb_eng 32768 bpe_unscaled 6 29510
```

## File locations

| What | Path |
|------|------|
| Raw FineWeb text | `/scratch/ssrivas9/catherinearnett/monolingual_training_data/fineweb_eng.txt` |
| Full tokenized (per tokenizer) | `…/monolingual_training_data_tokenized/{tok_basename}/fineweb_eng_1.0_tokenized_full.txt` |
| 7B-token train split | `…/monolingual_training_data_tokenized/{tok_basename}/fineweb_eng_1.0_tokenized.txt` |
| 8000-line eval split | `…/monolingual_training_data_tokenized/{tok_basename}/fineweb_eng_1.0_eval_tokenized.txt` |
| Model checkpoints | `./{run_name}/` |

Where `{tok_basename}` is e.g. `bpe_eng_latn_32768_300mb_unscaled` or `unigram_eng_latn_65536_300mb_unscaled`.

## Runtime estimates

| Stage | Per vocab size | Total (4 vocabs) |
|-------|---------------|-------------------|
| Download | ~1-3 hours (one-time) | ~1-3 hours |
| Tokenize (~40 GB) | ~7-14 hours | ~28-56 hours |
| Split (7B tokens) | ~10-30 minutes | ~40-120 minutes |
| Train (1 epoch, 7B tokens) | depends on GPU | depends on GPU |

## Future: Other languages

For non-English languages, use `split_tokenized.py` in byte-count mode:

```bash
python scripts/split_tokenized.py \
    --input  full_tokenized.txt \
    --train_output train_tokenized.txt \
    --eval_output  eval_tokenized.txt \
    --target_bytes <byte_premium * ref_bytes> \
    --eval_lines 8000
```

Where `ref_bytes` is the byte size of the English 7B-token training file (printed in step 2b). This will print the equivalent token count for the target language.
