# FineWeb 350M Training Guide

Trains 350M models on FineWeb (English) and FineWeb-2 (Amharic, Thai, Urdu, Vietnamese) with BPE + Unigram tokenizers across vocab sizes 8192, 32768, 65536, 98304.

---

## Part A: English (FineWeb 1)

**Must be completed before Part B** — the English tokenized files serve as the byte-size reference for all other languages.

### 1. Download (~40 GB raw text, one-time)

```bash
python scripts/download_fineweb.py \
    --output ${DATA_ROOT}/monolingual_training_data/fineweb_eng.txt
```

### 2. Tokenize + split + train

Run one launch script per tokenizer type. Each handles all 4 vocab sizes in sequence.

```bash
bash launch_350m_fineweb.sh           # BPE (bpe_unscaled)
bash launch_350m_fineweb_unigram.sh   # Unigram (unigram_unscaled)
```

Each script: tokenizes the full raw text → splits to first 7B tokens (train) + last 8000 lines (eval) → trains for 1 epoch.

#### Manual step-by-step (e.g. BPE vocab=32768)

```bash
# Tokenize
python scripts/tokenize_and_pack.py \
  --dataset fineweb_eng --tokenizer_type bpe_unscaled --tokenizer_vocabulary 32768 \
  --split train --max_seq_len 1024 --max_segments -1 \
  --prepend_cls True --include_sep True --shuffle False \
  --index_path configs/monolingual_bp_index.json \
  --output_root ${DATA_ROOT}/monolingual_training_data_tokenized

# Rename full file, then split
TOK_DIR="${DATA_ROOT}/monolingual_training_data_tokenized/bpe_eng_latn_32768_300mb_unscaled"
mv "${TOK_DIR}/fineweb_eng_1.0_tokenized.txt" "${TOK_DIR}/fineweb_eng_1.0_tokenized_full.txt"
python scripts/split_tokenized.py \
  --input  "${TOK_DIR}/fineweb_eng_1.0_tokenized_full.txt" \
  --train_output "${TOK_DIR}/fineweb_eng_1.0_tokenized.txt" \
  --eval_output  "${TOK_DIR}/fineweb_eng_1.0_eval_tokenized.txt" \
  --target_tokens 7000000000 --eval_lines 8000

# Train
bash monolingual_350m.sh pre fineweb_eng 32768 bpe_unscaled 6 29510
```

---

## Part B: FineWeb-2 Multilingual

**Prerequisite**: Part A (English) must be complete. Language configs and byte premiums:

| Dataset | Language | BP | Eval lines | FW2 raw size |
|---|---|---|---|---|
| `fineweb2_amh` | Amharic | 1.72 | 13760 | ~2.7 GiB (all downloaded) |
| `fineweb2_tha` | Thai | 2.74 | 21920 | ~322 GiB (capped download) |
| `fineweb2_urd` | Urdu | 1.71 | 13680 | ~22.5 GiB (all downloaded) |
| `fineweb2_vie` | Vietnamese | 1.35 | 10800 | ~403 GiB (capped download) |

Training data target: `target_bytes = BP × eng_ref_bytes` per tokenizer/vocab. For data-scarce languages (amh, urd), all available data is used and epochs are multiplied to compensate: `num_epochs = floor(target_bytes / actual_bytes)`.

### 1. Download raw text (one-time per language)

Amharic and Urdu: download everything (small datasets).

```bash
python scripts/download_fineweb2.py --language amh
python scripts/download_fineweb2.py --language urd
```

Thai and Vietnamese: capped download to avoid fetching hundreds of GiB unnecessarily. The budget is `2 × BP × eng_ref_bytes` (2× safety margin). Compute from your actual English reference file:

```bash
# Compute budget from English BPE vocab=98304 reference (largest ref = most conservative budget)
ENG_REF="${DATA_ROOT}/monolingual_training_data_tokenized/bpe_eng_latn_98304_300mb_unscaled/fineweb_eng_1.0_tokenized.txt"
ENG_BYTES=$(stat -c%s "$ENG_REF")

python scripts/download_fineweb2.py --language tha \
    --max_bytes $(python3 -c "import math; print(math.floor(2 * 2.74 * $ENG_BYTES))")

python scripts/download_fineweb2.py --language vie \
    --max_bytes $(python3 -c "import math; print(math.floor(2 * 1.35 * $ENG_BYTES))")
```

All outputs go to `${DATA_ROOT}/monolingual_training_data/fineweb2_{lang}.txt` by default.

### 2. Tokenize + split + train (full pipeline)

The launch scripts handle download (with budget, skipping if file exists), tokenize, split, epoch computation, and training — for all 4 languages and all 4 vocab sizes.

```bash
bash launch_350m_fineweb2.sh           # BPE (bpe_unscaled)
bash launch_350m_fineweb2_unigram.sh   # Unigram (unigram_unscaled)
```

> **Note**: the BPE launch script requires the English BPE tokenized files. The Unigram launch script requires the English Unigram tokenized files (falls back to BPE if available). Run Part A first.

### 3. Tokenize + split only (no training)

Per-language scripts covering both BPE and Unigram, 8 self-contained blocks each (copy any block to run independently):

```bash
bash tokenize_fineweb2_amh.sh
bash tokenize_fineweb2_tha.sh
bash tokenize_fineweb2_urd.sh
bash tokenize_fineweb2_vie.sh
```

#### Manual step-by-step (e.g. Thai, BPE vocab=32768)

```bash
# Compute target bytes
ENG_REF="${DATA_ROOT}/monolingual_training_data_tokenized/bpe_eng_latn_32768_300mb_unscaled/fineweb_eng_1.0_tokenized.txt"
ENG_BYTES=$(stat -c%s "$ENG_REF")
TARGET_BYTES=$(python3 -c "import math; print(math.floor(2.74 * $ENG_BYTES))")
TOK_DIR="${DATA_ROOT}/monolingual_training_data_tokenized/bpe_tha_thai_32768_300mb_unscaled"

# Tokenize
python scripts/tokenize_and_pack.py \
  --dataset fineweb2_tha --tokenizer_type bpe_unscaled --tokenizer_vocabulary 32768 \
  --split train --max_seq_len 1024 --max_segments -1 \
  --prepend_cls True --include_sep True --shuffle False \
  --index_path configs/monolingual_bp_index.json \
  --output_root ${DATA_ROOT}/monolingual_training_data_tokenized

# Rename, then split
mv "${TOK_DIR}/fineweb2_tha_2.74_tokenized.txt" "${TOK_DIR}/fineweb2_tha_2.74_tokenized_full.txt"
python scripts/split_tokenized.py \
  --input  "${TOK_DIR}/fineweb2_tha_2.74_tokenized_full.txt" \
  --train_output "${TOK_DIR}/fineweb2_tha_2.74_tokenized.txt" \
  --eval_output  "${TOK_DIR}/fineweb2_tha_2.74_eval_tokenized.txt" \
  --target_bytes "$TARGET_BYTES" --eval_lines 21920

# Compute epochs and train
ACTUAL_BYTES=$(stat -c%s "${TOK_DIR}/fineweb2_tha_2.74_tokenized.txt")
NUM_EPOCHS=$(python3 -c "print(max(1, $TARGET_BYTES // $ACTUAL_BYTES))")
bash monolingual_350m.sh pre fineweb2_tha 32768 bpe_unscaled 6 29510 $NUM_EPOCHS
```

---

## File locations

| What | Path |
|---|---|
| Raw text (English) | `${DATA_ROOT}/monolingual_training_data/fineweb_eng.txt` |
| Raw text (FW2 lang) | `${DATA_ROOT}/monolingual_training_data/fineweb2_{lang}.txt` |
| Tokenized train split | `…/monolingual_training_data_tokenized/{tok_dir}/{dataset}_{bp}_tokenized.txt` |
| Tokenized eval split | `…/monolingual_training_data_tokenized/{tok_dir}/{dataset}_{bp}_eval_tokenized.txt` |
| Full tokenized (pre-split) | `…/monolingual_training_data_tokenized/{tok_dir}/{dataset}_{bp}_tokenized_full.txt` |
| Data budget table (LaTeX) | `tables/fineweb2_data_budget.tex` |

`{tok_dir}` examples: `bpe_eng_latn_32768_300mb_unscaled`, `unigram_tha_thai_65536_300mb_unscaled`
