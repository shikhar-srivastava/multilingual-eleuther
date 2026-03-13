# FineWeb 350M Training Guide

## Overview

Train 350M parameter models on FineWeb data with BPE and Unigram tokenizers across vocab sizes (8192, 32768, 65536, 98304).

- **FineWeb 1** (English): Train on 7B tokens from the `sample-10BT` subset, 1 epoch.
- **FineWeb-2** (Amharic, Thai, Urdu, Vietnamese): Train on `byte_premium * English_ref_bytes` of tokenized data per language, with multi-epoch compensation when data is insufficient.

---

## Part A: English (FineWeb 1)

### Prerequisites

- HuggingFace `datasets` library installed (`pip install datasets`)
- `tqdm` installed (`pip install tqdm`)
- Access to eng_latn tokenizers at `/scratch/ssrivas9/catherinearnett/monolingual-tokenizers/`
- 4 GPUs available for training

### Step 1: Download FineWeb

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

### Step 2: Tokenize + Split + Train (all-in-one)

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

### Step-by-step breakdown (manual)

If you prefer to run each stage separately (e.g., to tokenize first and train later):

#### 2a. Tokenize full dataset for one tokenizer

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

**Output**: `…/monolingual_training_data_tokenized/bpe_eng_latn_32768_300mb_unscaled/fineweb_eng_1.0_tokenized.txt`

#### 2b. Split by 7B tokens

```bash
TOK_DIR="/scratch/ssrivas9/catherinearnett/monolingual_training_data_tokenized/bpe_eng_latn_32768_300mb_unscaled"
mv "${TOK_DIR}/fineweb_eng_1.0_tokenized.txt" "${TOK_DIR}/fineweb_eng_1.0_tokenized_full.txt"

python scripts/split_tokenized.py \
    --input  "${TOK_DIR}/fineweb_eng_1.0_tokenized_full.txt" \
    --train_output "${TOK_DIR}/fineweb_eng_1.0_tokenized.txt" \
    --eval_output  "${TOK_DIR}/fineweb_eng_1.0_eval_tokenized.txt" \
    --target_tokens 7000000000 \
    --eval_lines 8000
```

This will print:
- Number of lines and tokens in the training split
- **Byte size of the training file** (this is `eng_ref_bytes` used for other languages)
- Number of lines and tokens in the eval split

#### 2c. Train

```bash
bash monolingual_350m.sh pre fineweb_eng 32768 bpe_unscaled 6 29510
```

### English file locations

| What | Path |
|------|------|
| Raw FineWeb text | `…/monolingual_training_data/fineweb_eng.txt` |
| Full tokenized (per tokenizer) | `…/monolingual_training_data_tokenized/{tok_basename}/fineweb_eng_1.0_tokenized_full.txt` |
| 7B-token train split | `…/monolingual_training_data_tokenized/{tok_basename}/fineweb_eng_1.0_tokenized.txt` |
| 8000-line eval split | `…/monolingual_training_data_tokenized/{tok_basename}/fineweb_eng_1.0_eval_tokenized.txt` |
| Model checkpoints | `./{run_name}/` |

Where `{tok_basename}` is e.g. `bpe_eng_latn_32768_300mb_unscaled` or `unigram_eng_latn_65536_300mb_unscaled`.

---

## Part B: Multilingual (FineWeb-2)

### Languages and byte premiums

| Dataset ID | Language | HF Config | Byte Premium | Eval Lines | FW2 Raw Text |
|------------|----------|-----------|-------------|------------|-------------|
| `fineweb2_amh` | Amharic | `amh_Ethi` | 1.72 | 13760 | ~2.69 GiB |
| `fineweb2_tha` | Thai | `tha_Thai` | 2.74 | 21920 | ~321.88 GiB |
| `fineweb2_urd` | Urdu | `urd_Arab` | 1.71 | 13680 | ~22.48 GiB |
| `fineweb2_vie` | Vietnamese | `vie_Latn` | 1.35 | 10800 | ~402.69 GiB |

### How dataset sizing works

For each language, the training data target is:

```
target_bytes = byte_premium * eng_ref_bytes
```

Where `eng_ref_bytes` is the byte size of the English 7B-token tokenized training file **for the matching tokenizer type and vocab size**. This is computed per-(tokenizer, vocab) pair -- not a fixed number across all configs.

For example, with BPE vocab=32768:
- English ref = 31.11 GiB
- Thai target = 2.74 * 31.11 = 85.25 GiB tokenized
- Amharic target = 1.72 * 31.11 = 53.52 GiB tokenized

### Data insufficiency and epoch compensation

Amharic and Urdu have far less raw data in FineWeb-2 than the byte premium target requires. For these languages, the pipeline:

1. Downloads and tokenizes ALL available data
2. Uses all tokenized data (minus eval lines) for training
3. Compensates with multiple epochs: `num_epochs = floor(target_bytes / actual_bytes)`

| Language | Coverage | Approx Epochs (BPE) |
|----------|----------|---------------------|
| Amharic | ~5% | ~18-20 |
| Thai | sufficient | 1 |
| Urdu | ~42% | ~2 |
| Vietnamese | sufficient | 1 |

The exact epoch count is computed dynamically per tokenizer/vocab config after tokenization.

### Prerequisites

- **English FineWeb pipeline must be completed first** (the English tokenized files serve as the reference for computing target sizes and download budgets)
- Language-specific tokenizers (e.g., `amh_ethi`, `tha_thai`) at `/scratch/ssrivas9/catherinearnett/monolingual-tokenizers/`
- 4 GPUs available for training

### Step 1: Run the launch scripts

The launch scripts handle everything automatically: downloading (with byte budgets), tokenization, splitting, epoch computation, and training.

**BPE tokenizers (all 4 languages)**:
```bash
bash launch_350m_fineweb2.sh
```

**Unigram tokenizers (all 4 languages)**:
```bash
bash launch_350m_fineweb2_unigram.sh
```

If you only need to (re-)tokenize without running training, per-language tokenize+split scripts are available that combine both BPE and Unigram, with each block independently copy-pasteable:

```bash
bash tokenize_fineweb2_amh.sh   # Amharic (BP=1.72, eval=13760)
bash tokenize_fineweb2_tha.sh   # Thai    (BP=2.74, eval=21920)
bash tokenize_fineweb2_urd.sh   # Urdu    (BP=1.71, eval=13680)
bash tokenize_fineweb2_vie.sh   # Vietnamese (BP=1.35, eval=10800)
```

Each launch script does the following:

1. **Download** (once per language): Streams FineWeb-2 raw text via `download_fineweb2.py` with a byte budget of `2 * byte_premium * eng_ref_bytes` to avoid downloading the full dataset for large languages (saves ~468 GiB for Thai + Vietnamese).
2. **Tokenize** (per language/vocab): Tokenizes the full downloaded raw text using the language's CC-trained tokenizer.
3. **Split** (per language/vocab): Extracts training data up to `target_bytes` and eval data from the tail. Disjointness is enforced even when data is smaller than target.
4. **Epoch computation**: If `actual_bytes < target_bytes`, sets `num_epochs = floor(target_bytes / actual_bytes)`.
5. **Train**: Runs `monolingual_350m.sh` with the computed epoch count.

### Download byte budgets

The download budget is `2 * byte_premium * eng_ref_bytes` (2x safety margin over the tokenized target to account for the raw-to-tokenized size ratio):

| Language | Full FW2 | Download budget | Actual download |
|----------|----------|----------------|-----------------|
| Amharic | 2.69 GiB | ~108 GiB | 2.69 GiB (all) |
| Thai | 321.88 GiB | ~172 GiB | ~172 GiB |
| Urdu | 22.48 GiB | ~107 GiB | 22.48 GiB (all) |
| Vietnamese | 402.69 GiB | ~85 GiB | ~85 GiB |

### Step-by-step breakdown (manual)

#### 1a. Download one language

```bash
# Download all available Amharic (small dataset, no budget needed)
python scripts/download_fineweb2.py --language amh

# Download Thai with a byte budget
python scripts/download_fineweb2.py --language tha --max_bytes 172000000000

# Test with a small number of documents
python scripts/download_fineweb2.py --language vie --max_docs 1000
```

#### 1b. Tokenize for one tokenizer/vocab

```bash
python scripts/tokenize_and_pack.py \
    --dataset fineweb2_tha \
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

Then rename the output:
```bash
TOK_DIR="…/monolingual_training_data_tokenized/bpe_tha_thai_32768_300mb_unscaled"
mv "${TOK_DIR}/fineweb2_tha_2.74_tokenized.txt" "${TOK_DIR}/fineweb2_tha_2.74_tokenized_full.txt"
```

#### 1c. Split by target bytes

```bash
# Get English reference size for this tokenizer/vocab
ENG_REF="…/monolingual_training_data_tokenized/bpe_eng_latn_32768_300mb_unscaled/fineweb_eng_1.0_tokenized.txt"
ENG_BYTES=$(stat -c%s "$ENG_REF")
TARGET_BYTES=$(python3 -c "import math; print(math.floor(2.74 * $ENG_BYTES))")

python scripts/split_tokenized.py \
    --input  "${TOK_DIR}/fineweb2_tha_2.74_tokenized_full.txt" \
    --train_output "${TOK_DIR}/fineweb2_tha_2.74_tokenized.txt" \
    --eval_output  "${TOK_DIR}/fineweb2_tha_2.74_eval_tokenized.txt" \
    --target_bytes "$TARGET_BYTES" \
    --eval_lines 21920
```

#### 1d. Compute epochs and train

```bash
ACTUAL_BYTES=$(stat -c%s "${TOK_DIR}/fineweb2_tha_2.74_tokenized.txt")
if (( ACTUAL_BYTES < TARGET_BYTES )); then
    NUM_EPOCHS=$(python3 -c "print($TARGET_BYTES // $ACTUAL_BYTES)")
else
    NUM_EPOCHS=1
fi

bash monolingual_350m.sh pre fineweb2_tha 32768 bpe_unscaled 6 29510 $NUM_EPOCHS
```

### Multilingual file locations

| What | Path |
|------|------|
| Raw FineWeb-2 text | `…/monolingual_training_data/fineweb2_{lang}.txt` |
| Full tokenized | `…/monolingual_training_data_tokenized/{tok_basename}/fineweb2_{lang}_{bp}_tokenized_full.txt` |
| Train split | `…/monolingual_training_data_tokenized/{tok_basename}/fineweb2_{lang}_{bp}_tokenized.txt` |
| Eval split | `…/monolingual_training_data_tokenized/{tok_basename}/fineweb2_{lang}_{bp}_eval_tokenized.txt` |

Where `{tok_basename}` uses the language's CC tokenizer (e.g., `bpe_amh_ethi_32768_300mb_unscaled`), `{lang}` is `amh`/`tha`/`urd`/`vie`, and `{bp}` is the byte premium (e.g., `1.72`).

### Data budget table

A detailed LaTeX table with English reference sizes, required tokenized sizes, FineWeb-2 availability, and epoch multipliers for all 8 tokenizer/vocab configs is available at `tables/fineweb2_data_budget.tex`.

---

## Runtime estimates

| Stage | Per vocab size | Total (4 vocabs) |
|-------|---------------|-------------------|
| Download English (~40 GB) | ~1-3 hours (one-time) | ~1-3 hours |
| Tokenize English (~40 GB) | ~7-14 hours | ~28-56 hours |
| Split English (7B tokens) | ~10-30 minutes | ~40-120 minutes |
| Download FineWeb-2 (4 langs) | varies by language | one-time |
| Tokenize FineWeb-2 per lang | proportional to raw size | depends on lang |
| Train (1 epoch, 7B tokens) | depends on GPU | depends on GPU |
