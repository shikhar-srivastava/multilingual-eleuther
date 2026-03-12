<div align="center">
  
# Monolingual Language Model Training Pipeline

A pipeline for training monolingual LLaMA models across different languages, vocabulary sizes, and tokenization approaches. 

</div>

## 🚀 Quick Start

### Prerequisites

Before training monolingual models, you need to download the required data:

1. **Monolingual Training Data**: Download the preprocessed monolingual datasets
   - Contains text files for: `eng_latn`, `tha_thai`, `urd_arab`, `amh_ethi`, `vie_latn`

2. **Monolingual Tokenizers**: Download the pretrained tokenizers
   - Contains BPE and Unigram tokenizers with various vocabulary sizes

### Environment Setup

Set up your environment using conda:

```bash
conda create -n multi python=3.9 -y
conda activate multi
pip install -r exp_requirements.txt
```

### Path Configuration (`local.env`)

All data paths in this repo are read from a single **`local.env`** file at the
repo root. This file is **not committed to git** — each machine gets its own
copy.

```bash
cp local.env.example local.env
# Edit DATA_ROOT to point to your data directory
```

`local.env` defines one variable:

| Variable | Description |
|---|---|
| `DATA_ROOT` | Base directory containing training data, tokenized data, and tokenizers |

The repo expects these subdirectories under `DATA_ROOT`:

```
$DATA_ROOT/
├── monolingual_training_data/            Raw text files (eng_latn.txt, ...)
├── monolingual_training_data_bp/         Byte-premium train/eval splits
├── monolingual_training_data_tokenized/  Tokenized training data
└── monolingual-tokenizers/               Pretrained tokenizers
    ├── bpe_unscaled_tokenizers/
    └── unigram_unscaled_tokenizers/
```

The repo root itself is always resolved dynamically (`SCRIPT_DIR` in shell,
`Path(__file__)` in Python), so it works from any checkout location.

**Example configurations:**

| Machine | `DATA_ROOT` |
|---|---|
| Bluehive cluster | `DATA_ROOT=/scratch/ssrivas9/catherinearnett` |
| Docker / cloud | `DATA_ROOT=/root/data` |

To download tokenizers to the right location:

```bash
bash download_tokenizers.sh
```

## 📊 Data Preparation

### Step 1: Create Byte Premium Splits

Create balanced training and evaluation splits based on byte premium (BP) calculations:

```bash
bash create_byte_premium_sets.sh
```

Or manually (paths default from `local.env`):

```bash
python scripts/create_bp_splits.py
```

This creates balanced datasets with the following byte premiums:
- **English (eng_latn)**: 1.0× (1GB)
- **Thai (tha_thai)**: 2.74× (2.94GB)
- **Urdu (urd_arab)**: 1.71× (1.84GB)  
- **Amharic (amh_ethi)**: 1.72× (1.85GB)
- **Vietnamese (vie_latn)**: 1.35× (1.45GB)

## 🏋️ Model Training

### Monolingual Training Pipeline

The pipeline supports training 130M parameter models on individual languages with various normalization techniques. Each language has its dedicated training script:

#### English Training
```bash
bash launch_monolingual_eng.sh
```

#### Thai Training
```bash
bash launch_monolingual_tha.sh
```

#### Urdu Training
```bash
bash launch_monolingual_urd.sh
```

#### Amharic Training
```bash
bash launch_monolingual_amh.sh
```

#### Vietnamese Training
```bash
bash launch_monolingual_vie.sh
```

### Training Configuration

Each monolingual training script:
1. **Tokenizes** the data using multiple vocabulary sizes (16K, 32K, 49K, 65K, 81K)
2. **Trains** 130M parameter models 
3. **Supports** both BPE and Unigram tokenization (configurable in scripts)

Example tokenization command (executed automatically):
```bash
python scripts/tokenize_and_pack.py \
  --dataset "eng_latn" --tokenizer_type "bpe_unscaled" --tokenizer_vocabulary "32768" \
  --split train --max_seq_len 1024 --max_segments -1 --prepend_cls True --include_sep True --shuffle True
```

Example training command (executed automatically):
```bash
bash monolingual_130m.sh pre "eng_latn" "32768" "bpe_unscaled" 6 29510
```

### Direct Model Training

You can also train individual models directly:

```bash
# Train English model with 32K BPE vocabulary
bash monolingual_130m.sh pre eng_latn 32768 bpe_unscaled 6 29510

# Train Thai model with 49K BPE vocabulary  
bash monolingual_130m.sh pre tha_thai 49152 bpe_unscaled 6 29510

# Train with Unigram tokenization
bash monolingual_130m.sh pre eng_latn 32768 unigram_unscaled 6 29510
```

### Parameters:
- `norm_type`: Model configuration (use `pre` as default)
- `dataset`: Language dataset (`eng_latn`, `tha_thai`, `urd_arab`, `amh_ethi`, `vie_latn`)
- `vocab_size`: Tokenizer vocabulary size (`16384`, `32768`, `49152`, `65536`, `81920`)
- `tokenizer_type`: Tokenization algorithm (`bpe_unscaled`, `unigram_unscaled`)
- `post_num`: Configuration parameter (use `6` as default)
- `master_port`: Port for distributed training

## 🛠️ Model Architecture

### 130M Model Configuration
- **Layers**: 12 transformer blocks
- **Hidden Size**: 768
- **Attention Heads**: 12
- **Intermediate Size**: 2048
- **Max Sequence Length**: 1024
- **Position Encoding**: RoPE (Rotary Position Embedding)
- **Activation**: SiLU/Swish

### Training Hyperparameters
- **Learning Rate**: 1e-3
- **Batch Size**: 32 (per device)
- **Total Batch Size**: 512 (with gradient accumulation)
- **Epochs**: 10
- **Warmup Ratio**: 0.1
- **Optimizer**: Adam
- **Precision**: bfloat16

## 📁 Project Structure

```
multilingual-eleuther/
├── local.env.example          # Template — copy to local.env and edit
├── local.env                  # (gitignored) Machine-local DATA_ROOT
├── configs/                   # Model configurations
│   ├── llama_130m.json        # 130M model config
│   ├── llama_350m.json        # 350M model config
│   └── monolingual_bp_index.json  # Dataset index (uses ${DATA_ROOT})
├── scripts/                   # Data processing scripts
│   ├── create_bp_splits.py    # Create byte premium splits
│   ├── download_fineweb.py    # Download FineWeb dataset
│   ├── tokenize_and_pack.py   # Tokenization pipeline
│   └── split_tokenized.py     # Split tokenized data by token count
├── utils/                     # Shared utilities
│   └── local_config.py        # Reads DATA_ROOT from local.env
├── peft_pretraining/          # Core training modules
│   ├── modeling_llama.py      # Model implementations
│   ├── dataloader.py          # Data loading utilities
│   └── training_utils.py      # Training utilities
├── launch_monolingual_*.sh    # Language-specific training scripts
├── launch_350m_fineweb*.sh    # FineWeb 350M training scripts
├── monolingual_130m.sh        # Core 130M training script
├── monolingual_350m.sh        # Core 350M training script
├── download_tokenizers.sh     # Download tokenizers from HF
└── torchrun_main.py           # Main training entry point
```

## 🎯 Supported Languages

| Language | Script | Dataset Code | Byte Premium |
|----------|--------|--------------|--------------|
| English | Latin | `eng_latn` | 1.0× |
| Thai | Thai | `tha_thai` | 2.74× |
| Urdu | Arabic | `urd_arab` | 1.71× |
| Amharic | Ethiopic | `amh_ethi` | 1.72× |
| Vietnamese | Latin | `vie_latn` | 1.35× |


### Hyperparameter Tuning

Modify training parameters in `monolingual_130m.sh`:
- Learning rates: `1e-3`, `1e-4`, `5e-4`
- Batch sizes: `16`, `32`, `64`
- Sequence lengths: `512`, `1024`, `2048`

## 🙏 Acknowledgments

This code is built on top of:
- [@LayerNorm-Scaling](https://github.com/lmsdss/LayerNorm-Scaling): Core model training infrastructure
- [@word-acquisition-language-models](https://github.com/tylerachang/word-acquisition-language-models/tree/1182df1d388be189214da0184ee04a416dec18cc): Tokenization pipeline and data processing utilities

We thank the authors of these repositories for their work.
