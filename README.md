<div align="center">
  
# Multilingual Language Model Training Pipeline

A pipeline for training monolingual language models across different languages, vocabulary sizes, and tokenization approaches.

</div>

## 🚀 Quick Start

### Prerequisites

Before training monolingual models, you need to download the required data:

1. **Monolingual Training Data**: Download the preprocessed monolingual datasets
   - Expected location: `/scratch/ssrivas9/catherinearnett/monolingual_training_data`
   - Contains text files for: `eng_latn`, `tha_thai`, `urd_arab`, `amh_ethi`, `vie_latn`

2. **Monolingual Tokenizers**: Download the pretrained tokenizers
   - Expected location: `/scratch/ssrivas9/catherinearnett/monolingual-tokenizers`
   - Contains BPE and Unigram tokenizers with various vocabulary sizes

### Environment Setup

Set up your environment using conda:

```bash
conda create -n multi python=3.9 -y
conda activate multi
pip install -r exp_requirements.txt
```

## 📊 Data Preparation

### Step 1: Create Byte Premium Splits

Create balanced training and evaluation splits based on byte premium (BP) calculations:

```bash
# Train Split: Take 1 GB * Byte Premium of each monolingual dataset 
# Eval Split: Take last 8,000 lines of each monolingual dataset

python /scratch/ssrivas9/multilingual-eleuther/scripts/create_bp_splits.py \
  --input_root /scratch/ssrivas9/catherinearnett/monolingual_training_data \
  --output_root /scratch/ssrivas9/catherinearnett/monolingual_training_data_bp
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
python /scratch/ssrivas9/multilingual-eleuther/scripts/tokenize_and_pack.py \
  --dataset "eng_latn" --tokenizer_type "bpe_unscaled" --tokenizer_vocabulary "32768" \
  --split train --max_seq_len 1024 --max_segments -1 --prepend_cls True --include_sep True --shuffle True
```

Example training command (executed automatically):
```bash
bash /scratch/ssrivas9/multilingual-eleuther/monolingual_130m.sh pre "eng_latn" "32768" "bpe_unscaled" 6 29510
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
├── configs/                    # Model configurations
│   ├── llama_130m.json        # 130M model config
│   ├── llama_1b.json          # 1B model config
│   └── monolingual_bp_index.json  # Dataset index
├── scripts/                   # Data processing scripts
│   ├── create_bp_splits.py    # Create byte premium splits
│   └── tokenize_and_pack.py   # Tokenization pipeline
├── peft_pretraining/         # Core training modules
│   ├── modeling_llama.py     # Model implementations
│   ├── dataloader.py         # Data loading utilities
│   └── training_utils.py     # Training utilities
├── launch_monolingual_*.sh   # Language-specific training scripts
├── monolingual_130m.sh       # Core training script
└── torchrun_main.py         # Main training entry point
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