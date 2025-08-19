import os
import time
import json
import random
import argparse
import math
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

## datasets streaming removed
import wandb

try:
    from huggingface_hub import HfApi, create_repo
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    # Note: Logger warning moved below since logger not yet imported here

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import IntLineIterableDataset, build_intline_collate_fn
from peft_pretraining.modeling_llama import LlamaForCausalLM
from utils.activation_tracker import create_activation_tracker, log_activation_summary
from utils.enhanced_activation_tracker import create_enhanced_activation_tracker, log_enhanced_activation_summary
from utils.weight_tracker import create_weight_tracker, log_weight_summary
from utils.byte_consumption_plotter import log_byte_consumption_to_wandb, log_final_byte_summary

# # Optional import for bitsandbytes - not currently used in main training loop
# try:
#     import bitsandbytes as bnb
#     print("bitsandbytes imported successfully")
# except ImportError as e:
#     print(f"Warning: bitsandbytes not available ({e}). Continuing without it...")
#     bnb = None

import matplotlib.pyplot as plt
transformers.logging.set_verbosity_error()


def calculate_dataset_size(file_path, tokenizer=None, max_length=1024, sample_size=5000):
    """
    Calculate the exact number of lines and other stats for a dataset file.
    Caches results to avoid recalculation. Includes token estimation for packing.
    
    Args:
        file_path (str): Path to the text file
        tokenizer: HuggingFace tokenizer for token estimation (optional)
        max_length (int): Maximum sequence length for packing calculation
        sample_size (int): Number of lines to sample for token estimation
    
    Returns:
        dict: Dictionary containing dataset statistics including token estimates
    """
    import os
    import json
    import time
    from pathlib import Path
    
    # Create stats file path
    dataset_name = Path(file_path).stem  # Get filename without extension
    stats_dir = Path(file_path).parent / "dataset_stats"
    stats_dir.mkdir(exist_ok=True)
    stats_file = stats_dir / f"{dataset_name}_stats.json"
    
    # Get current file modification time and size
    file_mtime = os.path.getmtime(file_path)
    file_size_bytes = os.path.getsize(file_path)
    
    # Check if cached stats exist and are up-to-date
    if stats_file.exists():
        try:
            with open(stats_file, 'r', encoding='utf-8') as f:
                cached_stats = json.load(f)
            
            # Check if cached stats are still valid (file hasn't changed)
            if (cached_stats.get('file_mtime') == file_mtime and 
                cached_stats.get('file_size_bytes') == file_size_bytes):
                # If tokenizer was requested but token stats are missing, force recompute
                missing_token_stats = (
                    tokenizer is not None and (
                        cached_stats.get('estimated_total_tokens', 0) == 0 or
                        cached_stats.get('estimated_packed_sequences', 0) == 0 or
                        'tokens_per_char_ratio' not in cached_stats
                    )
                )
                if not missing_token_stats:
                    logger.info(f"Loading cached dataset stats from {stats_file}")
                    logger.info(f"Dataset '{dataset_name}': {cached_stats['total_lines']:,} lines, "
                               f"{cached_stats['file_size_mb']:.1f} MB, "
                               f"{cached_stats['avg_chars_per_line']:.1f} chars/line")
                    if cached_stats.get('estimated_total_tokens', 0) > 0:
                        logger.info(f"  Cached token stats: {cached_stats['estimated_total_tokens']:,} tokens")
                    return cached_stats
                else:
                    logger.info("Cached stats lack token-based estimates; recomputing with tokenizer...")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Cached stats file corrupted, recalculating: {e}")
    
    # Calculate exact statistics
    logger.info(f"Calculating exact dataset statistics for {file_path}")
    start_time = time.time()
    
    total_lines = 0
    total_chars = 0
    non_empty_lines = 0
    max_line_length = 0
    min_line_length = float('inf')
    sample_lines = []
    
    # Calculate rough sampling interval based on file size (will be refined during iteration)
    if tokenizer:
        sample_interval = max(1, int(file_size_bytes / (1024 * 1024 * 2)))  # Rough estimate: sample every ~2MB
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            line_length = len(line.rstrip('\n\r'))
            
            if line.strip():  # Non-empty line
                non_empty_lines += 1
                total_chars += line_length
                max_line_length = max(max_line_length, line_length)
                min_line_length = min(min_line_length, line_length)
                
                # Sample lines for token estimation
                if tokenizer and line_num % sample_interval == 0 and len(sample_lines) < sample_size:
                    sample_lines.append(line.strip())
            
            # Progress logging for large files
            if line_num % 1_000_000 == 0:
                elapsed = time.time() - start_time
                rate = line_num / elapsed
                logger.info(f"Processed {line_num:,} lines ({rate:,.0f} lines/sec)")
    
    calculation_time = time.time() - start_time
    
    # Handle edge case where all lines are empty
    if min_line_length == float('inf'):
        min_line_length = 0
    
    # Token estimation for sequence packing
    estimated_total_tokens = 0
    estimated_packed_sequences = 0
    packing_efficiency = 0.0
    tokens_per_char_ratio = 0.0
    
    if tokenizer and sample_lines:
        logger.info(f"Estimating token statistics from {len(sample_lines)} sample lines...")
        
        # Tokenize sample lines to estimate tokens per character
        sample_tokens = 0
        sample_chars = 0
        batch_size = 100  # Process in small batches to avoid memory issues
        
        for i in range(0, len(sample_lines), batch_size):
            batch_lines = sample_lines[i:i+batch_size]
            try:
                # Batch tokenize for efficiency without tensorization (works for any tokenizer)
                tokens = tokenizer(
                    batch_lines,
                    truncation=False,
                    padding=False,
                    return_tensors=None,
                )
                input_ids_list = tokens.get("input_ids", [])
                for j, token_ids in enumerate(input_ids_list):
                    sample_tokens += len(token_ids)
                    sample_chars += len(batch_lines[j])
            except Exception as e:
                logger.warning(f"Error tokenizing batch {i//batch_size + 1}: {e}")
                continue
        
        if sample_chars > 0:
            tokens_per_char_ratio = sample_tokens / sample_chars
            estimated_total_tokens = int(total_chars * tokens_per_char_ratio)
            
            # Add EOS tokens (one per document)
            estimated_total_tokens += non_empty_lines
            
            # (packing-related estimates no longer used)
    
    # Calculate statistics
    stats = {
        'dataset_name': dataset_name,
        'file_path': str(file_path),
        'file_size_bytes': file_size_bytes,
        'file_size_mb': file_size_bytes / (1024 * 1024),
        'file_size_gb': file_size_bytes / (1024 * 1024 * 1024),
        'file_mtime': file_mtime,
        'total_lines': total_lines,
        'non_empty_lines': non_empty_lines,
        'empty_lines': total_lines - non_empty_lines,
        'total_chars': total_chars,
        'avg_chars_per_line': total_chars / max(1, non_empty_lines),
        'max_line_length': max_line_length,
        'min_line_length': min_line_length,
        'tokens_per_char_ratio': tokens_per_char_ratio,
        'estimated_total_tokens': estimated_total_tokens,
        'estimated_packed_sequences': 0,
        'packing_efficiency': 0.0,
        'calculation_time_seconds': calculation_time,
        'calculated_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save stats to cache file
    try:
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved dataset stats to {stats_file}")
    except Exception as e:
        logger.warning(f"Failed to save stats cache: {e}")
    
    # Log comprehensive statistics
    logger.info(f"Dataset calculation completed in {calculation_time:.1f} seconds:")
    logger.info(f"  Dataset: {dataset_name}")
    logger.info(f"  File size: {stats['file_size_mb']:.1f} MB ({stats['file_size_gb']:.2f} GB)")
    logger.info(f"  Total lines: {stats['total_lines']:,}")
    logger.info(f"  Non-empty lines: {stats['non_empty_lines']:,}")
    logger.info(f"  Empty lines: {stats['empty_lines']:,}")
    logger.info(f"  Avg chars/line: {stats['avg_chars_per_line']:.1f}")
    logger.info(f"  Line length range: {stats['min_line_length']}-{stats['max_line_length']} chars")
    
    if tokenizer:
        logger.info(f"  Token estimation (from {len(sample_lines)} samples):")
        logger.info(f"    Tokens per char ratio: {tokens_per_char_ratio:.3f}")
        logger.info(f"    Estimated total tokens: {estimated_total_tokens:,}")
        # (packing-related logs removed)
    
    return stats


def calculate_checkpoint_intervals(epoch, steps_per_epoch):
    """
    Calculate checkpoint intervals for adaptive checkpointing strategy.
    
    Args:
        epoch (int): Current epoch (1-indexed)
        steps_per_epoch (int): Number of steps per epoch
    
    Returns:
        list: Steps at which to checkpoint within the epoch
    """
    if epoch == 1:
        checkpoints_per_epoch = 8
    elif epoch == 2:
        checkpoints_per_epoch = 4
    elif epoch == 3:
        checkpoints_per_epoch = 2
    else:
        checkpoints_per_epoch = 1
    
    if checkpoints_per_epoch >= steps_per_epoch:
        # If more checkpoints than steps, checkpoint at every step
        return list(range(1, steps_per_epoch + 1))
    
    interval = steps_per_epoch / checkpoints_per_epoch
    checkpoint_steps = []
    
    for i in range(checkpoints_per_epoch):
        step = int((i + 1) * interval)
        checkpoint_steps.append(step)
    
    # Ensure we don't exceed steps_per_epoch
    checkpoint_steps = [min(step, steps_per_epoch) for step in checkpoint_steps]
    
    return checkpoint_steps


def push_to_huggingface_hub(model_dir, repo_name, revision=None, commit_message=None):
    """
    Push model to HuggingFace Hub.
    
    Args:
        model_dir (str): Local directory containing the model
        repo_name (str): HuggingFace repository name (e.g., 'username/model-name')
        revision (str): Branch/revision name (optional)
        commit_message (str): Commit message (optional)
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not HF_HUB_AVAILABLE:
        logger.error("HuggingFace Hub not available. Cannot push model.")
        return False
    
    try:
        api = HfApi()
        
        # Create repo if it doesn't exist
        try:
            create_repo(repo_name, exist_ok=True)
            logger.info(f"Repository {repo_name} created/verified")
        except Exception as e:
            logger.warning(f"Could not create/verify repository: {e}")
        
        # Push the model
        if revision:
            logger.info(f"Pushing model to {repo_name} (revision: {revision})")
            api.upload_folder(
                folder_path=model_dir,
                repo_id=repo_name,
                revision=revision,
                commit_message=commit_message or f"Upload checkpoint at revision {revision}"
            )
        else:
            logger.info(f"Pushing model to {repo_name}")
            api.upload_folder(
                folder_path=model_dir,
                repo_id=repo_name,
                commit_message=commit_message or "Upload model checkpoint"
            )
        
        logger.info(f"Successfully pushed model to HuggingFace Hub: {repo_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to push model to HuggingFace Hub: {e}")
        return False

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps_ratio", type=float, default=0.05,
                        help="Ratio of warmup steps to total training steps (default: 0.05 = 5%%)")
    parser.add_argument("--eval_every", type=int, default=2_000)
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of epochs to train for (default: 10)")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_epochs. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=1.0)   
    parser.add_argument("--run_name", type=str, default="default")
    # beta1 for adafactor
    parser.add_argument("--beta1", type=float, default=0.0)
    
    # (removed) GaLore parameters
    
    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=False, action="store_true")
    
    # Activation tracking parameters
    parser.add_argument("--track_activations", default=True, action="store_true", 
                        help="Enable activation distribution tracking")
    parser.add_argument("--activation_track_every", type=int, default=100,
                        help="Track activations every N training steps")
    parser.add_argument("--activation_sample_ratio", type=float, default=1.0,
                        help="Ratio of activations to sample (0.1 = 10%%)")
    parser.add_argument("--track_gradients", default=False, action="store_true",
                        help="Also track gradient distributions (increases memory usage)")
    parser.add_argument("--track_head_activations", default=False, action="store_true",
                        help="Enable detailed head-wise activation tracking (Q,K,V states, attention values)")
    
    # Weight tracking parameters
    parser.add_argument("--track_weights", default=False, action="store_true",
                        help="Enable weight distribution tracking for individual attention heads and MLP components")
    parser.add_argument("--weight_track_every", type=int, default=100,
                        help="Track weights every N training steps")
    
    # Tokenizer selection parameter
    parser.add_argument("--tokenizer_name", type=str, default="t5-base",
                        help="Tokenizer model name to use (default: t5-base for backward compatibility)")
    
    
    # Tokenizer selection (pretrained local monolingual tokenizers)
    parser.add_argument("--tokenizer_type", type=str, default="bpe_unscaled",
                        choices=["bpe_unscaled", "unigram_unscaled"],
                        help="Which monolingual tokenizer family to use")
    parser.add_argument("--tokenizer_vocabulary", type=str, default="32768",
                        choices=["8192","16384","32768","49152","65536","81920","98304","114688","262144"],
                        help="Tokenizer vocabulary size for the selected tokenizer family")
    
    # Tokenizer pad token control
    def pad_token_type(value):
        """Parse pad token argument - can be either special token keyword or custom string"""
        value_str = str(value).lower()
        
        # Special keywords for existing tokens
        if value_str in ['eos', 'unk', 'bos']:
            return value_str  # Return lowercase for consistency
        else:
            # Custom token string - return as-is (preserve case)
            return str(value)
    
    parser.add_argument("--pad_token", type=pad_token_type, default=None,
                        help="Pad token to use. Can be: (1) Special token keywords: 'eos', 'unk', 'bos' - use existing special tokens, "
                             "or (2) Custom string (e.g., '<PAD>', '<pad>', '[PADDING]') - will be added to vocabulary. Default: None (no modification)")
    
    # Positional embedding type control
    parser.add_argument("--position_embedding_type", type=str, default="rope", 
                        choices=["rope", "learned", "sinusoidal", "none"],
                        help="Type of positional embeddings to use: 'rope' (RoPE, default), 'learned' (learnable absolute), "
                             "'sinusoidal' (fixed sinusoidal), 'none' (no positional embeddings)")
    
    # Monolingual dataset selection
    parser.add_argument("--monolingual-dataset", type=str, required=True,
                        choices=["eng_latn", "tha_thai", "urd_arab", "amh_ethi", "vie_latn"],
                        help="Monolingual dataset to use for training. Must be one of: eng_latn, tha_thai, urd_arab, amh_ethi, vie_latn. "
                             "Files are expected at /localdisk/ssrivas9/catherinearnett/monolingual_training_data/")
    
    # Hugging Face Hub integration flags
    parser.add_argument("--hf_repo_name", type=str, default=None,
                        help="Hugging Face repository to push to (e.g., 'username/model-name')")
    parser.add_argument("--hf_push_final", action="store_true",
                        help="If set, push the final model to HuggingFace Hub")
    parser.add_argument("--hf_push_checkpoints", action="store_true",
                        help="If set, push periodic checkpoints to HuggingFace Hub during training")

    # Back-compat no-op flags (accepted but unused)
    parser.add_argument("--track_dataset_bytes", action="store_true",
                        help="Deprecated/no-op: retained for CLI compatibility")
    parser.add_argument("--log_bytes_every", type=int, default=100,
                        help="Deprecated/no-op: retained for CLI compatibility")

    # Advanced efficiency optimizations
    # (removed: packing flags)
    
    # Evaluation dataset configuration
    # (removed: fraction/directional eval flags)
    
    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)
    return args


@torch.no_grad()
def evaluate_model(model, tokenizer, pad_idx, global_rank, world_size, device, batch_size, track_bytes=False, monolingual_dataset=None, args=None):
    """
    Evaluate strictly on pre-tokenized eval split (ints-per-line) matching the training tokenizer.
    """
    import math, os, json
    assert monolingual_dataset is not None, "monolingual_dataset is required for eval"
    tokenized_root = "/localdisk/ssrivas9/catherinearnett/monolingual_training_data_tokenized"
    bp_index_path = "/localdisk/ssrivas9/multilingual-eleuther/configs/monolingual_bp_index.json"
    with open(bp_index_path, 'r', encoding='utf-8') as f:
        bp_index = json.load(f)
    # Use the same tokenizer basename as training to guarantee identical tokenizer/eval path
    if hasattr(args, '_tokenizer_basename') and args._tokenizer_basename:
        tokenizer_basename = args._tokenizer_basename
    else:
        tok_root_map = {
            "bpe_unscaled": "/localdisk/ssrivas9/catherinearnett/monolingual-tokenizers/bpe_unscaled_tokenizers",
            "unigram_unscaled": "/localdisk/ssrivas9/catherinearnett/monolingual-tokenizers/unigram_unscaled_tokenizers",
        }
        tok_root = tok_root_map[args.tokenizer_type]
        tok_file = (f"bpe_{monolingual_dataset}_{args.tokenizer_vocabulary}_300mb_unscaled.json"
                    if args.tokenizer_type == "bpe_unscaled"
                    else f"unigram_{monolingual_dataset}_{args.tokenizer_vocabulary}_300mb_unscaled.json")
        tok_path = os.path.join(tok_root, tok_file)
        tokenizer_basename = os.path.splitext(os.path.basename(tok_path))[0]
    assert monolingual_dataset in bp_index, "Dataset missing in BP index for eval"
    bp = bp_index[monolingual_dataset]["byte_premium"]
    eval_path = os.path.join(tokenized_root, tokenizer_basename, f"{monolingual_dataset}_{bp}_eval_tokenized.txt")
    if not os.path.isfile(eval_path):
        raise FileNotFoundError(f"Pre-tokenized eval not found: {eval_path}")

    ds = IntLineIterableDataset(file_path=eval_path, block_size=args.max_length,
                                rank=global_rank, world_size=world_size)
    collate = build_intline_collate_fn(tokenizer.pad_token_id, args.max_length)
    eval_dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=1,
                                                 pin_memory=True, collate_fn=collate)

    total_loss = torch.tensor(0.0, device=device)
    total_batches = 0
    total_tokens = 0
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss
        total_loss += loss.detach()
        total_batches += 1
        total_tokens += (batch["attention_mask"]).sum().item() * world_size
    if total_batches == 0:
        return 0.0, 0, None, None
    avg_loss = (total_loss / total_batches).item()
    return avg_loss, total_tokens, None, None


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    # Normalize total batch size semantics to be GLOBAL across all devices
    # args_utils sets total_batch_size = batch_size * grad_accumulation (per device) by default; fix to global
    if args.total_batch_size is None:
        # Compute global total batch size
        effective_ga = args.gradient_accumulation if args.gradient_accumulation is not None else 1
        args.total_batch_size = args.batch_size * effective_ga * world_size
        args.gradient_accumulation = effective_ga
    else:
        # If args_utils default set per-device total, promote to global
        if args.gradient_accumulation is not None and args.total_batch_size == args.batch_size * args.gradient_accumulation:
            args.total_batch_size *= world_size
        # If gradient_accumulation not provided, derive from global total
        if args.gradient_accumulation is None:
            assert args.total_batch_size % (args.batch_size * world_size) == 0, "total_batch_size must be divisible by batch_size*world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, (
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"
    )

    # turn off logger
    if global_rank != 0: logger.remove()
            
    # initialize wandb without config (it is passed later)
    if global_rank == 0:
        wandb.init(project="cod", name=args.run_name)
        
    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    
    # Log HuggingFace Hub availability
    if not HF_HUB_AVAILABLE:
        logger.warning("huggingface_hub not available. HuggingFace Hub integration disabled.")
    
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)
    
    # Resolve pre-tokenized paths (BP index) and tokenizer settings
    bp_index_path = "/localdisk/ssrivas9/multilingual-eleuther/configs/monolingual_bp_index.json"
    bp_index = None
    if os.path.exists(bp_index_path):
        try:
            with open(bp_index_path, 'r', encoding='utf-8') as f:
                bp_index = json.load(f)
        except Exception:
            bp_index = None
    
    # Create tokenizer first
    # Resolve tokenizer path based on tokenizer_type and tokenizer_vocabulary if available
    resolved_tokenizer_name = args.tokenizer_name
    tok_root_map = {
        "bpe_unscaled": "/localdisk/ssrivas9/catherinearnett/monolingual-tokenizers/bpe_unscaled_tokenizers",
        "unigram_unscaled": "/localdisk/ssrivas9/catherinearnett/monolingual-tokenizers/unigram_unscaled_tokenizers",
    }
    if args.tokenizer_type in tok_root_map:
        tok_root = tok_root_map[args.tokenizer_type]
        tok_file = (f"bpe_{args.monolingual_dataset}_{args.tokenizer_vocabulary}_300mb_unscaled.json"
                    if args.tokenizer_type == "bpe_unscaled"
                    else f"unigram_{args.monolingual_dataset}_{args.tokenizer_vocabulary}_300mb_unscaled.json")
        candidate = os.path.join(tok_root, tok_file)
        if os.path.isfile(candidate):
            resolved_tokenizer_name = candidate
            logger.info(f"Resolved tokenizer file: {candidate}")
        else:
            logger.warning(f"Tokenizer file not found: {candidate}. Falling back to --tokenizer_name={args.tokenizer_name}")

    logger.info(f"Loading tokenizer: {resolved_tokenizer_name}")
    def _prepare_tokenizer_dir(json_path: str) -> str:
        base = os.path.splitext(os.path.basename(json_path))[0]
        out_dir = os.path.join(os.path.dirname(json_path), base + "_hf")
        os.makedirs(out_dir, exist_ok=True)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        with open(os.path.join(out_dir, 'tokenizer.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f)
        special_map = {}
        unk = data.get('model', {}).get('unk_token')
        if isinstance(unk, str):
            special_map['unk_token'] = unk
        for tok in data.get('added_tokens', []) or []:
            if not tok.get('special', False):
                continue
            content = tok.get('content')
            if not isinstance(content, str):
                continue
            upper = content.upper()
            if 'CLS' in upper and 'cls_token' not in special_map:
                special_map['cls_token'] = content
            if 'SEP' in upper and 'sep_token' not in special_map:
                special_map['sep_token'] = content
            if 'MASK' in upper and 'mask_token' not in special_map:
                special_map['mask_token'] = content
            if 'PAD' in upper and 'pad_token' not in special_map:
                special_map['pad_token'] = content
            if 'BOS' in upper and 'bos_token' not in special_map:
                special_map['bos_token'] = content
            if 'EOS' in upper and 'eos_token' not in special_map:
                special_map['eos_token'] = content
        with open(os.path.join(out_dir, 'special_tokens_map.json'), 'w', encoding='utf-8') as f:
            json.dump(special_map, f)
        return out_dir

    load_path = resolved_tokenizer_name
    if isinstance(load_path, str) and load_path.endswith('.json') and os.path.isfile(load_path):
        load_path = _prepare_tokenizer_dir(load_path)
    # Enforce AutoTokenizer load; fail loudly on error
    tokenizer = AutoTokenizer.from_pretrained(load_path, model_max_length=args.max_length, use_fast=True)
    
    # Use pre-tokenized ints-per-line
    tokenized_root = "/localdisk/ssrivas9/catherinearnett/monolingual_training_data_tokenized"
    tokenizer_basename = None
    if os.path.isfile(resolved_tokenizer_name):
        tokenizer_basename = os.path.splitext(os.path.basename(resolved_tokenizer_name))[0]
    # Persist tokenizer basename for eval to ensure exact match
    args._tokenizer_basename = tokenizer_basename
    tokenized_train_path = None
    if bp_index and args.monolingual_dataset in bp_index and tokenizer_basename:
        bp = bp_index[args.monolingual_dataset]["byte_premium"]
        candidate = os.path.join(tokenized_root, tokenizer_basename, f"{args.monolingual_dataset}_{bp}_tokenized.txt")
        if os.path.isfile(candidate):
            tokenized_train_path = candidate
            logger.info(f"Using pre-tokenized training data: {tokenized_train_path}")
    if tokenized_train_path is None:
        raise FileNotFoundError("Pre-tokenized training data not found. Please run scripts/create_bp_splits.py and scripts/tokenize_and_pack.py first.")

    # Determine dataset stats from pretokenized file only
    training_mode = "pretokenized ints-per-line"
    try:
        with open(tokenized_train_path, 'r', encoding='utf-8') as f:
            dataset_size = sum(1 for _ in f)
    except Exception:
        dataset_size = 1
    pre_file_size_bytes = os.path.getsize(tokenized_train_path) if os.path.exists(tokenized_train_path) else 0

    # Calculate training parameters
    steps_per_epoch = math.ceil(dataset_size / float(args.total_batch_size))
    if steps_per_epoch == 0:
        steps_per_epoch = 1  # Ensure at least 1 step per epoch

    total_training_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(total_training_steps * args.warmup_steps_ratio)

    logger.info(f"Epoch-based training configuration:")
    logger.info(f"  Training mode: {training_mode}")
    logger.info(f"  Dataset size: {dataset_size:,} training sequences")
    logger.info(f"  Pre-tokenized file size: {pre_file_size_bytes / (1024*1024):.1f} MB ({pre_file_size_bytes / (1024*1024*1024):.2f} GB)")
    logger.info(f"  Steps per epoch (planned): {steps_per_epoch:,}")
    logger.info(f"  Total epochs: {args.num_epochs}")
    logger.info(f"  Total training steps (planned): {total_training_steps:,}")
    logger.info(f"  Warmup steps: {warmup_steps:,} ({args.warmup_steps_ratio*100:.1f}%)")
    logger.info(f"  Avg sequences per step: {args.total_batch_size}")

    def create_dataset_for_epoch(epoch_num):
        logger.info("Building IntLineIterableDataset for pre-tokenized input")
        ds = IntLineIterableDataset(file_path=tokenized_train_path, block_size=args.max_length,
                                    rank=global_rank, world_size=world_size)
        return ds

    # Initial dataset creation for epoch 1
    # (Removed unused variable 'data')

    # Handle pad token configuration per user rule
    if args.pad_token is None:
        if tokenizer.pad_token is None:
            logger.warning("Tokenizer has no pad_token and --pad_token not set; batches will use placeholder -1.")
    else:
        # modify PAD only if explicitly provided
        if args.pad_token in ['eos', 'unk', 'bos']:
            name = args.pad_token
            tok = tokenizer.eos_token if name == 'eos' else tokenizer.unk_token if name == 'unk' else tokenizer.bos_token
            if tok is not None:
                tokenizer.pad_token = tok
                logger.info(f"Using {name.upper()} as pad_token: '{tokenizer.pad_token}'")
            else:
                logger.warning(f"Requested pad_token={name} but tokenizer lacks it; proceeding without pad_token")
        else:
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': args.pad_token})
                logger.info(f"Added custom pad token '{args.pad_token}'")
            else:
                tokenizer.pad_token = args.pad_token
                logger.info(f"Using custom pad token '{args.pad_token}'")

    # Load model config
    model_config = AutoConfig.from_pretrained(args.model_config)
    
    # Override positional embedding type if specified
    if hasattr(args, 'position_embedding_type') and args.position_embedding_type != "rope":
        logger.info(f"Overriding position_embedding_type from config to: {args.position_embedding_type}")
        model_config.position_embedding_type = args.position_embedding_type
    
    # Build dataloader from pretokenized ints
    from peft_pretraining.dataloader import build_intline_collate_fn
    ds = create_dataset_for_epoch(1)  # IntLineIterableDataset
    collate = build_intline_collate_fn(tokenizer.pad_token_id, args.max_length)
    dl_kwargs = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, collate_fn=collate)
    if args.workers > 0:
        dl_kwargs.update(dict(persistent_workers=True, prefetch_factor=2))
    dataloader = torch.utils.data.DataLoader(ds, **dl_kwargs)

    if args.use_hf_model:
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)
    
    # Always resize model embeddings to tokenizer vocab (tokenizer is authoritative)
    vocab_size = len(tokenizer.get_vocab())
    if getattr(model_config, 'vocab_size', None) != vocab_size:
        logger.info(f"Resizing model embeddings from {getattr(model_config, 'vocab_size', 'N/A')} to {vocab_size}")
        model.resize_token_embeddings(vocab_size)
        if hasattr(model, 'config'):
            model.config.vocab_size = vocab_size
        logger.info("Model embeddings resized ✓")

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    global_step = 0
    update_step = 0
    beginning_step = 0
    tokens_seen = 0
    tokens_seen_before = 0
    
    # (removed: dataset byte tracking state)

    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        
        from safetensors.torch import load_file
        state_dict = load_file(f"{args.continue_from}/model.safetensors")
        model.load_state_dict(state_dict)
        
        # checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")
        # model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        logger.info(f"Model successfully loaded (strict=True policy)")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(f"Loading training state like global_step, update_step, tokens_seen, and bytes_seen from {args.continue_from}")
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            # Load byte tracking state if available (for backward compatibility)
            # bytes_seen = _old_state.get("bytes_seen", 0) # Removed
            # bytes_seen_before = _old_state.get("bytes_seen_before", 0) # Removed
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            # (removed: dataset byte tracking logs)
            logger.info(f"Will train for {total_training_steps - update_step} update steps")
        else:
            logger.warning(f"Did not find training state in {args.continue_from}, global step will start from zero")
        logger.info("*" * 40)


    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "dataset": args.monolingual_dataset,
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
    })

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now") # save current script
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(total=total_training_steps - update_step, desc="Update steps", ncols=80)
    
    # (removed) GaLore parameter grouping
        
    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    # (removed) GaLore param count logging
    logger.info(f"Using adaptive checkpointing strategy (8,4,2,1,1... checkpoints per epoch)")
    
    # Initialize activation tracker before DDP wrapping
    activation_tracker = None
    if args.track_activations and global_rank == 0:  # Only track on rank 0 to avoid duplication
        if args.track_head_activations:
            logger.info("Initializing enhanced activation distribution tracker with head-wise tracking")
            activation_tracker = create_enhanced_activation_tracker(
                model=model,
                track_every_n_steps=args.activation_track_every,
                sample_ratio=args.activation_sample_ratio,
                track_gradients=args.track_gradients,
                track_head_activations=True,
                device=device,
            )
        else:
            logger.info("Initializing basic activation distribution tracker")
            activation_tracker = create_activation_tracker(
                model=model,
                track_every_n_steps=args.activation_track_every,
                sample_ratio=args.activation_sample_ratio,
                track_gradients=args.track_gradients,
                device=device,
            )
        logger.info(f"Activation tracker initialized: {activation_tracker.get_summary()}")
    
    # Initialize weight tracker before DDP wrapping
    weight_tracker = None
    if args.track_weights and global_rank == 0:  # Only track on rank 0 to avoid duplication
        logger.info("Initializing weight distribution tracker")
        weight_tracker = create_weight_tracker(
            model=model,
            track_every_n_steps=args.weight_track_every,
            device=device,
        )
        logger.info(f"Weight tracker initialized: {weight_tracker.get_summary()}")
    
    layer_wise_flag = False
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    if not layer_wise_flag:
        scheduler = training_utils.get_scheculer(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=total_training_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )

    if not args.single_gpu:
        model: LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1
    logger.info(f"Using pad_token_id={pad_idx} for training (-1 means placeholder)")
    
    # Epoch-based training variables
    current_epoch = 1
    epoch_step = 0  # Batch step counter (for debugging)
    epoch_update_step = 0  # Update step counter (for epoch completion and checkpointing)
    
    training_start_time = time.time()  # Track training start time for byte consumption analysis
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step

    # ##############################
    # EPOCH-BASED TRAINING LOOP
    # ##############################

    # Calculate adaptive checkpoint intervals for each epoch
    epoch_checkpoint_plans = {}
    for epoch in range(1, args.num_epochs + 1):
        epoch_checkpoint_plans[epoch] = calculate_checkpoint_intervals(epoch, steps_per_epoch)
        logger.info(f"Epoch {epoch} checkpoint plan: {epoch_checkpoint_plans[epoch]}")

    for epoch in range(current_epoch, args.num_epochs + 1):
        logger.info(f"Starting epoch {epoch}/{args.num_epochs}")
        current_epoch = epoch
        epoch_step = 0  # Reset batch step counter
        epoch_update_step = 0  # Reset update step counter for this epoch
        
        # Reshuffle dataset for this epoch (except epoch 1 which was already created)
        if epoch > 1:
            logger.info(f"Reshuffling dataset for epoch {epoch}")
            data = create_dataset_for_epoch(epoch)
            
            # Recreate dataset and dataloader with new shuffled data
            dataset = IntLineIterableDataset(file_path=tokenized_train_path, block_size=args.max_length,
                                    rank=global_rank, world_size=world_size)
            collate = build_intline_collate_fn(tokenizer.pad_token_id, args.max_length)
            _dl_kwargs = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, collate_fn=collate)
            if args.workers > 0:
                _dl_kwargs.update(dict(persistent_workers=True, prefetch_factor=2))
            dataloader = torch.utils.data.DataLoader(dataset, **_dl_kwargs)
        
        # Get checkpoint plan for this epoch
        checkpoint_steps = epoch_checkpoint_plans[epoch]
        
        for batch_idx, batch in enumerate(dataloader):
            global_step += 1
            local_step += 1
            epoch_step += 1

            # Do not early-break inside the epoch; finish consuming all lines in this epoch

            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            labels[labels == pad_idx] = -100
            tokens_seen += (batch["attention_mask"]).sum().item() * world_size

            # (dataset byte tracking removed in pretokenized-only pipeline)

            # Activation tracking logic
            if activation_tracker is not None and activation_tracker.step(global_step):
                activation_tracker.start_tracking()

            # Weight tracking logic - track weights before forward pass
            if weight_tracker is not None and weight_tracker.step(global_step):
                weight_tracker.track_weights()

            # Standard loss calculation
            loss = model(**batch, labels=labels).loss

            # Stop activation tracking after forward pass
            if activation_tracker is not None and activation_tracker.is_tracking:
                activation_tracker.stop_tracking()
            scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()

            # If not yet reached full accumulation, continue to next micro-batch
            if (epoch_step % args.gradient_accumulation) != 0:
                continue

            # The below code is only executed during the update step

            # add grad clipping
            if args.grad_clipping != 0.0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

            if global_rank == 0:
                pbar.update(1)

            if not layer_wise_flag:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            update_step += 1
            epoch_update_step += 1  # Increment epoch update step counter
            update_time = time.time() - update_time

            # Adaptive checkpointing based on epoch and step
            if (
                local_step > args.gradient_accumulation
                and epoch_update_step in checkpoint_steps
                and global_rank == 0
            ):
                checkpoint_name = f"epoch_{current_epoch}_step_{epoch_update_step}"
                current_model_directory = f"{args.save_dir}/model_{checkpoint_name}"
                logger.info(
                    f"Saving checkpoint to {current_model_directory} (Epoch {current_epoch}, Update Step {epoch_update_step}/{steps_per_epoch})"
                )

                # Save tokenizer first
                tokenizer.save_pretrained(current_model_directory)

                # Update model config with current tokenizer details
                model_to_save = model.module if hasattr(model, 'module') else model
                if hasattr(model_to_save, 'config'):
                    model_to_save.config.pad_token_id = tokenizer.pad_token_id
                    model_to_save.config.vocab_size = len(tokenizer.get_vocab())
                    logger.info(
                        f"Updated model config: pad_token_id={tokenizer.pad_token_id}, vocab_size={len(tokenizer.get_vocab())}"
                    )

                os.makedirs(args.save_dir, exist_ok=True)
                model_to_save.save_pretrained(current_model_directory, max_shard_size='100GB')

                # Push to HuggingFace Hub if requested
                if args.hf_push_checkpoints and args.hf_repo_name:
                    revision_name = f"checkpoint-{checkpoint_name}"
                    commit_msg = (
                        f"Checkpoint at epoch {current_epoch}, step {epoch_update_step} (global update step {update_step})"
                    )

                    logger.info(
                        f"Pushing checkpoint to HuggingFace Hub: {args.hf_repo_name} (revision: {revision_name})"
                    )
                    success = push_to_huggingface_hub(
                        model_dir=current_model_directory,
                        repo_name=args.hf_repo_name,
                        revision=revision_name,
                        commit_message=commit_msg,
                    )
                    if success:
                        logger.info(f"Successfully pushed checkpoint to HF Hub")
                    else:
                        logger.warning(f"Failed to push checkpoint to HF Hub")

                optimizer_checkpoint = {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "update_step": update_step,
                    "global_step": global_step,
                    "config": run_config,
                    "wandb": wandb.run.dir,
                    "dtype": args.dtype,
                }
                torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

                training_state_checkpoint = {
                    "global_step": global_step,
                    "update_step": update_step,
                    "tokens_seen": tokens_seen,
                    "tokens_seen_before": tokens_seen_before,
                    "update_time": update_time,
                }
                with open(f"{current_model_directory}/training_state.json", "w") as f:
                    json.dump(training_state_checkpoint, f, indent=4)

                # save wandb related info
                wandb_info = {
                    "wandb_id": wandb.run.id,
                }
                with open(f"{args.save_dir}/wandb.json", "w") as f:
                    json.dump(wandb_info, f, indent=4)

        # evaluation
            if update_step % args.eval_every == 0:
                logger.info(f"Performing evaluation at step {update_step}")
                eval_loss, eval_tokens, _, _ = evaluate_model(
                    model, tokenizer, pad_idx, global_rank, world_size, device, args.batch_size,
                    track_bytes=False, monolingual_dataset=args.monolingual_dataset, args=args
                )

                eval_metrics = {
                    "eval_loss": eval_loss,
                    "eval_tokens": eval_tokens,
                }

                # Add BPB metrics if byte tracking is enabled
                # (removed: BPB logging)

                if global_rank == 0:
                    wandb.log(eval_metrics, step=global_step)
                logger.info(f"Eval loss at step {update_step}: {eval_loss}")

            if not layer_wise_flag:
                lr = optimizer.param_groups[0]["lr"]
            else:
                pass
            tokens_in_update = tokens_seen - tokens_seen_before
            tokens_seen_before = tokens_seen
            # (removed: byte deltas)
            batches_in_update = args.gradient_accumulation * world_size

            if global_rank == 0:
                metrics_to_log = {
                    "loss": loss.item(),
                    "lr": lr,
                    "update_step": update_step,
                    "tokens_seen": tokens_seen,
                    "throughput_tokens": tokens_in_update / update_time,
                    "throughput_examples": args.total_batch_size / update_time,
                    "throughput_batches": batches_in_update / update_time,
                }

                # Add byte tracking metrics if enabled
                # (removed: dataset byte metrics)

                # Log activation distributions if tracking is enabled
                # Log at first update step (initial state) and then every N update steps
                if activation_tracker is not None and (update_step == 1 or update_step % args.activation_track_every == 0):
                    try:
                        activation_tracker.log_to_wandb(global_step)
                        if args.track_head_activations:
                            log_enhanced_activation_summary(activation_tracker, global_step)
                        else:
                            log_activation_summary(activation_tracker, global_step)
                        # Clear stored activations to free memory
                        activation_tracker.clear_stored_activations()
                        step_type = "INITIAL" if update_step == 1 else "REGULAR"
                        logger.info(
                            f"Logged {step_type} activation distributions at global_step {global_step}, update_step {update_step}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to log activation distributions: {e}")

                # Log weight distributions if tracking is enabled
                if weight_tracker is not None and (update_step == 1 or update_step % args.weight_track_every == 0):
                    try:
                        weight_tracker.log_to_wandb(global_step)
                        log_weight_summary(weight_tracker, global_step)
                        step_type = "INITIAL" if update_step == 1 else "REGULAR"
                        logger.info(
                            f"Logged {step_type} weight distributions at global_step {global_step}, update_step {update_step}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to log weight distributions: {e}")

                # Log byte consumption plots if tracking is enabled
                # (removed: byte consumption plots)

                wandb.log(metrics_to_log, step=global_step)
            update_time = time.time()

            # Continue until the dataloader is exhausted to ensure all lines are consumed
        
        # Break out of epoch loop if we've reached max training steps
        if update_step >= total_training_steps:
            break

        # Flush leftover micro-batches at end of epoch to ensure all lines contribute.
        # Adjust gradient magnitude so it matches a full accumulation step.
        leftover = epoch_step % args.gradient_accumulation
        if leftover != 0 and update_step < total_training_steps:
            scale_up = float(args.gradient_accumulation) / float(leftover)
            for param in trainable_params:
                if param.grad is not None:
                    param.grad.data.mul_(scale_up)
            if args.grad_clipping != 0.0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)
            if global_rank == 0:
                pbar.update(1)
            if not layer_wise_flag:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            update_step += 1
            epoch_update_step += 1
            # Metrics/logging for the flushed step
            if not layer_wise_flag:
                lr = optimizer.param_groups[0]["lr"]
            tokens_in_update = tokens_seen - tokens_seen_before
            tokens_seen_before = tokens_seen
            batches_in_update = args.gradient_accumulation * world_size
            if global_rank == 0:
                metrics_to_log = {
                    "loss": loss.item(),
                    "lr": lr,
                    "update_step": update_step,
                    "tokens_seen": tokens_seen,
                    "throughput_tokens": tokens_in_update / max(1e-9, (time.time() - update_time)),
                    "throughput_examples": args.total_batch_size / max(1e-9, (time.time() - update_time)),
                    "throughput_batches": batches_in_update / max(1e-9, (time.time() - update_time)),
                }
                wandb.log(metrics_to_log, step=global_step)
            # Checkpoint if this flushed step matches plan
            if (
                local_step > args.gradient_accumulation
                and epoch_update_step in checkpoint_steps
                and global_rank == 0
            ):
                checkpoint_name = f"epoch_{current_epoch}_step_{epoch_update_step}"
                current_model_directory = f"{args.save_dir}/model_{checkpoint_name}"
                logger.info(
                    f"Saving checkpoint to {current_model_directory} (Epoch {current_epoch}, Update Step {epoch_update_step}/{steps_per_epoch})"
                )
                tokenizer.save_pretrained(current_model_directory)
                model_to_save = model.module if hasattr(model, 'module') else model
                if hasattr(model_to_save, 'config'):
                    model_to_save.config.pad_token_id = tokenizer.pad_token_id
                    model_to_save.config.vocab_size = len(tokenizer.get_vocab())
                os.makedirs(args.save_dir, exist_ok=True)
                model_to_save.save_pretrained(current_model_directory, max_shard_size='100GB')
                if args.hf_push_checkpoints and args.hf_repo_name:
                    revision_name = f"checkpoint-{checkpoint_name}"
                    commit_msg = (
                        f"Checkpoint at epoch {current_epoch}, step {epoch_update_step} (global update step {update_step})"
                    )
                    push_to_huggingface_hub(
                        model_dir=current_model_directory,
                        repo_name=args.hf_repo_name,
                        revision=revision_name,
                        commit_message=commit_msg,
                    )
                optimizer_checkpoint = {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "update_step": update_step,
                    "global_step": global_step,
                    "config": run_config,
                    "wandb": wandb.run.dir,
                    "dtype": args.dtype,
                }
                torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")
                training_state_checkpoint = {
                    "global_step": global_step,
                    "update_step": update_step,
                    "tokens_seen": tokens_seen,
                    "tokens_seen_before": tokens_seen_before,
                    "update_time": 0.0,
                }
                with open(f"{current_model_directory}/training_state.json", "w") as f:
                    json.dump(training_state_checkpoint, f, indent=4)

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0: pbar.close()

    final_model_directory = f"{args.save_dir}/final_model"
    if global_rank == 0 and not os.path.exists(final_model_directory):
        logger.info(f"Saving final model to {final_model_directory}")
        
        # Save tokenizer
        tokenizer.save_pretrained(final_model_directory)
        
        # Update model config with current tokenizer details
        model_to_save = model.module if hasattr(model, 'module') else model
        if hasattr(model_to_save, 'config'):
            model_to_save.config.pad_token_id = tokenizer.pad_token_id
            model_to_save.config.vocab_size = len(tokenizer.get_vocab())
            logger.info(f"Final save - Updated model config: pad_token_id={tokenizer.pad_token_id}, vocab_size={len(tokenizer.get_vocab())}")
        
        os.makedirs(args.save_dir, exist_ok=True)
        model_to_save.save_pretrained(final_model_directory)
        
        # Push final model to HuggingFace Hub if requested
        if args.hf_push_final and args.hf_repo_name:
            logger.info(f"Pushing final model to HuggingFace Hub: {args.hf_repo_name}")
            success = push_to_huggingface_hub(
                model_dir=final_model_directory,
                repo_name=args.hf_repo_name,
                revision="main",
                commit_message=f"Final model after {args.num_epochs} epochs ({update_step} steps)"
            )
            if success:
                logger.info(f"Successfully pushed final model to HF Hub")
            else:
                logger.warning(f"Failed to push final model to HF Hub")

        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
        }
        with open(f"{final_model_directory}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc; gc.collect()
    torch.cuda.empty_cache()

    final_loss, final_tokens, _, _ = evaluate_model(
        model, tokenizer, pad_idx, global_rank, world_size, device, args.batch_size,
        track_bytes=False, monolingual_dataset=args.monolingual_dataset, args=args
    )

    final_eval_metrics = {
        "final_eval_loss": final_loss,
        "final_eval_tokens": final_tokens,
    }
    
    # Add final BPB metrics if byte tracking enabled
    # (removed: final BPB/byte summary)

    if global_rank == 0:
        wandb.log(final_eval_metrics, step=global_step)
        logger.info(f"Final eval loss: {final_loss}")
        
        # (removed: final BPB/byte summary)

    # Cleanup activation tracker
    if activation_tracker is not None:
        activation_tracker.cleanup()
        logger.info("Activation tracker cleaned up")
    
    # Cleanup weight tracker
    if weight_tracker is not None:
        weight_tracker.clear_stored_weights()
        logger.info("Weight tracker cleaned up")
    
    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)
