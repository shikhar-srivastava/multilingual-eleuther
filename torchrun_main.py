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

import datasets
import datasets.distributed
import wandb

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
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

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=2_000)
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
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
    
    # GaLore parameters
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")
    
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
    
    # Soft alpha parameter for label smoothing
    parser.add_argument("--soft_alpha", type=float, default=None,
                        help="Soft alpha value for label smoothing. If None, uses standard cross-entropy loss.")
    
    # Byte tracking parameters for dataset consumption measurement
    parser.add_argument("--track_dataset_bytes", default=False, action="store_true",
                        help="Track UTF-8 bytes consumed from dataset (for cross-tokenizer comparison)")
    parser.add_argument("--log_bytes_every", type=int, default=100,
                        help="Log byte consumption every N update steps")
    
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
    
    parser.add_argument("--pad_token", type=pad_token_type, default="<PAD>",
                        help="Pad token to use. Can be: (1) Special token keywords: 'eos', 'unk', 'bos' - use existing special tokens, "
                             "or (2) Custom string (e.g., '<PAD>', '<pad>', '[PADDING]') - will be added to vocabulary. Default: '<PAD>'")
    
    # Positional embedding type control
    parser.add_argument("--position_embedding_type", type=str, default="rope", 
                        choices=["rope", "learned", "sinusoidal", "none"],
                        help="Type of positional embeddings to use: 'rope' (RoPE, default), 'learned' (learnable absolute), "
                             "'sinusoidal' (fixed sinusoidal), 'none' (no positional embeddings)")
    
    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)
    return args


@torch.no_grad()
def evaluate_model(model, tokenizer, pad_idx, global_rank, world_size, device, batch_size, track_bytes=False):
    """
    Evaluate model with optional byte tracking for BPB calculation.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer to use for evaluation
        pad_idx: Padding token index
        global_rank: Distributed training rank
        world_size: Number of distributed processes
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        track_bytes: Whether to track UTF-8 bytes for BPB calculation
    
    Returns:
        tuple: (avg_loss, evaluated_tokens, evaluated_bytes, bits_per_byte)
    """
    import math
    
    _time = time.time()
    val_data = datasets.load_dataset("c4", "en", split="validation", streaming=True, trust_remote_code=True) #DGX
    val_data = val_data.shuffle(seed=42)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    if not args.single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

    # Use byte-tracking dataloader if requested
    if track_bytes:
        eval_dataset = PreprocessedIterableDataset(
            val_data, tokenizer, batch_size=batch_size, 
            max_length=args.max_length, track_bytes=True
        )
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=None, num_workers=1)
    else:
        # Original evaluation method for backward compatibility
        def preprocess_batched(batch):
            batch = tokenizer(
                batch["text"],
                max_length=args.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return batch
        
        val_data_mapped = val_data.map(
            preprocess_batched,
            batched=True,
            remove_columns=["text", "timestamp", "url"],
        )
        val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size)
        eval_dataloader = val_data_mapped.batch(batch_size=batch_size)

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    evaluated_on_bytes = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in eval_dataloader:
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        # Extract batch bytes if tracking
        batch_bytes = batch.pop("batch_bytes", 0) if track_bytes else 0
        
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss
        total_loss += loss.detach()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size
        
        # Track bytes if enabled
        if track_bytes and batch_bytes > 0:
            evaluated_on_bytes += batch_bytes * world_size

    total_loss = total_loss / total_batches

    # Gather losses and byte counts across all GPUs
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    avg_loss = sum([t.item() for t in gathered_losses]) / world_size
    
    # Calculate bits per byte (BPB) if byte tracking is enabled
    bits_per_byte = None
    if track_bytes and evaluated_on_bytes > 0:
        # Total loss = average loss * number of tokens
        total_loss_value = avg_loss * evaluated_on_tokens
        
        # Byte-level loss = total loss / total bytes
        byte_level_loss = total_loss_value / evaluated_on_bytes
        
        # Convert from nats to bits (divide by ln(2))
        bits_per_byte = byte_level_loss / math.log(2)
        
        logger.info(f"Evaluation BPB calculation:")
        logger.info(f"  Avg loss: {avg_loss:.4f}")
        logger.info(f"  Total tokens: {evaluated_on_tokens:,}")
        logger.info(f"  Total bytes: {evaluated_on_bytes:,}")
        logger.info(f"  Total loss: {total_loss_value:.4f}")
        logger.info(f"  Byte-level loss: {byte_level_loss:.6f}")
        logger.info(f"  Bits per byte: {bits_per_byte:.6f}")

    return avg_loss, evaluated_on_tokens, evaluated_on_bytes, bits_per_byte


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

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # turn off logger
    if global_rank != 0: logger.remove()
            
    # initialize wandb without config (it is passed later)
    if global_rank == 0:
        wandb.init(project="cod", name=args.run_name)
        
    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)
    
    data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)



    seed_for_shuffle = 32 
    
    logger.info(f"Shuffling data with seed {seed_for_shuffle}")
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)
    if not args.single_gpu:
        data = datasets.distributed.split_dataset_by_node(
            data, rank=global_rank, world_size=world_size,
        )

    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, model_max_length=args.max_length)

    # Set up pad token based on user specification
    if args.pad_token in ['eos', 'unk', 'bos']:
        # Use existing special token as pad token
        special_token_name = args.pad_token
        
        if special_token_name == 'eos':
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"Using EOS token '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id}) as pad token")
            else:
                raise ValueError("Tokenizer does not have an EOS token. Cannot use 'eos' as pad token.")
        
        elif special_token_name == 'unk':
            if tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
                logger.info(f"Using UNK token '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id}) as pad token")
            else:
                raise ValueError("Tokenizer does not have a UNK token. Cannot use 'unk' as pad token.")
        
        elif special_token_name == 'bos':
            if tokenizer.bos_token is not None:
                tokenizer.pad_token = tokenizer.bos_token
                logger.info(f"Using BOS token '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id}) as pad token")
            else:
                raise ValueError("Tokenizer does not have a BOS token. Cannot use 'bos' as pad token.")
    
    else:
        # Use custom string token (add new token if needed)
        pad_token_str = args.pad_token
        
        if tokenizer.pad_token is None:
            # Add the specified token as a new special token
            tokenizer.add_special_tokens({'pad_token': pad_token_str})
            logger.info(f"Added '{pad_token_str}' as new pad token")
        else:
            # Check if we should override existing pad token
            if tokenizer.pad_token != pad_token_str:
                logger.info(f"Tokenizer already has pad_token '{tokenizer.pad_token}', but user specified '{pad_token_str}'")
                # Add the user-specified token anyway
                tokenizer.add_special_tokens({'pad_token': pad_token_str})
                logger.info(f"Override: set '{pad_token_str}' as new pad token")
            else:
                logger.info(f"Tokenizer already has the requested pad token: '{tokenizer.pad_token}'")

    # Load model config early to validate tokenizer compatibility
    model_config = AutoConfig.from_pretrained(args.model_config)
    
    # Override positional embedding type if specified
    if hasattr(args, 'position_embedding_type') and args.position_embedding_type != "rope":
        logger.info(f"Overriding position_embedding_type from config to: {args.position_embedding_type}")
        model_config.position_embedding_type = args.position_embedding_type
    
    # Validate tokenizer vocab size matches model config
    # Use actual vocab size (len) instead of property which may be cached
    tokenizer_vocab_size = len(tokenizer.get_vocab())
    
    if tokenizer_vocab_size != model_config.vocab_size:
        if args.pad_token not in ['eos', 'unk', 'bos'] and tokenizer_vocab_size == model_config.vocab_size + 1:
            logger.info(f"Tokenizer vocab size ({tokenizer_vocab_size}) = model config ({model_config.vocab_size}) + 1 due to added pad token '{args.pad_token}' ✓")
            logger.info("This is expected and safe when training from scratch with custom padding token")
        else:
            logger.info(f"Tokenizer vocab size ({tokenizer_vocab_size}) != model config vocab size ({model_config.vocab_size})")
            if args.pad_token in ['eos', 'unk', 'bos']:
                logger.info(f"Using existing {args.pad_token.upper()} token as pad token, so vocab size difference may be expected")
            else:
                logger.info("Vocab size mismatch detected - will resize model embeddings")
            logger.info("Proceeding with model embedding resizing...")
    else:
        logger.info(f"Tokenizer vocab size ({tokenizer_vocab_size}) matches model config ✓")

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length, track_bytes=args.track_dataset_bytes)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)

    if args.use_hf_model:
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)
    
    # Resize model embeddings if tokenizer vocabulary size changed
    tokenizer_vocab_size = len(tokenizer.get_vocab())
    if tokenizer_vocab_size != model_config.vocab_size:
        logger.info(f"Resizing model embeddings from {model_config.vocab_size} to {tokenizer_vocab_size}")
        model.resize_token_embeddings(tokenizer_vocab_size)
        logger.info("Model embeddings resized successfully ✓")

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    global_step = 0
    update_step = 0
    beginning_step = 0
    tokens_seen = 0
    tokens_seen_before = 0
    
    # Initialize byte tracking variables
    bytes_seen = 0
    bytes_seen_before = 0

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
            bytes_seen = _old_state.get("bytes_seen", 0)
            bytes_seen_before = _old_state.get("bytes_seen_before", 0)
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            if args.track_dataset_bytes:
                logger.info(f"bytes_seen        : {bytes_seen}")
                logger.info(f"bytes_seen_before : {bytes_seen_before}")
            logger.info(f"Will train for {args.num_training_steps - update_step} update steps")
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
        "dataset": 'c4',
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
    })

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now") # save current script
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)
    
    if 'galore' in args.optimizer.lower():
        # make parameters with "rank" to a single group, if param_name has "mlp" or "attn"
        galore_params = []
        target_modules_list = ["attn", "mlp"]
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue
            
            print('enable GaLore for weights in module: ', module_name)
            galore_params.append(module.weight)
        id_galore_params = [id(p) for p in galore_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
        # then call galore_adamw
        param_groups = [{'params': regular_params}, 
                        {'params': galore_params, 'rank': args.rank, 'update_proj_gap': args.update_proj_gap, 'scale': args.galore_scale, 'proj_type': args.proj_type}]
        
    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    if 'galore' in args.optimizer.lower():
        logger.info(f"Total params with GaLore enabled: {sum(p.numel() for p in galore_params) / 1_000_000:.2f}M")
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")
    
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
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
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
    pad_idx = tokenizer.pad_token_id
    assert pad_idx is not None, f"pad_token_id should never be None at this point. Got: {pad_idx}"
    logger.info(f"Using pad_token_id={pad_idx} for training")
    
    training_start_time = time.time()  # Track training start time for byte consumption analysis
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step

    # ##############################
    # TRAINING LOOP
    # we'll never go through all the data, so no need for epochs
    # ##############################

    for batch_idx, batch in enumerate(dataloader):

        global_step += 1
        local_step += 1
        if update_step > args.num_training_steps:
            logger.info(f"Reached max number of update steps (f{args.num_training_steps}). Stopping training.")
            print(f"Rank {global_rank} stopping training.")
            break

        # Extract batch bytes before moving to device
        batch_bytes = batch.pop("batch_bytes", 0) if args.track_dataset_bytes else 0
        
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size
        
        # Track bytes consumed (distributed across all ranks)
        if args.track_dataset_bytes and batch_bytes > 0:
            bytes_seen += batch_bytes * world_size

        # Activation tracking logic
        if activation_tracker is not None and activation_tracker.step(global_step):
            activation_tracker.start_tracking()
        
        # Weight tracking logic - track weights before forward pass
        if weight_tracker is not None and weight_tracker.step(global_step):
            weight_tracker.track_weights()
        
        # Forward pass with soft alpha implementation
        if args.soft_alpha is not None:
            # Get both logits and labels for soft alpha computation
            model_output = model(**batch, labels=labels)
            
            # Implement soft alpha label smoothing - exact implementation from provided code
            shift_logits = model_output.logits[:, :-1, :].contiguous()
            valid_targets = labels[:, 1:].contiguous()

            num_classes = shift_logits.size(-1)

            # Handling special value -100 in targets
            mask = (valid_targets == -100)
            valid_targets[mask] = 0  # Replace -100 with 0 or another neutral index

            confidence = 1.0 - args.soft_alpha
            label_smoothing = args.soft_alpha / (num_classes - 1)
            targets_smooth = torch.full_like(shift_logits, label_smoothing)

            # Apply scatter only on valid targets
            targets_smooth.scatter_(-1, valid_targets.unsqueeze(-1), confidence)

            # Apply mask to neutralize the effect of -100 in loss calculation
            targets_smooth[mask.unsqueeze(-1).expand_as(targets_smooth)] = 0
            
            # Loss calculation with try-except block
            loss_fct = torch.nn.CrossEntropyLoss()
            try:
                loss = loss_fct(shift_logits.view(-1, num_classes), targets_smooth.view(-1, num_classes))
            except RuntimeError as e:
                print("Error during loss calculation:", e)
                print("Shapes at loss calculation:")
                print("shift_logits:", shift_logits.view(-1, num_classes).shape)
                print("targets_smooth:", targets_smooth.view(-1, num_classes).shape)
                raise e
        else:
            # Standard loss calculation
            loss = model(**batch, labels=labels).loss
        
        # Stop activation tracking after forward pass
        if activation_tracker is not None and activation_tracker.is_tracking:
            activation_tracker.stop_tracking()
        scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()

        if global_step % args.gradient_accumulation != 0:
            continue

        # The below code is only executed during the update step

        # add grad clipping
        if args.grad_clipping != 0.0: torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

        if global_rank == 0: pbar.update(1)
        
        if not layer_wise_flag:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        update_step += 1
        update_time = time.time() - update_time

        # save checkpoint by save_every
        if local_step > args.gradient_accumulation and update_step % args.save_every == 0 and global_rank == 0:
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
            tokenizer.save_pretrained(current_model_directory)
            os.makedirs(args.save_dir, exist_ok=True)
            model.module.save_pretrained(current_model_directory, max_shard_size='100GB')
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
                "bytes_seen": bytes_seen,
                "bytes_seen_before": bytes_seen_before,
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
            eval_loss, eval_tokens, eval_bytes, bits_per_byte = evaluate_model(
                model, tokenizer, pad_idx, global_rank, world_size, device, args.batch_size, 
                track_bytes=args.track_dataset_bytes
            )
            
            eval_metrics = {
                "eval_loss": eval_loss,
                "eval_tokens": eval_tokens,
            }
            
            # Add BPB metrics if byte tracking is enabled
            if args.track_dataset_bytes and bits_per_byte is not None:
                eval_metrics.update({
                    "eval_bytes": eval_bytes,
                    "eval_bits_per_byte": bits_per_byte,
                    "eval_byte_level_perplexity": math.exp(bits_per_byte * math.log(2)),
                    "tokenizer_efficiency/bits_per_byte": bits_per_byte,
                    f"tokenizers/{args.tokenizer_name}/eval_bpb": bits_per_byte,
                })
                logger.info(f"Eval BPB at step {update_step}: {bits_per_byte:.6f}")
            
            if global_rank == 0:
                wandb.log(eval_metrics, step=global_step)
            logger.info(f"Eval loss at step {update_step}: {eval_loss}")

        if not layer_wise_flag:
            lr = optimizer.param_groups[0]["lr"]
        else:
            pass
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        bytes_in_update = bytes_seen - bytes_seen_before
        bytes_seen_before = bytes_seen
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
            if args.track_dataset_bytes:
                metrics_to_log.update({
                    "dataset_bytes_seen": bytes_seen,
                    "dataset_bytes_per_update": bytes_in_update,
                    "dataset_throughput_bytes_per_sec": bytes_in_update / update_time,
                    "dataset_bytes_per_token": bytes_seen / max(1, tokens_seen),
                    "dataset_cumulative_GB": bytes_seen / (1024**3),  # Convert to GB
                    "dataset_megabytes_per_update": bytes_in_update / (1024**2),  # Convert to MB
                })
            
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
                    logger.info(f"Logged {step_type} activation distributions at global_step {global_step}, update_step {update_step}")
                except Exception as e:
                    logger.warning(f"Failed to log activation distributions: {e}")
            
            # Log weight distributions if tracking is enabled
            if weight_tracker is not None and (update_step == 1 or update_step % args.weight_track_every == 0):
                try:
                    weight_tracker.log_to_wandb(global_step)
                    log_weight_summary(weight_tracker, global_step)
                    step_type = "INITIAL" if update_step == 1 else "REGULAR"
                    logger.info(f"Logged {step_type} weight distributions at global_step {global_step}, update_step {update_step}")
                except Exception as e:
                    logger.warning(f"Failed to log weight distributions: {e}")
            
            # Log byte consumption plots if tracking is enabled
            if args.track_dataset_bytes and update_step % args.log_bytes_every == 0:
                try:
                    log_byte_consumption_to_wandb(
                        step=global_step,
                        bytes_seen=bytes_seen,
                        tokens_seen=tokens_seen,
                        model_config=model_config.to_dict(),
                        tokenizer_name=args.tokenizer_name,
                        create_plot=True,
                        current_loss=loss.item()
                    )
                    logger.info(f"Logged byte consumption metrics at step {update_step}")
                except Exception as e:
                    logger.warning(f"Failed to log byte consumption metrics: {e}")
            
            wandb.log(metrics_to_log, step=global_step)
        update_time = time.time()

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0: pbar.close()

    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if global_rank == 0 and not os.path.exists(current_model_directory):
        logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
        os.makedirs(args.save_dir, exist_ok=True)
        model.module.save_pretrained(current_model_directory)

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
            "bytes_seen": bytes_seen,
            "bytes_seen_before": bytes_seen_before,
            "update_time": update_time,
        }
        with open(f"{current_model_directory}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc; gc.collect()
    torch.cuda.empty_cache()

    final_loss, final_tokens, final_eval_bytes, final_bpb = evaluate_model(
        model, tokenizer, pad_idx, global_rank, world_size, device, args.batch_size,
        track_bytes=args.track_dataset_bytes
    )

    final_eval_metrics = {
        "final_eval_loss": final_loss,
        "final_eval_tokens": final_tokens,
    }
    
    # Add final BPB metrics if byte tracking enabled
    if args.track_dataset_bytes and final_bpb is not None:
        final_eval_metrics.update({
            "final_eval_bytes": final_eval_bytes,
            "final_eval_bits_per_byte": final_bpb,
            "final_eval_byte_level_perplexity": math.exp(final_bpb * math.log(2)),
            f"final_summary/{args.tokenizer_name}_bpb": final_bpb,
        })

    if global_rank == 0:
        wandb.log(final_eval_metrics, step=global_step)
        logger.info(f"Final eval loss: {final_loss}")
        if args.track_dataset_bytes and final_bpb is not None:
            logger.info(f"Final eval BPB: {final_bpb:.6f}")
            logger.info(f"Final eval byte-level perplexity: {math.exp(final_bpb * math.log(2)):.2f}")
        
        # Log final byte consumption summary if tracking was enabled
        if args.track_dataset_bytes:
            try:
                # Calculate total training time 
                total_training_time_hours = (time.time() - training_start_time) / 3600
                
                log_final_byte_summary(
                    final_bytes=bytes_seen,
                    final_tokens=tokens_seen,
                    training_steps=update_step,
                    model_config=model_config.to_dict(),
                    tokenizer_name=args.tokenizer_name,
                    training_time_hours=max(0.1, total_training_time_hours),  # Avoid division by zero
                    final_bpb=final_bpb
                )
                logger.info("Logged final byte consumption summary")
            except Exception as e:
                logger.warning(f"Failed to log final byte consumption summary: {e}")

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
