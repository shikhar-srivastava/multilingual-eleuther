"""
Coordinate Check Training Script for muP and CompleteP

This script verifies that THIS codebase is correctly configured for CompleteP
parameterization by running coordinate checks across different model widths/depths.

Key differences from the original (broken) version:
- Uses LLAMA 1 architecture (NOT LLAMA 2) with pre-normalization
- Matches the architecture in peft_pretraining/modeling_llama.py
- Follows the nanoGPT-mup reference implementation exactly

The coordinate check verifies:
1. Activations have stable magnitudes across different widths (for muP)
2. Activations have stable magnitudes across different depths (for CompleteP)
3. Optimal hyperparameters transfer across scales

Usage examples:
    # Standard parameterization (SP) - baseline
    python coord_check_train.py --n_layer 4 --hidden_size 256
    
    # muP only (width scaling)
    python coord_check_train.py --n_layer 4 --hidden_size 256 --mup_enabled --mup_width_multiplier 1.0
    
    # CompleteP (width + depth scaling)
    python coord_check_train.py --n_layer 4 --hidden_size 256 --mup_enabled --depth_alpha_enabled \
        --mup_width_multiplier 1.0 --depth_multiplier 2.0 --depth_alpha_exp 1.0
"""

import os
import sys
import time
import math
import json
import argparse
import numpy as np
from functools import partial
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from transformers import AutoConfig, AutoTokenizer

from peft_pretraining.modeling_llama import LlamaForCausalLM
from utils.csv_logging import CSVLogWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Coordinate Check Training for muP/CompleteP")
    
    # Model architecture
    parser.add_argument("--n_layer", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size / model width")
    parser.add_argument("--num_attention_heads", type=int, default=None, 
                        help="Number of attention heads (default: hidden_size // 64)")
    parser.add_argument("--intermediate_size", type=int, default=None,
                        help="MLP intermediate size (default: hidden_size * 4)")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--vocab_size", type=int, default=65, help="Vocabulary size (65 for char-level)")
    
    # Training
    parser.add_argument("--max_iters", type=int, default=10, help="Maximum training iterations")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--adam_eps", type=float, default=1e-12)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--init_std", type=float, default=0.02)
    parser.add_argument("--decay_lr", action="store_true", default=False)
    
    # muP settings
    parser.add_argument("--mup_enabled", action="store_true", default=False)
    parser.add_argument("--mup_disable_attention_scaling", action="store_true", default=False)
    parser.add_argument("--mup_disable_hidden_lr_scaling", action="store_true", default=False)
    parser.add_argument("--mup_width_multiplier", type=float, default=1.0)
    parser.add_argument("--mup_base_width", type=int, default=256)
    parser.add_argument("--mup_input_alpha", type=float, default=1.0)
    parser.add_argument("--mup_output_alpha", type=float, default=1.0)
    parser.add_argument("--mup_enable_coord_check_logging", action="store_true", default=True)
    
    # CompleteP settings
    parser.add_argument("--depth_alpha_enabled", action="store_true", default=False)
    parser.add_argument("--depth_multiplier", type=float, default=1.0)
    parser.add_argument("--depth_base_depth", type=int, default=2)
    parser.add_argument("--depth_alpha_exp", type=float, default=1.0)
    
    # Normalization
    parser.add_argument("--use_layernorm", action="store_true", default=False,
                        help="Use LayerNorm instead of RMSNorm (for debugging/comparison)")
    
    # Activation function
    parser.add_argument("--use_gelu_mlp", action="store_true", default=False,
                        help="Use GELU MLP (like GPT-2) instead of SwiGLU (LLAMA standard)")
    
    # Position embeddings
    parser.add_argument("--position_embedding_type", type=str, default="rope",
                        choices=["rope", "learned", "sinusoidal"],
                        help="Type of position embeddings (rope=rotary, learned=GPT-2 style, sinusoidal=fixed)")
    
    # Data
    parser.add_argument("--dataset", type=str, default="shakespeare_char",
                        choices=["shakespeare_char", "c4_sample"])
    parser.add_argument("--data_dir", type=str, default=None)
    
    # Output
    parser.add_argument("--out_dir", type=str, default="coord_check_out")
    parser.add_argument("--csv_log", action="store_true", default=True)
    parser.add_argument("--wandb_log", action="store_true", default=False)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--eval_iters", type=int, default=1)
    
    # System
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--compile", action="store_true", default=False)
    
    args = parser.parse_args()
    
    # Compute derived values
    if args.num_attention_heads is None:
        args.num_attention_heads = max(1, args.hidden_size // 64)
    if args.intermediate_size is None:
        args.intermediate_size = args.hidden_size * 4
    
    # Compute muP width multiplier
    if args.mup_enabled and args.mup_width_multiplier == 1.0:
        args.mup_width_multiplier = args.hidden_size / args.mup_base_width
    
    # Compute depth multiplier
    if args.depth_alpha_enabled and args.depth_multiplier == 1.0:
        args.depth_multiplier = args.n_layer / args.depth_base_depth
    
    return args


def get_batch(data, block_size, batch_size, device):
    """Get a batch of data for training."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def prepare_shakespeare_data(data_dir=None):
    """Prepare Shakespeare character-level data."""
    if data_dir is None:
        data_dir = '/localdisk/ssrivas9/multilingual-eleuther/data/shakespeare_char'
    
    # Create data dir if needed
    os.makedirs(data_dir, exist_ok=True)
    
    train_path = os.path.join(data_dir, 'train.bin')
    val_path = os.path.join(data_dir, 'val.bin')
    
    if not os.path.exists(train_path):
        # Download and prepare Shakespeare data
        print("Downloading Shakespeare data...")
        import urllib.request
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        input_file_path = os.path.join(data_dir, 'input.txt')
        urllib.request.urlretrieve(url, input_file_path)
        
        with open(input_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Character-level encoding
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        
        # Save metadata
        meta = {'vocab_size': vocab_size, 'stoi': stoi, 'itos': {i: ch for ch, i in stoi.items()}}
        with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
            import pickle
            pickle.dump(meta, f)
        
        # Encode and split
        data = np.array([stoi[ch] for ch in text], dtype=np.uint16)
        n = len(data)
        train_data = data[:int(n*0.9)]
        val_data = data[int(n*0.9):]
        
        train_data.tofile(train_path)
        val_data.tofile(val_path)
        print(f"Prepared Shakespeare data: {len(train_data)} train, {len(val_data)} val tokens")
    
    train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
    
    # Load vocab size
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        import pickle
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        vocab_size = meta['vocab_size']
    else:
        vocab_size = 65  # Default for Shakespeare char
    
    return train_data, val_data, vocab_size


def create_model_config(args, vocab_size):
    """Create model configuration with muP/CompleteP settings.
    
    This creates a LLAMA 1 config (not LLAMA 2) with pre-normalization,
    matching the architecture in peft_pretraining/modeling_llama.py
    """
    from transformers.models.llama.configuration_llama import LlamaConfig
    
    # Create config from scratch for LLAMA 1 architecture
    config = LlamaConfig(
        # Architecture params
        hidden_size=args.hidden_size,
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=args.max_length,
        max_sequence_length=args.max_length,
        
        # Activation and norm
        hidden_act="silu",
        rms_norm_eps=1e-6,
        
        # Position embeddings - configurable
        position_embedding_type=args.position_embedding_type,
        
        # Initialization
        initializer_range=args.init_std,
        
        # Model behavior
        use_cache=False,
        
        # Token IDs
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=1,
        
        # Model type
        model_type="llama",
    )
    
    # muP settings (for maximal update parameterization)
    config.mup_enabled = args.mup_enabled
    config.mup_disable_attention_scaling = args.mup_disable_attention_scaling
    config.mup_width_multiplier = args.mup_width_multiplier
    config.mup_input_alpha = args.mup_input_alpha
    config.mup_output_alpha = args.mup_output_alpha
    
    # CompleteP settings (depth scaling)
    config.depth_alpha_enabled = args.depth_alpha_enabled
    config.depth_multiplier = args.depth_multiplier
    config.depth_alpha_exp = args.depth_alpha_exp
    
    # Architecture options (for comparison with nanoGPT)
    config.use_layernorm = args.use_layernorm
    config.use_gelu_mlp = args.use_gelu_mlp
    
    return config


def configure_optimizers(model, args):
    """Configure optimizer with muP/CompleteP LR scaling.
    
    This follows the nanoGPT-mup reference implementation for CompleteP.
    Key points:
    - Embeddings (input and output/lm_head): lr_scale=1.0, no depth scaling
    - Hidden layer norms: depth_lr_scaling (if CompleteP), else 1.0
    - Hidden weights (Q,K,V,O,MLP): width_lr_scaling * depth_lr_scaling
    - Hidden biases: depth_lr_scaling (if CompleteP), else 1.0
    - Final norm: lr_scale=1.0, no depth scaling
    - Weight decay: applied per-group, scaled by 1/width_lr_scaling for hidden weights
    - Adam epsilon: scaled for CompleteP by (1/width) * (depth^-alpha)
    """
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    
    if args.mup_enabled and not args.mup_disable_hidden_lr_scaling:
        emb_params = []
        hidden_norm_params = []
        hidden_weight_params = []
        hidden_bias_params = []
        final_norm_params = []
        
        for name, param in param_dict.items():
            if 'embed_tokens' in name:
                emb_params.append(param)
            elif 'lm_head' in name:
                # Output embedding (lm_head) gets same treatment as input embeddings
                # lr_scale=1.0, no width or depth scaling
                emb_params.append(param)
            elif 'model.norm.' in name:
                # Final layer norm - no scaling
                final_norm_params.append(param)
            elif 'layernorm' in name.lower() or 'input_layernorm' in name or 'post_attention_layernorm' in name:
                # Hidden layer norms - get depth scaling if CompleteP
                hidden_norm_params.append(param)
            elif param.dim() >= 2:
                # Hidden layer weights (Q, K, V, O, MLP) - get both width and depth scaling
                hidden_weight_params.append(param)
            else:
                # Hidden layer biases - get depth scaling if CompleteP
                hidden_bias_params.append(param)
        
        # Compute LR scaling factors
        width_lr_scaling = 1.0 / args.mup_width_multiplier
        depth_lr_scaling = args.depth_multiplier ** (args.depth_alpha_exp - 1) if args.depth_alpha_enabled else 1.0
        
        # Compute adjusted adam_eps for CompleteP
        adam_eps = args.adam_eps
        if args.depth_alpha_enabled:
            adam_eps *= (1 / args.mup_width_multiplier) * (args.depth_multiplier ** (-1 * args.depth_alpha_exp))
        
        print(f"muP/CompleteP optimizer configuration:")
        print(f"  width_lr_scaling: {width_lr_scaling}")
        print(f"  depth_lr_scaling: {depth_lr_scaling}")
        print(f"  adam_eps: {adam_eps}")
        print(f"  Embeddings (input+output): lr_scale=1.0")
        print(f"  Hidden norms: lr_scale={depth_lr_scaling}")
        print(f"  Hidden weights: lr_scale={width_lr_scaling * depth_lr_scaling}, weight_decay={args.weight_decay / width_lr_scaling}")
        print(f"  Hidden biases: lr_scale={depth_lr_scaling}")
        print(f"  Final norm: lr_scale=1.0")
        
        if args.depth_alpha_enabled:
            ### Begin CompleteP code ###
            optim_groups = [
                {
                    'params': emb_params,
                    'weight_decay': args.weight_decay,
                    'lr_scale': 1.0,
                },
                {
                    'params': hidden_norm_params,
                    'weight_decay': 0.0,
                    'lr_scale': depth_lr_scaling,
                },
                {
                    'params': hidden_weight_params,
                    'weight_decay': args.weight_decay / width_lr_scaling,
                    'lr_scale': width_lr_scaling * depth_lr_scaling,
                },
                {
                    'params': hidden_bias_params,
                    'weight_decay': 0.0,
                    'lr_scale': depth_lr_scaling,
                },
                {
                    'params': final_norm_params,
                    'weight_decay': 0.0,
                    'lr_scale': 1.0,
                },
            ]
            ### End CompleteP code ###
        else:
            ### Begin muP code ###
            optim_groups = [
                {
                    'params': emb_params,
                    'weight_decay': args.weight_decay,
                    'lr_scale': 1.0,
                },
                {
                    'params': hidden_norm_params,
                    'weight_decay': 0.0,
                    'lr_scale': 1.0,
                },
                {
                    'params': hidden_weight_params,
                    'weight_decay': args.weight_decay,
                    'lr_scale': width_lr_scaling,
                },
                {
                    'params': hidden_bias_params,
                    'weight_decay': 0.0,
                    'lr_scale': 1.0,
                },
                {
                    'params': final_norm_params,
                    'weight_decay': 0.0,
                    'lr_scale': 1.0,
                },
            ]
            ### End muP code ###
        
        optim_groups = [g for g in optim_groups if len(g['params']) > 0]
        optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(args.beta1, args.beta2), eps=adam_eps)
    else:
        # Standard parameterization without muP/CompleteP scaling
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': args.weight_decay, 'lr_scale': 1.0},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr_scale': 1.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.adam_eps)
    
    return optimizer


@torch.no_grad()
def estimate_loss(model, train_data, val_data, args, ctx):
    """Estimate train and val loss."""
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            X, Y = get_batch(data, args.max_length, args.batch_size, args.device)
            with ctx:
                outputs = model(input_ids=X, labels=Y)
                loss = outputs.loss
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device and dtype
    device = args.device
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    ptdtype = dtype_map[args.dtype]
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    
    # Setup output
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Prepare data
    print(f"Preparing {args.dataset} data...")
    if args.dataset == 'shakespeare_char':
        train_data, val_data, vocab_size = prepare_shakespeare_data(args.data_dir)
        args.vocab_size = vocab_size
    else:
        raise ValueError(f"Dataset {args.dataset} not supported yet")
    
    # Create model - LLAMA 1 with pre-normalization
    print("="*80)
    print(f"Creating LLAMA 1 model with pre-normalization (NOT LLAMA 2)")
    print(f"  Architecture: {args.n_layer} layers, {args.hidden_size} hidden size")
    print(f"  Norm type: {os.getenv('NORM_TYPE', 'pre')} (set via NORM_TYPE env var)")
    print("="*80)
    
    config = create_model_config(args, args.vocab_size)
    model = LlamaForCausalLM(config)
    model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nModel created successfully:")
    print(f"  Parameters: {num_params:.2f}M")
    print(f"  Position embeddings: {config.position_embedding_type}")
    
    print(f"\nmuP configuration:")
    print(f"  Enabled: {args.mup_enabled}")
    print(f"  Width multiplier: {args.mup_width_multiplier}")
    print(f"  Base width: {args.mup_base_width}")
    print(f"  Input alpha: {args.mup_input_alpha}")
    print(f"  Output alpha: {args.mup_output_alpha}")
    
    print(f"\nCompleteP configuration:")
    print(f"  Enabled: {args.depth_alpha_enabled}")
    print(f"  Depth multiplier: {args.depth_multiplier}")
    print(f"  Base depth: {args.depth_base_depth}")
    print(f"  Alpha exponent: {args.depth_alpha_exp}")
    
    if args.depth_alpha_enabled:
        expected_residual_scaling = 1.0 / (args.depth_multiplier ** args.depth_alpha_exp)
        print(f"  Expected residual scaling: {expected_residual_scaling}")
    
    print(f"\nArchitecture Options:")
    print(f"  Normalization: {'LayerNorm' if args.use_layernorm else 'RMSNorm'}")
    print(f"  MLP activation: {'GELU (GPT-2 style)' if args.use_gelu_mlp else 'SwiGLU (LLAMA standard)'}")
    print(f"  Position embeddings: {args.position_embedding_type}")
    print(f"  Norm epsilon: {config.rms_norm_eps}")
    
    print(f"\nDtype:")
    print(f"  Training dtype: {args.dtype}")
    print(f"  PyTorch dtype: {ptdtype}")
    
    print("="*80)
    
    # Compile if requested
    if args.compile:
        print("Compiling model...")
        model = torch.compile(model)
    
    # Setup optimizer
    optimizer = configure_optimizers(model, args)
    
    # Setup CSV logging
    csv_logger = None
    if args.csv_log:
        csv_logger = CSVLogWrapper(out_dir=args.out_dir, config=vars(args))
    
    # Training loop
    print(f"Starting training for {args.max_iters} iterations...")
    iter_num = 0
    best_val_loss = 1e9
    
    coord_check_dict = None
    
    while iter_num < args.max_iters:
        # Get learning rate
        lr = args.lr  # No decay for coordinate checks
        
        # Apply LR scaling for muP
        if args.mup_enabled and not args.mup_disable_hidden_lr_scaling:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * param_group.get('lr_scale', 1.0)
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # Evaluate
        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, args, ctx)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            if csv_logger:
                csv_logger.log({
                    'iter': iter_num,
                    'train/loss': losses['train'],
                    'val/loss': losses['val'],
                    'lr': lr,
                })
        
        # Setup coordinate check hooks
        if args.mup_enable_coord_check_logging:
            coord_check_dict = {
                'token_embedding': [],
                'attn': [],
                'mlp': [],
                'lm_head': [],
                'last_layer': []
            }
            
            def coord_check_hook(module, input, output, key):
                with torch.no_grad():
                    if isinstance(output, tuple):
                        output = output[0]
                    coord_check_dict[key].append(output.abs().mean().item())
            
            handles = []
            n_layers = config.num_hidden_layers
            for module_name, module in model.named_modules():
                if module_name == 'model.embed_tokens':
                    handles.append(module.register_forward_hook(partial(coord_check_hook, key='token_embedding')))
                elif module_name.endswith('.self_attn'):
                    handles.append(module.register_forward_hook(partial(coord_check_hook, key='attn')))
                elif module_name.endswith('.mlp'):
                    handles.append(module.register_forward_hook(partial(coord_check_hook, key='mlp')))
                elif module_name == 'lm_head':
                    handles.append(module.register_forward_hook(partial(coord_check_hook, key='lm_head')))
                elif module_name == f'model.layers.{n_layers - 1}':
                    handles.append(module.register_forward_hook(partial(coord_check_hook, key='last_layer')))
        
        # Forward pass
        for micro_step in range(args.gradient_accumulation_steps):
            X, Y = get_batch(train_data, args.max_length, args.batch_size, device)
            with ctx:
                outputs = model(input_ids=X, labels=Y)
                loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
        
        # Remove hooks
        if args.mup_enable_coord_check_logging:
            for handle in handles:
                handle.remove()
        
        # Gradient clipping
        if args.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # Log coordinate check metrics
        if args.mup_enable_coord_check_logging and csv_logger:
            for key in coord_check_dict:
                if coord_check_dict[key]:
                    csv_logger.log({f'{key}_act_abs_mean': np.mean(coord_check_dict[key])})
        
        # Commit to CSV
        if csv_logger:
            csv_logger.step()
        
        iter_num += 1
    
    # Final evaluation
    losses = estimate_loss(model, train_data, val_data, args, ctx)
    print(f"Final: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # Close logger
    if csv_logger:
        csv_logger.close()
    
    print(f"Training complete. Logs saved to {args.out_dir}")


if __name__ == "__main__":
    main()

