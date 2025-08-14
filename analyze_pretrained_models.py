#!/usr/bin/env python3
"""
Pre-trained Model Analysis Script

Analyzes pre-trained models from key research papers with C4 dataset samples
to track activation and weight statistics and identify blow-up patterns.

Supported Papers:
- Scaling Laws for Neural Language Models (Kaplan et al., 2020)
- MoH: Multi-Head Attention as Mixture-of-Head Attention (2024)
- Multi-Head Mixture-of-Experts (2024) 
- SEAL: Scaling to Emphasize Attention for Long-Context Retrieval (2025)
"""

import torch
import torch.nn as nn
import numpy as np
import wandb
import argparse
import os
import time
import datasets
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import logging

# Import existing utilities
from utils.weight_tracker import WeightDistributionTracker
from utils.attention_analysis import AttentionHeadAnalyzer, AttentionScalingAnalyzer
from utils.moe_analysis import MoEAnalyzer
from peft_pretraining.dataloader import PreprocessedIterableDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelAnalysisConfig:
    """Configuration for model analysis"""
    
    def __init__(self, config_file: Optional[str] = None):
        # Load configuration from file if provided
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            self.models = config_data.get('models', {})
            self.analysis_configurations = config_data.get('analysis_configurations', {})
            self.dataset_configs = config_data.get('dataset_configs', {})
            self.tracking_configs = config_data.get('tracking_configs', {})
        else:
            # Fallback to default configuration
            self.models = {
                'scaling_laws_gpt2_small': {
                    'model_name': 'gpt2',
                    'model_type': 'standard',
                    'source': 'huggingface',
                    'description': 'GPT-2 representing scaling laws principles'
                }
            }
            self.analysis_configurations = {}
            self.dataset_configs = {}
            self.tracking_configs = {}
        
        # Analysis parameters (can be overridden)
        self.batch_size = 8
        self.max_length = 512
        self.max_batches = 100  # Number of C4 batches to process
        self.track_every_n_batches = 10
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        

class ActivationTracker:
    """Enhanced activation tracker for different model architectures"""
    
    def __init__(self, model: nn.Module, config: ModelAnalysisConfig):
        self.model = model
        self.config = config
        self.activation_stats = defaultdict(list)
        self.hooks = []
        self.current_batch = 0
        
        # Identify layer types
        self.attention_layers = []  # legacy: generic attention modules
        self.moe_layers = []
        self.mlp_layers = []       # legacy: generic mlp modules

        # Focused collections for outputs we want to measure
        self.attention_output_layers = []  # e.g., o_proj, out_proj, attention.dense, wo
        self.mlp_down_layers = []          # e.g., down_proj, dense_4h_to_h
        self._identify_layers()
        
    def _identify_layers(self):
        """Identify different types of layers in the model with robust name-based detection."""
        import re
        for name, module in self.model.named_modules():
            lowered = name.lower()

            # Identify specific attention output projection submodules
            if (
                ('attention' in lowered or 'attn' in lowered) and
                (
                    lowered.endswith('o_proj') or
                    lowered.endswith('out_proj') or
                    lowered.endswith('dense') or
                    lowered.endswith('wo') or
                    re.search(r'attention\.(o_proj|out_proj|dense|wo)$', lowered) is not None
                )
            ):
                self.attention_output_layers.append((name, module))

            # Also keep track of broader attention modules (fallback)
            if any(attn_key in lowered for attn_key in ['attention', 'attn']):
                self.attention_layers.append((name, module))

            # MoE layers (for Mixtral-style models)
            if 'moe' in lowered or 'expert' in lowered:
                self.moe_layers.append((name, module))

            # Identify MLP down-projection submodules across architectures
            if (
                ('mlp' in lowered or 'ffn' in lowered or 'feed_forward' in lowered or 'dense_4h_to_h' in lowered or 'down_proj' in lowered)
            ):
                # Capture specific down projection names seen in GPT-NeoX and LLaMA
                if lowered.endswith('down_proj') or 'dense_4h_to_h' in lowered:
                    self.mlp_down_layers.append((name, module))

                # Keep legacy broader MLP module references
                if any(key in lowered for key in ['mlp', 'feed_forward', 'ffn']):
                    self.mlp_layers.append((name, module))
    
    def register_hooks(self):
        """Register forward hooks for activation tracking"""
        def create_hook(layer_name, layer_type):
            def hook_fn(module, input, output):
                if self.current_batch % self.config.track_every_n_batches == 0:
                    self._track_activations(layer_name, layer_type, input, output)
            return hook_fn
        
        # Register hooks for attention output projection layers first (preferred)
        if self.attention_output_layers:
            for name, module in self.attention_output_layers:
                hook = module.register_forward_hook(create_hook(name, 'attention_output'))
                self.hooks.append(hook)
        else:
            # Fallback to broader attention modules
            for name, module in self.attention_layers:
                hook = module.register_forward_hook(create_hook(name, 'attention'))
                self.hooks.append(hook)
        
        # Register hooks for MoE layers
        for name, module in self.moe_layers:
            hook = module.register_forward_hook(create_hook(name, 'moe'))
            self.hooks.append(hook)
            
        # Register hooks for MLP down-projection layers (preferred)
        if self.mlp_down_layers:
            for name, module in self.mlp_down_layers:
                hook = module.register_forward_hook(create_hook(name, 'mlp_down'))
                self.hooks.append(hook)
        else:
            # Fallback to broader MLP modules
            for name, module in self.mlp_layers:
                hook = module.register_forward_hook(create_hook(name, 'mlp'))
                self.hooks.append(hook)
    
    def _track_activations(self, layer_name: str, layer_type: str, input_tensors, output_tensors):
        """Track activation statistics for blow-up pattern detection"""
        
        # Handle different input/output formats
        if isinstance(input_tensors, tuple):
            input_tensor = input_tensors[0]
        else:
            input_tensor = input_tensors
            
        if isinstance(output_tensors, tuple):
            output_tensor = output_tensors[0]
        else:
            output_tensor = output_tensors
        
        # Compute activation statistics
        if isinstance(output_tensor, torch.Tensor):
            # Convert to float32 for robust numpy ops
            if output_tensor.dtype in [torch.bfloat16, torch.half, torch.float16]:
                output_tensor = output_tensor.float()
            output_flat = output_tensor.detach().cpu().numpy().flatten()
            
            stats = {
                'batch': self.current_batch,
                'layer_name': layer_name,
                'layer_type': layer_type,
                'mean': float(np.mean(output_flat)),
                'std': float(np.std(output_flat)),
                'min': float(np.min(output_flat)),
                'max': float(np.max(output_flat)),
                'abs_mean': float(np.mean(np.abs(output_flat))),
                'abs_max': float(np.max(np.abs(output_flat))),
                'l2_norm': float(np.linalg.norm(output_flat)),
                'num_elements': len(output_flat),
                # Percentiles
                'q50': float(np.percentile(output_flat, 50)),
                'q75': float(np.percentile(output_flat, 75)),
                'q90': float(np.percentile(output_flat, 90)),
                'q95': float(np.percentile(output_flat, 95)),
                'q99': float(np.percentile(output_flat, 99)),
                # Blow-up indicators
                'has_inf': bool(np.any(np.isinf(output_flat))),
                'has_nan': bool(np.any(np.isnan(output_flat))),
                'max_abs_ratio': float(np.max(np.abs(output_flat)) / (np.mean(np.abs(output_flat)) + 1e-10)),
                'std_mean_ratio': float(np.std(output_flat) / (np.abs(np.mean(output_flat)) + 1e-10))
            }
            
            # Additional statistics for attention layers
            if layer_type in ['attention', 'attention_output'] and len(output_tensor.shape) >= 3:
                # Analyze attention head patterns
                if len(output_tensor.shape) == 4:  # [batch, heads, seq, dim]
                    head_stats = []
                    for head_idx in range(output_tensor.shape[1]):
                        head_output_tensor = output_tensor[:, head_idx, :, :]
                        if head_output_tensor.dtype in [torch.bfloat16, torch.half, torch.float16]:
                            head_output_tensor = head_output_tensor.float()
                        head_output = head_output_tensor.detach().cpu().numpy().flatten()
                        head_stats.append({
                            'head_idx': head_idx,
                            'mean': float(np.mean(head_output)),
                            'std': float(np.std(head_output)),
                            'abs_max': float(np.max(np.abs(head_output)))
                        })
                    stats['head_stats'] = head_stats
            
            self.activation_stats[f"{layer_type}_{layer_name}"].append(stats)
    
    def step(self, batch_idx: int):
        """Update batch counter"""
        self.current_batch = batch_idx
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class PretrainedModelAnalyzer:
    """Main analyzer for pre-trained models"""
    
    def __init__(self, config: ModelAnalysisConfig):
        self.config = config
        self.models_loaded = {}
        self.tokenizers = {}
        
    def load_model(self, model_key: str) -> Tuple[nn.Module, Any]:
        """Load a model and tokenizer based on configuration"""
        if model_key not in self.config.models:
            raise ValueError(f"Model {model_key} not found in configuration")
        
        model_config = self.config.models[model_key]
        model_name = model_config['model_name']
        
        logger.info(f"Loading model: {model_name}")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            self.models_loaded[model_key] = model
            self.tokenizers[model_key] = tokenizer
            
            logger.info(f"Successfully loaded {model_name}")
            logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def prepare_c4_dataset(self) -> torch.utils.data.DataLoader:
        """Prepare C4 dataset for analysis"""
        logger.info("Loading C4 dataset...")
        
        # Load C4 validation set for analysis
        dataset = datasets.load_dataset("c4", "en", split="validation", streaming=True, trust_remote_code=True)
        dataset = dataset.shuffle(seed=42)
        
        return dataset
    
    def analyze_model(self, model_key: str, max_batches: Optional[int] = None) -> Dict[str, Any]:
        """Analyze a specific model with C4 data"""
        if max_batches is None:
            max_batches = self.config.max_batches
            
        logger.info(f"Starting analysis of {model_key}")
        
        # Load model
        model, tokenizer = self.load_model(model_key)
        model.eval()
        
        # Setup trackers
        activation_tracker = ActivationTracker(model, self.config)
        weight_tracker = WeightDistributionTracker(
            model, 
            track_every_n_steps=self.config.track_every_n_batches
        )
        
        # Setup specialized analyzers based on model type
        attention_analyzer = None
        moe_analyzer = None
        scaling_analyzer = None
        
        model_type = self.config.models[model_key].get('model_type', 'standard')
        special_analysis = self.config.models[model_key].get('special_analysis', None)
        
        if special_analysis == 'attention_head_routing' or 'attention' in model_type:
            attention_analyzer = AttentionHeadAnalyzer(model)
            attention_analyzer.register_attention_hooks()
            logger.info(f"Enabled attention head analysis for {model_key}")
        
        if special_analysis == 'expert_routing' or model_type == 'moe':
            moe_analyzer = MoEAnalyzer(model)
            moe_analyzer.register_moe_hooks()
            logger.info(f"Enabled MoE analysis for {model_key}")
        
        if special_analysis == 'attention_scaling':
            scaling_analyzer = AttentionScalingAnalyzer(model)
            logger.info(f"Enabled attention scaling analysis for {model_key}")
        
        # Register hooks
        activation_tracker.register_hooks()
        
        # Prepare dataset
        c4_dataset = self.prepare_c4_dataset()
        preprocessed_dataset = PreprocessedIterableDataset(
            c4_dataset, 
            tokenizer, 
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            track_bytes=False
        )
        dataloader = torch.utils.data.DataLoader(preprocessed_dataset, batch_size=None, num_workers=1)
        
        # Analysis loop
        results = {
            'model_key': model_key,
            'model_config': self.config.models[model_key],
            'analysis_config': {
                'max_batches': max_batches,
                'batch_size': self.config.batch_size,
                'max_length': self.config.max_length
            },
            'batch_stats': [],
            'activation_stats': []
        }
        
        batch_idx = 0
        start_time = time.time()
        
        logger.info(f"Processing {max_batches} batches...")
        
        with torch.no_grad():
            for batch in dataloader:
                if batch_idx >= max_batches:
                    break
                
                # Move batch to device
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                
                # Update trackers
                activation_tracker.step(batch_idx)
                
                # Track weights if needed
                if weight_tracker.step(batch_idx):
                    weight_tracker.track_weights()
                
                # Forward pass
                try:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = outputs.loss if hasattr(outputs, 'loss') else None
                    
                    # Compute batch statistics
                    logits = outputs.logits
                    # Convert logits to float32 for numpy ops
                    logits_cpu = logits.detach().cpu()
                    if logits_cpu.dtype in [torch.bfloat16, torch.half, torch.float16]:
                        logits_cpu = logits_cpu.float()
                    logits_flat = logits_cpu.numpy().flatten()
                    
                    batch_stats = {
                        'batch_idx': batch_idx,
                        'loss': float(loss) if loss is not None else None,
                        'logits_mean': float(np.mean(logits_flat)),
                        'logits_std': float(np.std(logits_flat)),
                        'logits_max': float(np.max(logits_flat)),
                        'logits_min': float(np.min(logits_flat)),
                        'logits_abs_max': float(np.max(np.abs(logits_flat))),
                        'has_inf': bool(np.any(np.isinf(logits_flat))),
                        'has_nan': bool(np.any(np.isnan(logits_flat)))
                    }
                    
                    results['batch_stats'].append(batch_stats)
                    
                    # Collect current activation stats snapshot into results
                    if batch_idx % self.config.track_every_n_batches == 0:
                        for _, stats_list in activation_tracker.activation_stats.items():
                            if stats_list:
                                results['activation_stats'].append(stats_list[-1])

                    # Log to wandb
                    self._log_batch_to_wandb(model_key, batch_idx, batch_stats, 
                                           activation_tracker, weight_tracker)
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    continue
                
                batch_idx += 1
                
                if batch_idx % 10 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"Processed {batch_idx}/{max_batches} batches in {elapsed:.1f}s")
        
        # Cleanup
        activation_tracker.remove_hooks()
        if attention_analyzer:
            attention_analyzer.remove_hooks()
        if moe_analyzer:
            moe_analyzer.remove_hooks()
        
        # Final logging
        self._log_final_results(model_key, results, activation_tracker, weight_tracker)
        
        logger.info(f"Completed analysis of {model_key}")
        return results
    
    def _log_batch_to_wandb(self, model_key: str, batch_idx: int, batch_stats: Dict,
                           activation_tracker: ActivationTracker, weight_tracker: WeightDistributionTracker):
        """Log batch results to wandb"""
        
        # Basic batch metrics
        wandb_metrics = {
            f'{model_key}/batch_idx': batch_idx,
            f'{model_key}/loss': batch_stats.get('loss'),
            f'{model_key}/logits_mean': batch_stats['logits_mean'],
            f'{model_key}/logits_std': batch_stats['logits_std'],
            f'{model_key}/logits_abs_max': batch_stats['logits_abs_max'],
            f'{model_key}/has_inf': int(batch_stats['has_inf']),
            f'{model_key}/has_nan': int(batch_stats['has_nan'])
        }
        
        # Log activation statistics if available
        if batch_idx % self.config.track_every_n_batches == 0:
            for layer_key, stats_list in activation_tracker.activation_stats.items():
                if stats_list:
                    latest_stats = stats_list[-1]
                    layer_name = latest_stats['layer_name']
                    layer_type = latest_stats['layer_type']
                    
                    wandb_metrics.update({
                        f'{model_key}/activations/{layer_type}/{layer_name}/mean': latest_stats['mean'],
                        f'{model_key}/activations/{layer_type}/{layer_name}/std': latest_stats['std'],
                        f'{model_key}/activations/{layer_type}/{layer_name}/abs_max': latest_stats['abs_max'],
                        f'{model_key}/activations/{layer_type}/{layer_name}/l2_norm': latest_stats['l2_norm'],
                        f'{model_key}/activations/{layer_type}/{layer_name}/max_abs_ratio': latest_stats['max_abs_ratio'],
                        f'{model_key}/activations/{layer_type}/{layer_name}/has_inf': int(latest_stats['has_inf']),
                        f'{model_key}/activations/{layer_type}/{layer_name}/has_nan': int(latest_stats['has_nan'])
                    })
        
        # Log weight statistics
        weight_tracker.log_to_wandb(batch_idx)
        
        wandb.log(wandb_metrics, step=batch_idx)
    
    def _log_final_results(self, model_key: str, results: Dict, 
                          activation_tracker: ActivationTracker, weight_tracker: WeightDistributionTracker):
        """Log final analysis results"""
        
        # Compute summary statistics
        batch_stats = results['batch_stats']
        if batch_stats:
            losses = [bs['loss'] for bs in batch_stats if bs['loss'] is not None]
            logits_maxes = [bs['logits_abs_max'] for bs in batch_stats]
            
            summary = {
                f'{model_key}/summary/avg_loss': np.mean(losses) if losses else None,
                f'{model_key}/summary/max_logits_abs': np.max(logits_maxes),
                f'{model_key}/summary/num_inf_batches': sum(bs['has_inf'] for bs in batch_stats),
                f'{model_key}/summary/num_nan_batches': sum(bs['has_nan'] for bs in batch_stats),
                f'{model_key}/summary/total_batches': len(batch_stats)
            }
            
            wandb.log(summary)
        
        # Log model info
        model_info = {
            f'{model_key}/info/model_name': self.config.models[model_key]['model_name'],
            f'{model_key}/info/model_type': self.config.models[model_key]['model_type'],
            f'{model_key}/info/num_attention_layers': len(activation_tracker.attention_layers),
            f'{model_key}/info/num_moe_layers': len(activation_tracker.moe_layers),
            f'{model_key}/info/num_mlp_layers': len(activation_tracker.mlp_layers)
        }
        
        wandb.log(model_info)


def main():
    parser = argparse.ArgumentParser(description="Analyze pre-trained models with C4 dataset")
    parser.add_argument("--models", nargs="+", default=["scaling_laws_gpt2_small"], 
                       help="Models to analyze")
    parser.add_argument("--config", default="configs/pretrained_models_config.json",
                       help="Configuration file path")
    parser.add_argument("--max_batches", type=int, default=100,
                       help="Maximum number of C4 batches to process")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for processing")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--track_every", type=int, default=10,
                       help="Track detailed statistics every N batches")
    parser.add_argument("--wandb_project", default="pretrained-model-analysis",
                       help="WandB project name")
    parser.add_argument("--wandb_run", default=None,
                       help="WandB run name")
    parser.add_argument("--output_dir", default="./analysis_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Setup configuration
    config = ModelAnalysisConfig(config_file=args.config)
    config.max_batches = args.max_batches
    config.batch_size = args.batch_size
    config.max_length = args.max_length
    config.track_every_n_batches = args.track_every
    
    # Initialize wandb
    run_name = args.wandb_run or f"analysis_{'-'.join(args.models)}"
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = PretrainedModelAnalyzer(config)
    
    # Analyze each model
    all_results = {}
    for model_key in args.models:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Analyzing model: {model_key}")
            logger.info(f"{'='*50}")
            
            results = analyzer.analyze_model(model_key, args.max_batches)
            all_results[model_key] = results
            
            # Save intermediate results
            results_file = os.path.join(args.output_dir, f"{model_key}_results.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to analyze model {model_key}: {e}")
            continue
    
    # Save final results
    final_results_file = os.path.join(args.output_dir, "final_results.json")
    with open(final_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nAnalysis complete! Results saved to {args.output_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()