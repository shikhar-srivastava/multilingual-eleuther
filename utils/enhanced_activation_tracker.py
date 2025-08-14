"""
Enhanced Activation Distribution Tracker for Head-wise Analysis

This module extends the basic activation tracker to provide detailed
head-wise analysis of attention mechanisms including Q, K, V states
before/after rotary embedding and attention values.
"""

import torch
import torch.nn as nn
import numpy as np
import wandb
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict
import random
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .activation_tracker import ActivationDistributionTracker


class EnhancedActivationTracker(ActivationDistributionTracker):
    """
    Enhanced activation tracker with detailed head-wise analysis.
    Extends the base tracker to capture fine-grained attention mechanics.
    """
    
    def __init__(
        self,
        model: nn.Module,
        track_every_n_steps: int = 100,
        sample_ratio: float = 0.1,
        max_samples_per_activation: int = 10000,
        track_gradients: bool = False,
        device: str = "cuda",
        track_head_activations: bool = True,
    ):
        """
        Initialize enhanced activation tracker.
        
        Args:
            model: The neural network model to track
            track_every_n_steps: Track activations every N training steps
            sample_ratio: Ratio of tokens to sample from each batch
            max_samples_per_activation: Maximum number of samples to store per activation
            track_gradients: Whether to also track gradient distributions
            device: Device to store temporary activations
            track_head_activations: Whether to track detailed head-wise activations
        """
        super().__init__(
            model=model,
            track_every_n_steps=track_every_n_steps,
            sample_ratio=sample_ratio,
            max_samples_per_activation=max_samples_per_activation,
            track_gradients=track_gradients,
            device=device,
        )
        
        self.track_head_activations = track_head_activations
        
        # Storage for head-wise activations
        self.head_activations: Dict[str, List[torch.Tensor]] = defaultdict(list)
        
        # Additional hook handles for head-wise tracking
        self.head_hooks: List[torch.utils.hooks.RemovableHandle] = []
        
        if track_head_activations:
            self._register_head_hooks()
    
    def _register_head_hooks(self):
        """Register hooks for detailed head-wise activation tracking."""
        # Find attention layers and register detailed hooks
        for name, module in self.model.named_modules():
            if hasattr(module, 'q_proj') and hasattr(module, 'k_proj') and hasattr(module, 'v_proj'):
                self._register_attention_hooks(name, module)
    
    def _register_attention_hooks(self, layer_name: str, attention_module: nn.Module):
        """Register hooks for a specific attention layer."""
        
        # Hook for Q, K, V projections
        def create_qkv_hook(proj_name: str):
            def hook_fn(module, input, output):
                if self.is_tracking:
                    self._store_head_activation(f"{layer_name}/{proj_name}_proj", output, attention_module)
            return hook_fn
        
        # Register hooks for Q, K, V projections
        for proj_name in ['q', 'k', 'v']:
            proj_module = getattr(attention_module, f'{proj_name}_proj')
            hook = proj_module.register_forward_hook(create_qkv_hook(proj_name))
            self.head_hooks.append(hook)
        
        # Hook for output projection
        def o_proj_hook(module, input, output):
            if self.is_tracking:
                self._store_head_activation(f"{layer_name}/o_proj_output", output, attention_module)
        
        hook = attention_module.o_proj.register_forward_hook(o_proj_hook)
        self.head_hooks.append(hook)
        
        # Hook the entire attention forward to capture Q, K after rotary and attention values
        def attention_forward_hook(module, input, output):
            if self.is_tracking:
                # We'll capture internal states via monkey patching in the forward method
                pass
        
        hook = attention_module.register_forward_hook(attention_forward_hook)
        self.head_hooks.append(hook)
        
        # Monkey patch the attention forward to capture intermediate states
        self._monkey_patch_attention_forward(layer_name, attention_module)
    
    def _monkey_patch_attention_forward(self, layer_name: str, attention_module: nn.Module):
        """Monkey patch attention forward to capture intermediate states."""
        original_forward = attention_module.forward
        
        def enhanced_forward(
            self_attn,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
        ):
            if not self.is_tracking:
                return original_forward(
                    hidden_states, attention_mask, position_ids, 
                    past_key_value, output_attentions, use_cache
                )
            
            bsz, q_len, _ = hidden_states.size()

            # Get Q, K, V projections
            query_states = self_attn.q_proj(hidden_states).view(bsz, q_len, self_attn.num_heads, self_attn.head_dim).transpose(1, 2)
            key_states = self_attn.k_proj(hidden_states).view(bsz, q_len, self_attn.num_heads, self_attn.head_dim).transpose(1, 2)
            value_states = self_attn.v_proj(hidden_states).view(bsz, q_len, self_attn.num_heads, self_attn.head_dim).transpose(1, 2)

            # Store Q, K, V states before rotary embedding (by head)
            self._store_qkv_states_by_head(f"{layer_name}/q_before_rotary", query_states)
            self._store_qkv_states_by_head(f"{layer_name}/k_before_rotary", key_states)
            self._store_qkv_states_by_head(f"{layer_name}/v_states", value_states)

            # Apply rotary embeddings
            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value[0].shape[-2]
            cos, sin = self_attn.rotary_emb(value_states, seq_len=kv_seq_len)
            
            # Import the rotary embedding function
            from peft_pretraining.modeling_llama import apply_rotary_pos_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            # Store Q, K states after rotary embedding (by head)
            self._store_qkv_states_by_head(f"{layer_name}/q_after_rotary", query_states)
            self._store_qkv_states_by_head(f"{layer_name}/k_after_rotary", key_states)

            # Handle past key values
            if past_key_value is not None:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

            past_key_value = (key_states, value_states) if use_cache else None

            # Attention computation
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, dropout_p=0.0, is_causal=True,
            )

            # Store attention output by head (before o_proj)
            self._store_qkv_states_by_head(f"{layer_name}/attn_output_before_o_proj", attn_output)

            # Reshape and apply output projection
            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape(bsz, q_len, self_attn.hidden_size)
            attn_output = self_attn.o_proj(attn_output)

            # Store final attention output
            self._store_head_activation(f"{layer_name}/attn_final_output", attn_output, self_attn)

            attn_weights = None
            return attn_output, attn_weights, past_key_value
        
        # Replace the forward method
        attention_module.forward = enhanced_forward.__get__(attention_module, type(attention_module))
    
    def _store_head_activation(self, name: str, activation: torch.Tensor, attention_module: nn.Module):
        """Store activation for later analysis (general case)."""
        if not isinstance(activation, torch.Tensor):
            return
            
        # Sample a subset of the activation to save memory
        sampled_activation = self._sample_tensor(activation)
        
        # Store activation (detached to save memory)
        self.head_activations[name].append(sampled_activation.detach().cpu())
        
        # Limit memory usage
        if len(self.head_activations[name]) > self.max_samples_per_activation // 100:
            self.head_activations[name] = self.head_activations[name][-self.max_samples_per_activation // 100:]
    
    def _store_qkv_states_by_head(self, base_name: str, states: torch.Tensor):
        """Store Q, K, V or attention states by individual head."""
        if not isinstance(states, torch.Tensor):
            return
        
        # states shape: [batch, num_heads, seq_len, head_dim]
        num_heads = states.shape[1]
        
        for head_idx in range(num_heads):
            head_states = states[:, head_idx, :, :]  # [batch, seq_len, head_dim]
            sampled_head_states = self._sample_tensor(head_states)
            
            head_name = f"{base_name}/head_{head_idx}"
            self.head_activations[head_name].append(sampled_head_states.detach().cpu())
            
            # Limit memory usage
            if len(self.head_activations[head_name]) > self.max_samples_per_activation // 100:
                self.head_activations[head_name] = self.head_activations[head_name][-self.max_samples_per_activation // 100:]
    
    def compute_distributions(self) -> Dict[str, Dict[str, Any]]:
        """
        Compute distribution statistics for all tracked activations including head-wise.
        
        Returns:
            Dict containing distribution statistics for each layer and head
        """
        distributions = super().compute_distributions()
        
        # Process head-wise activations
        if self.track_head_activations:
            for name, activation_list in self.head_activations.items():
                if not activation_list:
                    continue
                    
                # Concatenate all stored activations for this head
                all_activations = torch.cat(activation_list, dim=0)
                
                # Compute distribution statistics
                dist_stats = self._compute_tensor_distribution(all_activations)
                distributions[f"head_activations/{name}"] = dist_stats
        
        return distributions
    
    def log_to_wandb(self, step: int):
        """Log activation distributions to wandb with head-wise organization."""
        distributions = self.compute_distributions()
        
        if not distributions:
            return
        
        # Organize metrics by layer type and head for better visualization
        organized_metrics = {}
        
        for full_name, stats in distributions.items():
            # Parse the metric name
            if full_name.startswith("head_activations/"):
                # Head-wise activations: "head_activations/layer_name/activation_type/head_X"
                name_parts = full_name.split("/", 1)[1]  # Remove "head_activations/"
                
                # Create hierarchical structure for wandb
                for stat_name, value in stats.items():
                    if stat_name == "histogram":
                        metric_name = f"head_activations/{name_parts}/histogram"
                    else:
                        metric_name = f"head_activations/{stat_name}/{name_parts}"
                    
                    organized_metrics[metric_name] = value
            else:
                # Regular activations - use parent class organization
                prefix, layer_name = full_name.split("/", 1)
                layer_type = layer_name.split("_")[0]
                
                for stat_name, value in stats.items():
                    if stat_name == "histogram":
                        metric_name = f"{prefix}/{layer_type}/{layer_name}/histogram"
                    else:
                        metric_name = f"{prefix}/{layer_type}/{stat_name}/{layer_name}"
                    
                    organized_metrics[metric_name] = value
        
        # Log to wandb
        wandb.log(organized_metrics, step=step)
    
    def clear_stored_activations(self):
        """Clear stored activations to free memory."""
        super().clear_stored_activations()
        if self.track_head_activations:
            self.head_activations.clear()
    
    def cleanup(self):
        """Remove all hooks and cleanup."""
        super().cleanup()
        
        # Remove head-specific hooks
        for handle in self.head_hooks:
            handle.remove()
        self.head_hooks.clear()
        
        if self.track_head_activations:
            self.head_activations.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of tracked layers and current state."""
        summary = super().get_summary()
        
        if self.track_head_activations:
            summary.update({
                "track_head_activations": True,
                "stored_head_activations": {name: len(acts) for name, acts in self.head_activations.items()},
                "num_head_activation_types": len(self.head_activations),
            })
        else:
            summary["track_head_activations"] = False
        
        return summary


# Utility functions for easy integration

def create_enhanced_activation_tracker(
    model: nn.Module,
    track_every_n_steps: int = 100,
    sample_ratio: float = 0.1,
    track_head_activations: bool = True,
    **kwargs
) -> EnhancedActivationTracker:
    """
    Factory function to create an enhanced activation tracker with sensible defaults.
    """
    return EnhancedActivationTracker(
        model=model,
        track_every_n_steps=track_every_n_steps,
        sample_ratio=sample_ratio,
        track_head_activations=track_head_activations,
        **kwargs
    )


def log_enhanced_activation_summary(tracker: EnhancedActivationTracker, step: int):
    """
    Log a summary of the enhanced activation tracker state to wandb.
    """
    summary = tracker.get_summary()
    
    base_metrics = {
        "activation_tracker/num_layers": summary["num_layers_tracked"],
        "activation_tracker/tracking_enabled": summary["is_tracking"],
        "activation_tracker/step": summary["current_step"],
    }
    
    if summary.get("track_head_activations", False):
        base_metrics.update({
            "activation_tracker/head_tracking_enabled": True,
            "activation_tracker/num_head_types": summary["num_head_activation_types"],
        })
    
    wandb.log(base_metrics, step=step)