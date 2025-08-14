"""
Activation Distribution Tracker for LLM Pre-training

This module provides efficient activation tracking and distribution analysis
for large language models during training, with wandb integration.
"""

import torch
import torch.nn as nn
import numpy as np
import wandb
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import random


class ActivationDistributionTracker:
    """
    Tracks activation distributions across all layers of a neural network.
    Designed for efficiency with large models and distributed training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        track_every_n_steps: int = 100,
        sample_ratio: float = 0.1,
        max_samples_per_activation: int = 10000,
        track_gradients: bool = False,
        device: str = "cuda",
    ):
        """
        Initialize activation tracker.
        
        Args:
            model: The neural network model to track
            track_every_n_steps: Track activations every N training steps
            sample_ratio: Ratio of tokens to sample from each batch
            max_samples_per_activation: Maximum number of samples to store per activation
            track_gradients: Whether to also track gradient distributions
            device: Device to store temporary activations
        """
        self.model = model
        self.track_every_n_steps = track_every_n_steps
        self.sample_ratio = sample_ratio
        self.max_samples_per_activation = max_samples_per_activation
        self.track_gradients = track_gradients
        self.device = device
        
        # Storage for activations
        self.activations: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.gradients: Dict[str, List[torch.Tensor]] = defaultdict(list) if track_gradients else {}
        
        # Hook handles for cleanup
        self.forward_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.backward_hooks: List[torch.utils.hooks.RemovableHandle] = []
        
        # Tracking state
        self.current_step = 0
        self.is_tracking = False
        
        # Layer name mapping for better organization
        self.layer_names: Dict[str, str] = {}
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks for all relevant layers."""
        
        def create_forward_hook(name: str):
            def hook_fn(module, input, output):
                if self.is_tracking:
                    self._store_activation(name, output)
            return hook_fn
        
        def create_backward_hook(name: str):
            def hook_fn(module, grad_input, grad_output):
                if self.is_tracking and self.track_gradients:
                    if grad_output[0] is not None:
                        self._store_gradient(name, grad_output[0])
            return hook_fn
        
        # Register hooks for different layer types
        for name, module in self.model.named_modules():
            # Skip the root model and containers
            if name == "" or len(list(module.children())) > 0:
                continue
                
            # Determine layer type and create descriptive name
            layer_type = self._get_layer_type(module)
            if layer_type:
                hook_name = f"{layer_type}_{name}"
                self.layer_names[hook_name] = name
                
                # Register forward hook
                handle = module.register_forward_hook(create_forward_hook(hook_name))
                self.forward_hooks.append(handle)
                
                # Register backward hook if tracking gradients
                if self.track_gradients:
                    handle = module.register_full_backward_hook(create_backward_hook(hook_name))
                    self.backward_hooks.append(handle)
    
    def _get_layer_type(self, module: nn.Module) -> Optional[str]:
        """Determine the type of layer for naming purposes."""
        if isinstance(module, nn.Embedding):
            return "embedding"
        elif isinstance(module, nn.Linear):
            return "linear"
        elif isinstance(module, nn.LayerNorm):
            return "layernorm"
        elif "RMSNorm" in module.__class__.__name__:
            return "rmsnorm"
        elif "Attention" in module.__class__.__name__:
            return "attention"
        elif "MLP" in module.__class__.__name__:
            return "mlp"
        elif "DecoderLayer" in module.__class__.__name__:
            return "decoder_layer"
        else:
            # Only track important layer types
            return None
    
    def _store_activation(self, name: str, activation: torch.Tensor):
        """Store sampled activation for later analysis."""
        if not isinstance(activation, torch.Tensor):
            return
            
        # Sample a subset of the activation to save memory
        sampled_activation = self._sample_tensor(activation)
        
        # Store activation (detached to save memory)
        self.activations[name].append(sampled_activation.detach().cpu())
        
        # Limit memory usage by keeping only recent activations
        if len(self.activations[name]) > self.max_samples_per_activation // 100:
            self.activations[name] = self.activations[name][-self.max_samples_per_activation // 100:]
    
    def _store_gradient(self, name: str, gradient: torch.Tensor):
        """Store sampled gradient for later analysis."""
        if not isinstance(gradient, torch.Tensor):
            return
            
        # Sample a subset of the gradient
        sampled_gradient = self._sample_tensor(gradient)
        
        # Store gradient (detached to save memory)
        self.gradients[name].append(sampled_gradient.detach().cpu())
        
        # Limit memory usage
        if len(self.gradients[name]) > self.max_samples_per_activation // 100:
            self.gradients[name] = self.gradients[name][-self.max_samples_per_activation // 100:]
    
    def _sample_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sample a subset of tensor elements to reduce memory usage."""
        # Flatten the tensor and sample - use contiguous() to handle non-contiguous tensors
        flat_tensor = tensor.contiguous().view(-1)
        n_samples = min(
            int(len(flat_tensor) * self.sample_ratio),
            self.max_samples_per_activation
        )
        
        if n_samples >= len(flat_tensor):
            return flat_tensor
        
        # Random sampling
        indices = torch.randperm(len(flat_tensor))[:n_samples]
        return flat_tensor[indices]
    
    def start_tracking(self):
        """Start tracking activations for the current step."""
        self.is_tracking = True
    
    def stop_tracking(self):
        """Stop tracking activations."""
        self.is_tracking = False
    
    def should_track(self, step: int) -> bool:
        """Determine if we should track activations at this step."""
        # Always track the first step (step 1) to capture initial distributions
        if step == 1:
            return True
        # Then track at regular intervals
        return step % self.track_every_n_steps == 0
    
    def step(self, step: int) -> bool:
        """
        Update step counter and return whether tracking should occur.
        
        Returns:
            bool: True if activations should be tracked this step
        """
        self.current_step = step
        return self.should_track(step)
    
    def compute_distributions(self) -> Dict[str, Dict[str, Any]]:
        """
        Compute distribution statistics for all tracked activations.
        
        Returns:
            Dict containing distribution statistics for each layer
        """
        distributions = {}
        
        # Process activations
        for name, activation_list in self.activations.items():
            if not activation_list:
                continue
                
            # Concatenate all stored activations for this layer
            all_activations = torch.cat(activation_list, dim=0)
            
            # Compute distribution statistics
            dist_stats = self._compute_tensor_distribution(all_activations)
            distributions[f"activations/{name}"] = dist_stats
        
        # Process gradients if tracking
        if self.track_gradients:
            for name, gradient_list in self.gradients.items():
                if not gradient_list:
                    continue
                    
                all_gradients = torch.cat(gradient_list, dim=0)
                dist_stats = self._compute_tensor_distribution(all_gradients)
                distributions[f"gradients/{name}"] = dist_stats
        
        return distributions
    
    def _compute_tensor_distribution(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Compute comprehensive distribution statistics for a tensor."""
        # Convert bfloat16 to float32 for numpy compatibility
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        tensor_np = tensor.numpy()
        
        # Basic statistics
        stats = {
            "mean": float(np.mean(tensor_np)),
            "std": float(np.std(tensor_np)),
            "min": float(np.min(tensor_np)),
            "max": float(np.max(tensor_np)),
            "median": float(np.median(tensor_np)),
            "q25": float(np.percentile(tensor_np, 25)),
            "q75": float(np.percentile(tensor_np, 75)),
            "q05": float(np.percentile(tensor_np, 5)),
            "q95": float(np.percentile(tensor_np, 95)),
        }
        
        # Histogram for wandb
        hist, bin_edges = np.histogram(tensor_np, bins=50, density=True)
        stats["histogram"] = wandb.Histogram(np_histogram=(hist, bin_edges))
        
        # Remove additional statistics for distribution analysis
        stats["num_samples"] = len(tensor_np)
        
        return stats
    
    def log_to_wandb(self, step: int):
        """Log activation distributions to wandb."""
        distributions = self.compute_distributions()
        
        if not distributions:
            return
        
        # Organize metrics by layer type for better visualization
        organized_metrics = {}
        
        for full_name, stats in distributions.items():
            # Parse the metric name (e.g., "activations/embedding_embed_tokens")
            prefix, layer_name = full_name.split("/", 1)
            layer_type = layer_name.split("_")[0]
            
            # Create hierarchical structure for wandb
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
        self.activations.clear()
        if self.track_gradients:
            self.gradients.clear()
    
    def cleanup(self):
        """Remove all hooks and cleanup."""
        for handle in self.forward_hooks:
            handle.remove()
        for handle in self.backward_hooks:
            handle.remove()
        
        self.forward_hooks.clear()
        self.backward_hooks.clear()
        self.clear_stored_activations()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of tracked layers and current state."""
        return {
            "num_layers_tracked": len(self.layer_names),
            "tracked_layers": list(self.layer_names.keys()),
            "current_step": self.current_step,
            "track_every_n_steps": self.track_every_n_steps,
            "sample_ratio": self.sample_ratio,
            "is_tracking": self.is_tracking,
            "stored_activations": {name: len(acts) for name, acts in self.activations.items()},
        }


# Utility functions for easy integration with training scripts

def create_activation_tracker(
    model: nn.Module,
    track_every_n_steps: int = 100,
    sample_ratio: float = 0.1,
    **kwargs
) -> ActivationDistributionTracker:
    """
    Factory function to create an activation tracker with sensible defaults.
    """
    return ActivationDistributionTracker(
        model=model,
        track_every_n_steps=track_every_n_steps,
        sample_ratio=sample_ratio,
        **kwargs
    )


def log_activation_summary(tracker: ActivationDistributionTracker, step: int):
    """
    Log a summary of the activation tracker state to wandb.
    """
    summary = tracker.get_summary()
    wandb.log({
        "activation_tracker/num_layers": summary["num_layers_tracked"],
        "activation_tracker/tracking_enabled": summary["is_tracking"],
        "activation_tracker/step": summary["current_step"],
    }, step=step) 