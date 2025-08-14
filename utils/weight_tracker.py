"""
Weight Distribution Tracker for Individual Attention Heads and MLP Components

This module provides detailed weight tracking for attention heads and MLP components,
analyzing distributions of weight matrices sliced by head dimensions.
"""

import torch
import torch.nn as nn
import numpy as np
import wandb
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import matplotlib.pyplot as plt


class WeightDistributionTracker:
    """
    Tracks weight distributions for individual attention heads and MLP components.
    Designed for detailed analysis of parameter evolution during training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        track_every_n_steps: int = 100,
        device: str = "cuda",
    ):
        """
        Initialize weight tracker.
        
        Args:
            model: The neural network model to track
            track_every_n_steps: Track weights every N training steps
            device: Device for computations
        """
        self.model = model
        self.track_every_n_steps = track_every_n_steps
        self.device = device
        
        # Storage for weight statistics
        self.weight_stats: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Tracking state
        self.current_step = 0
        self.max_weight_samples = 1000000  # Default, can be overridden for large models
        
        # Cache for layer identification
        self._attention_layers = []
        self._mlp_layers = []
        self._identify_layers()
    
    def _identify_layers(self):
        """Identify attention and MLP layers in the model."""
        for name, module in self.model.named_modules():
            if "self_attn" in name or "attention" in name.lower():
                if hasattr(module, 'q_proj') and hasattr(module, 'k_proj') and hasattr(module, 'v_proj') and hasattr(module, 'o_proj'):
                    self._attention_layers.append((name, module))
            elif "mlp" in name.lower():
                if hasattr(module, 'gate_proj') and hasattr(module, 'up_proj') and hasattr(module, 'down_proj'):
                    self._mlp_layers.append((name, module))
    
    def should_track(self, step: int) -> bool:
        """Determine if we should track weights at this step."""
        if step == 1:
            return True
        return step % self.track_every_n_steps == 0
    
    def step(self, step: int) -> bool:
        """
        Update step counter and return whether tracking should occur.
        
        Returns:
            bool: True if weights should be tracked this step
        """
        self.current_step = step
        return self.should_track(step)
    
    def track_weights(self):
        """Track weight distributions for all attention heads and MLP components."""
        # Track attention head weights
        for layer_name, attention_module in self._attention_layers:
            self._track_attention_weights(layer_name, attention_module)
        
        # Track MLP component weights
        for layer_name, mlp_module in self._mlp_layers:
            self._track_mlp_weights(layer_name, mlp_module)
    
    def _track_attention_weights(self, layer_name: str, attention_module: nn.Module):
        """Track weights for individual attention heads."""
        num_heads = attention_module.num_heads
        head_dim = attention_module.head_dim
        
        # Track Q, K, V projection weights by head
        for proj_name, proj_module in [('q_proj', attention_module.q_proj), 
                                       ('k_proj', attention_module.k_proj),
                                       ('v_proj', attention_module.v_proj)]:
            weight = proj_module.weight.data
            # Split weight matrix by heads: [hidden_size, num_heads * head_dim]
            weight_by_heads = weight.view(weight.shape[0], num_heads, head_dim)
            
            for head_idx in range(num_heads):
                head_weight = weight_by_heads[:, head_idx, :]  # [hidden_size, head_dim]
                stats = self._compute_weight_distribution(head_weight)
                
                weight_name = f"{layer_name}/{proj_name}/head_{head_idx}"
                self.weight_stats[weight_name].append({
                    'step': self.current_step,
                    **stats
                })
        
        # Track output projection weights by head
        o_proj_weight = attention_module.o_proj.weight.data  # [hidden_size, hidden_size]
        # Split by heads: each head contributes [head_dim, hidden_size] to output
        o_weight_by_heads = o_proj_weight.view(o_proj_weight.shape[0], num_heads, head_dim).transpose(0, 1)
        
        for head_idx in range(num_heads):
            head_o_weight = o_weight_by_heads[head_idx]  # [hidden_size, head_dim]
            stats = self._compute_weight_distribution(head_o_weight)
            
            weight_name = f"{layer_name}/o_proj/head_{head_idx}"
            self.weight_stats[weight_name].append({
                'step': self.current_step,
                **stats
            })
    
    def _track_mlp_weights(self, layer_name: str, mlp_module: nn.Module):
        """Track MLP weights sliced by head dimensions."""
        # Get reference attention layer to determine head structure
        layer_idx = self._extract_layer_index(layer_name)
        attention_module = None
        
        # Find corresponding attention layer
        for attn_name, attn_module in self._attention_layers:
            if self._extract_layer_index(attn_name) == layer_idx:
                attention_module = attn_module
                break
        
        if attention_module is None:
            # Fallback: use default head structure
            num_heads = 8  # Default assumption
            head_dim = mlp_module.gate_proj.in_features // num_heads
        else:
            num_heads = attention_module.num_heads
            head_dim = attention_module.head_dim
        
        # Track gate_proj weights by head dimension
        for proj_name, proj_module in [('gate_proj', mlp_module.gate_proj),
                                       ('up_proj', mlp_module.up_proj)]:
            weight = proj_module.weight.data  # [intermediate_size, hidden_size]
            # Split hidden dimension by heads
            weight_by_heads = weight.view(weight.shape[0], num_heads, head_dim)
            
            for head_idx in range(num_heads):
                head_weight = weight_by_heads[:, head_idx, :]  # [intermediate_size, head_dim]
                stats = self._compute_weight_distribution(head_weight)
                
                weight_name = f"{layer_name}/{proj_name}/head_{head_idx}"
                self.weight_stats[weight_name].append({
                    'step': self.current_step,
                    **stats
                })
        
        # Track down_proj weights by head dimension  
        down_weight = mlp_module.down_proj.weight.data  # [hidden_size, intermediate_size]
        # Split output dimension by heads
        down_weight_by_heads = down_weight.view(num_heads, head_dim, down_weight.shape[1])
        
        for head_idx in range(num_heads):
            head_down_weight = down_weight_by_heads[head_idx]  # [head_dim, intermediate_size]
            stats = self._compute_weight_distribution(head_down_weight)
            
            weight_name = f"{layer_name}/down_proj/head_{head_idx}"
            self.weight_stats[weight_name].append({
                'step': self.current_step,
                **stats
            })
    
    def _extract_layer_index(self, layer_name: str) -> int:
        """Extract layer index from layer name."""
        # Look for patterns like "layers.0", "layer_0", etc.
        import re
        match = re.search(r'layers?[._](\d+)', layer_name)
        if match:
            return int(match.group(1))
        return 0
    
    def _compute_weight_distribution(self, weight: torch.Tensor, max_samples: int = None) -> Dict[str, Any]:
        """Compute comprehensive distribution statistics for a weight tensor."""
        # Convert to float32 and numpy for computation
        if weight.dtype in [torch.bfloat16, torch.half, torch.float16]:
            weight = weight.float()
        weight_np = weight.detach().cpu().float().numpy().flatten()
        
        # Sample if tensor is too large to avoid computational overhead for large models
        original_size = len(weight_np)
        if max_samples is None:
            max_samples = self.max_weight_samples
        if original_size > max_samples:
            indices = np.random.choice(original_size, max_samples, replace=False)
            weight_np = weight_np[indices]
        
        # Basic statistics
        stats = {
            "mean": float(np.mean(weight_np)),
            "std": float(np.std(weight_np)),
            "min": float(np.min(weight_np)),
            "max": float(np.max(weight_np)),
            "median": float(np.median(weight_np)),
            "q25": float(np.percentile(weight_np, 25)),
            "q75": float(np.percentile(weight_np, 75)),
            "q05": float(np.percentile(weight_np, 5)),
            "q95": float(np.percentile(weight_np, 95)),
            "abs_mean": float(np.mean(np.abs(weight_np))),
            "abs_max": float(np.max(np.abs(weight_np))),
        }
        
        # Add original size information
        stats["num_params"] = original_size
        stats["sampled"] = original_size > max_samples
        stats["sample_size"] = len(weight_np) if original_size > max_samples else original_size
        
        return stats
    
    def log_to_wandb(self, step: int):
        """Log weight distributions to wandb with organized metrics and evolution plots."""
        if not self.weight_stats:
            return
        
        # Organize metrics by layer and component
        organized_metrics = {}
        
        for weight_name, stats_list in self.weight_stats.items():
            if not stats_list:
                continue
            
            # Get the latest stats
            latest_stats = stats_list[-1]
            
            # Parse weight name: layer_name/proj_name/head_idx
            parts = weight_name.split('/')
            if len(parts) >= 3:
                layer_name = parts[0]
                proj_name = parts[1] 
                head_name = parts[2]
                
                # Create hierarchical metrics
                for stat_name, value in latest_stats.items():
                    if stat_name == 'step':
                        continue
                    
                    metric_name = f"weights/{layer_name}/{proj_name}/{stat_name}/{head_name}"
                    organized_metrics[metric_name] = value
        
        # Generate and add weight evolution plots if we have enough data points
        has_multiple_timepoints = any(len(stats_list) >= 2 for stats_list in self.weight_stats.values())
        if has_multiple_timepoints:
            try:
                evolution_plots = self.create_weight_evolution_plots()
                organized_metrics.update(evolution_plots)
            except Exception as e:
                print(f"Warning: Failed to create weight evolution plots: {e}")
        
        # Log to wandb
        if organized_metrics:
            wandb.log(organized_metrics, step=step)
    
    def create_weight_evolution_plots(self, save_dir: Optional[str] = None) -> Dict[str, Any]:
        """Create plots showing weight evolution over training steps."""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set beautiful plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        plots = {}
        
        # Group weights by layer and projection type
        weight_groups = defaultdict(lambda: defaultdict(list))
        
        for weight_name, stats_list in self.weight_stats.items():
            if not stats_list:
                continue
                
            parts = weight_name.split('/')
            if len(parts) >= 3:
                layer_name = parts[0]
                proj_name = parts[1]
                head_name = parts[2]
                
                weight_groups[f"{layer_name}/{proj_name}"][head_name] = stats_list
        
        # Create plots for each weight group
        for group_name, heads_data in weight_groups.items():
            try:
                fig = self._create_evolution_plot(group_name, heads_data)
                
                if save_dir:
                    plt.savefig(f"{save_dir}/weight_evolution_{group_name.replace('/', '_')}.png", 
                              dpi=150, bbox_inches='tight')
                
                # Convert to wandb image
                plots[f"weight_evolution/{group_name}"] = wandb.Image(fig)
                plt.close(fig)
                
            except Exception as e:
                print(f"Warning: Failed to create plot for {group_name}: {e}")
                continue
        
        return plots
    
    def _create_evolution_plot(self, group_name: str, heads_data: Dict[str, List[Dict]]) -> plt.Figure:
        """Create evolution plot for a specific weight group."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Weight Evolution: {group_name}', fontsize=14)
        
        metrics_to_plot = ['mean', 'std', 'abs_mean', 'abs_max']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            for head_name, stats_list in heads_data.items():
                steps = [s['step'] for s in stats_list]
                values = [s[metric] for s in stats_list]
                ax.plot(steps, values, label=head_name, alpha=0.7)
            
            ax.set_xlabel('Training Step')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of tracked weights and current state."""
        return {
            "num_attention_layers": len(self._attention_layers),
            "num_mlp_layers": len(self._mlp_layers),
            "current_step": self.current_step,
            "track_every_n_steps": self.track_every_n_steps,
            "tracked_weights": list(self.weight_stats.keys()),
            "total_weight_components": len(self.weight_stats),
        }
    
    def clear_stored_weights(self):
        """Clear stored weight statistics to free memory."""
        self.weight_stats.clear()


# Utility functions for easy integration

def create_weight_tracker(
    model: nn.Module,
    track_every_n_steps: int = 100,
    **kwargs
) -> WeightDistributionTracker:
    """
    Factory function to create a weight tracker with sensible defaults.
    """
    return WeightDistributionTracker(
        model=model,
        track_every_n_steps=track_every_n_steps,
        **kwargs
    )


def log_weight_summary(tracker: WeightDistributionTracker, step: int):
    """
    Log a summary of the weight tracker state to wandb.
    """
    summary = tracker.get_summary()
    wandb.log({
        "weight_tracker/num_attention_layers": summary["num_attention_layers"],
        "weight_tracker/num_mlp_layers": summary["num_mlp_layers"],
        "weight_tracker/total_components": summary["total_weight_components"],
        "weight_tracker/step": summary["current_step"],
    }, step=step)