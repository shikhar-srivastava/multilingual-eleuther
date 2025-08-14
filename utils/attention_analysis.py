"""
Attention Analysis Utilities

Specialized analysis tools for attention mechanisms, particularly for:
- MoH (Mixture-of-Head) attention patterns
- SEAL attention scaling patterns
- Standard multi-head attention analysis
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import seaborn as sns
import logging

# Set up logging
logger = logging.getLogger(__name__)


class AttentionHeadAnalyzer:
    """Analyzer for attention head patterns and specialization"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.attention_layers = []
        self.attention_weights = defaultdict(list)
        self.head_statistics = defaultdict(list)
        self.hooks = []
        
        self._identify_attention_layers()
    
    def _identify_attention_layers(self):
        """Identify attention layers in the model"""
        for name, module in self.model.named_modules():
            # Look for attention modules
            if any(key in name.lower() for key in ['attention', 'attn']):
                # Check if it's a complete attention module
                if hasattr(module, 'q_proj') or hasattr(module, 'query'):
                    self.attention_layers.append((name, module))
                    
    def register_attention_hooks(self):
        """Register hooks to capture attention weights"""
        def create_attention_hook(layer_name):
            def hook_fn(module, input, output):
                # Try to extract attention weights if available
                if hasattr(module, 'attention_weights'):
                    attn_weights = module.attention_weights
                    self._analyze_attention_weights(layer_name, attn_weights)
                elif isinstance(output, tuple) and len(output) > 1:
                    # Some models return (output, attention_weights)
                    if len(output) > 1 and output[1] is not None:
                        attn_weights = output[1] 
                        self._analyze_attention_weights(layer_name, attn_weights)
            return hook_fn
        
        for layer_name, module in self.attention_layers:
            hook = module.register_forward_hook(create_attention_hook(layer_name))
            self.hooks.append(hook)
    
    def _analyze_attention_weights(self, layer_name: str, attention_weights: torch.Tensor):
        """Analyze attention weight patterns"""
        if attention_weights is None:
            return
            
        # Expected shape: [batch, heads, seq_len, seq_len]
        attn_weights = attention_weights.detach().cpu().numpy()
        
        if len(attn_weights.shape) == 4:
            batch_size, num_heads, seq_len, _ = attn_weights.shape
            
            # Analyze each head
            head_stats = []
            for head_idx in range(num_heads):
                head_weights = attn_weights[:, head_idx, :, :]  # [batch, seq, seq]
                
                # Compute head-specific statistics
                stats = {
                    'head_idx': head_idx,
                    'mean_attention': float(np.mean(head_weights)),
                    'std_attention': float(np.std(head_weights)),
                    'max_attention': float(np.max(head_weights)),
                    'entropy': self._compute_attention_entropy(head_weights),
                    'locality_score': self._compute_locality_score(head_weights),
                    'specialization_score': self._compute_specialization_score(head_weights)
                }
                head_stats.append(stats)
            
            self.head_statistics[layer_name].append({
                'layer_name': layer_name,
                'batch_size': batch_size,
                'num_heads': num_heads,
                'seq_len': seq_len,
                'head_stats': head_stats,
                'layer_entropy': float(np.mean([h['entropy'] for h in head_stats])),
                'layer_locality': float(np.mean([h['locality_score'] for h in head_stats])),
                'head_diversity': float(np.std([h['specialization_score'] for h in head_stats]))
            })
    
    def _compute_attention_entropy(self, attention_weights: np.ndarray) -> float:
        """Compute entropy of attention distribution"""
        # Average over batch and compute entropy for each position
        avg_weights = np.mean(attention_weights, axis=0)  # [seq, seq]
        
        entropies = []
        for i in range(avg_weights.shape[0]):
            attn_dist = avg_weights[i, :] 
            # Normalize to probability distribution
            attn_dist = attn_dist / (np.sum(attn_dist) + 1e-10)
            # Compute entropy
            entropy = -np.sum(attn_dist * np.log(attn_dist + 1e-10))
            entropies.append(entropy)
        
        return float(np.mean(entropies))
    
    def _compute_locality_score(self, attention_weights: np.ndarray) -> float:
        """Compute how local vs global the attention is"""
        # Average over batch
        avg_weights = np.mean(attention_weights, axis=0)  # [seq, seq]
        seq_len = avg_weights.shape[0]
        
        # Compute distance-weighted attention
        locality_scores = []
        for i in range(seq_len):
            distances = np.abs(np.arange(seq_len) - i)
            weighted_attention = avg_weights[i, :] * (1.0 / (distances + 1))
            local_score = np.sum(weighted_attention) / np.sum(avg_weights[i, :])
            locality_scores.append(local_score)
        
        return float(np.mean(locality_scores))
    
    def _compute_specialization_score(self, attention_weights: np.ndarray) -> float:
        """Compute how specialized this head is (variance in attention patterns)"""
        avg_weights = np.mean(attention_weights, axis=0)  # [seq, seq]
        
        # Compute variance across different positions
        position_variances = []
        for i in range(avg_weights.shape[0]):
            variance = np.var(avg_weights[i, :])
            position_variances.append(variance)
        
        return float(np.mean(position_variances))
    
    def analyze_head_similarity(self, layer_name: str) -> Dict[str, Any]:
        """Analyze similarity between different attention heads"""
        if layer_name not in self.head_statistics or not self.head_statistics[layer_name]:
            return {}
        
        latest_stats = self.head_statistics[layer_name][-1]
        head_stats = latest_stats['head_stats']
        
        # Create feature vectors for each head
        features = ['mean_attention', 'entropy', 'locality_score', 'specialization_score']
        head_features = []
        
        for head_stat in head_stats:
            feature_vector = [head_stat[feat] for feat in features]
            head_features.append(feature_vector)
        
        head_features = np.array(head_features)
        
        # Compute pairwise similarities
        num_heads = len(head_features)
        similarity_matrix = np.zeros((num_heads, num_heads))
        
        for i in range(num_heads):
            for j in range(num_heads):
                # Cosine similarity
                dot_product = np.dot(head_features[i], head_features[j])
                norm_i = np.linalg.norm(head_features[i])
                norm_j = np.linalg.norm(head_features[j])
                similarity = dot_product / (norm_i * norm_j + 1e-10)
                similarity_matrix[i, j] = similarity
        
        return {
            'similarity_matrix': similarity_matrix.tolist(),
            'avg_similarity': float(np.mean(similarity_matrix[np.triu_indices(num_heads, k=1)])),
            'diversity_score': float(1.0 - np.mean(similarity_matrix[np.triu_indices(num_heads, k=1)])),
            'num_heads': num_heads
        }
    
    def detect_head_specialization_patterns(self) -> Dict[str, Any]:
        """Detect common patterns in head specialization across layers"""
        patterns = {
            'local_heads': [],  # Heads with high locality scores
            'global_heads': [],  # Heads with low locality scores  
            'high_entropy_heads': [],  # Heads with uniform attention
            'low_entropy_heads': [],  # Heads with focused attention
            'specialized_heads': [],  # Heads with high specialization
            'general_heads': []  # Heads with low specialization
        }
        
        for layer_name, layer_stats_list in self.head_statistics.items():
            if not layer_stats_list:
                continue
                
            latest_stats = layer_stats_list[-1]
            head_stats = latest_stats['head_stats']
            
            for head_stat in head_stats:
                head_id = f"{layer_name}_head_{head_stat['head_idx']}"
                
                # Classify heads based on their properties
                if head_stat['locality_score'] > 0.7:
                    patterns['local_heads'].append(head_id)
                elif head_stat['locality_score'] < 0.3:
                    patterns['global_heads'].append(head_id)
                
                if head_stat['entropy'] > 2.0:
                    patterns['high_entropy_heads'].append(head_id)
                elif head_stat['entropy'] < 1.0:
                    patterns['low_entropy_heads'].append(head_id)
                
                if head_stat['specialization_score'] > 0.5:
                    patterns['specialized_heads'].append(head_id)
                elif head_stat['specialization_score'] < 0.1:
                    patterns['general_heads'].append(head_id)
        
        return patterns
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class AttentionScalingAnalyzer:
    """Analyzer for attention scaling patterns (SEAL-style analysis)"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.scaling_stats = defaultdict(list)
        self.context_length_effects = {}
        
    def analyze_attention_scaling(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, Any]:
        """Analyze how attention patterns scale with context length"""
        
        context_length = input_ids.shape[1]
        
        # Get attention weights with hook
        attention_weights = {}
        hooks = []
        
        def create_scaling_hook(layer_name):
            def hook_fn(module, input, output):
                if hasattr(module, 'attention_weights'):
                    attention_weights[layer_name] = module.attention_weights
                elif isinstance(output, tuple) and len(output) > 1:
                    if output[1] is not None:
                        attention_weights[layer_name] = output[1]
            return hook_fn
        
        # Register hooks
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                if hasattr(module, 'q_proj') or hasattr(module, 'query'):
                    hook = module.register_forward_hook(create_scaling_hook(name))
                    hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Analyze scaling effects
        scaling_results = {
            'context_length': context_length,
            'layer_scaling_stats': {}
        }
        
        for layer_name, attn_weights in attention_weights.items():
            if attn_weights is not None:
                stats = self._compute_scaling_statistics(attn_weights, context_length)
                scaling_results['layer_scaling_stats'][layer_name] = stats
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return scaling_results
    
    def _compute_scaling_statistics(self, attention_weights: torch.Tensor, context_length: int) -> Dict[str, Any]:
        """Compute statistics related to attention scaling"""
        attn_weights = attention_weights.detach().cpu().numpy()
        
        if len(attn_weights.shape) == 4:
            batch_size, num_heads, seq_len, _ = attn_weights.shape
            
            # Compute attention concentration (how much attention focuses on few tokens)
            concentration_scores = []
            for b in range(batch_size):
                for h in range(num_heads):
                    head_attn = attn_weights[b, h, :, :]
                    # Compute Gini coefficient as concentration measure
                    concentration = self._compute_gini_coefficient(head_attn)
                    concentration_scores.append(concentration)
            
            # Compute attention distance (average distance of attended tokens)
            distance_scores = []
            for b in range(batch_size):
                for h in range(num_heads):
                    head_attn = attn_weights[b, h, :, :]
                    avg_distance = self._compute_average_attention_distance(head_attn)
                    distance_scores.append(avg_distance)
            
            return {
                'context_length': context_length,
                'avg_concentration': float(np.mean(concentration_scores)),
                'std_concentration': float(np.std(concentration_scores)),
                'avg_attention_distance': float(np.mean(distance_scores)),
                'std_attention_distance': float(np.std(distance_scores)),
                'max_attention_value': float(np.max(attn_weights)),
                'min_attention_value': float(np.min(attn_weights)),
                'attention_variance': float(np.var(attn_weights))
            }
        
        return {}
    
    def _compute_gini_coefficient(self, attention_matrix: np.ndarray) -> float:
        """Compute Gini coefficient for attention concentration"""
        # Flatten and sort attention values
        values = attention_matrix.flatten()
        values = np.sort(values)
        n = len(values)
        
        if n == 0:
            return 0.0
        
        # Compute Gini coefficient
        cumsum = np.cumsum(values)
        gini = (2 * np.sum((np.arange(1, n + 1)) * values)) / (n * cumsum[-1]) - (n + 1) / n
        return float(gini)
    
    def _compute_average_attention_distance(self, attention_matrix: np.ndarray) -> float:
        """Compute average distance of attention weights"""
        seq_len = attention_matrix.shape[0]
        distances = []
        
        for i in range(seq_len):
            attn_row = attention_matrix[i, :]
            # Normalize to probability distribution
            attn_row = attn_row / (np.sum(attn_row) + 1e-10)
            
            # Compute weighted average distance
            weighted_distance = 0.0
            for j in range(seq_len):
                distance = abs(i - j)
                weighted_distance += distance * attn_row[j]
            
            distances.append(weighted_distance)
        
        return float(np.mean(distances))


def create_attention_heatmap(attention_weights: np.ndarray, layer_name: str, head_idx: int, 
                           save_path: Optional[str] = None) -> plt.Figure:
    """Create heatmap visualization of attention weights"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Take first batch if multiple batches
    if len(attention_weights.shape) == 4:
        attn_matrix = attention_weights[0, head_idx, :, :]
    elif len(attention_weights.shape) == 3:
        attn_matrix = attention_weights[head_idx, :, :]
    else:
        attn_matrix = attention_weights
    
    # Create heatmap
    sns.heatmap(attn_matrix, ax=ax, cmap='Blues', cbar=True)
    ax.set_title(f'Attention Weights - {layer_name} - Head {head_idx}')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def analyze_attention_head_routing(model: nn.Module, input_ids: torch.Tensor, 
                                 attention_mask: torch.Tensor) -> Dict[str, Any]:
    """Analyze attention head routing patterns (for MoH-style analysis)"""
    
    routing_stats = defaultdict(list)
    hooks = []
    
    def create_routing_hook(layer_name):
        def hook_fn(module, input, output):
            # Look for routing-related outputs or internal states
            if hasattr(module, 'router_outputs'):
                routing_stats[layer_name].append(module.router_outputs)
            elif hasattr(module, 'head_weights'):
                routing_stats[layer_name].append(module.head_weights)
        return hook_fn
    
    # Register hooks for attention layers
    for name, module in model.named_modules():
        if 'attention' in name.lower() or 'attn' in name.lower():
            hook = module.register_forward_hook(create_routing_hook(name))
            hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Analyze routing patterns
    analysis_results = {
        'total_layers_with_routing': len(routing_stats),
        'routing_patterns': {}
    }
    
    for layer_name, routing_data in routing_stats.items():
        if routing_data:
            # Analyze the routing data
            analysis_results['routing_patterns'][layer_name] = {
                'num_routing_decisions': len(routing_data),
                'routing_entropy': 'computed_if_available'  # Placeholder
            }
    
    return analysis_results


class DCFormerAnalyzer:
    """
    Analyzer for DCFormer (Dynamically Composable Multi-Head Attention)
    From the paper: Improving Transformers with Dynamically Composable Multi-Head Attention
    arxiv: 2405.08553
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.attention_modules = self._find_dcmha_modules()
        self.hooks = []
        self.composition_patterns: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.head_weights: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.dynamic_compositions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def _find_dcmha_modules(self):
        """Find DCFormer's Dynamically Composable Multi-Head Attention modules"""
        dcmha_modules = []
        for name, module in self.model.named_modules():
            # Look for DCFormer-specific attention modules
            # Based on the paper, DCFormer uses DCMHA (Dynamically Composable Multi-Head Attention)
            if 'dcmha' in name.lower() or 'composable' in name.lower():
                dcmha_modules.append((name, module))
            # Also check for standard attention modules that might be modified in DCFormer
            elif hasattr(module, 'q_proj') and hasattr(module, 'k_proj') and hasattr(module, 'v_proj'):
                # Check if this might be a DCFormer attention by looking for composition-related attributes
                if hasattr(module, 'composition') or hasattr(module, 'dynamic') or hasattr(module, 'head_weights'):
                    dcmha_modules.append((name, module))
                else:
                    # Even standard attention modules are interesting for comparison
                    dcmha_modules.append((name, module))
        
        logger.info(f"Found {len(dcmha_modules)} potential DCMHA modules")
        return dcmha_modules

    def _dcmha_hook(self, name: str):
        def hook_fn(module, input, output):
            """Hook function to capture dynamic composition patterns"""
            if isinstance(output, tuple):
                attention_output = output[0]
                # DCFormer might return additional outputs like composition weights
                if len(output) > 1:
                    additional_outputs = output[1:]
                    # Look for composition weights or head combination patterns
                    for i, additional_output in enumerate(additional_outputs):
                        if isinstance(additional_output, torch.Tensor):
                            self.composition_patterns[f"{name}_output_{i}"].append(additional_output.detach().cpu())
            else:
                attention_output = output
            
            # Analyze attention output for dynamic composition effects
            if isinstance(attention_output, torch.Tensor):
                # Extract head-wise statistics if possible
                if len(attention_output.shape) == 4:  # [batch, heads, seq, dim]
                    num_heads = attention_output.shape[1]
                    head_contributions = []
                    
                    for head_idx in range(num_heads):
                        head_output = attention_output[:, head_idx, :, :]
                        head_norm = torch.norm(head_output, dim=-1).mean().item()
                        head_contributions.append(head_norm)
                    
                    # Analyze composition diversity
                    head_variance = torch.var(torch.tensor(head_contributions)).item()
                    head_entropy = -sum([p * torch.log(torch.tensor(p + 1e-10)) for p in torch.softmax(torch.tensor(head_contributions), dim=0)]).item()
                    
                    composition_stats = {
                        'head_contributions': head_contributions,
                        'head_variance': head_variance,
                        'head_entropy': head_entropy,
                        'effective_heads': sum(1 for contrib in head_contributions if contrib > 0.1 * max(head_contributions)),
                        'composition_sparsity': sum(1 for contrib in head_contributions if contrib < 0.01 * max(head_contributions)) / num_heads
                    }
                    
                    self.dynamic_compositions[name].append(composition_stats)
                
                # Track overall attention output statistics
                output_stats = {
                    'mean': attention_output.mean().item(),
                    'std': attention_output.std().item(),
                    'max': attention_output.max().item(),
                    'l2_norm': torch.norm(attention_output).item()
                }
                
                self.head_weights[name].append(torch.tensor([output_stats['mean'], output_stats['std'], output_stats['max']]))

        return hook_fn

    def register_dcmha_hooks(self):
        for name, module in self.attention_modules:
            hook = module.register_forward_hook(self._dcmha_hook(name))
            self.hooks.append(hook)
        logger.info(f"Registered {len(self.hooks)} DCFormer DCMHA hooks")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        logger.info("Removed DCFormer DCMHA hooks")

    def log_to_wandb(self, step: int):
        """Log dynamic composition statistics to WandB"""
        for name, compositions in self.dynamic_compositions.items():
            if compositions:
                latest_comp = compositions[-1]
                
                wandb.log({
                    f"dcformer/{name}/head_variance": latest_comp['head_variance'],
                    f"dcformer/{name}/head_entropy": latest_comp['head_entropy'], 
                    f"dcformer/{name}/effective_heads": latest_comp['effective_heads'],
                    f"dcformer/{name}/composition_sparsity": latest_comp['composition_sparsity'],
                    f"dcformer/{name}/head_contributions": latest_comp['head_contributions']
                }, step=step)
        
        for name, weights in self.head_weights.items():
            if weights:
                latest_weight = weights[-1]
                wandb.log({
                    f"dcformer_weights/{name}/mean": latest_weight[0].item(),
                    f"dcformer_weights/{name}/std": latest_weight[1].item(), 
                    f"dcformer_weights/{name}/max": latest_weight[2].item()
                }, step=step)
        
        for name, patterns in self.composition_patterns.items():
            if patterns:
                # Log composition pattern statistics
                latest_pattern = patterns[-1]
                if latest_pattern.numel() > 0:
                    wandb.log({
                        f"dcformer_composition/{name}/pattern_mean": latest_pattern.mean().item(),
                        f"dcformer_composition/{name}/pattern_std": latest_pattern.std().item(),
                        f"dcformer_composition/{name}/pattern_sparsity": (latest_pattern.abs() < 0.01).float().mean().item()
                    }, step=step)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of DCFormer analysis"""
        summary = {
            'num_dcmha_modules': len(self.attention_modules),
            'total_compositions_analyzed': sum(len(comps) for comps in self.dynamic_compositions.values()),
            'avg_effective_heads': 0,
            'avg_composition_sparsity': 0
        }
        
        if self.dynamic_compositions:
            all_effective_heads = []
            all_sparsity = []
            for compositions in self.dynamic_compositions.values():
                for comp in compositions:
                    all_effective_heads.append(comp['effective_heads'])
                    all_sparsity.append(comp['composition_sparsity'])
            
            if all_effective_heads:
                summary['avg_effective_heads'] = sum(all_effective_heads) / len(all_effective_heads)
            if all_sparsity:
                summary['avg_composition_sparsity'] = sum(all_sparsity) / len(all_sparsity)
        
        return summary