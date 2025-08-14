"""
Mixture of Experts (MoE) Analysis Utilities

Specialized analysis tools for MoE models, particularly for:
- Expert utilization patterns
- Routing behavior analysis  
- Load balancing analysis
- Multi-Head MoE patterns
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import seaborn as sns


class MoEAnalyzer:
    """Analyzer for Mixture of Experts models"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.moe_layers = []
        self.expert_stats = defaultdict(list)
        self.routing_stats = defaultdict(list)
        self.load_balancing_stats = defaultdict(list)
        self.hooks = []
        
        self._identify_moe_layers()
    
    def _identify_moe_layers(self):
        """Identify MoE layers in the model"""
        for name, module in self.model.named_modules():
            # Look for MoE-related modules
            if any(key in name.lower() for key in ['moe', 'expert', 'switch', 'mixture']):
                self.moe_layers.append((name, module))
            # Also check for Mixtral-style MoE
            elif 'block_sparse_moe' in name.lower():
                self.moe_layers.append((name, module))
    
    def register_moe_hooks(self):
        """Register hooks to capture MoE routing and expert usage"""
        def create_moe_hook(layer_name):
            def hook_fn(module, input, output):
                # Analyze MoE layer outputs and routing
                self._analyze_moe_layer(layer_name, module, input, output)
            return hook_fn
        
        for layer_name, module in self.moe_layers:
            hook = module.register_forward_hook(create_moe_hook(layer_name))
            self.hooks.append(hook)
    
    def _analyze_moe_layer(self, layer_name: str, module: nn.Module, input_tensors, output_tensors):
        """Analyze MoE layer for expert utilization and routing patterns"""
        
        # Try to extract routing information from the module
        routing_weights = None
        expert_outputs = None
        
        # Common patterns for extracting MoE information
        if hasattr(module, 'router_logits'):
            routing_weights = module.router_logits
        elif hasattr(module, 'gate'):
            routing_weights = module.gate
        elif hasattr(module, 'routing_weights'):
            routing_weights = module.routing_weights
        
        # Extract expert information
        if hasattr(module, 'experts'):
            num_experts = len(module.experts) if hasattr(module.experts, '__len__') else getattr(module.experts, 'num_experts', None)
        else:
            num_experts = getattr(module, 'num_experts', None)
        
        # Analyze routing if available
        if routing_weights is not None:
            routing_analysis = self._analyze_routing_weights(routing_weights, num_experts)
            self.routing_stats[layer_name].append(routing_analysis)
        
        # Analyze load balancing
        if routing_weights is not None and num_experts is not None:
            load_balance_analysis = self._analyze_load_balancing(routing_weights, num_experts)
            self.load_balancing_stats[layer_name].append(load_balance_analysis)
        
        # General MoE statistics
        moe_stats = {
            'layer_name': layer_name,
            'num_experts': num_experts,
            'has_routing_weights': routing_weights is not None,
            'input_shape': input_tensors[0].shape if isinstance(input_tensors, tuple) else input_tensors.shape,
            'output_shape': output_tensors[0].shape if isinstance(output_tensors, tuple) else output_tensors.shape
        }
        
        self.expert_stats[layer_name].append(moe_stats)
    
    def _analyze_routing_weights(self, routing_weights: torch.Tensor, num_experts: int) -> Dict[str, Any]:
        """Analyze routing weight distributions"""
        
        # Convert to CPU numpy for analysis
        routing_weights = routing_weights.detach().cpu().numpy()
        
        if len(routing_weights.shape) == 3:  # [batch, seq_len, num_experts]
            batch_size, seq_len, n_experts = routing_weights.shape
            
            # Apply softmax to get probabilities if not already applied
            routing_probs = self._safe_softmax(routing_weights)
            
            # Compute per-expert utilization
            expert_utilization = np.mean(routing_probs, axis=(0, 1))  # Average over batch and sequence
            
            # Compute routing entropy (how evenly distributed the routing is)
            routing_entropy = []
            for b in range(batch_size):
                for s in range(seq_len):
                    probs = routing_probs[b, s, :]
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    routing_entropy.append(entropy)
            
            # Top-k analysis (which experts are most commonly selected)
            top_expert_indices = np.argmax(routing_probs, axis=2)  # [batch, seq_len]
            expert_selection_counts = np.bincount(top_expert_indices.flatten(), minlength=n_experts)
            expert_selection_freq = expert_selection_counts / np.sum(expert_selection_counts)
            
            return {
                'expert_utilization': expert_utilization.tolist(),
                'avg_routing_entropy': float(np.mean(routing_entropy)),
                'std_routing_entropy': float(np.std(routing_entropy)),
                'expert_selection_frequency': expert_selection_freq.tolist(),
                'most_used_expert': int(np.argmax(expert_selection_freq)),
                'least_used_expert': int(np.argmin(expert_selection_freq)),
                'utilization_variance': float(np.var(expert_utilization)),
                'max_routing_weight': float(np.max(routing_weights)),
                'min_routing_weight': float(np.min(routing_weights))
            }
        
        return {}
    
    def _analyze_load_balancing(self, routing_weights: torch.Tensor, num_experts: int) -> Dict[str, Any]:
        """Analyze load balancing across experts"""
        
        routing_weights = routing_weights.detach().cpu().numpy()
        routing_probs = self._safe_softmax(routing_weights)
        
        if len(routing_probs.shape) == 3:
            batch_size, seq_len, n_experts = routing_probs.shape
            
            # Compute load balancing loss (auxiliary loss used in MoE training)
            expert_assignment_freq = np.mean(routing_probs, axis=(0, 1))  # P_i in the literature
            expert_routing_prob = np.mean(np.argmax(routing_probs, axis=2) == np.arange(n_experts)[:, None, None], axis=(1, 2))  # D_i
            
            load_balance_loss = n_experts * np.sum(expert_assignment_freq * expert_routing_prob)
            
            # Compute coefficient of variation for load balancing
            cv_assignment = np.std(expert_assignment_freq) / (np.mean(expert_assignment_freq) + 1e-10)
            cv_routing = np.std(expert_routing_prob) / (np.mean(expert_routing_prob) + 1e-10)
            
            # Compute Gini coefficient for inequality measure
            gini_assignment = self._compute_gini_coefficient(expert_assignment_freq)
            gini_routing = self._compute_gini_coefficient(expert_routing_prob)
            
            return {
                'load_balance_loss': float(load_balance_loss),
                'assignment_cv': float(cv_assignment),
                'routing_cv': float(cv_routing),
                'assignment_gini': float(gini_assignment),
                'routing_gini': float(gini_routing),
                'expert_assignment_freq': expert_assignment_freq.tolist(),
                'expert_routing_prob': expert_routing_prob.tolist(),
                'perfect_balance_score': 1.0 / n_experts,  # What perfect balance would look like
                'balance_deviation': float(np.std(expert_assignment_freq))
            }
        
        return {}
    
    def _safe_softmax(self, logits: np.ndarray) -> np.ndarray:
        """Apply softmax safely to avoid overflow"""
        # Subtract max for numerical stability
        shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return softmax_probs
    
    def _compute_gini_coefficient(self, values: np.ndarray) -> float:
        """Compute Gini coefficient for inequality measurement"""
        values = np.sort(values)
        n = len(values)
        if n == 0 or np.sum(values) == 0:
            return 0.0
        
        cumsum = np.cumsum(values)
        gini = (2 * np.sum((np.arange(1, n + 1)) * values)) / (n * cumsum[-1]) - (n + 1) / n
        return float(gini)
    
    def analyze_expert_specialization(self) -> Dict[str, Any]:
        """Analyze how different experts specialize for different types of input"""
        
        specialization_results = {}
        
        for layer_name, routing_stats_list in self.routing_stats.items():
            if not routing_stats_list:
                continue
            
            # Aggregate routing statistics across batches
            all_utilizations = []
            all_entropies = []
            
            for stats in routing_stats_list:
                if 'expert_utilization' in stats:
                    all_utilizations.append(stats['expert_utilization'])
                if 'avg_routing_entropy' in stats:
                    all_entropies.append(stats['avg_routing_entropy'])
            
            if all_utilizations:
                utilizations = np.array(all_utilizations)
                
                # Compute specialization metrics
                expert_consistency = np.std(utilizations, axis=0)  # How consistent each expert's usage is
                avg_utilization = np.mean(utilizations, axis=0)
                specialization_index = expert_consistency / (avg_utilization + 1e-10)
                
                specialization_results[layer_name] = {
                    'avg_expert_utilization': avg_utilization.tolist(),
                    'expert_consistency': expert_consistency.tolist(),
                    'specialization_index': specialization_index.tolist(),
                    'most_specialized_expert': int(np.argmax(specialization_index)),
                    'least_specialized_expert': int(np.argmin(specialization_index)),
                    'avg_routing_entropy': float(np.mean(all_entropies)) if all_entropies else None
                }
        
        return specialization_results
    
    def detect_expert_collapse(self) -> Dict[str, Any]:
        """Detect if some experts are being underutilized (expert collapse)"""
        
        collapse_results = {}
        
        for layer_name, routing_stats_list in self.routing_stats.items():
            if not routing_stats_list:
                continue
            
            latest_stats = routing_stats_list[-1]
            if 'expert_selection_frequency' in latest_stats:
                selection_freq = np.array(latest_stats['expert_selection_frequency'])
                
                # Define thresholds for expert collapse
                very_low_usage_threshold = 0.01  # 1% usage
                low_usage_threshold = 0.05       # 5% usage
                
                very_low_usage_experts = np.sum(selection_freq < very_low_usage_threshold)
                low_usage_experts = np.sum(selection_freq < low_usage_threshold)
                
                # Compute effective number of experts (based on Shannon entropy)
                entropy = -np.sum(selection_freq * np.log(selection_freq + 1e-10))
                effective_experts = np.exp(entropy)
                
                collapse_results[layer_name] = {
                    'total_experts': len(selection_freq),
                    'very_low_usage_experts': int(very_low_usage_experts),
                    'low_usage_experts': int(low_usage_experts),
                    'effective_experts': float(effective_experts),
                    'collapse_severity': float(very_low_usage_experts / len(selection_freq)),
                    'usage_distribution': selection_freq.tolist(),
                    'max_usage': float(np.max(selection_freq)),
                    'min_usage': float(np.min(selection_freq))
                }
        
        return collapse_results
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class MultiHeadMoEAnalyzer:
    """Analyzer for Multi-Head MoE patterns (MH-MoE style)"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.multi_head_moe_stats = defaultdict(list)
        
    def analyze_multi_head_routing(self, layer_outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Analyze multi-head routing patterns"""
        
        routing_results = {}
        
        for layer_name, output in layer_outputs.items():
            if 'moe' in layer_name.lower() or 'expert' in layer_name.lower():
                # Analyze multi-head patterns if present
                if hasattr(output, 'shape') and len(output.shape) >= 3:
                    # Assume format includes head dimension
                    analysis = self._analyze_head_expert_interaction(output)
                    routing_results[layer_name] = analysis
        
        return routing_results
    
    def _analyze_head_expert_interaction(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Analyze interaction between heads and experts"""
        
        # This is a placeholder for more sophisticated analysis
        # Would need actual multi-head MoE architecture to implement fully
        
        tensor_np = tensor.detach().cpu().numpy()
        
        return {
            'tensor_shape': tensor_np.shape,
            'mean_activation': float(np.mean(tensor_np)),
            'std_activation': float(np.std(tensor_np)),
            'max_activation': float(np.max(tensor_np))
        }


def create_expert_utilization_plot(utilization_data: Dict[str, List[float]], 
                                 layer_name: str, save_path: Optional[str] = None) -> plt.Figure:
    """Create visualization of expert utilization patterns"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot of expert utilization
    experts = list(range(len(utilization_data['expert_utilization'])))
    utilizations = utilization_data['expert_utilization']
    
    ax1.bar(experts, utilizations)
    ax1.set_xlabel('Expert Index')
    ax1.set_ylabel('Utilization Frequency')
    ax1.set_title(f'Expert Utilization - {layer_name}')
    ax1.grid(True, alpha=0.3)
    
    # Heatmap of expert selection over time (if available)
    if 'expert_selection_frequency' in utilization_data:
        selection_freq = np.array(utilization_data['expert_selection_frequency']).reshape(1, -1)
        sns.heatmap(selection_freq, ax=ax2, cmap='Blues', cbar=True)
        ax2.set_xlabel('Expert Index')
        ax2.set_title(f'Expert Selection Frequency - {layer_name}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_load_balancing_plot(load_balance_data: Dict[str, Any], 
                              layer_name: str, save_path: Optional[str] = None) -> plt.Figure:
    """Create visualization of load balancing across experts"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Expert assignment frequency
    assignment_freq = load_balance_data['expert_assignment_freq']
    experts = list(range(len(assignment_freq)))
    
    ax1.bar(experts, assignment_freq)
    ax1.axhline(y=load_balance_data['perfect_balance_score'], color='r', linestyle='--', 
                label='Perfect Balance')
    ax1.set_xlabel('Expert Index')
    ax1.set_ylabel('Assignment Frequency')
    ax1.set_title('Expert Assignment Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Expert routing probability
    routing_prob = load_balance_data['expert_routing_prob']
    ax2.bar(experts, routing_prob)
    ax2.axhline(y=load_balance_data['perfect_balance_score'], color='r', linestyle='--',
                label='Perfect Balance')
    ax2.set_xlabel('Expert Index')
    ax2.set_ylabel('Routing Probability')
    ax2.set_title('Expert Routing Probability')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Load balance metrics
    metrics = ['load_balance_loss', 'assignment_cv', 'routing_cv', 'assignment_gini']
    values = [load_balance_data[metric] for metric in metrics]
    
    ax3.bar(metrics, values)
    ax3.set_ylabel('Metric Value')
    ax3.set_title('Load Balancing Metrics')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Distribution comparison
    ax4.hist(assignment_freq, alpha=0.5, label='Assignment Freq', bins=10)
    ax4.hist(routing_prob, alpha=0.5, label='Routing Prob', bins=10)
    ax4.set_xlabel('Frequency/Probability')
    ax4.set_ylabel('Count')
    ax4.set_title('Distribution of Expert Usage')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Load Balancing Analysis - {layer_name}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig