#!/usr/bin/env python3
"""
Professional Attention Layer Analysis Visualizations
Creates beautiful, clean plots for attention layer activation analysis.
"""

import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import wandb
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict
import re

# Set up beautiful, professional plotting style
plt.style.use('default')  # Use default to avoid seaborn issues
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttentionLayerVisualizer:
    """Generate professional attention layer activation analysis plots."""
    
    def __init__(self, results_dir: str = "analysis_results", plots_dir: str = "unified_analysis_plots/layer_analysis"):
        self.results_dir = Path(results_dir)
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(exist_ok=True)
        
        # Load activation data from analysis results
        self.activation_data = self.load_activation_data()
        
    def load_activation_data(self) -> Dict[str, List[Dict]]:
        """Load activation statistics from analysis results."""
        activation_data = {}
        
        # Load final results which should contain activation stats
        final_results_path = self.results_dir / "final_results.json"
        if final_results_path.exists():
            with open(final_results_path, 'r') as f:
                results = json.load(f)
                
            for model_name, model_data in results.items():
                if 'activation_stats' in model_data:
                    activation_data[model_name] = model_data['activation_stats']
                    logger.info(f"Loaded {len(model_data['activation_stats'])} activation stats for {model_name}")
                else:
                    logger.warning(f"No activation_stats found for {model_name}")
                    activation_data[model_name] = []
        
        return activation_data
    
    def extract_layer_number(self, layer_name: str) -> int:
        """Extract layer number from layer name."""
        # Look for patterns like "layers.0", "layer.0", "h.0", "blocks.0"
        patterns = [
            r'layers\.(\d+)',
            r'layer\.(\d+)', 
            r'h\.(\d+)', 
            r'blocks\.(\d+)',
            r'transformer\.h\.(\d+)',
            r'model\.layers\.(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, layer_name)
            if match:
                return int(match.group(1))
        
        # If no pattern found, try to extract any number
        numbers = re.findall(r'\d+', layer_name)
        if numbers:
            return int(numbers[0])
        
        return 0  # Default to 0 if no number found
    
    def organize_attention_data_by_layer(self, model_name: str) -> Dict[int, List[Dict]]:
        """Organize attention activation data by layer number."""
        if model_name not in self.activation_data:
            return {}
        
        organized_data = defaultdict(list)
        
        for activation_stat in self.activation_data[model_name]:
            layer_type = str(activation_stat.get('layer_type', 'unknown')).lower()
            layer_name = activation_stat.get('layer_name', '')
            name_lower = layer_name.lower()

            # Accept a broad set of attention output signals
            is_attention_like = (
                (layer_type in {"attention_output", "attention", "attn", "attn_output"}) or
                ("attention" in name_lower and ("dense" in name_lower or "o_proj" in name_lower or "out_proj" in name_lower or "wo" in name_lower))
            )

            if is_attention_like:
                layer_num = self.extract_layer_number(layer_name)
                organized_data[layer_num].append(activation_stat)
        
        return organized_data

    def organize_mlp_down_data_by_layer(self, model_name: str) -> Dict[int, List[Dict]]:
        """Organize MLP down-projection activation data by layer number."""
        if model_name not in self.activation_data:
            return {}

        organized_data = defaultdict(list)

        for activation_stat in self.activation_data[model_name]:
            layer_type = str(activation_stat.get('layer_type', 'unknown')).lower()
            layer_name = activation_stat.get('layer_name', '')
            name_lower = layer_name.lower()

            is_mlp_down_like = (
                (layer_type in {"mlp_down", "mlp", "ffn"}) and ("down" in name_lower or "proj" in name_lower) or
                ("dense_4h_to_h" in name_lower) or  # GPT-NeoX down projection
                ("down_proj" in name_lower)
            )

            if is_mlp_down_like:
                layer_num = self.extract_layer_number(layer_name)
                organized_data[layer_num].append(activation_stat)

        return organized_data
    
    def create_attention_q95_comparison_plot(self) -> str:
        """Create Q95 percentile comparison plot for attention outputs across layers."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Attention Output Activations - Q95 Percentile Analysis by Transformer Layer', 
                    fontsize=24, fontweight='bold', y=1.02)
        plt.subplots_adjust(top=0.90)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Professional blue, orange, green, red
        markers = ['o', 's', '^', 'D']
        
        model_data = {}
        # Prefer plotting DCFormer and Pythia if present, otherwise first two
        preferred_order = []
        if 'dcformer_2_8b' in self.activation_data:
            preferred_order.append('dcformer_2_8b')
        if 'pythia_2_8b_baseline' in self.activation_data:
            preferred_order.append('pythia_2_8b_baseline')
        # Fill remaining with any others not already included
        for k in self.activation_data.keys():
            if k not in preferred_order:
                preferred_order.append(k)

        for model_idx, model_name in enumerate(preferred_order[:2]):
            activation_data = self.activation_data[model_name]
                
            organized_data = self.organize_attention_data_by_layer(model_name)
            
            if not organized_data:
                logger.warning(f"No attention output data found for {model_name}")
                continue
            
            color = colors[model_idx]
            marker = markers[model_idx]
            # Pretty names
            pretty_map = {
                'dcformer_2_8b': 'DCFormer-2.8B',
                'pythia_2_8b_baseline': 'Pythia-2.8B'
            }
            clean_name = pretty_map.get(model_name, model_name.replace("_", " ").title())
            
            # Aggregate statistics by layer
            layer_nums = sorted(organized_data.keys())
            q95_values = []
            max_values = []
            mean_values = []
            std_values = []
            
            for layer_num in layer_nums:
                layer_stats = organized_data[layer_num]
                if layer_stats:
                    # Average across batches for this layer
                    q95_avg = np.mean([stat.get('q95', 0) for stat in layer_stats])
                    max_avg = np.mean([stat.get('abs_max', 0) for stat in layer_stats])
                    mean_avg = np.mean([stat.get('abs_mean', 0) for stat in layer_stats])
                    std_avg = np.mean([stat.get('std', 0) for stat in layer_stats])
                    
                    q95_values.append(q95_avg)
                    max_values.append(max_avg)
                    mean_values.append(mean_avg)
                    std_values.append(std_avg)
                else:
                    q95_values.append(0)
                    max_values.append(0)
                    mean_values.append(0)
                    std_values.append(0)
            
            model_data[clean_name] = {
                'layer_nums': layer_nums,
                'q95_values': q95_values,
                'max_values': max_values,
                'mean_values': mean_values,
                'std_values': std_values,
                'color': color,
                'marker': marker
            }
        
        # Plot Q95 percentiles
        ax1 = axes[0, 0]
        for model_name, data in model_data.items():
            ax1.plot(data['layer_nums'], data['q95_values'], 
                    marker=data['marker'], color=data['color'], 
                    label=model_name, linewidth=3, markersize=8)
        ax1.set_title('Q95 Percentile of Attention Output Activations', fontweight='bold', fontsize=16)
        ax1.set_xlabel('Transformer Layer', fontsize=14)
        ax1.set_ylabel('Q95 Activation Value', fontsize=14)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot Max activations
        ax2 = axes[0, 1]
        for model_name, data in model_data.items():
            ax2.plot(data['layer_nums'], data['max_values'], 
                    marker=data['marker'], color=data['color'], 
                    label=model_name, linewidth=3, markersize=8)
        ax2.set_title('Maximum Attention Output Activations', fontweight='bold', fontsize=16)
        ax2.set_xlabel('Transformer Layer', fontsize=14)
        ax2.set_ylabel('Max Activation Value', fontsize=14)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Plot Mean activations
        ax3 = axes[1, 0]
        for model_name, data in model_data.items():
            ax3.plot(data['layer_nums'], data['mean_values'], 
                    marker=data['marker'], color=data['color'], 
                    label=model_name, linewidth=3, markersize=8)
        ax3.set_title('Mean Absolute Attention Output Activations', fontweight='bold', fontsize=16)
        ax3.set_xlabel('Transformer Layer', fontsize=14)
        ax3.set_ylabel('Mean |Activation|', fontsize=14)
        ax3.legend(fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Plot Standard deviation
        ax4 = axes[1, 1]
        for model_name, data in model_data.items():
            ax4.plot(data['layer_nums'], data['std_values'], 
                    marker=data['marker'], color=data['color'], 
                    label=model_name, linewidth=3, markersize=8)
        ax4.set_title('Standard Deviation of Attention Output Activations', fontweight='bold', fontsize=16)
        ax4.set_xlabel('Transformer Layer', fontsize=14)
        ax4.set_ylabel('Activation Std Dev', fontsize=14)
        ax4.legend(fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.plots_dir / "attention_output_activations_by_layer.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Created attention output activation plot: {plot_path}")
        return str(plot_path)
    
    def create_side_by_side_comparison_plot(self) -> str:
        """Create a side-by-side comparison plot focusing on Q95 and Max values."""
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))
        fig.suptitle('DCFormer vs Pythia: Attention Output Activation Comparison', 
                    fontsize=26, fontweight='bold', y=1.02)
        plt.subplots_adjust(top=0.90)
        
        colors = ['#1f77b4', '#ff7f0e']  # Blue for DCFormer, Orange for Pythia
        markers = ['o', 's']
        
        model_data = {}
        preferred_order = []
        if 'dcformer_2_8b' in self.activation_data:
            preferred_order.append('dcformer_2_8b')
        if 'pythia_2_8b_baseline' in self.activation_data:
            preferred_order.append('pythia_2_8b_baseline')
        for k in self.activation_data.keys():
            if k not in preferred_order:
                preferred_order.append(k)

        for model_idx, model_name in enumerate(preferred_order[:2]):
            activation_data = self.activation_data[model_name]
                
            organized_data = self.organize_attention_data_by_layer(model_name)
            
            if not organized_data:
                continue
            
            color = colors[model_idx]
            marker = markers[model_idx]
            pretty_map = {
                'dcformer_2_8b': 'DCFormer-2.8B',
                'pythia_2_8b_baseline': 'Pythia-2.8B'
            }
            clean_name = pretty_map.get(model_name, model_name.replace("_", " ").title())
            
            # Aggregate statistics by layer
            layer_nums = sorted(organized_data.keys())
            q95_values = []
            max_values = []
            
            for layer_num in layer_nums:
                layer_stats = organized_data[layer_num]
                if layer_stats:
                    q95_avg = np.mean([stat.get('q95', 0) for stat in layer_stats])
                    max_avg = np.mean([stat.get('abs_max', 0) for stat in layer_stats])
                    
                    q95_values.append(q95_avg)
                    max_values.append(max_avg)
                else:
                    q95_values.append(0)
                    max_values.append(0)
            
            model_data[clean_name] = {
                'layer_nums': layer_nums,
                'q95_values': q95_values,
                'max_values': max_values,
                'color': color,
                'marker': marker
            }
        
        # Q95 Comparison
        ax1 = axes[0]
        for model_name, data in model_data.items():
            ax1.plot(data['layer_nums'], data['q95_values'], 
                    marker=data['marker'], color=data['color'], 
                    label=model_name, linewidth=4, markersize=10)
        ax1.set_title('Q95 Percentile - Attention Output Activations', fontweight='bold', fontsize=18)
        ax1.set_xlabel('Transformer Layer', fontsize=16)
        ax1.set_ylabel('Q95 Activation Value', fontsize=16)
        ax1.legend(fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Max Comparison  
        ax2 = axes[1]
        for model_name, data in model_data.items():
            ax2.plot(data['layer_nums'], data['max_values'], 
                    marker=data['marker'], color=data['color'], 
                    label=model_name, linewidth=4, markersize=10)
        ax2.set_title('Maximum - Attention Output Activations', fontweight='bold', fontsize=18)
        ax2.set_xlabel('Transformer Layer', fontsize=16)
        ax2.set_ylabel('Max Activation Value', fontsize=16)
        ax2.legend(fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.plots_dir / "dcformer_vs_pythia_attention_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Created side-by-side comparison plot: {plot_path}")
        return str(plot_path)
    
    def create_detailed_statistics_plot(self) -> str:
        """Create a detailed statistics plot with multiple metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(28, 16))
        fig.suptitle('Comprehensive Attention Output Statistics by Layer', 
                    fontsize=26, fontweight='bold', y=1.02)
        plt.subplots_adjust(top=0.90)
        
        colors = ['#1f77b4', '#ff7f0e']
        markers = ['o', 's']
        
        model_data = {}
        for model_idx, (model_name, activation_data) in enumerate(self.activation_data.items()):
            if model_idx >= 2:
                break
                
            organized_data = self.organize_attention_data_by_layer(model_name)
            
            if not organized_data:
                continue
            
            color = colors[model_idx]
            marker = markers[model_idx]
            pretty_map = {
                'dcformer_2_8b': 'DCFormer-2.8B',
                'pythia_2_8b_baseline': 'Pythia-2.8B'
            }
            clean_name = pretty_map.get(model_name, model_name.replace("_", " ").title())
            
            layer_nums = sorted(organized_data.keys())
            metrics = {
                'q50': [],
                'q95': [],
                'q99': [],
                'abs_max': [],
                'abs_mean': [],
                'std': []
            }
            
            for layer_num in layer_nums:
                layer_stats = organized_data[layer_num]
                if layer_stats:
                    for metric in metrics:
                        avg_val = np.mean([stat.get(metric, 0) for stat in layer_stats])
                        metrics[metric].append(avg_val)
                else:
                    for metric in metrics:
                        metrics[metric].append(0)
            
            model_data[clean_name] = {
                'layer_nums': layer_nums,
                'metrics': metrics,
                'color': color,
                'marker': marker
            }
        
        # Plot configurations
        plot_configs = [
            ('q50', 'Q50 Percentile', axes[0, 0]),
            ('q95', 'Q95 Percentile', axes[0, 1]),
            ('q99', 'Q99 Percentile', axes[0, 2]),
            ('abs_max', 'Maximum Absolute', axes[1, 0]),
            ('abs_mean', 'Mean Absolute', axes[1, 1]),
            ('std', 'Standard Deviation', axes[1, 2])
        ]
        
        for metric, title, ax in plot_configs:
            for model_name, data in model_data.items():
                ax.plot(data['layer_nums'], data['metrics'][metric], 
                       marker=data['marker'], color=data['color'], 
                       label=model_name, linewidth=3, markersize=8)
            ax.set_title(f'{title} of Attention Activations', fontweight='bold', fontsize=16)
            ax.set_xlabel('Transformer Layer', fontsize=14)
            ax.set_ylabel(f'{title} Value', fontsize=14)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.plots_dir / "comprehensive_attention_statistics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Created comprehensive statistics plot: {plot_path}")
        return str(plot_path)

    def create_mlp_down_projection_plot(self) -> Optional[str]:
        """Create MLP down-projection activation comparison plot across layers, if data present."""
        fig, ax = plt.subplots(1, 1, figsize=(20, 8))
        fig.suptitle('MLP Down-Projection Activations by Layer (Q95 and Max)', 
                    fontsize=24, fontweight='bold', y=1.02)
        plt.subplots_adjust(top=0.90)

        colors = ['#1f77b4', '#ff7f0e']
        markers = ['o', 's']

        any_plotted = False

        preferred_order = []
        if 'dcformer_2_8b' in self.activation_data:
            preferred_order.append('dcformer_2_8b')
        if 'pythia_2_8b_baseline' in self.activation_data:
            preferred_order.append('pythia_2_8b_baseline')
        for k in self.activation_data.keys():
            if k not in preferred_order:
                preferred_order.append(k)

        for model_idx, model_name in enumerate(preferred_order[:2]):
            organized_data = self.organize_mlp_down_data_by_layer(model_name)
            if not organized_data:
                logger.warning(f"No MLP down projection data found for {model_name}")
                continue

            any_plotted = True
            layer_nums = sorted(organized_data.keys())
            q95_values = []
            max_values = []

            for layer_num in layer_nums:
                layer_stats = organized_data[layer_num]
                q95_avg = np.mean([stat.get('q95', 0) for stat in layer_stats])
                max_avg = np.mean([stat.get('abs_max', 0) for stat in layer_stats])
                q95_values.append(q95_avg)
                max_values.append(max_avg)

            pretty_map = {
                'dcformer_2_8b': 'DCFormer-2.8B',
                'pythia_2_8b_baseline': 'Pythia-2.8B'
            }
            clean_name = pretty_map.get(model_name, model_name.replace("_", " ").title())
            color = colors[model_idx]
            marker = markers[model_idx]

            ax.plot(layer_nums, q95_values, marker=marker, color=color, linewidth=3, label=f"{clean_name} - Q95")
            ax.plot(layer_nums, max_values, marker=marker, color=color, linewidth=3, linestyle='--', label=f"{clean_name} - Max")

        if not any_plotted:
            plt.close(fig)
            return None

        ax.set_xlabel('Transformer Layer', fontsize=14)
        ax.set_ylabel('Activation Value', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.plots_dir / "mlp_down_projection_by_layer.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Created MLP down-projection plot: {plot_path}")
        return str(plot_path)
    
    def generate_all_plots(self) -> List[str]:
        """Generate all attention layer plots."""
        logger.info("Generating attention layer activation plots...")
        
        if not self.activation_data:
            logger.error("No activation data found! Please run the analysis first.")
            return []
        
        generated_plots = []
        
        try:
            # Generate Q95 comparison plot
            q95_plot = self.create_attention_q95_comparison_plot()
            if q95_plot:
                generated_plots.append(q95_plot)
            
            # Generate side-by-side comparison
            comparison_plot = self.create_side_by_side_comparison_plot()
            if comparison_plot:
                generated_plots.append(comparison_plot)
            
            # Generate detailed statistics plot
            stats_plot = self.create_detailed_statistics_plot()
            if stats_plot:
                generated_plots.append(stats_plot)

            # Generate MLP down projection plot, if data available
            mlp_plot = self.create_mlp_down_projection_plot()
            if mlp_plot:
                generated_plots.append(mlp_plot)
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            raise
        
        logger.info(f"Generated {len(generated_plots)} attention layer plots")
        return generated_plots
    
    def upload_to_wandb(self, project_name: str = "attention-layer-analysis", 
                       run_name: str = "attention-plots") -> None:
        """Upload all generated plots to WandB."""
        logger.info("Uploading attention layer plots to WandB...")
        
        # Generate all plots first
        generated_plots = self.generate_all_plots()
        
        if not generated_plots:
            logger.error("No plots to upload!")
            return
        
        # Initialize WandB
        wandb.init(project=project_name, name=run_name, job_type="attention-visualization")
        
        try:
            # Upload plots as images
            for plot_path in generated_plots:
                plot_name = Path(plot_path).stem
                wandb.log({f"attention_plots/{plot_name}": wandb.Image(plot_path)})
            
            logger.info(f"Successfully uploaded {len(generated_plots)} plots to WandB")
            
        except Exception as e:
            logger.error(f"Error uploading to WandB: {e}")
        finally:
            wandb.finish()

def main():
    """Main function to generate attention layer visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate attention layer activation analysis plots")
    parser.add_argument("--results_dir", default="analysis_results", help="Directory containing analysis results")
    parser.add_argument("--plots_dir", default="attention_layer_plots", help="Directory to save plots")
    parser.add_argument("--wandb_project", default="attention-layer-analysis", help="WandB project name")
    parser.add_argument("--wandb_run", default="attention-plots", help="WandB run name")
    parser.add_argument("--upload_wandb", action="store_true", help="Upload to WandB")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = AttentionLayerVisualizer(args.results_dir, args.plots_dir)
    
    # Generate all plots
    generated_plots = visualizer.generate_all_plots()
    
    print(f"\nGenerated {len(generated_plots)} attention layer plots:")
    for plot_path in generated_plots:
        print(f"  - {plot_path}")
    
    # Upload to WandB if requested
    if args.upload_wandb:
        visualizer.upload_to_wandb(args.wandb_project, args.wandb_run)
        print(f"\nUploaded plots to WandB project: {args.wandb_project}")
    
    print("\nAttention layer visualization generation complete!")

if __name__ == "__main__":
    main()