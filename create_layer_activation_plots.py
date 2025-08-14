#!/usr/bin/env python3
"""
Layer-by-Layer Activation Analysis Visualizations
Creates beautiful, professional plots for activation analysis across transformer layers.
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
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 8
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LayerActivationVisualizer:
    """Generate professional layer-by-layer activation analysis plots."""
    
    def __init__(self, results_dir: str = "analysis_results", plots_dir: str = "layer_activation_plots"):
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
                else:
                    logger.warning(f"No activation_stats found for {model_name}")
                    activation_data[model_name] = []
        
        # Also try to load from individual model result files
        for json_file in self.results_dir.glob("*_results.json"):
            if json_file.name != "final_results.json":
                model_name = json_file.stem.replace('_results', '')
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    if 'activation_stats' in data:
                        activation_data[model_name] = data['activation_stats']
                except Exception as e:
                    logger.warning(f"Could not load activation data from {json_file}: {e}")
        
        return activation_data
    
    def extract_layer_number(self, layer_name: str) -> int:
        """Extract layer number from layer name."""
        # Look for patterns like "layers.0", "layer.0", "h.0", "blocks.0"
        patterns = [
            r'layers?\.(\d+)',
            r'h\.(\d+)', 
            r'blocks?\.(\d+)',
            r'transformer\.h\.(\d+)',
            r'model\.layers\.(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, layer_name.lower())
            if match:
                return int(match.group(1))
        
        # If no pattern found, try to extract any number
        numbers = re.findall(r'\d+', layer_name)
        if numbers:
            return int(numbers[0])
        
        return 0  # Default to 0 if no number found
    
    def organize_activation_data_by_layer(self, model_name: str) -> Dict[str, Dict[int, List[Dict]]]:
        """Organize activation data by layer type and layer number."""
        if model_name not in self.activation_data:
            return {}
        
        organized_data = {
            'mlp_down': defaultdict(list),
            'attention_output': defaultdict(list),
            'attention': defaultdict(list),
            'mlp': defaultdict(list)
        }
        
        for activation_stat in self.activation_data[model_name]:
            layer_type = activation_stat.get('layer_type', 'unknown')
            layer_name = activation_stat.get('layer_name', '')
            layer_num = self.extract_layer_number(layer_name)
            
            if layer_type in organized_data:
                organized_data[layer_type][layer_num].append(activation_stat)
        
        return organized_data
    
    def create_mlp_down_activation_plots(self) -> str:
        """Create plots showing MLP down projection activations for each transformer block."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('MLP Down Projection Activations by Transformer Layer', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']  # Professional color palette
        models_plotted = []
        
        for model_idx, (model_name, activation_data) in enumerate(self.activation_data.items()):
            if model_idx >= 2:  # Limit to 2 models for clarity
                break
                
            organized_data = self.organize_activation_data_by_layer(model_name)
            mlp_down_data = organized_data.get('mlp_down', {})
            
            if not mlp_down_data:
                logger.warning(f"No MLP down projection data found for {model_name}")
                continue
            
            models_plotted.append(model_name)
            color = colors[model_idx]
            
            # Aggregate statistics by layer
            layer_nums = sorted(mlp_down_data.keys())
            q95_values = []
            max_values = []
            mean_values = []
            std_values = []
            
            for layer_num in layer_nums:
                layer_stats = mlp_down_data[layer_num]
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
            
            # Plot Q95 percentiles
            ax1 = axes[0, 0]
            ax1.plot(layer_nums, q95_values, marker='o', color=color, 
                    label=f'{model_name.replace("_", " ").title()}', linewidth=2.5)
            ax1.set_title('Q95 Percentile of MLP Down Projection Activations', fontweight='bold', fontsize=14)
            ax1.set_xlabel('Transformer Layer')
            ax1.set_ylabel('Q95 Activation Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot Max activations
            ax2 = axes[0, 1]
            ax2.plot(layer_nums, max_values, marker='s', color=color, 
                    label=f'{model_name.replace("_", " ").title()}', linewidth=2.5)
            ax2.set_title('Maximum MLP Down Projection Activations', fontweight='bold', fontsize=14)
            ax2.set_xlabel('Transformer Layer')
            ax2.set_ylabel('Max Activation Value')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot Mean activations
            ax3 = axes[1, 0]
            ax3.plot(layer_nums, mean_values, marker='^', color=color, 
                    label=f'{model_name.replace("_", " ").title()}', linewidth=2.5)
            ax3.set_title('Mean Absolute MLP Down Projection Activations', fontweight='bold', fontsize=14)
            ax3.set_xlabel('Transformer Layer')
            ax3.set_ylabel('Mean |Activation|')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot Standard deviation
            ax4 = axes[1, 1]
            ax4.plot(layer_nums, std_values, marker='d', color=color, 
                    label=f'{model_name.replace("_", " ").title()}', linewidth=2.5)
            ax4.set_title('Standard Deviation of MLP Down Projection Activations', fontweight='bold', fontsize=14)
            ax4.set_xlabel('Transformer Layer')
            ax4.set_ylabel('Activation Std Dev')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.plots_dir / "mlp_down_projection_activations_by_layer.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Created MLP down projection activation plot: {plot_path}")
        return str(plot_path)
    
    def create_attention_output_activation_plots(self) -> str:
        """Create plots showing attention output activations (after o projection) for each transformer block."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Attention Output (O Projection) Activations by Transformer Layer', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        models_plotted = []
        
        for model_idx, (model_name, activation_data) in enumerate(self.activation_data.items()):
            if model_idx >= 2:  # Limit to 2 models for clarity
                break
                
            organized_data = self.organize_activation_data_by_layer(model_name)
            attention_output_data = organized_data.get('attention_output', {})
            
            if not attention_output_data:
                logger.warning(f"No attention output data found for {model_name}")
                continue
            
            models_plotted.append(model_name)
            color = colors[model_idx]
            
            # Aggregate statistics by layer
            layer_nums = sorted(attention_output_data.keys())
            q95_values = []
            max_values = []
            mean_values = []
            std_values = []
            
            for layer_num in layer_nums:
                layer_stats = attention_output_data[layer_num]
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
            
            # Plot Q95 percentiles
            ax1 = axes[0, 0]
            ax1.plot(layer_nums, q95_values, marker='o', color=color, 
                    label=f'{model_name.replace("_", " ").title()}', linewidth=2.5)
            ax1.set_title('Q95 Percentile of Attention Output Activations', fontweight='bold', fontsize=14)
            ax1.set_xlabel('Transformer Layer')
            ax1.set_ylabel('Q95 Activation Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot Max activations
            ax2 = axes[0, 1]
            ax2.plot(layer_nums, max_values, marker='s', color=color, 
                    label=f'{model_name.replace("_", " ").title()}', linewidth=2.5)
            ax2.set_title('Maximum Attention Output Activations', fontweight='bold', fontsize=14)
            ax2.set_xlabel('Transformer Layer')
            ax2.set_ylabel('Max Activation Value')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot Mean activations
            ax3 = axes[1, 0]
            ax3.plot(layer_nums, mean_values, marker='^', color=color, 
                    label=f'{model_name.replace("_", " ").title()}', linewidth=2.5)
            ax3.set_title('Mean Absolute Attention Output Activations', fontweight='bold', fontsize=14)
            ax3.set_xlabel('Transformer Layer')
            ax3.set_ylabel('Mean |Activation|')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot Standard deviation
            ax4 = axes[1, 1]
            ax4.plot(layer_nums, std_values, marker='d', color=color, 
                    label=f'{model_name.replace("_", " ").title()}', linewidth=2.5)
            ax4.set_title('Standard Deviation of Attention Output Activations', fontweight='bold', fontsize=14)
            ax4.set_xlabel('Transformer Layer')
            ax4.set_ylabel('Activation Std Dev')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.plots_dir / "attention_output_activations_by_layer.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Created attention output activation plot: {plot_path}")
        return str(plot_path)
    
    def create_combined_layer_analysis_plot(self) -> str:
        """Create a comprehensive combined plot showing both MLP and attention activations."""
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        fig.suptitle('Comprehensive Layer-by-Layer Activation Analysis', 
                    fontsize=22, fontweight='bold', y=0.96)
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        markers = ['o', 's', '^', 'd']
        
        for model_idx, (model_name, activation_data) in enumerate(self.activation_data.items()):
            if model_idx >= 2:  # Limit to 2 models
                break
                
            organized_data = self.organize_activation_data_by_layer(model_name)
            color = colors[model_idx]
            marker = markers[model_idx]
            clean_name = model_name.replace("_", " ").title()
            
            # Process MLP down data
            mlp_down_data = organized_data.get('mlp_down', {})
            if mlp_down_data:
                layer_nums = sorted(mlp_down_data.keys())
                q95_mlp = [np.mean([stat.get('q95', 0) for stat in mlp_down_data[ln]]) 
                          for ln in layer_nums]
                max_mlp = [np.mean([stat.get('abs_max', 0) for stat in mlp_down_data[ln]]) 
                          for ln in layer_nums]
                
                # MLP Q95 plot
                axes[0, 0].plot(layer_nums, q95_mlp, marker=marker, color=color, 
                               label=clean_name, linewidth=2.5, markersize=6)
                
                # MLP Max plot
                axes[0, 1].plot(layer_nums, max_mlp, marker=marker, color=color, 
                               label=clean_name, linewidth=2.5, markersize=6)
            
            # Process attention output data
            attention_output_data = organized_data.get('attention_output', {})
            if attention_output_data:
                layer_nums = sorted(attention_output_data.keys())
                q95_att = [np.mean([stat.get('q95', 0) for stat in attention_output_data[ln]]) 
                          for ln in layer_nums]
                max_att = [np.mean([stat.get('abs_max', 0) for stat in attention_output_data[ln]]) 
                          for ln in layer_nums]
                
                # Attention Q95 plot
                axes[1, 0].plot(layer_nums, q95_att, marker=marker, color=color, 
                               label=clean_name, linewidth=2.5, markersize=6)
                
                # Attention Max plot
                axes[1, 1].plot(layer_nums, max_att, marker=marker, color=color, 
                               label=clean_name, linewidth=2.5, markersize=6)
                
                # Combined comparison - Q95
                axes[2, 0].plot(layer_nums, q95_mlp, marker='o', color=color, 
                               label=f'{clean_name} MLP', linewidth=2.5, markersize=6, linestyle='-')
                axes[2, 0].plot(layer_nums, q95_att, marker='s', color=color, 
                               label=f'{clean_name} Attention', linewidth=2.5, markersize=6, linestyle='--')
                
                # Combined comparison - Max
                axes[2, 1].plot(layer_nums, max_mlp, marker='o', color=color, 
                               label=f'{clean_name} MLP', linewidth=2.5, markersize=6, linestyle='-')
                axes[2, 1].plot(layer_nums, max_att, marker='s', color=color, 
                               label=f'{clean_name} Attention', linewidth=2.5, markersize=6, linestyle='--')
        
        # Configure subplots
        axes[0, 0].set_title('MLP Down Projection - Q95 Percentile', fontweight='bold', fontsize=14)
        axes[0, 0].set_xlabel('Transformer Layer')
        axes[0, 0].set_ylabel('Q95 Activation Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('MLP Down Projection - Maximum', fontweight='bold', fontsize=14)
        axes[0, 1].set_xlabel('Transformer Layer')
        axes[0, 1].set_ylabel('Max Activation Value')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Attention Output - Q95 Percentile', fontweight='bold', fontsize=14)
        axes[1, 0].set_xlabel('Transformer Layer')
        axes[1, 0].set_ylabel('Q95 Activation Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Attention Output - Maximum', fontweight='bold', fontsize=14)
        axes[1, 1].set_xlabel('Transformer Layer')
        axes[1, 1].set_ylabel('Max Activation Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[2, 0].set_title('Combined Comparison - Q95 Percentile', fontweight='bold', fontsize=14)
        axes[2, 0].set_xlabel('Transformer Layer')
        axes[2, 0].set_ylabel('Q95 Activation Value')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        axes[2, 1].set_title('Combined Comparison - Maximum', fontweight='bold', fontsize=14)
        axes[2, 1].set_xlabel('Transformer Layer')
        axes[2, 1].set_ylabel('Max Activation Value')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.plots_dir / "comprehensive_layer_activation_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Created comprehensive layer activation analysis plot: {plot_path}")
        return str(plot_path)
    
    def generate_all_layer_plots(self) -> List[str]:
        """Generate all layer-by-layer activation plots."""
        logger.info("Generating layer-by-layer activation plots...")
        
        if not self.activation_data:
            logger.error("No activation data found! Please run the analysis first.")
            return []
        
        generated_plots = []
        
        try:
            # Generate MLP down projection plots
            mlp_plot = self.create_mlp_down_activation_plots()
            if mlp_plot:
                generated_plots.append(mlp_plot)
            
            # Generate attention output plots
            attention_plot = self.create_attention_output_activation_plots()
            if attention_plot:
                generated_plots.append(attention_plot)
            
            # Generate combined analysis plot
            combined_plot = self.create_combined_layer_analysis_plot()
            if combined_plot:
                generated_plots.append(combined_plot)
            
        except Exception as e:
            logger.error(f"Error generating layer plots: {e}")
            raise
        
        logger.info(f"Generated {len(generated_plots)} layer activation plots")
        return generated_plots
    
    def upload_to_wandb(self, project_name: str = "layer-activation-analysis", 
                       run_name: str = "layer-plots") -> None:
        """Upload all generated plots to WandB."""
        logger.info("Uploading layer activation plots to WandB...")
        
        # Generate all plots first
        generated_plots = self.generate_all_layer_plots()
        
        if not generated_plots:
            logger.error("No plots to upload!")
            return
        
        # Initialize WandB
        wandb.init(project=project_name, name=run_name, job_type="layer-visualization")
        
        try:
            # Upload plots as images
            for plot_path in generated_plots:
                plot_name = Path(plot_path).stem
                wandb.log({f"layer_plots/{plot_name}": wandb.Image(plot_path)})
            
            logger.info(f"Successfully uploaded {len(generated_plots)} plots to WandB")
            
        except Exception as e:
            logger.error(f"Error uploading to WandB: {e}")
        finally:
            wandb.finish()

def main():
    """Main function to generate layer activation visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate layer-by-layer activation analysis plots")
    parser.add_argument("--results_dir", default="analysis_results", help="Directory containing analysis results")
    parser.add_argument("--plots_dir", default="layer_activation_plots", help="Directory to save plots")
    parser.add_argument("--wandb_project", default="layer-activation-analysis", help="WandB project name")
    parser.add_argument("--wandb_run", default="layer-plots", help="WandB run name")
    parser.add_argument("--upload_wandb", action="store_true", help="Upload to WandB")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = LayerActivationVisualizer(args.results_dir, args.plots_dir)
    
    # Generate all plots
    generated_plots = visualizer.generate_all_layer_plots()
    
    print(f"\nGenerated {len(generated_plots)} layer activation plots:")
    for plot_path in generated_plots:
        print(f"  - {plot_path}")
    
    # Upload to WandB if requested
    if args.upload_wandb:
        visualizer.upload_to_wandb(args.wandb_project, args.wandb_run)
        print(f"\nUploaded plots to WandB project: {args.wandb_project}")
    
    print("\nLayer activation visualization generation complete!")

if __name__ == "__main__":
    main()