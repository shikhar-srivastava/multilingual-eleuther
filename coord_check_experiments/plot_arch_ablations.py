#!/usr/bin/env python3
"""
Plot architecture ablation studies for CompleteP.

This script creates beautiful plots comparing different architectural components:
- Baseline: RMSNorm + SwiGLU + RoPE (LLAMA standard)
- LayerNorm only: LayerNorm + SwiGLU + RoPE
- GELU only: RMSNorm + GELU + RoPE
- Learned Pos only: RMSNorm + SwiGLU + Learned
- All GPT-2-like: LayerNorm + GELU + Learned

Each plot clearly indicates which architectural variant is being tested.
"""

import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set matplotlib style for beautiful plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['font.size'] = 10


def load_experiment_data(base_dir, experiment_name):
    """Load all CSV files from an experiment directory."""
    pattern = os.path.join(base_dir, experiment_name, "out", "*", "log.csv")
    files = glob.glob(pattern)
    
    data = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dir_name = os.path.basename(os.path.dirname(f))
            parts = dir_name.split("_")
            
            config = {}
            for part in parts:
                if part.startswith("width"):
                    config["width"] = int(part.replace("width", ""))
                elif part.startswith("seed"):
                    config["seed"] = int(part.replace("seed", ""))
            
            df["experiment"] = experiment_name
            for k, v in config.items():
                df[k] = v
            
            data.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
    
    if data:
        return pd.concat(data, ignore_index=True)
    return None


def plot_arch_comparison(base_dir, output_dir="coord_check_experiments"):
    """Create comparison plots for all architectural ablations."""
    
    experiments = [
        ("width_scaling/completep", "Baseline\n(RMSNorm + SwiGLU + RoPE)", "#1f77b4", "baseline"),
        ("width_scaling/completep_swiglu_fixed", "SwiGLU Fixed\n(RMSNorm + SwiGLU+fix + RoPE)", "#17becf", "swiglu_fixed"),
        ("width_scaling/completep_layernorm", "LayerNorm\n(LayerNorm + SwiGLU + RoPE)", "#ff7f0e", "layernorm"),
        ("width_scaling/completep_gelu", "GELU MLP\n(RMSNorm + GELU + RoPE)", "#2ca02c", "gelu"),
        ("width_scaling/completep_learned_pos", "Learned Pos\n(RMSNorm + SwiGLU + Learned)", "#d62728", "learned_pos"),
        ("width_scaling/completep_gpt2like", "GPT-2-like\n(LayerNorm + GELU + Learned)", "#9467bd", "gpt2like"),
    ]
    
    metrics = [
        ("last_layer_act_abs_mean", "Last Layer Output"),
        ("attn_act_abs_mean", "Attention Output"),
        ("mlp_act_abs_mean", "MLP Output"),
    ]
    
    # Create one large comparison plot
    fig, axes = plt.subplots(len(metrics), len(experiments), figsize=(20, 12))
    if len(metrics) == 1:
        axes = axes.reshape(1, -1)
    
    for col_idx, (exp_name, exp_label, color, short_name) in enumerate(experiments):
        data = load_experiment_data(base_dir, exp_name)
        
        if data is None:
            print(f"No data found for {exp_name} - skipping")
            for row_idx in range(len(metrics)):
                ax = axes[row_idx, col_idx]
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{exp_label}\n(No data)")
            continue
        
        for row_idx, (metric_col, metric_name) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            
            if metric_col not in data.columns:
                ax.text(0.5, 0.5, f"No {metric_col}", ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Plot each width
            for width in sorted(data["width"].unique()):
                width_data = data[data["width"] == width]
                
                # Average over seeds
                avg_data = width_data.groupby("iter")[metric_col].mean()
                std_data = width_data.groupby("iter")[metric_col].std()
                
                steps = avg_data.index
                ax.plot(steps, avg_data, label=f"width={width}", linewidth=2, alpha=0.8)
                ax.fill_between(steps, avg_data - std_data, avg_data + std_data, alpha=0.2)
            
            ax.set_xlabel("Training Step", fontsize=10)
            ax.set_ylabel("|output|.mean()", fontsize=10)
            ax.set_title(f"{exp_label}\n{metric_name}", fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
            
            # Set consistent y-axis limits for comparison
            ax.set_ylim(bottom=0)
    
    plt.suptitle("CompleteP Architecture Ablation Study: Width Coordinate Checks", 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "width_coord_check_arch_ablation.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved architecture ablation plot to {output_path}")
    plt.close()
    
    # Create individual plots for each variant
    for exp_name, exp_label, color, short_name in experiments:
        plot_single_variant(base_dir, exp_name, exp_label, short_name, metrics, output_dir, color)


def plot_single_variant(base_dir, exp_name, exp_label, short_name, metrics, output_dir, color):
    """Create a focused plot for a single architectural variant."""
    
    data = load_experiment_data(base_dir, exp_name)
    if data is None:
        print(f"No data for {exp_name} - skipping individual plot")
        return
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    if len(metrics) == 1:
        axes = [axes]
    
    for idx, (metric_col, metric_name) in enumerate(metrics):
        ax = axes[idx]
        
        if metric_col not in data.columns:
            continue
        
        # Plot each width
        widths = sorted(data["width"].unique())
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(widths)))
        
        for width_idx, width in enumerate(widths):
            width_data = data[data["width"] == width]
            
            # Average over seeds
            avg_data = width_data.groupby("iter")[metric_col].mean()
            std_data = width_data.groupby("iter")[metric_col].std()
            
            steps = avg_data.index
            ax.plot(steps, avg_data, label=f"width={width}", 
                   linewidth=2.5, alpha=0.9, color=colors[width_idx])
            ax.fill_between(steps, avg_data - std_data, avg_data + std_data, 
                          alpha=0.15, color=colors[width_idx])
        
        ax.set_xlabel("Training Step", fontsize=12, fontweight='bold')
        ax.set_ylabel("|output|.mean()", fontsize=12, fontweight='bold')
        ax.set_title(metric_name, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)
    
    # Clean up exp_label for title (remove newlines)
    title_label = exp_label.replace('\n', ' ')
    plt.suptitle(f"CompleteP Width Coordinate Check - {title_label}", 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"width_coord_check_{short_name}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved {short_name} plot to {output_path}")
    plt.close()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("="*80)
    print("Plotting Architecture Ablation Studies")
    print("="*80)
    print()
    print("Experiments:")
    print("  1. Baseline: RMSNorm + SwiGLU + RoPE (LLAMA standard)")
    print("  2. SwiGLU Fixed: RMSNorm + SwiGLU+fix + RoPE (with muP multiplicative scaling)")
    print("  3. LayerNorm: LayerNorm + SwiGLU + RoPE")
    print("  4. GELU: RMSNorm + GELU + RoPE")
    print("  5. Learned Pos: RMSNorm + SwiGLU + Learned")
    print("  6. GPT-2-like: LayerNorm + GELU + Learned")
    print()
    
    plot_arch_comparison(base_dir, base_dir)
    
    print()
    print("="*80)
    print("✨ Plotting complete!")
    print("="*80)
    print("\nGenerated plots:")
    print("  - width_coord_check_arch_ablation.png (full comparison)")
    print("  - width_coord_check_baseline.png")
    print("  - width_coord_check_swiglu_fixed.png (NEW: with muP fix)")
    print("  - width_coord_check_layernorm.png")
    print("  - width_coord_check_gelu.png")
    print("  - width_coord_check_learned_pos.png")
    print("  - width_coord_check_gpt2like.png")
    print("="*80)


if __name__ == "__main__":
    main()

