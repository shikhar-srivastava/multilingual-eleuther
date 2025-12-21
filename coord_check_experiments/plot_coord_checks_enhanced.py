#!/usr/bin/env python3
"""
Enhanced coordinate check plotting in the style of nanoGPT-mup.

Creates beautiful plots with:
- Width/Depth on X-axis (log scale)
- Separate lines for each training step
- Multiple activation types in subplots
- Professional styling matching the reference implementation
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Beautiful plot styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


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
                if part.startswith("depth"):
                    config["depth"] = int(part.replace("depth", ""))
                elif part.startswith("width"):
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


def plot_width_scaling_enhanced(base_dir, output_dir="coord_check_experiments"):
    """
    Create width scaling plots in nanoGPT-mup style.
    X-axis: Width (log scale)
    Y-axis: Activation magnitude (log scale)
    Lines: Different training steps
    """
    
    experiments = [
        ("width_scaling/sp", "Standard Parameterization (SP)", "SP"),
        ("width_scaling/mup", "muP", "muP"),
        ("width_scaling/completep", "CompleteP (α=1.0)", "CompleteP"),
        ("width_scaling/completep_swiglu_fixed", "CompleteP + SwiGLU Fix", "CompleteP-SwiGLU-Fix"),
        ("width_scaling/completep_layernorm", "CompleteP + LayerNorm", "CompleteP-LN"),
        ("width_scaling/completep_gelu", "CompleteP + GELU", "CompleteP-GELU"),
        ("width_scaling/completep_learned_pos", "CompleteP + Learned Pos", "CompleteP-Learned"),
        ("width_scaling/completep_gpt2like", "CompleteP + GPT-2-like", "CompleteP-GPT2"),
        ("width_scaling/ablation_no_swiglu_scaling", "CompleteP (No SwiGLU Scale)", "Ablation-NoScale"),
    ]
    
    metrics = [
        ("token_embedding_act_abs_mean", "Word Embedding"),
        ("attn_act_abs_mean", "Attention Output"),
        ("mlp_act_abs_mean", "FFN Output"),
        ("last_layer_act_abs_mean", "Output Logits"),
    ]
    
    # Select steps to plot (similar to reference: steps 1-10)
    steps_to_plot = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Color map for steps (blue to red)
    colors = cm.coolwarm(np.linspace(0.1, 0.9, len(steps_to_plot)))
    
    for exp_name, exp_label, exp_short in experiments:
        data = load_experiment_data(base_dir, exp_name)
        if data is None or "width" not in data.columns:
            print(f"Skipping {exp_name} - no width data")
            continue
        
        print(f"Plotting width scaling for {exp_short}...")
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))
        if len(metrics) == 1:
            axes = [axes]
        
        for metric_idx, (metric_col, metric_name) in enumerate(metrics):
            ax = axes[metric_idx]
            
            if metric_col not in data.columns:
                ax.text(0.5, 0.5, f"No {metric_col}", ha='center', va='center', transform=ax.transAxes)
                continue
            
            widths = sorted(data["width"].unique())
            width_values = np.array(widths)
            
            # Plot each step as a separate line
            for step_idx, step in enumerate(steps_to_plot):
                step_data = data[data["iter"] == step]
                if len(step_data) == 0:
                    continue
                
                # Average over seeds for each width at this step
                means = []
                stds = []
                for width in widths:
                    width_step_data = step_data[step_data["width"] == width][metric_col]
                    if len(width_step_data) > 0:
                        means.append(width_step_data.mean())
                        stds.append(width_step_data.std())
                    else:
                        means.append(np.nan)
                        stds.append(np.nan)
                
                means = np.array(means)
                stds = np.array(stds)
                
                # Plot line
                ax.plot(width_values, means, color=colors[step_idx], 
                       linewidth=2, alpha=0.8, label=f"Step {step}")
                # Shaded error region
                ax.fill_between(width_values, means - stds, means + stds,
                               color=colors[step_idx], alpha=0.15)
            
            # Styling
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.set_xlabel('Width', fontweight='bold', fontsize=11)
            ax.set_ylabel('||activation||.mean()', fontweight='bold', fontsize=11)
            ax.set_title(metric_name, fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
            
            # Only show legend in the last subplot to keep it clean
            if metric_idx == len(metrics) - 1:
                ax.legend(ncol=2, framealpha=0.95, loc='best', fontsize=8, 
                         title='Training Step', title_fontsize=9)
            
            # Format x-axis ticks as powers of 2
            ax.set_xticks(widths)
            ax.set_xticklabels([f'$2^{{{int(np.log2(w))}}}$' for w in widths])
        
        plt.suptitle(f"Width Coordinate Check: {exp_label}", 
                     fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f"width_coord_{exp_short.lower().replace('-', '_')}.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"  ✅ Saved to {output_path}")
        plt.close()


def plot_depth_scaling_enhanced(base_dir, output_dir="coord_check_experiments"):
    """
    Create depth scaling plots in nanoGPT-mup style.
    X-axis: Depth (log scale)
    Y-axis: Activation magnitude (log scale)
    Lines: Different training steps
    """
    
    experiments = [
        ("sp_and_mup", "SP/muP (no depth scaling)", "sp_mup"),
        ("completep", "CompleteP α=1.0 (baseline)", "completep_baseline_old"),
        ("depth_alpha_05", "CompleteP α=0.5", "completep_05"),
        ("depth_scaling/completep_baseline", "CompleteP α=1.0 - Baseline (RMSNorm+SwiGLU+RoPE)", "completep_baseline"),
        ("depth_scaling/completep_layernorm", "CompleteP α=1.0 + LayerNorm", "completep_ln"),
        ("depth_scaling/completep_gelu", "CompleteP α=1.0 + GELU", "completep_gelu"),
        ("depth_scaling/completep_learned_pos", "CompleteP α=1.0 + Learned Pos", "completep_learned"),
        ("depth_scaling/completep_gpt2like", "CompleteP α=1.0 + GPT-2-like", "completep_gpt2"),
        ("depth_scaling/ablation_no_swiglu_scaling", "CompleteP (No SwiGLU Scale)", "ablation_noscale"),
    ]
    
    metrics = [
        ("token_embedding_act_abs_mean", "Word Embedding"),
        ("attn_act_abs_mean", "Attention Output"),
        ("mlp_act_abs_mean", "FFN Output"),
        ("last_layer_act_abs_mean", "Final residual"),
    ]
    
    # Select steps to plot
    steps_to_plot = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Color map for steps
    colors = cm.coolwarm(np.linspace(0.1, 0.9, len(steps_to_plot)))
    
    for exp_name, exp_label, exp_short in experiments:
        data = load_experiment_data(base_dir, exp_name)
        if data is None or "depth" not in data.columns:
            print(f"Skipping {exp_name} - no depth data")
            continue
        
        print(f"Plotting depth scaling for {exp_short}...")
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))
        if len(metrics) == 1:
            axes = [axes]
        
        for metric_idx, (metric_col, metric_name) in enumerate(metrics):
            ax = axes[metric_idx]
            
            if metric_col not in data.columns:
                ax.text(0.5, 0.5, f"No {metric_col}", ha='center', va='center', transform=ax.transAxes)
                continue
            
            depths = sorted(data["depth"].unique())
            depth_values = np.array(depths)
            
            # Plot each step as a separate line
            for step_idx, step in enumerate(steps_to_plot):
                step_data = data[data["iter"] == step]
                if len(step_data) == 0:
                    continue
                
                # Average over seeds for each depth at this step
                means = []
                stds = []
                for depth in depths:
                    depth_step_data = step_data[step_data["depth"] == depth][metric_col]
                    if len(depth_step_data) > 0:
                        means.append(depth_step_data.mean())
                        stds.append(depth_step_data.std())
                    else:
                        means.append(np.nan)
                        stds.append(np.nan)
                
                means = np.array(means)
                stds = np.array(stds)
                
                # Plot line
                ax.plot(depth_values, means, color=colors[step_idx],
                       linewidth=2, alpha=0.8, label=f"Step {step}")
                # Shaded error region
                ax.fill_between(depth_values, means - stds, means + stds,
                               color=colors[step_idx], alpha=0.15)
            
            # Styling
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.set_xlabel('Depth (L)', fontweight='bold', fontsize=11)
            ax.set_ylabel('||activation||.mean()', fontweight='bold', fontsize=11)
            ax.set_title(metric_name, fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
            
            # Only show legend in the last subplot to keep it clean
            if metric_idx == len(metrics) - 1:
                ax.legend(ncol=2, framealpha=0.95, loc='best', fontsize=8,
                         title='Training Step', title_fontsize=9)
            
            # Format x-axis ticks as powers of 2
            ax.set_xticks(depths)
            ax.set_xticklabels([f'$2^{{{int(np.log2(d))}}}$' for d in depths])
        
        plt.suptitle(f"Depth Coordinate Check: {exp_label}",
                     fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f"depth_coord_{exp_short}.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"  ✅ Saved to {output_path}")
        plt.close()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("="*80)
    print("Enhanced Coordinate Check Plotting (nanoGPT-mup style)")
    print("="*80)
    print()
    
    # Plot width scaling
    print("Plotting width coordinate checks...")
    plot_width_scaling_enhanced(base_dir, base_dir)
    
    print()
    
    # Plot depth scaling
    print("Plotting depth coordinate checks...")
    plot_depth_scaling_enhanced(base_dir, base_dir)
    
    print()
    print("="*80)
    print("✨ Enhanced plotting complete!")
    print("="*80)
    print()
    print("Look for files matching:")
    print("  - width_coord_*.png")
    print("  - depth_coord_*.png")
    print("="*80)


if __name__ == "__main__":
    main()

