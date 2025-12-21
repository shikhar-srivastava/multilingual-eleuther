#!/usr/bin/env python3
"""
Plotting Script for Coordinate Check Results

This script generates plots to visualize the coordinate check results,
comparing SP+muP, CompleteP, and depth_alpha_05 parameterizations.

Usage:
    python coord_check_experiments/plot_coord_checks.py
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_experiment_data(base_dir, experiment_name):
    """Load all CSV files from an experiment directory."""
    pattern = os.path.join(base_dir, experiment_name, "out", "*", "log.csv")
    files = glob.glob(pattern)
    
    # Try to load architecture config from first experiment
    arch_config = {"use_layernorm": False, "use_gelu_mlp": False, "position_embedding_type": "rope"}
    if files:
        config_path = os.path.join(os.path.dirname(files[0]), "config.json")
        if os.path.exists(config_path):
            try:
                import json
                with open(config_path, 'r') as cf:
                    saved_config = json.load(cf)
                    arch_config["use_layernorm"] = saved_config.get("use_layernorm", False)
                    arch_config["use_gelu_mlp"] = saved_config.get("use_gelu_mlp", False)
                    arch_config["position_embedding_type"] = saved_config.get("position_embedding_type", "rope")
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
    
    data = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Parse directory name to get depth/width and seed
            dir_name = os.path.basename(os.path.dirname(f))
            parts = dir_name.split("_")
            
            # Extract depth/width and seed
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
            
            # Add architecture config
            for k, v in arch_config.items():
                df[k] = v
            
            data.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
    
    if data:
        combined = pd.concat(data, ignore_index=True)
        # Store architecture config as metadata
        combined.attrs['arch_config'] = arch_config
        return combined
    return None


def get_arch_suffix(arch_config):
    """Generate a suffix for the filename based on architecture options."""
    parts = []
    if arch_config.get("use_layernorm", False):
        parts.append("LN")
    if arch_config.get("use_gelu_mlp", False):
        parts.append("GELU")
    if arch_config.get("position_embedding_type", "rope") != "rope":
        pos_type = arch_config["position_embedding_type"]
        parts.append(f"{pos_type.upper()}")
    
    return "_" + "_".join(parts) if parts else ""


def get_arch_label(arch_config):
    """Generate a label for the plot title based on architecture options."""
    parts = []
    norm = "LayerNorm" if arch_config.get("use_layernorm", False) else "RMSNorm"
    parts.append(norm)
    
    mlp = "GELU" if arch_config.get("use_gelu_mlp", False) else "SwiGLU"
    parts.append(mlp)
    
    pos = arch_config.get("position_embedding_type", "rope").upper()
    parts.append(pos)
    
    return " | ".join(parts)


def plot_depth_coord_check(base_dir, output_path="depth_coord_check.png"):
    """Plot depth coordinate check results - one row per metric, one column per experiment."""
    experiments = [
        ("sp_and_mup", "SP + muP\n(no depth scaling)"),
        ("completep", "CompleteP (α=1.0)"),
        ("depth_alpha_05", "CompleteP (α=0.5)"),
    ]
    
    metrics = [
        ("last_layer_act_abs_mean", "Last Layer Output"),
        ("attn_act_abs_mean", "Attention Output"),
        ("mlp_act_abs_mean", "MLP Output"),
    ]
    
    fig, axes = plt.subplots(len(metrics), len(experiments), figsize=(15, 12))
    
    # Get architecture config from first available experiment
    arch_config = None
    for exp_name, _ in experiments:
        data = load_experiment_data(base_dir, exp_name)
        if data is not None and hasattr(data, 'attrs') and 'arch_config' in data.attrs:
            arch_config = data.attrs['arch_config']
            break
    
    if arch_config is None:
        arch_config = {"use_layernorm": False, "use_gelu_mlp": False, "position_embedding_type": "rope"}
    
    for col_idx, (exp_name, exp_label) in enumerate(experiments):
        data = load_experiment_data(base_dir, exp_name)
        if data is None:
            print(f"No data found for {exp_name}")
            continue
        
        for row_idx, (metric, metric_name) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            
            if metric not in data.columns:
                ax.set_title(f"{exp_label}\n{metric_name} (no data)")
                ax.set_xlabel("Training Step")
                continue
            
            # Group by depth and step
            grouped = data.groupby(["depth", "step"])[metric].agg(["mean", "std"]).reset_index()
            
            depths = sorted(data["depth"].unique())
            colors = plt.cm.viridis(np.linspace(0, 1, len(depths)))
            
            for depth, color in zip(depths, colors):
                depth_data = grouped[grouped["depth"] == depth]
                ax.plot(depth_data["step"], depth_data["mean"], 
                       label=f"depth={depth}", color=color, linewidth=2)
                ax.fill_between(depth_data["step"], 
                               depth_data["mean"] - depth_data["std"],
                               depth_data["mean"] + depth_data["std"],
                               alpha=0.2, color=color)
            
            ax.set_xlabel("Training Step")
            ax.set_ylabel(f"|output|.mean()")
            ax.set_title(f"{exp_label}\n{metric_name}")
            ax.legend(fontsize=7, loc='upper left')
            ax.grid(True, alpha=0.3)
    
    # Add architecture info to title
    arch_label = get_arch_label(arch_config)
    plt.suptitle(f"Depth Coordinate Check: Activation Magnitudes vs Depth\n{arch_label}", 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Add architecture suffix to filename
    arch_suffix = get_arch_suffix(arch_config)
    if arch_suffix:
        output_path = output_path.replace(".png", f"{arch_suffix}.png")
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved depth coordinate check plot to {output_path}")
    plt.close()


def plot_width_coord_check(base_dir, output_path="width_coord_check.png"):
    """Plot width coordinate check results."""
    experiments = [
        ("width_scaling/sp", "Standard Parameterization (SP)"),
        ("width_scaling/mup", "muP"),
        ("width_scaling/completep", "CompleteP"),
    ]
    
    metrics = [
        ("last_layer_act_abs_mean", "Last Layer Output"),
        ("attn_act_abs_mean", "Attention Output"),
        ("mlp_act_abs_mean", "MLP Output"),
    ]
    
    fig, axes = plt.subplots(len(metrics), len(experiments), figsize=(18, 12))
    
    # Get architecture config from first available experiment
    arch_config = None
    for exp_name, _ in experiments:
        alt_base = os.path.join(base_dir, exp_name)
        pattern = os.path.join(alt_base, "out", "*", "config.json")
        config_files = glob.glob(pattern)
        if config_files:
            try:
                import json
                with open(config_files[0], 'r') as cf:
                    saved_config = json.load(cf)
                    arch_config = {
                        "use_layernorm": saved_config.get("use_layernorm", False),
                        "use_gelu_mlp": saved_config.get("use_gelu_mlp", False),
                        "position_embedding_type": saved_config.get("position_embedding_type", "rope")
                    }
                    break
            except:
                pass
    
    if arch_config is None:
        arch_config = {"use_layernorm": False, "use_gelu_mlp": False, "position_embedding_type": "rope"}
    
    for col_idx, (exp_name, exp_label) in enumerate(experiments):
        # Load data from width_scaling subdirectory
        alt_base = os.path.join(base_dir, exp_name)
        pattern = os.path.join(alt_base, "out", "*", "log.csv")
        files = glob.glob(pattern)
        
        if not files:
            print(f"No data found for {exp_name}")
            continue
            
        data_list = []
        for f in files:
            try:
                df = pd.read_csv(f)
                dir_name = os.path.basename(os.path.dirname(f))
                parts = dir_name.split("_")
                for part in parts:
                    if part.startswith("width"):
                        df["width"] = int(part.replace("width", ""))
                    elif part.startswith("depth"):
                        df["depth"] = int(part.replace("depth", ""))
                    elif part.startswith("seed"):
                        df["seed"] = int(part.replace("seed", ""))
                data_list.append(df)
            except Exception as e:
                print(f"Warning: Failed to load {f}: {e}")
        
        if not data_list:
            continue
        data = pd.concat(data_list, ignore_index=True)
        
        for row_idx, (metric, metric_name) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            
            if metric not in data.columns:
                ax.set_title(f"{exp_label}\n{metric_name} (no data)")
                continue
            
            widths = sorted(data["width"].unique())
            colors = plt.cm.plasma(np.linspace(0, 1, len(widths)))
            
            grouped = data.groupby(["width", "step"])[metric].agg(["mean", "std"]).reset_index()
            
            for width, color in zip(widths, colors):
                width_data = grouped[grouped["width"] == width]
                ax.plot(width_data["step"], width_data["mean"],
                       label=f"width={width}", color=color, linewidth=2)
                ax.fill_between(width_data["step"],
                               width_data["mean"] - width_data["std"],
                               width_data["mean"] + width_data["std"],
                               alpha=0.2, color=color)
            
            ax.set_xlabel("Training Step")
            ax.set_ylabel("|output|.mean()")
            ax.set_title(f"{exp_label}\n{metric_name}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    # Add architecture info to title
    arch_label = get_arch_label(arch_config)
    plt.suptitle(f"Width Coordinate Check: Activation Magnitudes vs Width\n{arch_label}", 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Add architecture suffix to filename
    arch_suffix = get_arch_suffix(arch_config)
    if arch_suffix:
        output_path = output_path.replace(".png", f"{arch_suffix}.png")
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved width coordinate check plot to {output_path}")
    plt.close()


def main():
    base_dir = Path(__file__).parent
    
    print("Generating coordinate check plots...")
    
    # Plot depth checks
    plot_depth_coord_check(base_dir, base_dir / "depth_coord_check.png")
    
    # Plot width checks
    plot_width_coord_check(base_dir, base_dir / "width_coord_check.png")
    
    print("\nPlotting complete!")
    print("Check the following files:")
    print(f"  - {base_dir / 'depth_coord_check.png'}")
    print(f"  - {base_dir / 'width_coord_check.png'}")


if __name__ == "__main__":
    main()
