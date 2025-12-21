#!/usr/bin/env python3
"""
Plot muTransfer Learning Rate Sweep Results

This script generates a figure comparing:
- SP (Standard Parameterization): Different widths have different optimal LRs
- μP (muTransfer): All widths converge to the same optimal LR

This replicates the classic muTransfer demonstration from the μP paper.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import seaborn as sns

# Set style
sns.set(style='whitegrid')

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PARAMETERIZATIONS = [
    ('sp', r'SP'),
    ('mup', r'$\mu$P'),
]

SEEDS = [1, 2, 3]

WIDTHS = [
    256,
    512,
    1024,
    2048,
]

# Learning rates (powers of 2 from 2^-14 to 2^-3)
LRS = [
    0.125,          # 2^-3
    0.0625,         # 2^-4
    0.03125,        # 2^-5
    0.015625,       # 2^-6
    0.0078125,      # 2^-7
    0.00390625,     # 2^-8
    0.001953125,    # 2^-9
    0.0009765625,   # 2^-10
    0.00048828125,  # 2^-11
    0.000244140625, # 2^-12
    0.0001220703125, # 2^-13
    0.00006103515625, # 2^-14
    0.00003051757812, # 2^-15
]

LAYERS = 2


class MplColorHelper:
    """Helper class to get colors from a colormap."""
    
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)


def load_results(base_dir, parameterization, widths, seeds, lrs, layers, out_dir='out', verbose=True):
    """Load training results from CSV files."""
    results = {}
    
    for width in widths:
        results[width] = {}
        for lr in lrs:
            losses = []
            for seed in seeds:
                # Format learning rate to match directory naming
                lr_str = f'{lr:.20f}'.rstrip('0')
                if lr_str.endswith('.'):
                    lr_str += '0'
                
                # Try multiple naming conventions
                possible_names = [
                    f'width{width}_depth{layers}_seed{seed}_lr{lr_str}',  # Full experiment naming
                    f'width{width}_lr{lr_str}',  # Quick test naming
                ]
                
                found = False
                for job_name in possible_names:
                    csv_path = os.path.join(base_dir, parameterization, out_dir, job_name, 'log.csv')
                    
                    if os.path.exists(csv_path):
                        try:
                            df = pd.read_csv(csv_path)
                            # Use exponential moving average of training loss
                            if 'train/loss' in df.columns:
                                final_loss = df['train/loss'].ewm(alpha=0.9).mean().values[-1]
                                losses.append(final_loss)
                                found = True
                                break
                            else:
                                print(f"Warning: 'train/loss' column not found in {csv_path}")
                        except Exception as e:
                            print(f"Error reading {csv_path}: {e}")
                
                if not found and verbose:
                    # Only print missing for the first naming convention to reduce noise
                    csv_path = os.path.join(base_dir, parameterization, out_dir, possible_names[0], 'log.csv')
                    print(f"Missing: {csv_path}")
            
            if losses:
                results[width][lr] = {
                    'mean': np.mean(losses),
                    'std': np.std(losses, ddof=1) if len(losses) > 1 else 0,
                    'sem': np.std(losses, ddof=1) / np.sqrt(len(losses)) if len(losses) > 1 else 0,
                    'n': len(losses)
                }
    
    return results


def plot_mutransfer(results_sp, results_mup, widths, lrs, output_path=None):
    """Plot muTransfer learning rate sweep comparison."""
    
    color_helper = MplColorHelper('viridis', 0, len(widths) - 1)
    
    n_cols = 2
    n_rows = 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    
    for param_idx, (results, title) in enumerate([(results_sp, r'SP'), (results_mup, r'$\mu$P')]):
        ax = axes[param_idx]
        optimal_lrs = []
        optimal_losses = []
        
        for width_idx, width in enumerate(widths):
            if width not in results:
                continue
                
            avg_losses = []
            sem_losses = []
            lrs_to_plot = []
            
            for lr in lrs:
                if lr in results[width]:
                    avg_losses.append(results[width][lr]['mean'])
                    sem_losses.append(results[width][lr]['sem'])
                    lrs_to_plot.append(lr)
            
            if not lrs_to_plot:
                continue
            
            avg_losses = np.array(avg_losses)
            sem_losses = np.array(sem_losses)
            
            # Plot loss vs LR curve
            ax.plot(lrs_to_plot, avg_losses, label=width, marker='o', 
                   color=color_helper.get_rgb(width_idx), markersize=4)
            ax.fill_between(lrs_to_plot, avg_losses - sem_losses, avg_losses + sem_losses, 
                           color=color_helper.get_rgb(width_idx), alpha=0.33)
            
            # Track optimal LR
            if len(avg_losses) > 0:
                optimum_idx = np.argmin(avg_losses)
                optimal_lrs.append(lrs_to_plot[optimum_idx])
                optimal_losses.append(avg_losses[optimum_idx])
        
        # Mark optimal points
        if optimal_lrs:
            ax.plot(optimal_lrs, optimal_losses, color='red', linestyle='none', 
                   marker='o', markersize=6, markeredgecolor='black', markeredgewidth=0.5)
        
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Learning rate')
        ax.set_title(title)
        
        # Set y-axis limits based on data
        all_losses = [results[w][lr]['mean'] for w in results for lr in results[w] if 'mean' in results[w][lr]]
        if all_losses:
            ymin = max(min(all_losses) - 0.2, 0)
            ymax = min(max(all_losses) + 0.5, 10)
            ax.set_ylim(ymin, ymax)
    
    # Add legend and labels
    axes[0].legend(title='Width', loc='upper left', fontsize=8)
    axes[0].set_ylabel('Train Loss on\nShakespeare')
    axes[1].yaxis.set_ticklabels([])
    axes[1].tick_params(axis='y', length=0, width=0)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    plt.show()
    plt.close()
    
    return fig


def main():
    """Main function to generate muTransfer plot."""
    
    # First try the full experiment output (out/)
    print("Loading SP results from out/...")
    results_sp = load_results(SCRIPT_DIR, 'sp', WIDTHS, SEEDS, LRS, LAYERS, out_dir='out', verbose=False)
    
    # If no results found, try quick test output (out_test/)
    sp_has_results = any(results_sp[w] for w in results_sp)
    if not sp_has_results:
        print("No results in out/, trying out_test/...")
        results_sp = load_results(SCRIPT_DIR, 'sp', WIDTHS, [1], LRS, LAYERS, out_dir='out_test', verbose=True)
    
    print("\nLoading μP results from out/...")
    results_mup = load_results(SCRIPT_DIR, 'mup', WIDTHS, SEEDS, LRS, LAYERS, out_dir='out', verbose=False)
    
    # If no results found, try quick test output (out_test/)
    mup_has_results = any(results_mup[w] for w in results_mup)
    if not mup_has_results:
        print("No results in out/, trying out_test/...")
        results_mup = load_results(SCRIPT_DIR, 'mup', WIDTHS, [1], LRS, LAYERS, out_dir='out_test', verbose=True)
    
    print("\nGenerating plot...")
    output_path = os.path.join(SCRIPT_DIR, 'mutransfer_lr_shakespeare.png')
    plot_mutransfer(results_sp, results_mup, WIDTHS, LRS, output_path)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary: Optimal Learning Rates by Width")
    print("=" * 60)
    
    for param_name, results in [('SP', results_sp), ('μP', results_mup)]:
        print(f"\n{param_name}:")
        for width in WIDTHS:
            if width in results:
                best_lr = None
                best_loss = float('inf')
                for lr, data in results[width].items():
                    if data['mean'] < best_loss:
                        best_loss = data['mean']
                        best_lr = lr
                if best_lr:
                    print(f"  Width {width}: optimal LR = {best_lr:.6f} (2^{np.log2(best_lr):.1f}), loss = {best_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("Key Insight:")
    print("  - SP: Optimal LR shifts with width (different for each width)")
    print("  - μP: Optimal LR is approximately CONSTANT across widths")
    print("=" * 60)


if __name__ == "__main__":
    main()

