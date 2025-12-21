#!/usr/bin/env python3
"""
Multilingual muTransfer Scaling Visualization

This script generates beautiful comparison plots showing:
1. Width scaling transfer across different languages (SP vs CompleteP)
2. Depth scaling transfer across different languages (SP vs CompleteP)
3. Combined multi-panel figures for publication

The key insight to demonstrate:
- SP: Optimal learning rate SHIFTS with model scale (different for each width/depth)
- CompleteP: Optimal learning rate REMAINS CONSTANT across scales

Usage:
    python plot_mutransfer_multilingual.py [--out_dir out] [--experiment width|depth|both]
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Language configurations with beautiful display names and colors
LANGUAGES = {
    'eng_latn': {'name': 'English', 'short': 'ENG', 'color': '#2E86AB', 'script': 'Latin'},
    'tha_thai': {'name': 'Thai', 'short': 'THA', 'color': '#A23B72', 'script': 'Thai'},
    'urd_arab': {'name': 'Urdu', 'short': 'URD', 'color': '#F18F01', 'script': 'Arabic'},
    'amh_ethi': {'name': 'Amharic', 'short': 'AMH', 'color': '#C73E1D', 'script': 'Ethiopic'},
    'vie_latn': {'name': 'Vietnamese', 'short': 'VIE', 'color': '#3B1F2B', 'script': 'Latin'},
}

# Width and depth configurations (matching 9M base model)
WIDTHS = [128, 256, 512, 768]
DEPTHS = [4, 8, 12, 16]
BASE_DEPTH = 4
BASE_WIDTH = 128

# Learning rates
LRS = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]

SEEDS = [1, 2, 3]

# Color palette for width/depth gradients
CMAP_WIDTH = 'viridis'
CMAP_DEPTH = 'magma'


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def setup_matplotlib_style():
    """Set up a beautiful, publication-ready matplotlib style."""
    plt.rcParams.update({
        # Font settings - using clean sans-serif
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size': 11,
        
        # Figure settings
        'figure.facecolor': '#FAFAFA',
        'figure.edgecolor': 'none',
        'figure.dpi': 150,
        
        # Axes settings
        'axes.facecolor': '#FFFFFF',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 0.8,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'axes.labelweight': 'medium',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        
        # Grid settings
        'grid.color': '#E0E0E0',
        'grid.linewidth': 0.5,
        'grid.linestyle': '-',
        'grid.alpha': 0.7,
        
        # Legend settings
        'legend.frameon': True,
        'legend.facecolor': '#FFFFFF',
        'legend.edgecolor': '#CCCCCC',
        'legend.fontsize': 9,
        'legend.title_fontsize': 10,
        
        # Tick settings
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
    })


class GradientColorHelper:
    """Helper to get colors from a colormap for width/depth gradients."""
    
    def __init__(self, cmap_name, values):
        self.cmap = plt.get_cmap(cmap_name)
        self.values = sorted(values)
        self.norm = mpl.colors.Normalize(vmin=0, vmax=len(values) - 1)
        self.scalar_map = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
    
    def get_color(self, value):
        idx = self.values.index(value) if value in self.values else 0
        return self.scalar_map.to_rgba(idx)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_results(base_dir, parameterization, language, widths, depths, seeds, lrs, 
                 out_dir='out', experiment_type='width', verbose=False,
                 use_best_loss=True):
    """
    Load training results from experiment output directories.
    
    Returns a nested dict: results[scale_value][lr] = {'mean': ..., 'std': ..., 'sem': ...}
    where scale_value is width (for width experiments) or depth (for depth experiments).
    
    Args:
        use_best_loss: If True, use the minimum (best) loss across training epochs.
                       If False, use the final loss. Default is True for muTransfer experiments
                       as this shows the optimal achievable performance at each LR.
    """
    results = {}
    
    # Determine which dimension to sweep
    if experiment_type == 'width':
        scale_values = widths
        fixed_dim = BASE_DEPTH
        scale_name = 'width'
        fixed_name = 'depth'
    else:
        scale_values = depths
        fixed_dim = BASE_WIDTH
        scale_name = 'depth'
        fixed_name = 'width'
    
    for scale_val in scale_values:
        results[scale_val] = {}
        
        for lr in lrs:
            losses = []
            
            for seed in seeds:
                # Format LR for directory naming
                lr_str = f'{lr}'
                
                # Build directory pattern based on experiment type
                if experiment_type == 'width':
                    job_name = f'width{scale_val}_depth{fixed_dim}_seed{seed}_lr{lr_str}'
                else:
                    job_name = f'width{fixed_dim}_depth{scale_val}_seed{seed}_lr{lr_str}'
                
                csv_path = os.path.join(
                    base_dir, parameterization, out_dir, language, job_name, 'log.csv'
                )
                
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        
                        # Determine which loss column to use
                        loss_col = None
                        if 'train/loss' in df.columns:
                            loss_col = 'train/loss'
                        elif 'loss' in df.columns:
                            loss_col = 'loss'
                        
                        if loss_col is not None:
                            # Apply EWM smoothing for stability
                            smoothed_loss = df[loss_col].ewm(alpha=0.9).mean()
                            
                            if use_best_loss:
                                # Use the MINIMUM (best) loss across all epochs
                                # This is standard for muTransfer experiments
                                best_loss = smoothed_loss.min()
                                losses.append(best_loss)
                            else:
                                # Use the final loss (last value)
                                final_loss = smoothed_loss.values[-1]
                                losses.append(final_loss)
                                
                    except Exception as e:
                        if verbose:
                            print(f"Error reading {csv_path}: {e}")
                else:
                    if verbose:
                        print(f"Missing: {csv_path}")
            
            if losses:
                results[scale_val][lr] = {
                    'mean': np.mean(losses),
                    'std': np.std(losses, ddof=1) if len(losses) > 1 else 0,
                    'sem': np.std(losses, ddof=1) / np.sqrt(len(losses)) if len(losses) > 1 else 0,
                    'n': len(losses)
                }
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting Functions
# ═══════════════════════════════════════════════════════════════════════════════

def plot_lr_sweep_single(ax, results, scale_values, lrs, color_helper, title, 
                          scale_label='Width', ylabel='Best Loss'):
    """
    Plot learning rate sweep curves for a single parameterization.
    
    Shows loss vs LR curves for different scale values (width or depth),
    with markers at optimal LR points.
    """
    optimal_lrs = []
    optimal_losses = []
    scale_at_optimal = []
    
    for idx, scale_val in enumerate(scale_values):
        if scale_val not in results:
            continue
        
        avg_losses = []
        sem_losses = []
        lrs_to_plot = []
        
        for lr in lrs:
            if lr in results[scale_val]:
                avg_losses.append(results[scale_val][lr]['mean'])
                sem_losses.append(results[scale_val][lr]['sem'])
                lrs_to_plot.append(lr)
        
        if not lrs_to_plot:
            continue
        
        avg_losses = np.array(avg_losses)
        sem_losses = np.array(sem_losses)
        
        color = color_helper.get_color(scale_val)
        
        # Plot curve with confidence band
        ax.plot(lrs_to_plot, avg_losses, 
                label=f'{scale_val}', 
                marker='o', 
                color=color, 
                markersize=5,
                linewidth=1.5,
                alpha=0.9)
        ax.fill_between(lrs_to_plot, 
                        avg_losses - sem_losses, 
                        avg_losses + sem_losses,
                        color=color, 
                        alpha=0.15)
        
        # Track optimal LR
        if len(avg_losses) > 0:
            opt_idx = np.argmin(avg_losses)
            optimal_lrs.append(lrs_to_plot[opt_idx])
            optimal_losses.append(avg_losses[opt_idx])
            scale_at_optimal.append(scale_val)
    
    # Mark optimal points with red circles
    if optimal_lrs:
        ax.plot(optimal_lrs, optimal_losses, 
                color='#E63946',
                linestyle='none',
                marker='o', 
                markersize=8,
                markeredgecolor='white',
                markeredgewidth=1.5,
                zorder=10,
                label='_nolegend_')
        
        # Draw line connecting optimal points to show transfer
        sorted_indices = np.argsort(scale_at_optimal)
        sorted_lrs = [optimal_lrs[i] for i in sorted_indices]
        sorted_losses = [optimal_losses[i] for i in sorted_indices]
        ax.plot(sorted_lrs, sorted_losses,
                color='#E63946',
                linestyle='--',
                linewidth=1.5,
                alpha=0.5,
                zorder=5,
                label='_nolegend_')
    
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Set reasonable y-axis limits
    all_losses = [results[s][lr]['mean'] for s in results for lr in results[s]]
    if all_losses:
        ymin = max(min(all_losses) - 0.3, 0)
        ymax = min(max(all_losses) + 0.5, max(all_losses) * 1.5)
        ax.set_ylim(ymin, ymax)
    
    return optimal_lrs, optimal_losses


def plot_sp_vs_completep_comparison(results_sp, results_completep, scale_values, lrs,
                                     language_info, experiment_type='width', 
                                     output_path=None):
    """
    Create a 2-panel comparison plot: SP (left) vs CompleteP (right).
    """
    setup_matplotlib_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Select colormap based on experiment type
    cmap = CMAP_WIDTH if experiment_type == 'width' else CMAP_DEPTH
    color_helper = GradientColorHelper(cmap, scale_values)
    
    scale_label = 'Width' if experiment_type == 'width' else 'Depth'
    lang_name = language_info['name']
    
    # Plot SP
    plot_lr_sweep_single(
        axes[0], results_sp, scale_values, lrs, color_helper,
        title=f'SP ({lang_name})',
        scale_label=scale_label,
        ylabel=f'Best Loss ({lang_name})'
    )
    
    # Plot CompleteP
    plot_lr_sweep_single(
        axes[1], results_completep, scale_values, lrs, color_helper,
        title=f'CompleteP ({lang_name})',
        scale_label=scale_label,
        ylabel=''
    )
    
    # Add shared legend
    axes[0].legend(title=scale_label, loc='upper left', fontsize=8)
    axes[1].yaxis.set_ticklabels([])
    axes[1].tick_params(axis='y', length=0)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    plt.close()
    return fig


def plot_multilingual_grid(all_results, languages, scale_values, lrs,
                           experiment_type='width', output_path=None):
    """
    Create a publication-quality grid plot showing all languages and parameterizations.
    
    Layout: 
    - Rows: Languages
    - Columns: SP vs CompleteP
    """
    setup_matplotlib_style()
    
    n_langs = len(languages)
    fig, axes = plt.subplots(n_langs, 2, figsize=(11, 3 * n_langs))
    
    if n_langs == 1:
        axes = axes.reshape(1, 2)
    
    cmap = CMAP_WIDTH if experiment_type == 'width' else CMAP_DEPTH
    color_helper = GradientColorHelper(cmap, scale_values)
    scale_label = 'Width' if experiment_type == 'width' else 'Depth'
    
    for lang_idx, lang in enumerate(languages):
        lang_info = LANGUAGES.get(lang, {'name': lang, 'short': lang.upper()})
        
        for param_idx, param in enumerate(['sp', 'completep']):
            ax = axes[lang_idx, param_idx]
            
            results = all_results.get(lang, {}).get(param, {})
            
            if not results:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                        transform=ax.transAxes, fontsize=12, color='gray')
                ax.set_title(f'{param.upper()} - {lang_info["name"]}')
                continue
            
            plot_lr_sweep_single(
                ax, results, scale_values, lrs, color_helper,
                title=f'{"SP" if param == "sp" else "CompleteP"} - {lang_info["name"]}',
                scale_label=scale_label,
                ylabel='Best Loss' if param_idx == 0 else ''
            )
            
            # Add legend only to first row
            if lang_idx == 0:
                ax.legend(title=scale_label, loc='upper left', fontsize=7)
            
            # Remove y labels for right column
            if param_idx == 1:
                ax.yaxis.set_ticklabels([])
                ax.tick_params(axis='y', length=0)
    
    # Add column headers
    fig.text(0.25, 0.98, 'Standard Parameterization (SP)', ha='center', 
             fontsize=14, fontweight='bold')
    fig.text(0.75, 0.98, 'CompleteP (μP + Depth)', ha='center', 
             fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    plt.close()
    return fig


def plot_optimal_lr_transfer(all_results, languages, scale_values, lrs,
                              experiment_type='width', output_path=None):
    """
    Create a focused plot showing how optimal LR changes across scales.
    
    This is the key visualization for demonstrating muTransfer:
    - SP: Lines diverge (different optimal LRs for different scales)
    - CompleteP: Lines converge (same optimal LR across scales)
    """
    setup_matplotlib_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    scale_label = 'Width' if experiment_type == 'width' else 'Depth'
    
    for param_idx, param in enumerate(['sp', 'completep']):
        ax = axes[param_idx]
        
        for lang in languages:
            lang_info = LANGUAGES.get(lang, {'name': lang, 'color': '#666666'})
            results = all_results.get(lang, {}).get(param, {})
            
            if not results:
                continue
            
            optimal_lrs = []
            scale_vals = []
            
            for scale_val in scale_values:
                if scale_val not in results:
                    continue
                
                # Find optimal LR for this scale
                best_lr = None
                best_loss = float('inf')
                for lr in lrs:
                    if lr in results[scale_val]:
                        if results[scale_val][lr]['mean'] < best_loss:
                            best_loss = results[scale_val][lr]['mean']
                            best_lr = lr
                
                if best_lr is not None:
                    optimal_lrs.append(best_lr)
                    scale_vals.append(scale_val)
            
            if optimal_lrs:
                ax.plot(scale_vals, optimal_lrs,
                        marker='o',
                        markersize=8,
                        linewidth=2,
                        color=lang_info['color'],
                        label=lang_info['name'])
        
        ax.set_xlabel(scale_label)
        ax.set_ylabel('Optimal Learning Rate')
        ax.set_yscale('log')
        ax.set_title('SP' if param == 'sp' else 'CompleteP', fontsize=14, fontweight='bold')
        
        if param_idx == 0:
            ax.legend(loc='best', fontsize=9)
    
    # Add annotation explaining the key insight
    fig.text(0.5, 0.02, 
             'SP: Optimal LR shifts with scale  |  CompleteP: Optimal LR remains constant',
             ha='center', fontsize=11, style='italic', color='#555555')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    plt.close()
    return fig


def plot_combined_width_depth(all_results_width, all_results_depth, languages,
                               widths, depths, lrs, output_path=None):
    """
    Create a comprehensive 4-panel figure showing both width and depth scaling.
    
    Layout:
    - Row 1: Width scaling (SP | CompleteP)
    - Row 2: Depth scaling (SP | CompleteP)
    """
    setup_matplotlib_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    width_color = GradientColorHelper(CMAP_WIDTH, widths)
    depth_color = GradientColorHelper(CMAP_DEPTH, depths)
    
    experiments = [
        (all_results_width, widths, width_color, 'Width', 0),
        (all_results_depth, depths, depth_color, 'Depth', 1),
    ]
    
    for all_results, scale_values, color_helper, scale_label, row in experiments:
        for param_idx, param in enumerate(['sp', 'completep']):
            ax = axes[row, param_idx]
            
            # Aggregate across languages (use English as primary example)
            for lang in ['eng_latn']:  # Primary language for main figure
                results = all_results.get(lang, {}).get(param, {})
                if results:
                    plot_lr_sweep_single(
                        ax, results, scale_values, lrs, color_helper,
                        title=f'{"SP" if param == "sp" else "CompleteP"} - {scale_label} Scaling',
                        scale_label=scale_label,
                        ylabel=f'Best Loss' if param_idx == 0 else ''
                    )
                    ax.legend(title=scale_label, loc='upper left', fontsize=8)
            
            if param_idx == 1:
                ax.yaxis.set_ticklabels([])
    
    # Row labels
    fig.text(0.02, 0.75, 'Width\nScaling', ha='center', va='center', 
             fontsize=12, fontweight='bold', rotation=90)
    fig.text(0.02, 0.25, 'Depth\nScaling', ha='center', va='center', 
             fontsize=12, fontweight='bold', rotation=90)
    
    plt.tight_layout(rect=[0.04, 0, 1, 1])
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Summary Statistics
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(all_results, languages, scale_values, lrs, experiment_type='width'):
    """Print summary statistics about optimal LRs across scales."""
    
    scale_label = 'Width' if experiment_type == 'width' else 'Depth'
    
    print("\n" + "=" * 70)
    print(f"Summary: Optimal Learning Rates by {scale_label}")
    print("=" * 70)
    
    for param in ['sp', 'completep']:
        print(f"\n{'SP' if param == 'sp' else 'CompleteP'}:")
        print("-" * 60)
        
        for lang in languages:
            lang_info = LANGUAGES.get(lang, {'name': lang})
            results = all_results.get(lang, {}).get(param, {})
            
            if not results:
                continue
            
            print(f"\n  {lang_info['name']}:")
            
            optimal_lrs = []
            for scale_val in scale_values:
                if scale_val not in results:
                    continue
                
                best_lr = None
                best_loss = float('inf')
                for lr in lrs:
                    if lr in results[scale_val]:
                        if results[scale_val][lr]['mean'] < best_loss:
                            best_loss = results[scale_val][lr]['mean']
                            best_lr = lr
                
                if best_lr:
                    optimal_lrs.append(best_lr)
                    print(f"    {scale_label} {scale_val}: LR = {best_lr:.6f} (log2={np.log2(best_lr):.1f}), loss = {best_loss:.4f}")
            
            if len(optimal_lrs) > 1:
                lr_range = max(optimal_lrs) / min(optimal_lrs)
                print(f"    → LR range factor: {lr_range:.2f}x")
    
    print("\n" + "=" * 70)
    print("Key Insight:")
    print("  • SP:       Optimal LR shifts across scales (factor varies)")
    print("  • CompleteP: Optimal LR stays constant (factor ≈ 1.0)")
    print("=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Plot multilingual muTransfer results')
    parser.add_argument('--out_dir', type=str, default='out',
                        help='Output directory (out or out_test)')
    parser.add_argument('--experiment', type=str, default='both',
                        choices=['width', 'depth', 'both'],
                        help='Which experiment type to plot')
    parser.add_argument('--languages', type=str, nargs='+', 
                        default=['eng_latn', 'tha_thai', 'urd_arab', 'amh_ethi', 'vie_latn'],
                        help='Languages to include')
    parser.add_argument('--use_final_loss', action='store_true',
                        help='Use final loss instead of best (minimum) loss. '
                             'Default is to use best loss across epochs (standard for muTransfer).')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose loading information')
    args = parser.parse_args()
    
    use_best_loss = not args.use_final_loss
    
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║      Multilingual muTransfer Scaling Visualization                 ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"\nLoading results from: {SCRIPT_DIR}/{{sp,completep}}/{args.out_dir}/")
    print(f"Languages: {', '.join(args.languages)}")
    print(f"Experiments: {args.experiment}")
    print(f"Loss metric: {'Best (minimum) loss across epochs' if use_best_loss else 'Final loss'}")
    
    # Load all results
    all_results_width = {}
    all_results_depth = {}
    
    for lang in args.languages:
        all_results_width[lang] = {}
        all_results_depth[lang] = {}
        
        for param in ['sp', 'completep']:
            # Width scaling
            if args.experiment in ['width', 'both']:
                results_w = load_results(
                    SCRIPT_DIR, param, lang, WIDTHS, DEPTHS, SEEDS, LRS,
                    out_dir=args.out_dir, experiment_type='width', verbose=args.verbose,
                    use_best_loss=use_best_loss
                )
                if any(results_w[w] for w in results_w):
                    all_results_width[lang][param] = results_w
                    print(f"  ✓ Loaded width results: {lang} / {param}")
            
            # Depth scaling
            if args.experiment in ['depth', 'both']:
                results_d = load_results(
                    SCRIPT_DIR, param, lang, WIDTHS, DEPTHS, SEEDS, LRS,
                    out_dir=args.out_dir, experiment_type='depth', verbose=args.verbose,
                    use_best_loss=use_best_loss
                )
                if any(results_d[d] for d in results_d):
                    all_results_depth[lang][param] = results_d
                    print(f"  ✓ Loaded depth results: {lang} / {param}")
    
    # Generate plots
    print("\n" + "─" * 70)
    print("Generating plots...")
    print("─" * 70)
    
    # Width scaling plots
    if args.experiment in ['width', 'both'] and any(all_results_width.values()):
        # Per-language comparisons
        for lang in args.languages:
            if lang in all_results_width and all_results_width[lang]:
                plot_sp_vs_completep_comparison(
                    all_results_width[lang].get('sp', {}),
                    all_results_width[lang].get('completep', {}),
                    WIDTHS, LRS, LANGUAGES[lang], experiment_type='width',
                    output_path=os.path.join(SCRIPT_DIR, f'width_scaling_{lang}.png')
                )
        
        # Multi-language grid
        plot_multilingual_grid(
            all_results_width, args.languages, WIDTHS, LRS,
            experiment_type='width',
            output_path=os.path.join(SCRIPT_DIR, 'width_scaling_multilingual.png')
        )
        
        # Optimal LR transfer plot
        plot_optimal_lr_transfer(
            all_results_width, args.languages, WIDTHS, LRS,
            experiment_type='width',
            output_path=os.path.join(SCRIPT_DIR, 'width_optimal_lr_transfer.png')
        )
        
        print_summary(all_results_width, args.languages, WIDTHS, LRS, 'width')
    
    # Depth scaling plots
    if args.experiment in ['depth', 'both'] and any(all_results_depth.values()):
        # Per-language comparisons
        for lang in args.languages:
            if lang in all_results_depth and all_results_depth[lang]:
                plot_sp_vs_completep_comparison(
                    all_results_depth[lang].get('sp', {}),
                    all_results_depth[lang].get('completep', {}),
                    DEPTHS, LRS, LANGUAGES[lang], experiment_type='depth',
                    output_path=os.path.join(SCRIPT_DIR, f'depth_scaling_{lang}.png')
                )
        
        # Multi-language grid
        plot_multilingual_grid(
            all_results_depth, args.languages, DEPTHS, LRS,
            experiment_type='depth',
            output_path=os.path.join(SCRIPT_DIR, 'depth_scaling_multilingual.png')
        )
        
        # Optimal LR transfer plot
        plot_optimal_lr_transfer(
            all_results_depth, args.languages, DEPTHS, LRS,
            experiment_type='depth',
            output_path=os.path.join(SCRIPT_DIR, 'depth_optimal_lr_transfer.png')
        )
        
        print_summary(all_results_depth, args.languages, DEPTHS, LRS, 'depth')
    
    # Combined width + depth plot
    if args.experiment == 'both' and all_results_width and all_results_depth:
        plot_combined_width_depth(
            all_results_width, all_results_depth, args.languages,
            WIDTHS, DEPTHS, LRS,
            output_path=os.path.join(SCRIPT_DIR, 'combined_width_depth_scaling.png')
        )
    
    print("\n" + "═" * 70)
    print("Plotting complete!")
    print("═" * 70)


if __name__ == "__main__":
    main()

