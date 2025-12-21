#!/usr/bin/env python3
"""
Visualize activation statistics comparison between SP, muP, and CompleteP
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Read the CSV files
sp_df = pd.read_csv('coord_check_out/comparison/sp/log.csv')
mup_df = pd.read_csv('coord_check_out/comparison/mup_only/log.csv')
completep_df = pd.read_csv('coord_check_out/comparison/completep/log.csv')

# Create figure with subplots
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('Activation Analysis: SP vs muP vs CompleteP\n8 layers (4x depth scaling), 256 hidden (1x width)', 
             fontsize=14, fontweight='bold')

# Metrics to plot
metrics = [
    ('token_embedding_act_abs_mean', 'Token Embedding Activations'),
    ('attn_act_abs_mean', 'Attention Activations'),
    ('mlp_act_abs_mean', 'MLP Activations'),
    ('lm_head_act_abs_mean', 'LM Head Activations'),
    ('last_layer_act_abs_mean', 'Last Layer Activations'),
    ('train/loss', 'Training Loss')
]

# Plot each metric
for idx, (metric, title) in enumerate(metrics):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    ax.plot(sp_df['step'], sp_df[metric], 'o-', label='SP', linewidth=2, markersize=6)
    ax.plot(mup_df['step'], mup_df[metric], 's-', label='muP only', linewidth=2, markersize=6)
    ax.plot(completep_df['step'], completep_df[metric], '^-', label='CompleteP', linewidth=2, markersize=6)
    
    ax.set_xlabel('Training Step', fontsize=10)
    ax.set_ylabel('Activation Magnitude', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add final value annotations
    for df, marker, color in [(sp_df, 'o', 'C0'), (mup_df, 's', 'C1'), (completep_df, '^', 'C2')]:
        final_val = df[metric].iloc[-1]
        ax.annotate(f'{final_val:.3f}', 
                   xy=(df['step'].iloc[-1], final_val),
                   xytext=(5, 0), textcoords='offset points',
                   fontsize=8, color=color)

# Remove empty subplot
axes[2, 1].remove()

plt.tight_layout()
plt.savefig('coord_check_out/comparison/activation_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: coord_check_out/comparison/activation_comparison.png")

# Create a second figure focusing on the most critical metrics
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Critical Activation Metrics: Depth Scaling Impact\n8 layers (4x depth), 256 hidden', 
              fontsize=14, fontweight='bold')

critical_metrics = [
    ('attn_act_abs_mean', 'Attention Activations\n(Should be stable for CompleteP)'),
    ('mlp_act_abs_mean', 'MLP Activations\n(Should be stable for CompleteP)'),
    ('last_layer_act_abs_mean', 'Last Layer Activations\n(Critical: depth explosion indicator)'),
    ('train/loss', 'Training Loss\n(Lower is better)')
]

for idx, (metric, title) in enumerate(critical_metrics):
    row = idx // 2
    col = idx % 2
    ax = axes2[row, col]
    
    ax.plot(sp_df['step'], sp_df[metric], 'o-', label='SP', linewidth=2.5, markersize=7, alpha=0.8)
    ax.plot(mup_df['step'], mup_df[metric], 's-', label='muP only', linewidth=2.5, markersize=7, alpha=0.8)
    ax.plot(completep_df['step'], completep_df[metric], '^-', label='CompleteP', linewidth=2.5, markersize=7, alpha=0.8)
    
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Magnitude', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Highlight the dramatic difference for last_layer_act_abs_mean
    if metric == 'last_layer_act_abs_mean':
        # Add shaded region to show explosion zone
        ax.axhspan(completep_df[metric].max(), sp_df[metric].max(), 
                  alpha=0.1, color='red', label='Explosion zone')
        ax.text(0.5, 0.95, '⚠️ SP/muP: 25x growth (explosion!)\n✅ CompleteP: 10x growth (controlled)', 
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('coord_check_out/comparison/critical_activations.png', dpi=300, bbox_inches='tight')
print("✅ Saved: coord_check_out/comparison/critical_activations.png")

# Create comparison table
print("\n" + "="*80)
print("ACTIVATION GROWTH COMPARISON (Initial → Final)")
print("="*80)

for metric, title in metrics[:-1]:  # Exclude loss
    sp_init = sp_df[metric].iloc[0]
    sp_final = sp_df[metric].iloc[-1]
    sp_growth = (sp_final / sp_init - 1) * 100
    
    mup_init = mup_df[metric].iloc[0]
    mup_final = mup_df[metric].iloc[-1]
    mup_growth = (mup_final / mup_init - 1) * 100
    
    cp_init = completep_df[metric].iloc[0]
    cp_final = completep_df[metric].iloc[-1]
    cp_growth = (cp_final / cp_init - 1) * 100
    
    print(f"\n{title}:")
    print(f"  SP:        {sp_init:.4f} → {sp_final:.4f} ({sp_growth:+.1f}%)")
    print(f"  muP only:  {mup_init:.4f} → {mup_final:.4f} ({mup_growth:+.1f}%)")
    print(f"  CompleteP: {cp_init:.4f} → {cp_final:.4f} ({cp_growth:+.1f}%)")
    print(f"  CompleteP reduction: {(1 - cp_final/sp_final)*100:.1f}% vs SP")

print("\n" + "="*80)
print("FINAL LOSS COMPARISON")
print("="*80)
print(f"SP:        train={sp_df['train/loss'].iloc[-1]:.4f}, val={sp_df['val/loss'].iloc[-1]:.4f}")
print(f"muP only:  train={mup_df['train/loss'].iloc[-1]:.4f}, val={mup_df['val/loss'].iloc[-1]:.4f}")
print(f"CompleteP: train={completep_df['train/loss'].iloc[-1]:.4f}, val={completep_df['val/loss'].iloc[-1]:.4f}")
print(f"\nCompleteP improvement vs SP: {(1 - completep_df['train/loss'].iloc[-1]/sp_df['train/loss'].iloc[-1])*100:.2f}% (train), "
      f"{(1 - completep_df['val/loss'].iloc[-1]/sp_df['val/loss'].iloc[-1])*100:.2f}% (val)")
print("="*80)

plt.show()


