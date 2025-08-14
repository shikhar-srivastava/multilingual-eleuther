"""
Byte Consumption Plotting Utilities for C4 Dataset Training

This module provides utilities to create visualizations of UTF-8 byte consumption
during model training, particularly useful for cross-tokenizer comparisons.
"""

import matplotlib.pyplot as plt
import wandb
import numpy as np
from typing import Dict, List, Optional, Tuple


def create_byte_consumption_plot(
    steps: List[int],
    cumulative_bytes: List[int],
    cumulative_tokens: List[int],
    model_name: str = "Model",
    tokenizer_name: str = "Tokenizer"
) -> plt.Figure:
    """
    Create a comprehensive byte consumption plot.
    
    Args:
        steps: List of training steps
        cumulative_bytes: List of cumulative bytes consumed
        cumulative_tokens: List of cumulative tokens processed
        model_name: Name of the model being trained
        tokenizer_name: Name of the tokenizer being used
    
    Returns:
        matplotlib Figure object
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Convert bytes to GB for readability
    cumulative_gb = [b / (1024**3) for b in cumulative_bytes]
    
    # Plot 1: Cumulative bytes over time
    ax1.plot(steps, cumulative_gb, 'b-', linewidth=2, label='UTF-8 Bytes (GB)')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Cumulative Bytes (GB)')
    ax1.set_title(f'C4 Dataset Byte Consumption - {model_name}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Bytes per token ratio over time
    if cumulative_tokens and len(cumulative_tokens) == len(cumulative_bytes):
        bytes_per_token = [b/max(1, t) for b, t in zip(cumulative_bytes, cumulative_tokens)]
        ax2.plot(steps, bytes_per_token, 'g-', linewidth=2, label='Bytes/Token')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Bytes per Token')
        ax2.set_title(f'Compression Ratio - {tokenizer_name}')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'Token data not available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Compression Ratio (No Token Data)')
    
    # Plot 3: Byte throughput (derivative)
    if len(steps) > 1:
        byte_throughput = []
        step_throughput = []
        for i in range(1, len(steps)):
            if steps[i] - steps[i-1] > 0:
                throughput = (cumulative_bytes[i] - cumulative_bytes[i-1]) / (steps[i] - steps[i-1])
                byte_throughput.append(throughput / (1024**2))  # Convert to MB per step
                step_throughput.append(steps[i])
        
        if byte_throughput:
            ax3.plot(step_throughput, byte_throughput, 'r-', linewidth=2, label='MB/Step')
            ax3.set_xlabel('Training Steps')
            ax3.set_ylabel('Bytes per Step (MB)')
            ax3.set_title('Byte Consumption Rate')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
    
    # Plot 4: Cumulative statistics
    if cumulative_bytes:
        final_gb = cumulative_gb[-1]
        final_tokens = cumulative_tokens[-1] if cumulative_tokens else 0
        final_ratio = cumulative_bytes[-1] / max(1, final_tokens) if cumulative_tokens else 0
        
        stats_text = f"""Final Statistics:
        
Total Bytes: {final_gb:.2f} GB
Total Tokens: {final_tokens:,}
Avg Bytes/Token: {final_ratio:.2f}
Tokenizer: {tokenizer_name}
Model: {model_name}"""
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Training Summary')
    
    plt.tight_layout()
    return fig


def log_byte_consumption_to_wandb(
    step: int,
    bytes_seen: int,
    tokens_seen: int,
    model_config: Dict,
    tokenizer_name: str,
    create_plot: bool = False,
    current_loss: Optional[float] = None
):
    """
    Log byte consumption metrics to wandb with optional plotting.
    
    Args:
        step: Current training step
        bytes_seen: Total bytes consumed so far
        tokens_seen: Total tokens processed so far
        model_config: Model configuration dictionary
        tokenizer_name: Name of the tokenizer
        create_plot: Whether to create and log a plot
        current_loss: Current training loss for BPB estimation
    """
    import math
    
    # Basic metrics
    metrics = {
        "dataset/bytes_total_GB": bytes_seen / (1024**3),
        "dataset/bytes_total_MB": bytes_seen / (1024**2),
        "dataset/tokens_total": tokens_seen,
        "dataset/bytes_per_token": bytes_seen / max(1, tokens_seen),
    }
    
    # Model-specific metrics for comparison
    model_size = model_config.get('num_hidden_layers', 'unknown')
    hidden_size = model_config.get('hidden_size', 'unknown')
    
    metrics.update({
        f"dataset/bytes_per_M_params": bytes_seen / (model_config.get('total_params_M', 1) * 1024**2),
        f"dataset/compression_ratio": tokens_seen / max(1, bytes_seen),  # tokens per byte
    })
    
    # Add tokenizer comparison metrics
    metrics.update({
        f"tokenizers/{tokenizer_name}/bytes_per_token": bytes_seen / max(1, tokens_seen),
        f"tokenizers/{tokenizer_name}/total_bytes_GB": bytes_seen / (1024**3),
        f"tokenizers/{tokenizer_name}/compression_efficiency": tokens_seen / max(1, bytes_seen),
    })
    
    # Add training BPB estimation if loss is provided
    if current_loss is not None and bytes_seen > 0:
        # Estimate training BPB (rough approximation)
        total_loss_estimate = current_loss * tokens_seen
        byte_level_loss_estimate = total_loss_estimate / bytes_seen
        training_bpb_estimate = byte_level_loss_estimate / math.log(2)
        
        metrics.update({
            f"tokenizers/{tokenizer_name}/training_bpb_estimate": training_bpb_estimate,
            "dataset/training_bpb_estimate": training_bpb_estimate,
        })
    
    wandb.log(metrics, step=step)
    
    # Create visualization if requested
    if create_plot and step > 0:
        try:
            # This would need historical data - for now just log current state
            model_name = f"{model_size}L-{hidden_size}H"
            
            # Create a simple current state plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Create a simple bar chart of current metrics
            metric_names = ['Total GB', 'Bytes/Token', 'Compression\nRatio']
            metric_values = [
                bytes_seen / (1024**3),
                bytes_seen / max(1, tokens_seen),
                tokens_seen / max(1, bytes_seen) * 1000  # Scale for visibility
            ]
            
            bars = ax.bar(metric_names, metric_values, 
                         color=['blue', 'green', 'orange'], alpha=0.7)
            
            ax.set_title(f'C4 Dataset Consumption - Step {step}\n{model_name} with {tokenizer_name}')
            ax.set_ylabel('Value')
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Log to wandb
            wandb.log({
                "dataset/consumption_summary": wandb.Image(fig),
            }, step=step)
            
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Failed to create byte consumption plot: {e}")


def create_tokenizer_comparison_plot(
    tokenizer_data: Dict[str, Dict],
    model_name: str = "Model"
) -> plt.Figure:
    """
    Create a comparison plot across different tokenizers.
    
    Args:
        tokenizer_data: Dict mapping tokenizer names to their metrics
        model_name: Name of the model for the title
    
    Returns:
        matplotlib Figure object
    """
    if not tokenizer_data:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, 'No tokenizer data available', 
               ha='center', va='center', transform=ax.transAxes)
        return fig
    
    tokenizer_names = list(tokenizer_data.keys())
    
    # Extract metrics
    total_bytes = [data.get('total_bytes', 0) / (1024**3) for data in tokenizer_data.values()]  # GB
    bytes_per_token = [data.get('bytes_per_token', 0) for data in tokenizer_data.values()]
    compression_ratios = [data.get('compression_ratio', 0) for data in tokenizer_data.values()]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Total bytes consumed
    bars1 = ax1.bar(tokenizer_names, total_bytes, color='skyblue', alpha=0.7)
    ax1.set_ylabel('Total Bytes (GB)')
    ax1.set_title('Total C4 Bytes Consumed')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars1, total_bytes):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:.2f}', ha='center', va='bottom')
    
    # Plot 2: Bytes per token
    bars2 = ax2.bar(tokenizer_names, bytes_per_token, color='lightgreen', alpha=0.7)
    ax2.set_ylabel('Bytes per Token')
    ax2.set_title('Tokenizer Efficiency')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars2, bytes_per_token):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:.2f}', ha='center', va='bottom')
    
    # Plot 3: Compression ratio
    bars3 = ax3.bar(tokenizer_names, compression_ratios, color='coral', alpha=0.7)
    ax3.set_ylabel('Tokens per Byte')
    ax3.set_title('Compression Efficiency')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars3, compression_ratios):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.suptitle(f'Tokenizer Comparison for {model_name} Training on C4', fontsize=16)
    plt.tight_layout()
    
    return fig


def log_final_byte_summary(
    final_bytes: int,
    final_tokens: int,
    training_steps: int,
    model_config: Dict,
    tokenizer_name: str,
    training_time_hours: float,
    final_bpb: Optional[float] = None
):
    """
    Log a comprehensive final summary of byte consumption.
    
    Args:
        final_bytes: Total bytes consumed
        final_tokens: Total tokens processed  
        training_steps: Total training steps
        model_config: Model configuration
        tokenizer_name: Tokenizer used
        training_time_hours: Total training time in hours
        final_bpb: Final bits per byte metric from evaluation
    """
    import math
    
    # Calculate comprehensive metrics
    summary_metrics = {
        "final_summary/total_bytes_GB": final_bytes / (1024**3),
        "final_summary/total_tokens_M": final_tokens / 1_000_000,
        "final_summary/bytes_per_token": final_bytes / max(1, final_tokens),
        "final_summary/tokens_per_byte": final_tokens / max(1, final_bytes),
        "final_summary/bytes_per_step": final_bytes / max(1, training_steps),
        "final_summary/tokens_per_step": final_tokens / max(1, training_steps),
        "final_summary/bytes_per_hour": final_bytes / max(1, training_time_hours),
        "final_summary/GB_per_hour": (final_bytes / (1024**3)) / max(1, training_time_hours),
    }
    
    # Model-specific metrics
    total_params = model_config.get('total_params_M', 1)
    summary_metrics.update({
        "final_summary/bytes_per_M_params": final_bytes / (total_params * 1024**2),
        "final_summary/GB_per_M_params": (final_bytes / (1024**3)) / total_params,
    })
    
    # Create summary table data
    table_data = [
        ["Total Bytes", f"{final_bytes / (1024**3):.2f}", "GB"],
        ["Total Tokens", f"{final_tokens / 1_000_000:.2f}", "M"],
        ["Bytes per Token", f"{final_bytes / max(1, final_tokens):.3f}", "bytes"],
        ["Compression Ratio", f"{final_tokens / max(1, final_bytes):.6f}", "tokens/byte"],
        ["Training Steps", f"{training_steps:,}", "steps"],
        ["Training Time", f"{training_time_hours:.2f}", "hours"],
        ["Model Size", f"{total_params:.1f}", "M params"],
        ["Tokenizer", tokenizer_name, "-"],
        ["Bytes/Hour", f"{(final_bytes / (1024**3)) / max(1, training_time_hours):.2f}", "GB/hour"],
    ]
    
    # Add BPB metrics if available
    if final_bpb is not None:
        byte_level_perplexity = math.exp(final_bpb * math.log(2))
        table_data.extend([
            ["Bits per Byte (BPB)", f"{final_bpb:.6f}", "bits"],
            ["Byte-level Perplexity", f"{byte_level_perplexity:.2f}", "-"],
        ])
        summary_metrics.update({
            f"final_summary/{tokenizer_name}_final_bpb": final_bpb,
            "final_summary/byte_level_perplexity": byte_level_perplexity,
        })
    
    # Create a summary table for wandb
    summary_table = wandb.Table(
        columns=["Metric", "Value", "Unit"],
        data=table_data
    )
    
    summary_metrics["final_summary/training_summary"] = summary_table
    
    wandb.log(summary_metrics)
    
    print(f"\n{'='*60}")
    print(f"FINAL C4 DATASET CONSUMPTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total UTF-8 Bytes Consumed: {final_bytes:,} bytes ({final_bytes/(1024**3):.2f} GB)")
    print(f"Total Tokens Processed: {final_tokens:,} tokens")
    print(f"Bytes per Token: {final_bytes/max(1, final_tokens):.3f}")
    print(f"Compression Ratio: {final_tokens/max(1, final_bytes):.6f} tokens/byte")
    if final_bpb is not None:
        print(f"Final Bits per Byte (BPB): {final_bpb:.6f}")
        print(f"Byte-level Perplexity: {math.exp(final_bpb * math.log(2)):.2f}")
    print(f"Model: {model_config.get('num_hidden_layers', '?')} layers, {total_params:.1f}M parameters")
    print(f"Tokenizer: {tokenizer_name}")
    print(f"Training Steps: {training_steps:,}")
    print(f"Training Time: {training_time_hours:.2f} hours")
    print(f"Data Rate: {(final_bytes/(1024**3))/max(1, training_time_hours):.2f} GB/hour")
    print(f"{'='*60}\n")


def create_bpb_comparison_plot(
    tokenizer_results: Dict[str, Dict],
    model_name: str = "Model"
) -> plt.Figure:
    """
    Create a bits per byte comparison plot across tokenizers.
    
    Args:
        tokenizer_results: Dict mapping tokenizer names to their BPB results
                          Format: {"tokenizer_name": {"bpb": float, "bytes_per_token": float, "eval_loss": float}}
        model_name: Name of the model for the title
    
    Returns:
        matplotlib Figure object
    """
    import math
    
    if not tokenizer_results:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, 'No BPB data available', 
               ha='center', va='center', transform=ax.transAxes)
        return fig
    
    tokenizer_names = list(tokenizer_results.keys())
    
    # Extract metrics
    bpb_values = [data.get('bpb', 0) for data in tokenizer_results.values()]
    bytes_per_token = [data.get('bytes_per_token', 0) for data in tokenizer_results.values()]
    eval_losses = [data.get('eval_loss', 0) for data in tokenizer_results.values()]
    perplexities = [math.exp(data.get('bpb', 0) * math.log(2)) if data.get('bpb') else 0 
                   for data in tokenizer_results.values()]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Bits per Byte comparison
    bars1 = ax1.bar(tokenizer_names, bpb_values, color='lightcoral', alpha=0.7)
    ax1.set_ylabel('Bits per Byte')
    ax1.set_title('Tokenizer-Independent Performance (Lower is Better)')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars1, bpb_values):
        if value > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{value:.4f}', ha='center', va='bottom')
    
    # Plot 2: Byte-level perplexity
    bars2 = ax2.bar(tokenizer_names, perplexities, color='lightblue', alpha=0.7)
    ax2.set_ylabel('Byte-level Perplexity')
    ax2.set_title('Byte-level Perplexity (Lower is Better)')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars2, perplexities):
        if value > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{value:.2f}', ha='center', va='bottom')
    
    # Plot 3: Bytes per token (efficiency)
    bars3 = ax3.bar(tokenizer_names, bytes_per_token, color='lightgreen', alpha=0.7)
    ax3.set_ylabel('Bytes per Token')
    ax3.set_title('Tokenization Efficiency')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars3, bytes_per_token):
        if value > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{value:.2f}', ha='center', va='bottom')
    
    # Plot 4: Traditional eval loss (for reference)
    bars4 = ax4.bar(tokenizer_names, eval_losses, color='gold', alpha=0.7)
    ax4.set_ylabel('Evaluation Loss')
    ax4.set_title('Traditional Token-based Loss (For Reference)')
    ax4.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars4, eval_losses):
        if value > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{value:.3f}', ha='center', va='bottom')
    
    plt.suptitle(f'Bits Per Byte Analysis for {model_name} Training on C4', fontsize=16)
    plt.tight_layout()
    
    return fig 