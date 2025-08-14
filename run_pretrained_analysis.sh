#!/bin/bash

# Pre-trained Model Analysis Launcher
# This script launches different types of analysis based on the research papers

set -e

# Default parameters
ANALYSIS_TYPE="moh_comparison"
OUTPUT_DIR="./analysis_results"
WANDB_PROJECT="pretrained-model-analysis"
MAX_BATCHES=150
BATCH_SIZE=8

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --analysis-type)
            ANALYSIS_TYPE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --max-batches)
            MAX_BATCHES="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --analysis-type TYPE     Type of analysis to run (scaling_laws, moe_attention, attention_scaling, blow_up_detection)"
            echo "  --output-dir DIR         Output directory for results (default: ./analysis_results)"
            echo "  --wandb-project PROJECT  WandB project name (default: pretrained-model-analysis)"
            echo "  --max-batches N          Maximum number of C4 batches to process (default: 100)"
            echo "  --batch-size N           Batch size for processing (default: 8)"
            echo "  --help                   Show this help message"
            echo ""
            echo "Available analysis types:"
            echo "  moh_comparison          - Compare MoH-LLaMA3-8B vs LLaMA3-8B baseline"
            echo "  dcformer_comparison     - Compare DCFormer-2.8B vs Pythia-2.8B baseline"
            echo "  blow_up_detection       - Focus on detecting numerical instabilities across all models"
            echo "  comprehensive           - Run all comparison types"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Pre-trained Model Analysis"
echo "=========================================="
echo "Analysis Type: $ANALYSIS_TYPE"
echo "Output Directory: $OUTPUT_DIR"
echo "WandB Project: $WANDB_PROJECT"
echo "Max Batches: $MAX_BATCHES"
echo "Batch Size: $BATCH_SIZE"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run analysis with specific configuration
run_analysis() {
    local analysis_name=$1
    local models=$2
    local track_every=$3
    local run_name="${analysis_name}_$(date +%Y%m%d_%H%M%S)"
    
    echo "Running $analysis_name analysis..."
    echo "Models: $models"
    echo "Run name: $run_name"
    
    python analyze_pretrained_models.py \
        --models $models \
        --analysis_config "$analysis_name" \
        --max_batches "$MAX_BATCHES" \
        --batch_size "$BATCH_SIZE" \
        --track_every "$track_every" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run "$run_name" \
        --output_dir "$OUTPUT_DIR/$analysis_name"
}

# Run analysis based on type
case $ANALYSIS_TYPE in
    "moh_comparison")
        echo "Comparing MoH-LLaMA3-8B vs LLaMA3-8B baseline..."
        run_analysis "moh_comparison" "moh_llama3_8b llama3_8b_baseline" 10
        ;;
    
    "dcformer_comparison")
        echo "Comparing DCFormer-2.8B vs Pythia-2.8B baseline..."
        run_analysis "dcformer_comparison" "dcformer_2_8b pythia_2_8b_baseline" 10
        ;;
    
    "blow_up_detection")
        echo "Analyzing blow-up patterns across all models..."
        run_analysis "blow_up_detection" "moh_llama3_8b llama3_8b_baseline dcformer_2_8b pythia_2_8b_baseline" 5
        ;;
    
    "comprehensive")
        echo "Running comprehensive analysis..."
        
        # Run MoH comparison
        echo "Step 1/3: MoH vs LLaMA3 Baseline Comparison"
        run_analysis "moh_comparison" "moh_llama3_8b llama3_8b_baseline" 10
        
        # Run DCFormer comparison
        echo "Step 2/3: DCFormer vs Pythia Baseline Comparison"
        run_analysis "dcformer_comparison" "dcformer_2_8b pythia_2_8b_baseline" 10
        
        # Run blow-up detection
        echo "Step 3/3: Blow-up Detection Analysis"
        run_analysis "blow_up_detection" "moh_llama3_8b llama3_8b_baseline dcformer_2_8b pythia_2_8b_baseline" 5
        
        echo "Comprehensive analysis complete!"
        ;;
    
    *)
        echo "Unknown analysis type: $ANALYSIS_TYPE"
        echo "Available types: moh_comparison, dcformer_comparison, blow_up_detection, comprehensive"
        exit 1
        ;;
esac

echo "Analysis complete! Results saved to: $OUTPUT_DIR"
echo "Check WandB project '$WANDB_PROJECT' for detailed logs and visualizations."

# Generate summary report
echo "Generating summary report..."
cat > "$OUTPUT_DIR/analysis_summary.md" << EOF
# Pre-trained Model Analysis Summary

**Analysis Type:** $ANALYSIS_TYPE  
**Date:** $(date)  
**Max Batches:** $MAX_BATCHES  
**Batch Size:** $BATCH_SIZE  

## Analysis Configuration

Based on the following research papers:

1. **Scaling Laws for Neural Language Models** (Kaplan et al., 2020)
   - Paper: https://arxiv.org/abs/2003.02436
   - Focus: Power-law relationships between model size, compute, and performance

2. **MoH: Multi-Head Attention as Mixture-of-Head Attention** (2024)
   - Paper: https://arxiv.org/abs/2410.11842  
   - Focus: Treating attention heads as experts with dynamic routing

3. **Multi-Head Mixture-of-Experts** (2024)
   - Paper: https://arxiv.org/abs/2405.08553
   - Focus: Splitting tokens into sub-tokens processed by different experts

4. **SEAL: Scaling to Emphasize Attention for Long-Context Retrieval** (2025)
   - Paper: https://arxiv.org/abs/2501.15225
   - Focus: Scalar gains on attention heads for long-context processing

## Results

Results are saved in subdirectories and logged to WandB project: \`$WANDB_PROJECT\`

### Key Metrics Tracked

- **Activation Statistics:** Mean, std, min, max, L2 norm per layer
- **Weight Distributions:** Distribution statistics for attention heads and MLP components  
- **Attention Patterns:** Head specialization, routing patterns, scaling effects
- **Blow-up Detection:** Inf/NaN detection, gradient explosion patterns
- **MoE Analysis:** Expert utilization, load balancing, routing behavior

### Files Generated

- \`final_results.json\`: Complete analysis results
- \`*_results.json\`: Per-model analysis results
- WandB logs with detailed metrics and visualizations

## Next Steps

1. Review WandB dashboards for interactive visualizations
2. Analyze JSON results for quantitative patterns
3. Compare scaling behaviors across different model architectures
4. Investigate any detected blow-up patterns or anomalies

EOF

echo "Summary report saved to: $OUTPUT_DIR/analysis_summary.md"