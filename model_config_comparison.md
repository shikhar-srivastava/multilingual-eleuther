# LLaMA Model Configuration Comparison

## Main Model Scale Variations

| Model | Hidden Size (N) | Layers (L) | N/L Ratio | Attention Heads | Intermediate Size | Vocab Size | Max Seq Length |
|-------|----------------|-----------|-----------|-----------------|-------------------|------------|----------------|
| llama_9m | 128 | 4 | 32.0 | 4 | 352 | 32,000 | 1024 |
| llama_20m | 256 | 4 | 64.0 | 4 | 688 | 32,000 | 1024 |
| llama_35m | 384 | 6 | 64.0 | 8 | 1,024 | 32,000 | 1024 |
| llama_40m | 416 | 8 | 52.0 | 8 | 1,024 | 32,000 | 1024 |
| llama_60m | 512 | 8 | 64.0 | 8 | 1,376 | 32,000 | 1024 |
| llama_71m | 512 | 12 | 42.67 | 8 | 1,368 | 32,000 | 1024 |
| llama_100m | 640 | 12 | 53.33 | 10 | 1,708 | 32,100 | 1024 |
| llama_130m | 768 | 12 | 64.0 | 12 | 2,048 | 32,000 | 1024 |
| llama_250m | 768 | 24 | 32.0 | 16 | 2,560 | 32,000 | 1024 |
| llama_350m | 1,024 | 24 | 42.67 | 16 | 2,736 | 32,000 | 1024 |
| llama_1b | 2,048 | 24 | 85.33 | 32 | 5,461 | 32,000 | 1024 |
| llama_3b | 2,560 | 32 | 80.0 | 32 | 6,848 | 32,000 | 1024 |
| llama_7b | 4,096 | 32 | 128.0 | 32 | 11,008 | 32,000 | 2048 |

## Scaling Patterns Analysis

### Width (Hidden Size) Progression
- **9M → 20M**: 2× increase (128 → 256)
- **20M → 35M**: 1.5× increase (256 → 384)
- **35M → 40M**: 1.08× increase (384 → 416)
- **40M → 60M**: 1.23× increase (416 → 512)
- **60M → 71M**: No change (512)
- **71M → 100M**: 1.25× increase (512 → 640)
- **100M → 130M**: 1.2× increase (640 → 768)
- **130M → 250M**: No change (768)
- **250M → 350M**: 1.33× increase (768 → 1,024)
- **350M → 1B**: 2× increase (1,024 → 2,048)
- **1B → 3B**: 1.25× increase (2,048 → 2,560)
- **3B → 7B**: 1.6× increase (2,560 → 4,096)

### Depth (Layers) Progression
- **9M → 20M**: No change (4 layers)
- **20M → 35M**: 1.5× increase (4 → 6)
- **35M → 40M**: 1.33× increase (6 → 8)
- **40M → 60M**: No change (8 layers)
- **60M → 71M**: 1.5× increase (8 → 12)
- **71M → 130M**: No change (12 layers)
- **130M → 250M**: 2× increase (12 → 24)
- **250M → 350M**: No change (24 layers)
- **350M → 1B**: No change (24 layers)
- **1B → 3B**: 1.33× increase (24 → 32)
- **3B → 7B**: No change (32 layers)

### N/L Ratio Analysis
- **Lowest ratio**: 32.0 (llama_9m, llama_250m)
- **Highest ratio**: 128.0 (llama_7b)
- **Most common ratio**: 64.0 (appears in 20M, 35M, 60M, 130M models)
- **Pattern observation**: Smaller models tend to balance width and depth more evenly, while larger models increasingly favor width over depth.

### Attention Heads Progression
- **4 heads**: 9M, 20M (smallest models)
- **8 heads**: 35M, 40M, 60M, 71M (small-to-medium models)
- **10 heads**: 100M (unique configuration)
- **12 heads**: 130M (medium model)
- **16 heads**: 250M, 350M (large models)
- **32 heads**: 1B, 3B, 7B (largest models)

### Intermediate Size Analysis
The intermediate size (FFN hidden dimension) generally follows the pattern:
- **FFN/Hidden ratio** varies between ~2.67× to ~2.75× for most models
- Notable exceptions in smaller models where the ratio is slightly different

## Positional Encoding Variants (130M base)

All four variants of the 130M model share the same architectural parameters (768 hidden size, 12 layers, 12 attention heads) but differ in positional encoding:

| Variant | Position Embedding Type | Notes |
|---------|------------------------|-------|
| llama_130m | rope (RoPE) | Standard rotary position embeddings (default for LLaMA) |
| llama_130m_learned_pos | learned | Learned position embeddings |
| llama_130m_no_pos | none | No positional encoding |
| llama_130m_sinusoidal_pos | sinusoidal | Fixed sinusoidal position embeddings |

### Common Parameters for all 130M variants:
- **Hidden Size**: 768
- **Layers**: 12
- **Attention Heads**: 12
- **Intermediate Size**: 2,048
- **Vocab Size**: 32,000
- **Max Sequence Length**: 1,024

## Key Configuration Parameters (Constant Across Models)

The following parameters remain constant across all model scales:

- **hidden_act**: `silu` (Swish/SiLU activation function)
- **initializer_range**: 0.02
- **rms_norm_eps**: 1e-06 (RMSNorm epsilon)
- **bos_token_id**: 0
- **eos_token_id**: 1
- **pad_token_id**: -1
- **model_type**: `llama`
- **architectures**: `LLaMAForCausalLM`

## Notable Exceptions

1. **Vocabulary Size**: 
   - Most models use 32,000
   - llama_100m uses 32,100 (slightly larger)

2. **Max Sequence Length**:
   - Most models use 1,024
   - llama_7b uses 2,048 (doubled context length)

3. **Position Embedding**:
   - Only llama_130m and its variants explicitly specify `position_embedding_type`
   - Standard config uses RoPE (rotary position embeddings) implicitly

## Architectural Insights

1. **Scaling Strategy**: The model family uses both width and depth scaling, with a tendency to increase width more aggressively in larger models (note the increasing N/L ratio from 32.0 to 128.0).

2. **Attention Head Scaling**: Attention heads scale less aggressively than hidden size, with head dimension per attention head increasing as models grow.

3. **FFN Ratio**: The intermediate size (FFN hidden dimension) maintains a relatively consistent ratio to hidden size (~2.67×), following the standard transformer scaling pattern.

4. **Memory-Efficient Small Models**: The smallest models (9M-40M) use shallower architectures (4-8 layers) to keep memory footprint low while still maintaining reasonable model capacity.

5. **Compute-Efficient Medium Models**: The 60M-350M range balances depth and width more carefully, with N/L ratios between 32-64.

6. **Capacity-Focused Large Models**: The 1B+ models prioritize width (hidden dimension) over depth, with N/L ratios of 80-128, likely for better parallelization and computational efficiency.

