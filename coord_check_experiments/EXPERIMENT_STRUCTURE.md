# Experiment Data Structure

This document clarifies which experiments are used for which plots.

## Directory Structure

```
coord_check_experiments/
├── OLD DEPTH EXPERIMENTS (baseline - using standard LLAMA architecture)
│   ├── sp_and_mup/          # 6 depths (2,4,8,16,32,64) - muP without depth scaling
│   ├── completep/           # 6 depths (2,4,8,16,32,64) - CompleteP α=1.0
│   └── depth_alpha_05/      # 6 depths (2,4,8,16,32,64) - CompleteP α=0.5
│
└── NEW WIDTH EXPERIMENTS (architecture ablations)
    └── width_scaling/
        ├── sp/              # 5 widths (128,256,512,1024,2048) - Standard Param
        ├── mup/             # 5 widths (128,256,512,1024,2048) - muP
        ├── completep/       # 5 widths - CompleteP baseline (RMSNorm+SwiGLU+RoPE)
        ├── completep_layernorm/    # 5 widths - CompleteP + LayerNorm
        ├── completep_gelu/         # 5 widths - CompleteP + GELU
        ├── completep_learned_pos/  # 5 widths - CompleteP + Learned PosEmb
        └── completep_gpt2like/     # 5 widths - CompleteP + all GPT-2 components
```

## Plot Mappings

### Depth Coordinate Check Plots (from OLD experiments)
- `depth_coord_sp_mup.png` ← `sp_and_mup/` (RMSNorm+SwiGLU+RoPE, no depth scaling)
- `depth_coord_completep_1.png` ← `completep/` (RMSNorm+SwiGLU+RoPE, α=1.0)
- `depth_coord_completep_05.png` ← `depth_alpha_05/` (RMSNorm+SwiGLU+RoPE, α=0.5)

**Architecture**: All use standard LLAMA (RMSNorm + SwiGLU + RoPE)
**Variable**: Depth (2, 4, 8, 16, 32, 64 layers)
**Fixed**: Width=256

### Width Coordinate Check Plots (from NEW experiments)
- `width_coord_sp.png` ← `width_scaling/sp/`
- `width_coord_mup.png` ← `width_scaling/mup/`
- `width_coord_completep.png` ← `width_scaling/completep/` (baseline)
- `width_coord_completep_ln.png` ← `width_scaling/completep_layernorm/`
- `width_coord_completep_gelu.png` ← `width_scaling/completep_gelu/`
- `width_coord_completep_learned.png` ← `width_scaling/completep_learned_pos/`
- `width_coord_completep_gpt2.png` ← `width_scaling/completep_gpt2like/`

**Variable**: Width (128, 256, 512, 1024, 2048)
**Fixed**: Depth=6 layers
**Architecture**: Varies by experiment (testing LayerNorm, GELU, Learned Pos, etc.)

## Data Isolation

✅ **Depth plots** and **width plots** use completely separate data sources.

✅ **No mixing** between old baseline depth experiments and new architecture width experiments.

✅ Each plot clearly shows which experiment it represents.

## If You Want Depth Ablations

If you want to test architecture ablations across depths (e.g., CompleteP with LayerNorm at different depths), you would need to run new depth experiments for each architecture variant. Currently:

- Depth experiments: Only baseline LLAMA architecture
- Width experiments: Multiple architecture variants

To add depth ablations, run:
```bash
# Example: CompleteP + LayerNorm depth scaling
# (Would need to create these scripts)
./depth_scaling/run_completep_layernorm.sh
```

This would create:
- `depth_scaling/completep_layernorm/` with 6 depths
- New plot: `depth_coord_completep_ln.png`

