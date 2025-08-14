<div align="center">
  
# The Curse of Depth in Large Language Models
[[`Arxiv`](https://arxiv.org/abs/2502.05795)] 
[[X(Twitter)](https://x.com/Shiwei_Liu66/status/1889257901346152844)]
[[Model](https://huggingface.co/pengxiang/LNS_1B)]
[[Talk](https://www.youtube.com/watch?v=sVN7wgmmNms)]

We present the Curse of Depth, a phenomenon in Large Language Models (LLMs) where deeper layers contribute less effectively to training due to the widespread use of Pre-Layer Normalization (Pre-LN). Our analysis identifies this issue as a key bottleneck in LLM optimization and proposes LayerNorm Scaling as a solution to mitigate its impact.

<div align="center">
  <img src="scaling.png" alt="Image 2" style="width: 900px; margin: 0 auto;">
</div>
</div>

## Abstract

In this paper, we introduce the Curse of Depth, a concept that highlights, explains, and addresses the recent observation in modern Large Language Models (LLMs) where nearly half of the layers are less effective than expected. We first confirm the wide existence of this phenomenon across the most popular families of LLMs such as Llama, Mistral, DeepSeek, and Qwen. Our analysis, theoretically and empirically, identifies that the underlying reason for the ineffectiveness of deep layers in LLMs is the widespread usage of Pre-Layer Normalization (Pre-LN). While Pre-LN stabilizes the training of Transformer LLMs, its output variance exponentially grows with the model depth, which undesirably causes the derivative of the deep Transformer blocks to be an identity matrix, and therefore barely contributes to the training. To resolve this training pitfall, we propose LayerNorm Scaling, which scales the variance of output of the layer normalization inversely by the square root of its depth. This simple modification mitigates the output variance explosion of deeper Transformer layers, improving their contribution. Our experimental results, spanning model sizes from 130M to 1B, demonstrate that LayerNorm Scaling significantly enhances LLM pre-training performance compared to Pre-LN. Moreover, this improvement seamlessly carries over to supervised fine-tuning. All these gains can be attributed to the fact that LayerNorm Scaling enables deeper layers to contribute more effectively during training.

## Caveat 

Combining LNS with Scaled Initialization (which scales the initialization of W0 and W2 by the overall depth $1/\sqrt{2L}$) undermines the effectiveness of LNS, performing worse than using LNS alone. This highlights the importance of eliminating conflicting initialization strategies before adopting LNS.

<div align="center">
  <img src="scaled_init.png" alt="Image 2" style="width: 600px; margin: 0 auto;">
</div>
</div>

## Hugging Face
We have uploaded the trained weights for the 1B model using LayerNorm Scaling (LNS). 
You can download them from [https://huggingface.co/pengxiang/LNS_1B].

## Quick Start

### Install experiment dependencies

You can configure the environment using the following command lines:

```bash
conda create -n LNS python=3.9 -y
conda activate LNS
pip install -r exp_requirements.txt
```

### Training Examples
We provide scripts to train models of different sizes using Pre-LN, Post-LN, Mix-LN, and LayerNorm Scaling (LNS).

Train a 130M Model:
```bash
bash run_130m.sh pre      3   # Pre-LN
bash run_130m.sh post     3   # Post-LN
bash run_130m.sh post_pre 3   # Mix-LN
bash run_130m.sh LNS      3   # LayerNorm Scaling (LNS)

(Note: 3 represents the number of Post-LN layers in Mix-LN.)
```


Train a 250M Mode:
```bash
bash run_250m.sh pre      6   # Pre-LN
bash run_250m.sh post     6   # Post-LN
bash run_250m.sh post_pre 6   # Mix-LN
bash run_250m.sh LNS      6   # LayerNorm Scaling (LNS)

(Note: 6 represents the number of Post-LN layers in Mix-LN.)
```

Train a 350M Mode:
```bash
bash run_350m.sh pre      6   # Pre-LN
bash run_350m.sh post     6   # Post-LN
bash run_350m.sh post_pre 6   # Mix-LN
bash run_350m.sh LNS      6   # LayerNorm Scaling (LNS)
```

Train a 1B Mode:
```bash
bash run_1b.sh pre        6   # Pre-LN
bash run_1b.sh post       6   # Post-LN
bash run_1b.sh post_pre   6   # Mix-LN
bash run_1b.sh LNS        6   # LayerNorm Scaling (LNS)
```

## Normalization Variants

This repository implements 15 different normalization variants for experimental research. All variants are implemented in [`peft_pretraining/modeling_llama.py`](peft_pretraining/modeling_llama.py).

| Variant | Description | Constructor Lines | Forward Lines |
|---------|-------------|-------------------|---------------|
| `pre` | Standard Pre-LayerNorm (Pre-LN) - applies normalization before attention/MLP | [L291](peft_pretraining/modeling_llama.py#L291) | [L404](peft_pretraining/modeling_llama.py#L404) |
| `post` | Standard Post-LayerNorm (Post-LN) - applies normalization after attention/MLP | [L294](peft_pretraining/modeling_llama.py#L294) | [L473](peft_pretraining/modeling_llama.py#L473) |
| `deeppost` | Post-LN with DeepNet-style residual scaling (2√2 factor) | [L294](peft_pretraining/modeling_llama.py#L294) | [L473](peft_pretraining/modeling_llama.py#L473) |
| `LNS` / `lns` | **LayerNorm Scaling** - Pre-LN with depth-dependent scaling (1/√layer_index) | [L291](peft_pretraining/modeling_llama.py#L291) | [L426](peft_pretraining/modeling_llama.py#L426) |
| `scale_pre` | Pre-LN with scaled weight initialization (1/√2L) | [L291](peft_pretraining/modeling_llama.py#L291) | [L404](peft_pretraining/modeling_llama.py#L404) |
| `sandwich` | Sandwich normalization - both pre and post norms around attention/MLP | [L297](peft_pretraining/modeling_llama.py#L297) | [L504](peft_pretraining/modeling_llama.py#L504) |
| `post_pre` | **Mix-LN** - First POST_NUM layers use post-norm, remaining use pre-norm | [L322](peft_pretraining/modeling_llama.py#L322) | [L603](peft_pretraining/modeling_llama.py#L603) |
| `scale_post_pre` | Mix-LN with scaled initialization for pre-norm layers | [L322](peft_pretraining/modeling_llama.py#L322) | [L603](peft_pretraining/modeling_llama.py#L603) |
| `pre_post` | First 7 layers use pre-norm, remaining use post-norm (hardcoded) | [L302](peft_pretraining/modeling_llama.py#L302) | [L525](peft_pretraining/modeling_llama.py#L525) |
| `pre_sandwich` | First 7 layers use pre-norm, remaining use sandwich norm (hardcoded) | [L311](peft_pretraining/modeling_llama.py#L311) | [L563](peft_pretraining/modeling_llama.py#L563) |
| `scale_res_pre_norm` | Pre-LN with learnable residual scaling factors (separate for attention/MLP) | [L331](peft_pretraining/modeling_llama.py#L331) | [L455](peft_pretraining/modeling_llama.py#L455) |
| `pre_post_pre_post` | Alternating pre/post every 4 layers (3 pre, 1 post pattern) | [L336](peft_pretraining/modeling_llama.py#L336) | ⚠️ *No forward implementation* |
| `post_pre_post_pre` | Alternating post/pre every 4 layers (3 post, 1 pre pattern) | [L345](peft_pretraining/modeling_llama.py#L345) | [L769](peft_pretraining/modeling_llama.py#L769) |
| `mono` | Pre-norm with Gumbel-Softmax routing between pre/post every 4th layer | [L354](peft_pretraining/modeling_llama.py#L354) | [L640](peft_pretraining/modeling_llama.py#L640) |
| `mono_reverse` | Post-norm with Gumbel-Softmax routing between pre/post every 4th layer | [L366](peft_pretraining/modeling_llama.py#L366) | [L704](peft_pretraining/modeling_llama.py#L704) |

### Key Notes:
- **POST_NUM Parameter**: Only `post_pre` and `scale_post_pre` use the `POST_NUM` environment variable to control the transition point between normalization strategies
- **Main Contribution**: `LNS`/`lns` is the primary contribution of this paper - LayerNorm Scaling that mitigates the curse of depth
- **Mix-LN**: Refers specifically to `post_pre` - the hybrid approach combining post-norm early layers with pre-norm later layers
- **Experimental Variants**: Many variants (`mono`, `mono_reverse`, etc.) are for research exploration of different normalization strategies

### Usage:
```bash
# Standard variants
bash run_1b.sh pre 1        # Pre-LN (POST_NUM ignored)
bash run_1b.sh post 1       # Post-LN (POST_NUM ignored) 
bash run_1b.sh LNS 1        # LayerNorm Scaling (POST_NUM ignored)

# Hybrid variants (POST_NUM matters)
bash run_1b.sh post_pre 3   # First 3 layers post-norm, rest pre-norm
bash run_1b.sh post_pre 12  # First 12 layers post-norm, rest pre-norm
```

## Performance Drop
Calculate the performance drop after removing different layers. We use [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness) to obtain evaluation results. Please refer to its installation instructions to configure `lm_eval``.
```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

Then, you can run the following command to remove different layers and save the weights to a new model. The performance drop will be calculated based on the new model:
```bash
# LLaMA2-7B, Remove Layer 1
python layer_remove.py \
    --model_path meta-llama/Llama-2-7b-hf \
    --layer_index 1 \
    --save_path ./llama_7b_removed_1
```

### 📚Citation

```bibtex
@article{sun2025curse,
  title={The Curse of Depth in Large Language Models},
  author={Sun, Wenfang and Song, Xinyuan and Li, Pengxiang and Yin, Lu and Zheng, Yefeng and Liu, Shiwei},
  journal={arXiv preprint arXiv:2502.05795},
  year={2025}
}
```


### Acknowledgement
This repository is built upon the [Mix-LN](https://github.com/pixeli99/MixLN/tree/main) repositories. Thanks for their great work!
