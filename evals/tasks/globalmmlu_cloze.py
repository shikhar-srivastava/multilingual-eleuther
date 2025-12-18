import logging
from typing import Dict, List, Optional

import datasets
import torch

from ..tasks import register
from ..utils_modeling import create_model_and_tokenizer, compute_choice_losses

logger = logging.getLogger(__name__)

SUPPORTED_LANGS = ["eng_latn", "amh_ethi", "vie_latn"]


@register("globalmmlu_cloze")
def run_globalmmlu_cloze(
    model_id: str,
    revision: Optional[str] = None,
    languages: Optional[List[str]] = None,
    limit_samples: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict:
    model, tokenizer, _device = create_model_and_tokenizer(model_id, revision=revision)
    ds = datasets.load_dataset("CohereLabs/Global-MMLU")
    langs = languages or SUPPORTED_LANGS

    out: Dict[str, float] = {}
    for lang in langs:
        subset = ds[lang]
        prompts: List[str] = []
        choices_list: List[List[str]] = []
        labels: List[int] = []
        for ex in subset:
            context = ex.get("context", "")
            question = ex.get("question", "")
            choices = ex.get("choices", []) or [ex.get(f"choice_{i}", "") for i in range(4)]
            answer_idx = int(ex.get("label", 0))
            prompt = f"{context}\n\n{question}\n"
            prompts.append(prompt)
            choices_list.append(choices)
            labels.append(answer_idx)
            if limit_samples and len(prompts) >= limit_samples:
                break

        per_choice_losses = compute_choice_losses(model, tokenizer, prompts, choices_list, max_length=1024, batch_size=32)
        correct = 0
        for y, losses in zip(labels, per_choice_losses):
            pred = int(min(range(len(losses)), key=lambda i: losses[i]))
            correct += 1 if pred == y else 0
        out[lang] = correct / max(len(labels), 1)

    return out
