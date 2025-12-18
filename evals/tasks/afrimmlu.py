import logging
from typing import Dict, List, Optional

import datasets
import torch

from ..tasks import register
from ..utils_modeling import create_model_and_tokenizer, compute_choice_losses

logger = logging.getLogger(__name__)

# Dataset: https://huggingface.co/datasets/masakhane/afrimmlu
# Language: Amharic (amh_ethi)


@register("afrimmlu")
def run_afrimmlu(
    model_id: str,
    revision: Optional[str] = None,
    languages: Optional[List[str]] = None,
    limit_samples: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict:
    model, tokenizer, _device = create_model_and_tokenizer(model_id, revision=revision)
    ds = datasets.load_dataset("masakhane/afrimmlu")
    split = ds.get("test") or ds.get("validation") or ds.get("dev") or ds.get("train")
    if split is None:
        raise RuntimeError("AfriMMLU: no split found")

    if "language" in split.column_names:
        split = split.filter(lambda ex: str(ex["language"]).lower().startswith("am"))

    prompts: List[str] = []
    choices_list: List[List[str]] = []
    labels: List[int] = []

    for ex in split:
        question = ex.get("question", "")
        context = ex.get("context", "")
        choices = ex.get("choices") or [ex.get("A", ""), ex.get("B", ""), ex.get("C", ""), ex.get("D", "")]
        choices = [c for c in choices if isinstance(c, str)]
        if not choices:
            continue
        label = ex.get("answer") or ex.get("label") or ex.get("correct")
        if isinstance(label, str):
            label = {"A": 0, "B": 1, "C": 2, "D": 3}.get(label.strip().upper(), 0)
        answer_idx = int(label)
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
    return {"amh_ethi": correct / max(len(labels), 1)}
