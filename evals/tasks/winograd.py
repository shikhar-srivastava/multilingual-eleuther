import logging
from typing import Dict, Optional

import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..tasks import register

logger = logging.getLogger(__name__)

# Thai: https://huggingface.co/datasets/pakphum/winograd_th
# English: https://huggingface.co/datasets/allenai/winogrande


def _score_pair(model, tokenizer, prompt: str, option1: str, option2: str, device: str) -> int:
    scores = []
    for option in [option1, option2]:
        text = prompt.replace("_", option)
        with torch.no_grad():
            enc = tokenizer(text, return_tensors="pt")
            input_ids = enc.input_ids.to(device)
            attn = enc.attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
            scores.append(outputs.loss.detach().float().item())
    return 0 if scores[0] <= scores[1] else 1


@register("winograd")
def run_winograd(
    model_id: str,
    revision: Optional[str] = None,
    languages: Optional[list] = None,
    limit_samples: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None
    )
    model.to(device)
    model.eval()

    langs = languages or ["eng_latn", "tha_thai"]
    results: Dict[str, float] = {}

    if "eng_latn" in langs:
        ds_en = datasets.load_dataset("allenai/winogrande", "winogrande_xl")
        correct = 0
        total = 0
        for ex in ds_en["validation"]:
            prompt = ex["sentence"].replace("@pronoun", "_")
            option1, option2 = ex["option1"], ex["option2"]
            answer_idx = int(ex["answer"]) - 1
            pred = _score_pair(model, tokenizer, prompt, option1, option2, device)
            correct += 1 if pred == answer_idx else 0
            total += 1
            if limit_samples and total >= limit_samples:
                break
        results["eng_latn"] = correct / max(total, 1)

    if "tha_thai" in langs:
        ds_th = datasets.load_dataset("pakphum/winograd_th")
        correct = 0
        total = 0
        for ex in ds_th["test"]:
            prompt = ex["sentence"].replace("@pronoun", "_")
            option1, option2 = ex["option1"], ex["option2"]
            answer_idx = int(ex["answer"]) - 1
            pred = _score_pair(model, tokenizer, prompt, option1, option2, device)
            correct += 1 if pred == answer_idx else 0
            total += 1
            if limit_samples and total >= limit_samples:
                break
        results["tha_thai"] = correct / max(total, 1)

    return results
