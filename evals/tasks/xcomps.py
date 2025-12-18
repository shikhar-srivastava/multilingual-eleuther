import logging
import os
from typing import Dict, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..tasks import register

logger = logging.getLogger(__name__)

# Dataset repo: https://github.com/LinyangHe/XCOMPS
# This evaluator expects an environment variable XCOMPS_PATH pointing to the cloned repo.
# It will look for English/Vietnamese CSVs and evaluate using cloze-like scoring.


def _score_choices(model, tokenizer, prompt: str, choices: List[str], device: str) -> int:
    scores = []
    for c in choices:
        text = prompt + c
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt")
            input_ids = inputs.input_ids.to(device)
            attn = inputs.attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
            scores.append(outputs.loss.detach().float().item())
    return int(min(range(len(scores)), key=lambda i: scores[i]))


@register("xcomps")
def run_xcomps(
    model_id: str,
    revision: Optional[str] = None,
    languages: Optional[List[str]] = None,
    limit_samples: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    xcomps_path = os.environ.get("XCOMPS_PATH")
    if not xcomps_path or not os.path.isdir(xcomps_path):
        # Skip gracefully
        return {"skipped": "XCOMPS_PATH not set"}

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None
    )
    model.to(device)
    model.eval()

    # Expect CSVs for English/Vietnamese
    file_map = {
        "eng_latn": os.path.join(xcomps_path, "data", "en", "xcomps_en.csv"),
        "vie_latn": os.path.join(xcomps_path, "data", "vi", "xcomps_vi.csv"),
    }

    langs = languages or list(file_map.keys())
    out: Dict[str, float] = {}
    for lang in langs:
        csv_path = file_map.get(lang)
        if not csv_path or not os.path.exists(csv_path):
            out[lang] = None
            continue
        df = pd.read_csv(csv_path)
        correct = 0
        total = 0
        for _, row in df.iterrows():
            context = row.get("context", "")
            question = row.get("question", "")
            choices = [row.get("A", ""), row.get("B", ""), row.get("C", ""), row.get("D", "")]
            answer_idx = int(row.get("label", 0))
            prompt = f"{context}\n\n{question}\n"
            pred = _score_choices(model, tokenizer, prompt, choices, device)
            correct += 1 if pred == answer_idx else 0
            total += 1
            if limit_samples and total >= limit_samples:
                break
        out[lang] = correct / max(total, 1)

    return out
