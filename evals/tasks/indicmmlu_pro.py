import logging
from typing import Dict, List, Optional

import datasets
import torch

from ..tasks import register
from ..utils_modeling import (
    create_model_and_tokenizer,
    prepare_tokenizer_for_padding,
    get_prepend_token_id,
    build_batch_from_token_ids,
)

logger = logging.getLogger(__name__)

# Dataset: https://huggingface.co/datasets/LinguaLift/IndicMMLU-Pro
# Language: Urdu (urd_arab)


def _label_to_index(label) -> int:
    if isinstance(label, int):
        return label
    if isinstance(label, str):
        m = {"A": 0, "B": 1, "C": 2, "D": 3}
        return m.get(label.strip().upper(), 0)
    return 0


def _score_batched(
    model,
    tokenizer,
    prompts: List[str],
    choices_list: List[List[str]],
    device: str,
    batch_size: int = 16,
) -> List[int]:
    prepend_id = get_prepend_token_id(tokenizer)
    unk_id = tokenizer.unk_token_id
    pad_id = -100 if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    preds: List[int] = []
    for i in range(0, len(prompts)):
        prompt = prompts[i]
        choices = choices_list[i]
        scores = []
        for c in choices:
            text = prompt + c
            ids = tokenizer.encode(text, add_special_tokens=False)
            token_ids = [[prepend_id] + ids]
            batch = build_batch_from_token_ids(token_ids, pad_token_id=pad_id, unk_token_id=unk_id, max_length=1024)
            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                scores.append(outputs.loss.detach().float().item())
        pred = int(min(range(len(scores)), key=lambda j: scores[j]))
        preds.append(pred)
    return preds


@register("indicmmlu_pro")
def run_indicmmlu_pro(
    model_id: str,
    revision: Optional[str] = None,
    languages: Optional[List[str]] = None,
    limit_samples: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict:
    device0 = device
    model, tokenizer, auto_device = create_model_and_tokenizer(model_id, revision=revision)
    device = device0 or auto_device
    prepare_tokenizer_for_padding(tokenizer)

    ds = datasets.load_dataset("LinguaLift/IndicMMLU-Pro")
    split = None
    for candidate in ["test", "validation", "dev", "train"]:
        if candidate in ds:
            split = ds[candidate]
            break
    if split is None:
        raise RuntimeError("IndicMMLU-Pro: no split found")

    if "language" in split.column_names:
        split = split.filter(lambda ex: str(ex["language"]).lower().startswith("ur"))

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
        answer_idx = _label_to_index(label)
        prompt = f"{context}\n\n{question}\n"
        prompts.append(prompt)
        choices_list.append(choices)
        labels.append(answer_idx)
        if limit_samples and len(prompts) >= limit_samples:
            break

    preds = _score_batched(model, tokenizer, prompts, choices_list, device=device)
    correct = sum(int(p == y) for p, y in zip(preds, labels))
    total = len(labels)

    return {"urd_arab": correct / max(total, 1)}
