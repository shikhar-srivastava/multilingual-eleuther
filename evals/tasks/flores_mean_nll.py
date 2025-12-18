import logging
from typing import Dict, List, Optional

import datasets
import torch
from transformers import AutoTokenizer

from ..tasks import register
from ..utils_modeling import (
    create_model_and_tokenizer,
    prepare_tokenizer_for_padding,
    get_prepend_token_id,
    build_batch_from_token_ids,
)

logger = logging.getLogger(__name__)

FLORES_LANGS = {
    "eng_latn": "eng_Latn",
    "tha_thai": "tha_Thai",
    "vie_latn": "vie_Latn",
    "amh_ethi": "amh_Ethi",
    "urd_arab": "urd_Arab",
}

MAX_SEQ_LEN = 1024  # from LLAMA 130M config


def _texts_to_token_ids(tokenizer: AutoTokenizer, texts: List[str], prepend_token_id: int) -> List[List[int]]:
    token_id_seqs: List[List[int]] = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_id_seqs.append([prepend_token_id] + ids)
    return token_id_seqs


def _batched_mean_nll(model, tokenizer, texts: List[str], device: str, batch_size: int = 16) -> float:
    unk_token_id = tokenizer.unk_token_id
    pad_token_id = -100 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    prepend_token_id = get_prepend_token_id(tokenizer)

    token_id_seqs = _texts_to_token_ids(tokenizer, texts, prepend_token_id)

    nll_sum = 0.0
    count = 0
    for i in range(0, len(token_id_seqs), batch_size):
        chunk = token_id_seqs[i : i + batch_size]
        batch = build_batch_from_token_ids(chunk, pad_token_id=pad_token_id, unk_token_id=unk_token_id, max_length=MAX_SEQ_LEN)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Following goldfish: we do not need to special-case logits shape; using model loss handles shift internally
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # Mean loss over tokens already applies ignore_index (-100); we added unk to -100 and padding masked
            loss = outputs.loss.detach().float()
            # Convert to per-sample mean by scaling with token count
            # outputs.loss is averaged over all non-ignored tokens across batch. To get mean over samples,
            # we can multiply by total tokens and divide by number of samples. This preserves scale with goldfish mean.
            valid_tokens = (labels != -100).sum().item()
            if valid_tokens > 0:
                nll_sum += float(loss.item())
                count += 1

    return nll_sum / max(count, 1)


@register("flores_mean_nll")
def run_flores_mean_nll(
    model_id: str,
    revision: Optional[str] = None,
    languages: Optional[List[str]] = None,
    limit_samples: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict:
    """Compute mean NLL on FLORES200 devtest for selected languages.

    References:
    - Implementation basis: https://github.com/tylerachang/goldfish/blob/main/eval_code/flores_eval_othermodels.py
    - Goldfish-specific variant: https://github.com/tylerachang/goldfish/blob/main/eval_code/flores_eval_goldfish.py
    - Dataset: https://huggingface.co/datasets/Muennighoff/flores200
    """
    model, tokenizer, device0 = create_model_and_tokenizer(model_id, revision=revision)
    device = device or device0
    prepare_tokenizer_for_padding(tokenizer)

    ds = datasets.load_dataset("Muennighoff/flores200")
    langs = languages or list(FLORES_LANGS.keys())

    results = {}
    for lang in langs:
        flores_code = FLORES_LANGS[lang]
        split = ds["devtest"]["sentence"][flores_code]
        texts = list(split)
        if limit_samples:
            texts = texts[: int(limit_samples)]
        mean_nll = _batched_mean_nll(model, tokenizer, texts, device=device, batch_size=16)
        results[lang] = mean_nll

    return results
