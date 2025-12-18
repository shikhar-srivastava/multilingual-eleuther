import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def prepare_tokenizer_for_padding(tokenizer: PreTrainedTokenizerBase) -> int:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return tokenizer.pad_token_id

def get_prepend_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    if getattr(tokenizer, "cls_token_id", None) is not None:
        return tokenizer.cls_token_id
    if tokenizer.bos_token_id is not None:
        return tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    return tokenizer.unk_token_id


def wrap_model_for_multi_gpu(model: PreTrainedModel) -> PreTrainedModel:
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return torch.nn.DataParallel(model)
    return model

def build_batch_from_token_ids(
    token_id_sequences: List[List[int]],
    pad_token_id: int,
    unk_token_id: Optional[int],
    max_length: int,
) -> Dict[str, torch.Tensor]:
    truncated = [seq[:max_length] for seq in token_id_sequences]
    seq_lengths = [len(seq) for seq in truncated]
    batch_size = len(truncated)
    max_len = max(seq_lengths) if seq_lengths else 0
    max_len = min(max_len, max_length)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

    for i, seq in enumerate(truncated):
        L = min(len(seq), max_len)
        if L == 0:
            continue
        input_ids[i, :L] = torch.tensor(seq[:L], dtype=torch.long)
        attention_mask[i, :L] = 1
        labels[i, :L] = input_ids[i, :L]
        if unk_token_id is not None:
            labels[i, :L][labels[i, :L] == unk_token_id] = -100

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def create_model_and_tokenizer(
    model_id: str,
    revision: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, str]:
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None
    )
    model = wrap_model_for_multi_gpu(model)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _left_truncate_pair(prompt_ids: List[int], choice_ids: List[int], max_length: int) -> Tuple[List[int], List[int]]:
    total = len(prompt_ids) + len(choice_ids)
    if total <= max_length:
        return prompt_ids, choice_ids
    # Keep the tail so that the choice is preserved; drop from the start of prompt
    overflow = total - max_length
    if overflow >= len(prompt_ids):
        # If prompt is fully truncated, keep only the tail part of choice
        cut_choice = choice_ids[overflow - len(prompt_ids) :]
        return [], cut_choice
    else:
        cut_prompt = prompt_ids[overflow:]
        return cut_prompt, choice_ids


def build_cloze_batch(
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    choices_list: List[List[str]],
    max_length: int,
    ignore_unk: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    pad_id = prepare_tokenizer_for_padding(tokenizer)
    unk_id = tokenizer.unk_token_id if ignore_unk else None

    all_input_ids: List[List[int]] = []
    all_attention: List[List[int]] = []
    all_labels: List[List[int]] = []
    group_sizes: List[int] = []

    for i in range(len(prompts)):
        prompt_ids = tokenizer.encode(prompts[i], add_special_tokens=False)
        group_count = 0
        for choice in choices_list[i]:
            choice_ids = tokenizer.encode(choice, add_special_tokens=False)
            p_ids, c_ids = _left_truncate_pair(prompt_ids, choice_ids, max_length)
            seq = p_ids + c_ids
            attn = [1] * len(seq)
            # labels: mask prompt tokens (-100), score only choice tokens; also ignore UNK if requested
            labels = [-100] * len(p_ids) + c_ids[:]
            if unk_id is not None:
                labels = [(-100 if t == unk_id else t) for t in labels]
            all_input_ids.append(seq)
            all_attention.append(attn)
            all_labels.append(labels)
            group_count += 1
        group_sizes.append(group_count)

    # Pad to batch tensors
    max_len = max((len(x) for x in all_input_ids), default=0)
    input_ids = torch.full((len(all_input_ids), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(all_input_ids), max_len), dtype=torch.long)
    labels = torch.full((len(all_labels), max_len), -100, dtype=torch.long)

    for i, (seq, attn, lab) in enumerate(zip(all_input_ids, all_attention, all_labels)):
        L = len(seq)
        input_ids[i, :L] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, :L] = torch.tensor(attn, dtype=torch.long)
        labels[i, :L] = torch.tensor(lab, dtype=torch.long)

    return input_ids, attention_mask, labels, group_sizes


def compute_mean_nll_per_sequence(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # logits: [B, T, V]; labels: [B, T]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    # mask
    active = shift_labels != -100
    if active.sum().item() == 0:
        return torch.zeros((logits.size(0),), dtype=logits.dtype, device=logits.device)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    target_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    target_log_probs = torch.where(active, target_log_probs, torch.zeros_like(target_log_probs))
    token_counts = active.sum(dim=-1).clamp_min(1)
    nll = -target_log_probs.sum(dim=-1) / token_counts
    return nll


def compute_choice_losses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    choices_list: List[List[str]],
    max_length: int = 1024,
    batch_size: int = 32,
    ignore_unk: bool = True,
) -> List[List[float]]:
    input_ids, attention_mask, labels, group_sizes = build_cloze_batch(
        tokenizer, prompts, choices_list, max_length=max_length, ignore_unk=ignore_unk
    )

    losses: List[float] = []
    device = next(model.parameters()).device if not isinstance(model, torch.nn.DataParallel) else model.module.device

    for start in range(0, input_ids.size(0), batch_size):
        end = min(start + batch_size, input_ids.size(0))
        batch_in = input_ids[start:end].to(device)
        batch_attn = attention_mask[start:end].to(device)
        batch_labels = labels[start:end].to(device)
        with torch.no_grad():
            outputs = model(input_ids=batch_in, attention_mask=batch_attn)
            per_seq_nll = compute_mean_nll_per_sequence(outputs.logits, batch_labels)
            losses.extend(per_seq_nll.detach().float().cpu().tolist())

    # Unflatten into groups per example
    result: List[List[float]] = []
    idx = 0
    for g in group_sizes:
        result.append(losses[idx : idx + g])
        idx += g
    return result
