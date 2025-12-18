import re
import logging
from typing import Generator, Iterable, List, Optional

from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

CHECKPOINT_PATTERN = re.compile(r"^checkpoint-epoch_(\d+)_step_(\d+)$")


def iter_models(
    owner: str,
    monolingual_datasets: Iterable[str],
    tokenizer_types: Iterable[str],
    vocab_sizes: Iterable[int],
    base_name: str = "mono_gold_130m_pre_lr1e-4",
) -> Generator[str, None, None]:
    """Yield expected model repo ids for the provided grid.

    Example output:
      shikhar-srivastava/mono_gold_130m_pre_lr1e-4_tha_thai_bpe_unscaled_8192
    """
    for dataset in monolingual_datasets:
        for tok in tokenizer_types:
            for vs in vocab_sizes:
                repo_id = f"{owner}/{base_name}_{dataset}_{tok}_{vs}"
                yield repo_id


def list_revisions(repo_id: str, api: Optional[HfApi] = None) -> List[str]:
    """List all revision names (refs) for a repo id.

    Returns tag names including checkpoints and possibly "main".
    """
    api = api or HfApi()
    try:
        refs = api.list_repo_refs(repo_id)
    except Exception as exc:
        logger.warning("Failed to list refs for %s: %s", repo_id, exc)
        return []

    tags = [t.name for t in refs.tags or []]
    branches = [b.name for b in refs.branches or []]
    # Prefer tags for checkpoints, but include branches for completeness
    return sorted(set(tags + branches))


def filter_checkpoint_revisions(revisions: Iterable[str]) -> List[str]:
    """Return only revisions that match the checkpoint pattern.

    Example: checkpoint-epoch_1_step_500
    """
    return [r for r in revisions if CHECKPOINT_PATTERN.match(r)]


def has_model(repo_id: str, api: Optional[HfApi] = None) -> bool:
    api = api or HfApi()
    try:
        card = api.model_info(repo_id)
        return card is not None
    except Exception as exc:
        logger.info("Model not found or inaccessible %s: %s", repo_id, exc)
        return False
