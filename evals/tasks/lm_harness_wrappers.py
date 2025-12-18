import logging
from typing import Dict, List, Optional

from lm_eval import evaluator, tasks as lm_tasks

from ..tasks import register

logger = logging.getLogger(__name__)

# Note: Requires lm-eval installed. We will use HF-based model from CLI via lm-eval's hf-causal model interface.
# For simplicity, we route through lm_eval.evaluator with model_args specifying model & revision.


def _run_lm_eval(model_id: str, revision: Optional[str], task_names: List[str], limit: Optional[int]) -> Dict[str, float]:
    model_args = f"pretrained={model_id}"
    if revision:
        model_args += f",revision={revision}"
    if limit:
        # Few-shot limit for speed: use --limit argument emulation
        _limit = int(limit)
    else:
        _limit = None

    results = evaluator.simple_evaluate(
        model="hf-causal",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=0,
        limit=_limit,
    )

    out: Dict[str, float] = {}
    task_results = results.get("results", {})
    for task, metrics in task_results.items():
        # choose accuracy if available, else take first metric
        if "acc" in metrics:
            out[task] = float(metrics["acc"])
        else:
            for k, v in metrics.items():
                out[task] = float(v)
                break
    return out


@register("lm_harness")
def run_lm_harness(
    model_id: str,
    revision: Optional[str] = None,
    tasks: Optional[List[str]] = None,
    limit_samples: Optional[int] = None,
) -> Dict:
    # Default collection to cover what's requested
    default_tasks = [
        # MultiBLiMP
        "multiblimp",
        # XQuAD
        "xquad_en",
        "xquad_vi",
        "xquad_th",
        # XCOPA
        "xcopa_en",
        "xcopa_vi",
        "xcopa_th",
        # HellaSwag
        "hellaswag_en",
        "okapi_hellaswag_vi",
        # ARC
        "arc_easy",
        "arc_challenge",
        "okapi_arc_vi",
    ]
    task_list = tasks or default_tasks
    return _run_lm_eval(model_id, revision, task_list, limit_samples)
