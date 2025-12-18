import argparse
import logging
from typing import List, Optional

# Ensure task modules are imported so they register themselves
from .tasks import TASK_REGISTRY  # noqa: F401
from .tasks import (
    flores_mean_nll as _t1,  # noqa: F401
    belebele_wrapper as _t2,  # noqa: F401
    globalmmlu_cloze as _t3,  # noqa: F401
    winograd as _t4,  # noqa: F401
    lm_harness_wrappers as _t5,  # noqa: F401
    indicmmlu_pro as _t6,  # noqa: F401
    afrimmlu as _t7,  # noqa: F401
    xcomps as _t8,  # noqa: F401
)

from .hf_utils import iter_models, list_revisions, filter_checkpoint_revisions, has_model
from .results_store import update_result, record_skip

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run evaluations across models and checkpoints")
    p.add_argument("--owner", default="shikhar-srivastava")
    p.add_argument("--base-name", default="mono_gold_130m_pre_lr1e-4")
    p.add_argument("--monolingual-datasets", nargs="*", default=["eng_latn", "tha_thai", "urd_arab", "amh_ethi", "vie_latn"])  # noqa
    p.add_argument("--tokenizer-types", nargs="*", default=["bpe_unscaled", "unigram_unscaled"])  # noqa
    p.add_argument("--vocab-sizes", nargs="*", type=int, default=[8192, 16384, 32768, 49152, 65536, 81920, 98304, 114688, 262144])  # noqa

    p.add_argument(
        "--tasks",
        nargs="*",
        default=[
            "flores_mean_nll",
            "belebele",
            "globalmmlu_cloze",
            "indicmmlu_pro",
            "afrimmlu",
            "winograd",
            "xcomps",
        ],
    )
    p.add_argument("--lm-harness", action="store_true", help="Also run LM eval harness wrappers")

    p.add_argument("--limit-samples", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-if-exists", action="store_true")
    p.add_argument("--only-final", action="store_true", help="Only evaluate final (main) revision")

    return p.parse_args()


def _run_task(task_name: str, model_id: str, revision: Optional[str], limit_samples: Optional[int]):
    fn = TASK_REGISTRY.get(task_name)
    if not fn:
        raise ValueError(f"Unknown task: {task_name}")
    kwargs = {"model_id": model_id, "revision": revision, "limit_samples": limit_samples}
    return fn(**kwargs)


def main():
    args = parse_args()

    all_models = list(
        iter_models(
            owner=args.owner,
            monolingual_datasets=args.monolingual_datasets,
            tokenizer_types=args.tokenizer_types,
            vocab_sizes=args.vocab_sizes,
            base_name=args.base_name,
        )
    )

    logger.info("Planned %d models.", len(all_models))

    for model_id in all_models:
        if args.dry_run:
            logger.info("[DRY] Would process model %s", model_id)
            continue
        if not has_model(model_id):
            logger.info("Model missing, skipping: %s", model_id)
            record_skip(model_id, "final", "model_missing")
            continue

        revisions = list_revisions(model_id)
        checkpoints = filter_checkpoint_revisions(revisions)
        final_revisions: List[str] = ["final"]
        if not args.only_final:
            final_revisions.extend(sorted(checkpoints))

        for rev in final_revisions:
            revision = None if rev == "final" else rev
            logger.info("Evaluating %s @ %s", model_id, rev)

            for task in args.tasks:
                try:
                    result = _run_task(task, model_id, revision, args.limit_samples)
                except Exception as exc:
                    logger.exception("Task %s failed for %s@%s: %s", task, model_id, rev, exc)
                    record_skip(model_id, rev, f"task_failed:{task}")
                    continue
                update_result(model_id, rev, task, result, skip_if_exists=args.skip_if_exists)

            if args.lm_harness:
                try:
                    from .tasks.lm_harness_wrappers import run_lm_harness

                    lm_out = run_lm_harness(model_id, revision=revision, limit_samples=args.limit_samples)
                    update_result(model_id, rev, "lm_harness", lm_out, skip_if_exists=args.skip_if_exists)
                except Exception as exc:
                    logger.exception("LM Harness failed for %s@%s: %s", model_id, rev, exc)
                    record_skip(model_id, rev, "lm_harness_failed")


if __name__ == "__main__":
    main()
