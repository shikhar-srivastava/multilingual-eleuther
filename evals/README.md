# Evals

This package provides a reproducible evaluation framework for multilingual models and their checkpoints hosted on Hugging Face Hub.

What it includes:
- Orchestrator CLI to evaluate multiple models/checkpoints across tasks
- Task implementations/wrappers (FLORES mean NLL, Belebele, GlobalMMLU cloze, LM Harness tasks, Winograd)
- Hugging Face Hub utilities to discover models and list checkpoint revisions
- Results manager to atomically persist cumulative results in `results.json`
- Progress table generator to produce `PROGRESS.md`

## Quick start

```bash
# (Optional) create a venv
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r evals/requirements.txt

# Dry run: list models/checkpoints that would be evaluated
python -m evals.orchestrator --dry-run

# Run a small evaluation on a subset
python -m evals.orchestrator \
  --tasks flores_mean_nll belebele \
  --languages eng_latn tha_thai \
  --limit-samples 200

# Generate/update the progress table
python -m evals.progress_table
```

## Results JSON format

Results are stored at `evals/results.json` as a single cumulative JSON object:

```json
{
  "shikhar-srivastava/mono_gold_130m_pre_lr1e-4_tha_thai_bpe_unscaled_8192": {
    "final": { "flores_mean_nll": {"tha_thai": 2.31, "eng_latn": 1.98}, "timestamp": "2025-01-01T00:00:00Z" },
    "checkpoint-epoch_1_step_500": { "belebele_acc": {"tha_thai": 0.47} }
  }
}
```

- Top-level keys are model ids
- Under each model, keys are either `final` or checkpoint tags (e.g., `checkpoint-epoch_1_step_500`)
- Each checkpoint contains one or more task result sections and a `timestamp`

## Notes
- Missing models or tasks without required packages are skipped gracefully and logged in the JSON under a `skipped` reason.
- Checkpoint revisions are discovered as tags on the Hugging Face repo whose names match `checkpoint-epoch_.*_step_.*`.
- Use `--skip-if-exists` to avoid re-computing existing results entries.
