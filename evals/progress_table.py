import json
import os
from typing import Dict, Set

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")
PROGRESS_MD = os.path.join(os.path.dirname(__file__), "PROGRESS.md")

LANG_COLUMNS = ["English", "Vietnamese", "Thai", "Amharic", "Urdu"]
LANG_KEYS = {
    "English": "eng_latn",
    "Vietnamese": "vie_latn",
    "Thai": "tha_thai",
    "Amharic": "amh_ethi",
    "Urdu": "urd_arab",
}

TASK_ROWS = [
    ("Benchmark 1", "flores_mean_nll"),
    ("Benchmark 2", "belebele"),
    ("Benchmark 3", "globalmmlu_cloze"),
    # Additional rows could summarize lm_harness subtasks if desired
]


def _load_results() -> Dict:
    if not os.path.exists(RESULTS_PATH):
        return {}
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def _collect_coverage(results: Dict) -> Dict[str, Set[str]]:
    # Return mapping task_key -> set of language keys covered by any model/revision
    coverage: Dict[str, Set[str]] = {}
    for _model, revs in results.items():
        for _rev, entries in revs.items():
            for task_key, task_res in entries.items():
                if task_key in ("timestamp", "skipped"):
                    continue
                if isinstance(task_res, dict):
                    langs_present = {k for k in task_res.keys() if k in LANG_KEYS.values()}
                else:
                    langs_present = set()
                s = coverage.setdefault(task_key, set())
                s.update(langs_present)
    return coverage


def generate_md():
    results = _load_results()
    coverage = _collect_coverage(results)

    lines = []
    lines.append("# Evaluation Progress\n")
    lines.append("\n")

    # Header
    header = [""] + LANG_COLUMNS
    sep = ["---"] * (len(LANG_COLUMNS) + 1)
    lines.append(" | ".join(header) + "\n")
    lines.append(" | ".join(sep) + "\n")

    for row_name, task_key in TASK_ROWS:
        row = [row_name]
        cov = coverage.get(task_key, set())
        for lang_col in LANG_COLUMNS:
            lang_key = LANG_KEYS[lang_col]
            cell = "✅" if lang_key in cov else ""
            row.append(cell)
        lines.append(" | ".join(row) + "\n")

    with open(PROGRESS_MD, "w", encoding="utf-8") as f:
        f.writelines(lines)


if __name__ == "__main__":
    generate_md()
