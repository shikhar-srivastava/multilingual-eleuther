import json
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")


@contextmanager
def atomic_write(path: str):
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            yield f
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def load_results(path: str = RESULTS_PATH) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def save_results(data: Dict[str, Any], path: str = RESULTS_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with atomic_write(path) as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


def update_result(
    model_id: str,
    revision: str,
    task_key: str,
    result_obj: Dict[str, Any],
    path: str = RESULTS_PATH,
    skip_if_exists: bool = True,
) -> Dict[str, Any]:
    """Merge a task result into the cumulative results structure.

    Structure:
    {
      model_id: {
        revision: { task_key: {...}, "timestamp": ISO8601 }
      }
    }
    """
    data = load_results(path)
    data.setdefault(model_id, {})
    if skip_if_exists and model_id in data and revision in data[model_id] and task_key in data[model_id][revision]:
        return data

    rev_entry: Dict[str, Any] = data[model_id].setdefault(revision, {})
    rev_entry.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    rev_entry[task_key] = result_obj

    save_results(data, path)
    return data


def record_skip(model_id: str, revision: str, reason: str, path: str = RESULTS_PATH) -> None:
    data = load_results(path)
    entry = data.setdefault(model_id, {}).setdefault(revision, {})
    entry.setdefault("skipped", reason)
    save_results(data, path)
