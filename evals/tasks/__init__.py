from typing import Dict, Callable

TaskFn = Callable[..., Dict]

TASK_REGISTRY: Dict[str, TaskFn] = {}


def register(task_name: str):
    def _decorator(fn: TaskFn) -> TaskFn:
        TASK_REGISTRY[task_name] = fn
        return fn

    return _decorator
