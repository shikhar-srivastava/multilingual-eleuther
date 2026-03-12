"""
Load machine-local path configuration from local.env at the repo root.

local.env is not committed to git and must be created on each machine.
It must define DATA_ROOT — the base directory for training data, tokenized
data, and tokenizers on that machine.

Example local.env:
    DATA_ROOT=/scratch/ssrivas9/catherinearnett
"""
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent


def _load_local_env() -> dict:
    env_file = _REPO_ROOT / "local.env"
    if not env_file.exists():
        raise FileNotFoundError(
            f"local.env not found at {env_file}. "
            "Create it with: DATA_ROOT=/path/to/your/data/base"
        )
    config = {}
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            config[k.strip()] = v.strip()
    return config


def get_data_root() -> str:
    """Return the DATA_ROOT path defined in local.env."""
    cfg = _load_local_env()
    if "DATA_ROOT" not in cfg:
        raise KeyError("DATA_ROOT not defined in local.env")
    return cfg["DATA_ROOT"]


def resolve_path(path: str) -> str:
    """Replace ${DATA_ROOT} placeholder in a path string with the actual value."""
    return path.replace("${DATA_ROOT}", get_data_root())
