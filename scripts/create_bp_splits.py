"""
Create byte-premium (BP) sized train/eval splits for monolingual datasets.

For each dataset, we:
- Write the first BP GiB of bytes to {output_dir}/{dataset}_{bp}.txt
- Write the last round(8000 * BP) lines to {output_dir}/{dataset}_{bp}_eval.txt

After generating files, update a JSON index in the codebase:
  configs/monolingual_bp_index.json

Usage:
  python scripts/create_bp_splits.py \
    --input_root /localdisk/ssrivas9/catherinearnett/monolingual_training_data \
    --output_root /localdisk/ssrivas9/catherinearnett/monolingual_training_data_bp

Notes:
- BP sizes are interpreted in GiB (1024**3 bytes) for deterministic slicing.
- Handles files smaller than the requested BP size by copying the whole file.
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, Tuple


DATASET_TO_BP: Dict[str, float] = {
    "eng_latn": 1.0,
    "tha_thai": 2.74,
    "urd_arab": 1.71,
    "amh_ethi": 1.72,
    "vie_latn": 1.35,
}


def _copy_first_bytes(src_path: str, dst_path: str, num_bytes: int, chunk_size: int = 4 * 1024 * 1024) -> int:
    """Copy the first num_bytes from src_path to dst_path. Returns bytes written."""
    written = 0
    with open(src_path, "rb") as fin, open(dst_path, "wb") as fout:
        remaining = num_bytes
        while remaining > 0:
            to_read = min(chunk_size, remaining)
            data = fin.read(to_read)
            if not data:
                break
            fout.write(data)
            written += len(data)
            remaining -= len(data)
    return written


def _tail_last_n_lines(src_path: str, n_lines: int, chunk_size: int = 1024 * 1024) -> bytes:
    """Return the last n_lines from file as bytes, including newlines, UTF-8 safe."""
    if n_lines <= 0:
        return b""
    with open(src_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        buffer = bytearray()
        pos = file_size
        lines_found = 0
        while pos > 0 and lines_found <= n_lines:
            read_size = min(chunk_size, pos)
            pos -= read_size
            f.seek(pos)
            chunk = f.read(read_size)
            buffer[:0] = chunk  # prepend
            lines_found = buffer.count(b"\n")
            if pos == 0:
                break
        # Split and take last n_lines
        parts = buffer.split(b"\n")
        # If file ends without newline, last element is trailing text; we still count it as a line.
        tail_parts = parts[-n_lines:]
        data = b"\n".join(tail_parts)
        # Ensure trailing newline like `tail -n` typically produces
        if not data.endswith(b"\n"):
            data += b"\n"
        return data


def build_paths(input_root: str, output_root: str, dataset: str, bp: float) -> Tuple[str, str, str]:
    src = os.path.join(input_root, f"{dataset}.txt")
    train_out = os.path.join(output_root, f"{dataset}_{bp}.txt")
    eval_out = os.path.join(output_root, f"{dataset}_{bp}_eval.txt")
    return src, train_out, eval_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--codebase_root", type=str, default="/localdisk/ssrivas9/multilingual-eleuther")
    parser.add_argument("--decimal_gb", action="store_true", help="Use 1e9 bytes per GB instead of GiB (1024**3).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = args.input_root
    output_root = args.output_root
    os.makedirs(output_root, exist_ok=True)

    bytes_per_unit = 1024 ** 3  # GiB by default
    if args.decimal_gb:
        bytes_per_unit = int(1e9)

    index: Dict[str, Dict] = {}

    for dataset, bp in DATASET_TO_BP.items():
        src_path, train_out_path, eval_out_path = build_paths(input_root, output_root, dataset, bp)
        if not os.path.isfile(src_path):
            print(f"[WARN] Missing source file: {src_path}")
            continue

        # Train split: first BP units of bytes.
        target_bytes = int(bp * bytes_per_unit)
        file_size = os.path.getsize(src_path)
        if target_bytes > file_size:
            print(f"[INFO] Requested {target_bytes} bytes > file size {file_size}. Copying full file for {dataset}.")
            target_bytes = file_size

        if not os.path.exists(train_out_path) or args.overwrite:
            written = _copy_first_bytes(src_path, train_out_path, target_bytes)
            print(f"[OK] Wrote {written} bytes to {train_out_path}")
        else:
            print(f"[SKIP] Exists: {train_out_path}")

        # Eval split: last round(8000 * BP) lines.
        n_eval_lines = max(1, int(round(8000 * bp)))
        if not os.path.exists(eval_out_path) or args.overwrite:
            data = _tail_last_n_lines(src_path, n_eval_lines)
            with open(eval_out_path, "wb") as fout:
                fout.write(data)
            print(f"[OK] Wrote last {n_eval_lines} lines to {eval_out_path}")
        else:
            print(f"[SKIP] Exists: {eval_out_path}")

        index[dataset] = {
            "byte_premium": bp,
            "train_path": train_out_path,
            "train_bytes": target_bytes,
            "eval_lines": n_eval_lines,
            "eval_path": eval_out_path,
        }

    # Write/update index in codebase configs.
    config_dir = os.path.join(args.codebase_root, "configs")
    os.makedirs(config_dir, exist_ok=True)
    index_path = os.path.join(config_dir, "monolingual_bp_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote index: {index_path}")


if __name__ == "__main__":
    main()


