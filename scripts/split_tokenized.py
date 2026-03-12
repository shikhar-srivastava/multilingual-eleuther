"""
Split a fully-tokenized ints-per-line file into train and eval portions.

Two splitting modes (exactly one of --target_tokens or --target_bytes must be given):

  Token-count mode (--target_tokens):
      Reads lines from the beginning of the input and writes them to the
      train output until the cumulative token count reaches the target.
      Prints the byte size of the resulting train file.
      Use this for English (FineWeb).

  Byte-count mode (--target_bytes):
      Reads lines from the beginning and writes them to the train output
      until the train file reaches the target byte size.
      Prints the token count of the resulting train file.
      Use this for other languages (future).

In both modes, the last --eval_lines lines from the FULL input are written
to the eval output.  Since train lines come from the beginning and eval
lines from the end, they are guaranteed disjoint as long as the input is
large enough.

Usage (English / token-count):
    python scripts/split_tokenized.py \
        --input  full_tokenized.txt \
        --train_output train_tokenized.txt \
        --eval_output  eval_tokenized.txt \
        --target_tokens 7000000000 \
        --eval_lines 8000

Usage (other language / byte-count):
    python scripts/split_tokenized.py \
        --input  full_tokenized.txt \
        --train_output train_tokenized.txt \
        --eval_output  eval_tokenized.txt \
        --target_bytes 12345678 \
        --eval_lines 8000
"""

import argparse
import os
import sys
from collections import deque


def count_tokens_in_line(line: str) -> int:
    """Count space-separated token IDs in a single ints-per-line entry."""
    stripped = line.strip()
    if not stripped:
        return 0
    return stripped.count(" ") + 1


def extract_train_by_tokens(input_path: str, output_path: str, target_tokens: int) -> tuple:
    """Write lines from the start of input until target_tokens is reached.

    Returns (total_tokens_written, total_lines_written).
    """
    total_tokens = 0
    total_lines = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            n_tok = count_tokens_in_line(line)
            if n_tok == 0:
                continue
            fout.write(line if line.endswith("\n") else line + "\n")
            total_tokens += n_tok
            total_lines += 1
            if total_lines % 1_000_000 == 0:
                print(f"  [train-tokens] {total_lines:,} lines, {total_tokens:,} tokens")
            if total_tokens >= target_tokens:
                break
    return total_tokens, total_lines


def extract_train_by_bytes(input_path: str, output_path: str, target_bytes: int) -> tuple:
    """Write lines from the start of input until file size reaches target_bytes.

    Returns (total_tokens_written, total_lines_written, total_bytes_written).
    """
    total_tokens = 0
    total_lines = 0
    total_bytes = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            n_tok = count_tokens_in_line(line)
            if n_tok == 0:
                continue
            encoded = line if line.endswith("\n") else line + "\n"
            line_bytes = len(encoded.encode("utf-8"))
            if total_bytes + line_bytes > target_bytes and total_lines > 0:
                break
            fout.write(encoded)
            total_tokens += n_tok
            total_lines += 1
            total_bytes += line_bytes
            if total_lines % 1_000_000 == 0:
                print(f"  [train-bytes] {total_lines:,} lines, {total_bytes:,} bytes, {total_tokens:,} tokens")
    return total_tokens, total_lines, total_bytes


def extract_eval_tail(input_path: str, output_path: str, eval_lines: int) -> int:
    """Write the last eval_lines non-empty lines from input to output.

    Returns total tokens in the eval set.
    """
    ring: deque = deque(maxlen=eval_lines)
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                ring.append(line if line.endswith("\n") else line + "\n")

    total_tokens = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for line in ring:
            fout.write(line)
            total_tokens += count_tokens_in_line(line)
    return total_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Split tokenized file into train/eval by token count or byte count")
    parser.add_argument("--input", required=True, help="Full tokenized ints-per-line file")
    parser.add_argument("--train_output", required=True, help="Output path for training split")
    parser.add_argument("--eval_output", required=True, help="Output path for eval split")
    parser.add_argument("--target_tokens", type=int, default=None,
                        help="Stop writing train lines after this many tokens (token-count mode)")
    parser.add_argument("--target_bytes", type=int, default=None,
                        help="Stop writing train lines after this many bytes (byte-count mode)")
    parser.add_argument("--eval_lines", type=int, default=8000,
                        help="Number of lines to take from the end for eval")
    args = parser.parse_args()

    if (args.target_tokens is None) == (args.target_bytes is None):
        print("Error: specify exactly one of --target_tokens or --target_bytes", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.train_output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.eval_output) or ".", exist_ok=True)

    # --- Extract training split ---
    if args.target_tokens is not None:
        print(f"Extracting train split: first {args.target_tokens:,} tokens from {args.input}")
        train_tokens, train_lines = extract_train_by_tokens(
            args.input, args.train_output, args.target_tokens
        )
        train_bytes = os.path.getsize(args.train_output)
        print(f"Train split: {train_lines:,} lines, {train_tokens:,} tokens")
        print(f"Train file size: {train_bytes:,} bytes ({train_bytes / (1024**3):.4f} GiB)")
    else:
        print(f"Extracting train split: first {args.target_bytes:,} bytes from {args.input}")
        train_tokens, train_lines, train_bytes = extract_train_by_bytes(
            args.input, args.train_output, args.target_bytes
        )
        print(f"Train split: {train_lines:,} lines, {train_tokens:,} tokens")
        print(f"Train file size: {train_bytes:,} bytes ({train_bytes / (1024**3):.4f} GiB)")

    # --- Extract eval split (last N lines) ---
    print(f"Extracting eval split: last {args.eval_lines} lines from {args.input}")
    eval_tokens = extract_eval_tail(args.input, args.eval_output, args.eval_lines)
    eval_bytes = os.path.getsize(args.eval_output)
    print(f"Eval split: {args.eval_lines} lines, {eval_tokens:,} tokens, {eval_bytes:,} bytes")

    print("Done.")


if __name__ == "__main__":
    main()
