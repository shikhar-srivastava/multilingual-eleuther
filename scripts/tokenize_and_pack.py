"""
Tokenize BP train/eval splits using the same approach as the reference script:
- Batch-wise tokenization of raw lines
- Concatenate consecutive non-empty lines into a single example until
  max_seq_len is reached or max_segments is reached
- Optionally prepend CLS and append SEP after each segment (line)
- Truncate to max_seq_len and write a single line of space-separated token ids
- No padding on disk

This mirrors: /scratch/ssrivas9/word-acquisition-language-models/scripts/tokenize_dataset.py
"""

import argparse
import json
import os
import codecs
import shutil
import subprocess
from typing import List, Optional

from transformers import AutoTokenizer, AlbertTokenizer


VALID_VOCABS = {8192, 16384, 32768, 49152, 65536, 81920, 98304, 114688, 262144}
DATASETS = ["eng_latn", "tha_thai", "urd_arab", "amh_ethi", "vie_latn"]

TOKENIZER_ROOTS = {
    "bpe_unscaled": "/scratch/ssrivas9/catherinearnett/monolingual-tokenizers/bpe_unscaled_tokenizers",
    "unigram_unscaled": "/scratch/ssrivas9/catherinearnett/monolingual-tokenizers/unigram_unscaled_tokenizers",
}


def build_tokenizer_path(dataset: str, tokenizer_type: str, vocab: int) -> str:
    root = TOKENIZER_ROOTS[tokenizer_type]
    if tokenizer_type == "bpe_unscaled":
        fname = f"bpe_{dataset}_{vocab}_300mb_unscaled.json"
    else:
        fname = f"unigram_{dataset}_{vocab}_300mb_unscaled.json"
    return os.path.join(root, fname)


MAX_STORED_LINE_COUNT = 10000


def prepare_tokenizer_dir(json_path: str) -> str:
    """
    Create a minimal HF-compatible tokenizer directory next to the given JSON tokenizer file,
    so AutoTokenizer.from_pretrained(dir) works without hard-coding special tokens.
    - copies the JSON to 'tokenizer.json'
    - generates 'special_tokens_map.json' by inspecting model.unk_token and added_tokens
    """
    base = os.path.splitext(os.path.basename(json_path))[0]
    out_dir = os.path.join(os.path.dirname(json_path), base + "_hf")
    os.makedirs(out_dir, exist_ok=True)
    # Copy tokenizer.json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(os.path.join(out_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(data, f)
    # Build special tokens map
    special_map = {}
    # unk
    try:
        unk = data.get("model", {}).get("unk_token")
        if isinstance(unk, str):
            special_map["unk_token"] = unk
    except Exception:
        pass
    # added tokens
    for tok in data.get("added_tokens", []) or []:
        if not tok.get("special", False):
            continue
        content = tok.get("content")
        if not isinstance(content, str):
            continue
        upper = content.upper()
        if "CLS" in upper and "cls_token" not in special_map:
            special_map["cls_token"] = content
        if "SEP" in upper and "sep_token" not in special_map:
            special_map["sep_token"] = content
        if "MASK" in upper and "mask_token" not in special_map:
            special_map["mask_token"] = content
        if "PAD" in upper and "pad_token" not in special_map:
            special_map["pad_token"] = content
        if "BOS" in upper and "bos_token" not in special_map:
            special_map["bos_token"] = content
        if "EOS" in upper and "eos_token" not in special_map:
            special_map["eos_token"] = content
    # If eos_token missing but sep exists, do not guess here; let model config decide if needed
    with open(os.path.join(out_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_map, f)
    return out_dir


def tokenize_file(input_path: str, output_path: str, tokenizer, max_seq_len: int,
                  max_examples: int, max_segments: int,
                  prepend_cls: bool, include_sep: bool) -> None:
    print(f"Tokenizing file: {input_path}")
    cls_token_id: Optional[int] = tokenizer.cls_token_id if prepend_cls else None
    sep_token_id: Optional[int] = tokenizer.sep_token_id if include_sep else None
    if prepend_cls and cls_token_id is None:
        print("Warning: [CLS] token does not exist.")
    if include_sep and sep_token_id is None:
        print("Warning: [SEP] token does not exist.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.isfile(output_path):
        print(f"File already exists: {output_path}")
        return

    infile = codecs.open(input_path, "rb", encoding="utf-8", errors="replace")
    outfile = codecs.open(output_path, "wb", encoding="utf-8")
    example_count = 0
    line_count = 0
    stored_lines: List[str] = []

    def tokenize_batch() -> bool:
        nonlocal stored_lines, example_count
        curr_example: List[int] = [] if cls_token_id is None else [cls_token_id]
        curr_n_segments = 0
        enc = tokenizer(stored_lines, add_special_tokens=False, truncation=True, max_length=max_seq_len)
        for tokenized_line in enc["input_ids"]:
            curr_example = curr_example + tokenized_line
            if sep_token_id is not None:
                curr_example.append(sep_token_id)
            curr_n_segments += 1
            if len(curr_example) >= max_seq_len or curr_n_segments >= max_segments:
                curr_example = curr_example[:max_seq_len]
                outfile.write(" ".join(str(t) for t in curr_example))
                outfile.write("\n")
                curr_example = [] if cls_token_id is None else [cls_token_id]
                curr_n_segments = 0
                example_count += 1
                if example_count >= max_examples:
                    print("Finished tokenization.")
                    return True
        stored_lines = []
        return False

    for line in infile:
        line_count += 1
        s = line.strip()
        if s != "":
            stored_lines.append(s)
        if line_count % MAX_STORED_LINE_COUNT == 0:
            completed = tokenize_batch()
            print(f"Processed up to line {line_count} ({example_count} examples)")
            if completed:
                break
    if len(stored_lines) > 0:
        tokenize_batch()
    outfile.close()
    infile.close()
    print(f"Finished tokenization: {example_count} examples.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--tokenizer_type", default="bpe_unscaled", choices=list(TOKENIZER_ROOTS.keys()))
    parser.add_argument("--tokenizer_vocabulary", default="32768")
    parser.add_argument("--split", choices=["train", "eval"], required=True)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--max_segments", type=int, default=-1)
    parser.add_argument("--prepend_cls", type=str, default="True")
    parser.add_argument("--include_sep", type=str, default="True")
    parser.add_argument("--index_path", default="/scratch/ssrivas9/multilingual-eleuther/configs/monolingual_bp_index.json")
    parser.add_argument("--output_root", default="/scratch/ssrivas9/catherinearnett/monolingual_training_data_tokenized")
    parser.add_argument("--shuffle", type=str, default="auto",
                        help="Shuffle output lines. 'auto' = shuffle for train only; 'True'/'False' to force.")
    args = parser.parse_args()

    vocab = int(args.tokenizer_vocabulary)
    if vocab not in VALID_VOCABS:
        raise ValueError(f"tokenizer_vocabulary must be one of {sorted(VALID_VOCABS)}")

    with open(args.index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    if args.dataset not in index:
        raise ValueError(f"Dataset {args.dataset} not found in index {args.index_path}")
    entry = index[args.dataset]
    bp = entry["byte_premium"]
    src_path = entry["train_path"] if args.split == "train" else entry["eval_path"]

    tok_path = build_tokenizer_path(args.dataset, args.tokenizer_type, vocab)
    if not os.path.isfile(tok_path):
        raise FileNotFoundError(f"Tokenizer not found: {tok_path}")
    try:
        # Prefer loading via a prepared directory so AutoTokenizer can discover special tokens cleanly
        prepared_dir = prepare_tokenizer_dir(tok_path)
        tokenizer = AutoTokenizer.from_pretrained(prepared_dir, trust_remote_code=True, use_fast=True)
    except Exception as e:
        print(f"Error loading tokenizer with AutoTokenizer: {e}")
        raise

    tokenizer_name = os.path.splitext(os.path.basename(tok_path))[0]
    subdir = os.path.join(args.output_root, tokenizer_name)
    os.makedirs(subdir, exist_ok=True)

    suffix = "" if args.split == "train" else "_eval"
    out_file = os.path.join(subdir, f"{args.dataset}_{bp}{suffix}_tokenized.txt")

    print(f"Tokenizing {src_path} with {tok_path}; writing to {out_file}")
    import math
    max_examples = math.inf if args.max_examples == -1 else args.max_examples
    max_segments = math.inf if args.max_segments == -1 else args.max_segments
    prepend_cls = args.prepend_cls.lower() == "true"
    include_sep = args.include_sep.lower() == "true"
    tokenize_file(src_path, out_file, tokenizer, args.max_seq_len,
                  max_examples=max_examples, max_segments=max_segments,
                  prepend_cls=prepend_cls, include_sep=include_sep)

    # Decide whether to shuffle
    if args.shuffle.lower() in {"true", "false"}:
        do_shuffle = args.shuffle.lower() == "true"
    else:
        # auto: shuffle train, keep eval fixed
        do_shuffle = args.split == "train"

    def _shuffle_inplace(path: str) -> None:
        print(f"Shuffling file in-place: {path}")
        tmp_path = path + ".shuf.tmp"
        terashuf = shutil.which("terashuf")
        shuf = shutil.which("shuf")
        try:
            if terashuf is not None:
                with open(tmp_path, "w", encoding="utf-8") as out:
                    subprocess.run([terashuf, path], check=True, stdout=out)
            elif shuf is not None:
                with open(tmp_path, "w", encoding="utf-8") as out:
                    subprocess.run([shuf, path], check=True, stdout=out)
            else:
                print("Warning: Neither 'terashuf' nor 'shuf' found on PATH. Skipping shuffle.")
                return
            os.replace(tmp_path, path)
            print("Shuffle complete.")
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    if do_shuffle:
        _shuffle_inplace(out_file)
    print("Done.")


if __name__ == "__main__":
    main()


