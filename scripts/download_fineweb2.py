"""
Download FineWeb-2 data for a specified language from HuggingFace and write to
a flat raw text file.

FineWeb-2 (HuggingFaceFW/fineweb-2) is a multilingual dataset with language
configs like amh_Ethi, tha_Thai, urd_Arab, vie_Latn. The output format matches
the existing monolingual training data: a flat text file with documents written
as-is, internal newlines preserved.

Usage:
    python scripts/download_fineweb2.py --language amh
    python scripts/download_fineweb2.py --language tha --max_bytes 180000000000
    python scripts/download_fineweb2.py --language urd --max_docs 1000
    python scripts/download_fineweb2.py --language vie --output /tmp/vie_test.txt
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.local_config import get_data_root

LANG_TO_HF_CONFIG = {
    "amh": "amh_Ethi",
    "tha": "tha_Thai",
    "urd": "urd_Arab",
    "vie": "vie_Latn",
}


def main() -> None:
    _data_root = get_data_root()
    parser = argparse.ArgumentParser(
        description="Download FineWeb-2 language data to a flat text file"
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=list(LANG_TO_HF_CONFIG.keys()),
        help="Language to download (maps to HF config name)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: ${DATA_ROOT}/monolingual_training_data/fineweb2_{lang}.txt)",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=-1,
        help="Maximum number of documents to download (-1 = all)",
    )
    parser.add_argument(
        "--max_bytes",
        type=int,
        default=-1,
        help="Stop downloading after writing approximately this many bytes of raw text (-1 = no limit)",
    )
    args = parser.parse_args()

    hf_config = LANG_TO_HF_CONFIG[args.language]
    if args.output is None:
        args.output = f"{_data_root}/monolingual_training_data/fineweb2_{args.language}.txt"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if os.path.isfile(args.output):
        size = os.path.getsize(args.output)
        print(f"Output file already exists: {args.output} ({size / (1024**3):.2f} GiB)")
        print("Delete it first if you want to re-download.")
        sys.exit(0)

    from datasets import load_dataset

    dataset_name = "HuggingFaceFW/fineweb-2"
    limit_msg = ""
    if args.max_bytes > 0:
        limit_msg += f", max_bytes={args.max_bytes:,} ({args.max_bytes / (1024**3):.2f} GiB)"
    if args.max_docs > 0:
        limit_msg += f", max_docs={args.max_docs:,}"
    print(f"Streaming {dataset_name} (config={hf_config}{limit_msg}) ...")
    ds = load_dataset(dataset_name, name=hf_config, split="train", streaming=True)

    doc_count = 0
    bytes_written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for example in ds:
            text = example.get("text", "")
            if not text:
                continue
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
            doc_count += 1
            bytes_written += len(text.encode("utf-8")) + 1
            if doc_count % 100_000 == 0:
                print(f"  {doc_count:,} docs written ({bytes_written / (1024**3):.2f} GiB)")
            if args.max_docs > 0 and doc_count >= args.max_docs:
                print(f"  Reached max_docs limit ({args.max_docs:,})")
                break
            if args.max_bytes > 0 and bytes_written >= args.max_bytes:
                print(f"  Reached max_bytes limit ({args.max_bytes:,} bytes)")
                break

    final_size = os.path.getsize(args.output)
    print(f"Done. Wrote {doc_count:,} documents to {args.output}")
    print(f"File size: {final_size:,} bytes ({final_size / (1024**3):.2f} GiB)")


if __name__ == "__main__":
    main()
