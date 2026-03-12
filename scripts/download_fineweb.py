"""
Download FineWeb 1 (sample-10BT) from HuggingFace and write to a flat raw text file.

The output format matches the existing monolingual training data (e.g. eng_latn.txt):
a flat text file where each document's text is written as-is with internal newlines
preserved. No document-per-line assumption.

Usage:
    python scripts/download_fineweb.py \
        --output /scratch/ssrivas9/catherinearnett/monolingual_training_data/fineweb_eng.txt

    # For testing with a limited number of documents:
    python scripts/download_fineweb.py \
        --output /tmp/fineweb_test.txt --max_docs 1000
"""

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FineWeb sample-10BT to a flat text file")
    parser.add_argument(
        "--output",
        type=str,
        default="/scratch/ssrivas9/catherinearnett/monolingual_training_data/fineweb_eng.txt",
        help="Output path for the raw text file",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=-1,
        help="Maximum number of documents to download (-1 = all)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="HuggingFaceFW/fineweb",
        help="HuggingFace dataset identifier",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="sample-10BT",
        help="Dataset subset/configuration name",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if os.path.isfile(args.output):
        size = os.path.getsize(args.output)
        print(f"Output file already exists: {args.output} ({size / (1024**3):.2f} GiB)")
        print("Delete it first if you want to re-download.")
        sys.exit(0)

    from datasets import load_dataset

    print(f"Streaming {args.dataset_name} ({args.subset}) ...")
    ds = load_dataset(args.dataset_name, name=args.subset, split="train", streaming=True)

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
                break

    final_size = os.path.getsize(args.output)
    print(f"Done. Wrote {doc_count:,} documents to {args.output}")
    print(f"File size: {final_size:,} bytes ({final_size / (1024**3):.2f} GiB)")


if __name__ == "__main__":
    main()
