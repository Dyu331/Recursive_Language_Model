import argparse
import json
import os
from typing import Any

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Tevatron/browsecomp-plus-corpus and save as a local JSONL file."
    )
    parser.add_argument(
        "--output",
        default="bench_BrowseComp-Plus/corpus.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of docs to write (for testing).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
    with open(args.output, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if args.limit is not None and i >= args.limit:
                break
            json.dump(dict(row), f, ensure_ascii=False)
            f.write("\n")

    print(f"Wrote corpus JSONL to {args.output}")


if __name__ == "__main__":
    main()
