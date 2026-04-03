import argparse
import json
import os
from typing import Any

from datasets import load_dataset

DATASET_NAME = "oolongbench/oolong-real"
DATASET_CONFIG = "dnd"
DATASET_SPLIT = "validation"
DEFAULT_OUTPUT_PATH = "bench_Oolong_real/data/validation_single_episode.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--allow_two_episodes",
        action="store_true",
        help="Write only 2-episode examples instead of the default 1-episode examples.",
    )
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def normalize_episodes(record: dict[str, Any]) -> list[int]:
    value = record.get("episodes")
    if not isinstance(value, list) or not value:
        raise ValueError("Expected non-empty list field `episodes` in Oolong Real record")
    normalized: list[int] = []
    for episode in value:
        if not isinstance(episode, int):
            raise ValueError("Expected all `episodes` entries to be ints in Oolong Real record")
        normalized.append(episode)
    return normalized


def validate_record(record: dict[str, Any], allowed_episode_count: int) -> dict[str, Any]:
    required_str_fields = [
        "id",
        "context_window_id",
        "context_window_text",
        "question",
        "answer",
        "question_type",
        "campaign",
    ]
    for key in required_str_fields:
        value = record.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Expected non-empty string field `{key}` in Oolong Real record")

    episodes = normalize_episodes(record)
    if len(episodes) != allowed_episode_count:
        raise ValueError(f"Expected only {allowed_episode_count}-episode records after filtering")

    return {
        "id": record["id"],
        "context_window_id": record["context_window_id"],
        "context_window_text": record["context_window_text"],
        "question": record["question"],
        "answer": record["answer"],
        "question_type": record["question_type"],
        "episodes": episodes,
        "campaign": record["campaign"],
    }


def main() -> None:
    args = parse_args()
    allowed_episode_count = 2 if args.allow_two_episodes else 1
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    temp_output = f"{args.output}.tmp"

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        streaming=args.streaming,
    )

    written = 0
    with open(temp_output, "w", encoding="utf-8") as f:
        for record in dataset:
            if not isinstance(record, dict):
                raise ValueError("Expected dict records from Oolong Real dataset")
            episodes = normalize_episodes(record)
            if len(episodes) != allowed_episode_count:
                continue
            validated = validate_record({**record, "episodes": episodes}, allowed_episode_count)
            json.dump(validated, f, ensure_ascii=False)
            f.write("\n")
            written += 1
            if args.limit is not None and written >= args.limit:
                break

    os.replace(temp_output, args.output)

    print(f"Wrote {written} {allowed_episode_count}-episode validation examples to {args.output}")


if __name__ == "__main__":
    main()
