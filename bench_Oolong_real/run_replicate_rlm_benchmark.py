"""
Replicate Oolong Real RLM benchmark: 50 frozen example IDs, multiple baselines,
per-baseline trial counts, JSONL under replicate_rlm_benchmarks/.

Results directory: <baseline>/ when --max_depth 1 (default), else <baseline>_depthN/.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import run_benchmark as ob
from dotenv import load_dotenv

from rlm import RLM
from rlm.logger.rlm_logger import RLMLogger

load_dotenv()

_BENCH_DIR = Path(__file__).resolve().parent
_REPLICATE_ROOT = _BENCH_DIR / "replicate_rlm_benchmarks"

# First 50 `id` values in validation_single_episode.jsonl file order (1-episode rows only).
REPLICATE_EXAMPLE_IDS_SINGLE: tuple[str, ...] = (
    "3952f2d5-082f-14b2-5ec4-d9cbedd2f865",
    "c0d82ee3-6b75-ed66-9c43-44f82e42de14",
    "f463ea50-0d40-5de6-1a65-9101296423f6",
    "12f05ebb-5c26-734f-521f-c214e5d8a524",
    "474dc32e-ebe2-69d8-ac3b-93d90d30ccb5",
    "8838b31b-2d0a-3cb1-6b70-9a2cd1c67bec",
    "a4f74314-0397-3684-913b-b8800043d63c",
    "06785154-ec6f-33c7-3a9d-270495c5185a",
    "2096a48f-bb2c-fd18-3c29-1c6ccd303ac7",
    "6feb239a-df9b-3895-61d5-6a4f4152c5cb",
    "82937a6b-e623-0baf-d4a0-a3ed084a2199",
    "cb74a9ec-60a1-920d-e8b6-014d444da449",
    "2b909102-3410-13e9-c4f8-bdb54299dd7f",
    "6e81cccf-2f57-648a-7319-805b21a671a2",
    "fe2fc969-2c78-07b6-e61c-2cc0b804758e",
    "ffc17a05-fddc-dac6-9321-eacb6a9030eb",
    "9ce1e2a1-4f9a-5aa4-5425-73907717ac6b",
    "b585666b-46dc-55c7-5da6-255540f1f1cf",
    "92cbd631-3d37-29a8-e1ff-b4137b0d13fa",
    "8989c0de-e04e-c827-8663-f491e7556a5c",
    "87b851dc-e92d-2d34-21b0-8950c64fd82c",
    "17ea4835-8da0-9866-8d3c-753836fa2bcc",
    "a16cfc46-05e6-1b05-3905-6784833656ea",
    "2c8fbe30-91da-e4b3-2754-ee9ad345b395",
    "2d31a20b-97b7-57a6-de6d-f36d256dbe24",
    "63af62d7-2086-b3b0-7505-090ee6776475",
    "cf7d1502-97d6-8d7d-a20e-83fc460afd75",
    "595151e4-3b2c-d036-a9b8-d4d250ef81fc",
    "7d2b78af-cc54-dd5e-d5e5-1455d09add1b",
    "5384f5a4-b8df-e975-2e35-b50cd9959d41",
    "27baddde-a3a2-796b-fa65-0a0b6dd590cc",
    "8aa29626-fd33-1987-b955-24a252bf3396",
    "e741133f-7f9a-556c-f4a2-73a6dcf69d9c",
    "f2fff8ad-c410-02c9-3d59-04f4f3fe094d",
    "fdb58b9f-8640-fe54-9bf9-37304961f9cb",
    "361e2479-f15f-9f4b-51cc-bddb75c0e444",
    "12ac7af5-8191-6d90-aaf2-c0f7a72c1484",
    "af9e9e51-d65e-b020-68c5-57d2b3dd89f1",
    "eb37a6c5-b42c-13ac-c791-9876cb1b682f",
    "14343980-5f0d-1d47-4bc0-a3d96b3401ec",
    "54786d8e-4f51-7696-6cd1-9e298ce97a06",
    "1be6f26f-806a-1bb4-7f5b-9819239be347",
    "8b90baca-2ca6-8732-de2e-fa5045ef36ec",
    "d9268029-2d04-e093-0fa0-d285e3e386ae",
    "b03de106-f8a3-d920-3395-cc109083fd64",
    "6cd21630-f5c2-f92e-79b2-41fcd62ece87",
    "d178704b-0018-97d5-6e45-05fcf50b4a50",
    "1db77c85-50d4-93e0-8817-9e9512b3a47c",
    "38fc693d-6fe6-8ecd-88e7-fc14e7577092",
    "5a088bcb-825c-e0bb-9944-ec727e9a50a3",
)

# First 50 `id` values in validation_two_episode.jsonl file order (2-episode rows only).
REPLICATE_EXAMPLE_IDS_TWO: tuple[str, ...] = (
    "a76add1a-e5c9-107e-34d3-894f4b30a4d8",
    "d2e65233-86cd-c067-3a35-9b5c5206d66e",
    "46d3403c-75c1-b801-afdc-b2612651e0e3",
    "21e65cfd-e739-b799-9daa-80a8438217bd",
    "3eb8fd90-1761-6f30-8657-da4deaad8928",
    "7805b9b4-9976-6304-2f8a-cea39be02640",
    "2ede4be8-778e-692b-274e-593321b2aaee",
    "8d6491cf-6d65-298e-0f74-23e33a86532f",
    "d2c3af09-8f93-5365-8dda-959721678442",
    "668fa20e-42ac-8944-ccb5-66bf28b2ce3f",
    "4abcd845-62d0-843d-9817-ed85290787dd",
    "df200593-c41a-5c4e-0cdf-105c5649b678",
    "5a682a98-36d5-319d-1cdb-23eb8c98776d",
    "ade56ce8-a77e-adef-b025-7e927e01c885",
    "33aae5eb-5164-9019-97b4-902ad88f55ba",
    "6813ac1f-740f-5105-f8f2-07958a26b102",
    "bc629d1c-0b91-4755-c37a-3d2e5447f3a0",
    "272372f7-6750-bc56-1271-10ef097e90aa",
    "5d4cbe0d-9c05-6c70-53f4-c96128ba8cca",
    "d6fae750-a41f-a7da-d78d-82de9c2eb9bb",
    "a1bce92a-5b5d-e6d4-8200-852cf5c94eaa",
    "fc7adc94-e2b3-b868-f6d3-40181db415e8",
    "3392466b-f168-0b2b-aa47-6effbd67624a",
    "371f2607-454c-bf8f-ead2-120dd285975c",
    "9c5a1ad1-70a0-9ead-7f30-6ba609e00c1f",
    "02fa2572-9a4c-5f01-6312-b283da465839",
    "ccd8f3f5-ce28-6289-e2aa-94b779395d34",
    "8a5e4e97-055b-1045-d4d7-c89fe3223e6d",
    "95dc72c1-e76a-b5dc-05c1-2a5dcb4867f4",
    "0bda1a82-6be9-376e-9ca9-e15daa162b50",
    "cb205527-aae5-ddea-c1e6-3f8f4911a027",
    "e585d7f6-acf1-d1b1-ebb7-98fe3c4c2e5a",
    "73ef8008-476e-d70e-71e7-014eb5e0765c",
    "10ef54c3-039b-3b02-32aa-372d811a02ce",
    "79299799-51d0-cf90-390f-c7bd21b25d81",
    "1163ffe6-2808-b559-877a-30aa0e66d2d8",
    "4c6aa81a-4a65-11b9-3686-90928db97b87",
    "bddb2ec9-13f6-6d4e-9358-7623089c7689",
    "518d3a8d-9464-ac2e-acbf-b9aa12b4422c",
    "e8778960-2139-f0ad-0f16-22e293efb5d0",
    "f96ec067-2ccc-4d7d-0dc4-81e62bbe44ed",
    "b4da5785-bf28-b2b0-7425-b88cb5d1f651",
    "bbe9472f-f92a-f502-b902-f8499a75bc25",
    "49f135f5-72dd-f6b6-9a33-2cdb20a23307",
    "eddd05c3-679f-f83f-c538-eeabe325cf14",
    "78040f2e-e70a-fbaf-b04c-3f75be41e2c7",
    "55fa6f6d-814c-b6e7-d3fb-b6bdbde0f3e8",
    "e0bc1bee-4294-3358-4412-538c636308fc",
    "189b4e96-754e-28b3-6341-1bf014ab4a59",
    "d8ab4ed6-7f0a-9a9c-06e9-6601ce296318",
)

# (baseline_name, root_model, sub_model, num_trials)
# gpt-5 / gpt-5-mini: OpenAI Responses API model ids (distinct from gpt-5.4 family).
REPLICATE_BASELINES: tuple[tuple[str, str, str, int], ...] = (
    ("flagship-root_mini-sub", "gpt-5.4", "gpt-5.4-mini", 1),
    ("flagship-root_mini-sub_gpt5", "gpt-5", "gpt-5-mini", 1),
    ("mini-root_mini-sub", "gpt-5.4-mini", "gpt-5.4-mini", 1),
    ("flagship-root_nano-sub", "gpt-5.4", "gpt-5.4-nano", 1),
    ("mini-root_nano-sub", "gpt-5.4-mini", "gpt-5.4-nano", 1),
)

_REPLICATE_BASELINE_NAMES: tuple[str, ...] = tuple(b[0] for b in REPLICATE_BASELINES)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replicate RLM Oolong Real benchmark runner")
    p.add_argument(
        "--baseline",
        default=None,
        choices=_REPLICATE_BASELINE_NAMES,
        help="Run only this baseline (default: all configured baselines)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip trials already present in the output JSONL for that baseline",
    )
    p.add_argument(
        "--timestamped",
        action="store_true",
        help=(
            "Write to results_YYYYMMDD_HHMMSS.jsonl and point latest.jsonl at it; "
            "default overwrites replicate_rlm_benchmarks/<baseline>[_depthN]/results.jsonl"
        ),
    )
    p.add_argument(
        "--data_path",
        default=None,
        help=(
            "Validation JSONL path (default: single-episode or two-episode file from "
            "--allow_two_episodes)"
        ),
    )
    p.add_argument(
        "--allow_two_episodes",
        action="store_true",
        help="Use 2-episode examples and REPLICATE_EXAMPLE_IDS_TWO (default: 1-episode).",
    )
    p.add_argument(
        "--max_depth",
        type=int,
        default=1,
        help="RLM max_depth (default: 1). When >1, results go under <baseline>_depth<N>/",
    )
    p.add_argument(
        "--system_prompt",
        choices=[
            "default",
            "subagent_encouraging",
            "subagent_confidence_selfeval",
            "dynamic_model_picker",
        ],
        default="default",
    )
    return p.parse_args()


def baseline_results_dir_name(baseline_name: str, max_depth: int) -> str:
    if max_depth <= 0:
        raise ValueError("max_depth must be >= 1")
    if max_depth == 1:
        return baseline_name
    return f"{baseline_name}_depth{max_depth}"


def ensure_validation_dataset(data_path: str) -> None:
    if os.path.isfile(data_path):
        return
    raise FileNotFoundError(f"Missing Oolong Real validation JSONL at '{data_path}'.")


def replicate_example_ids(*, allow_two_episodes: bool) -> tuple[str, ...]:
    return REPLICATE_EXAMPLE_IDS_TWO if allow_two_episodes else REPLICATE_EXAMPLE_IDS_SINGLE


def default_data_path(*, allow_two_episodes: bool) -> str:
    if allow_two_episodes:
        return str(_BENCH_DIR / "data" / "validation_two_episode.jsonl")
    return str(_BENCH_DIR / "data" / "validation_single_episode.jsonl")


def load_replicate_examples(
    data_path: str,
    *,
    allowed_episode_counts: set[int],
    example_ids: tuple[str, ...],
) -> list[ob.OolongRealExample]:
    all_in_file = ob.load_examples_from_jsonl(data_path, allowed_episode_counts)
    by_id = {ex.example_id: ex for ex in all_in_file}
    out: list[ob.OolongRealExample] = []
    for eid in example_ids:
        if eid not in by_id:
            raise ValueError(f"Replicate example_id not found in '{data_path}': {eid}")
        out.append(by_id[eid])
    return out


def load_completed_keys(jsonl_path: Path) -> set[tuple[str, str, int]]:
    done: set[tuple[str, str, int]] = set()
    if not jsonl_path.is_file():
        return done
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add((str(row["task_id"]), str(row["baseline"]), int(row["trial"])))
    return done


def results_dir(baseline_name: str, max_depth: int) -> Path:
    return _REPLICATE_ROOT / baseline_results_dir_name(baseline_name, max_depth)


def resolve_output_file(
    baseline_name: str, max_depth: int, *, resume: bool, timestamped: bool
) -> Path:
    base_dir = results_dir(baseline_name, max_depth)
    base_dir.mkdir(parents=True, exist_ok=True)
    if not timestamped:
        return base_dir / "results.jsonl"
    latest = base_dir / "latest.jsonl"
    if resume and latest.is_symlink():
        target = latest.resolve()
        if target.is_file():
            return target
    if resume and latest.is_file() and not latest.is_symlink():
        return latest
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"results_{stamp}.jsonl"


# Keep bucket rules in sync with bench_BrowseComp-Plus/run_replicate_rlm_benchmark.py.
_SUBCALL_NANO_MARKERS: tuple[str, ...] = ("gpt-5.4-nano",)
_SUBCALL_MINI_MARKERS: tuple[str, ...] = ("gpt-5.4-mini", "gpt-5-mini")


def _increment_subcall_bucket(counts: dict[str, int], model: str) -> None:
    m = model
    if any(marker in m for marker in _SUBCALL_NANO_MARKERS):
        counts["nano"] += 1
    elif any(marker in m for marker in _SUBCALL_MINI_MARKERS):
        counts["mini"] += 1
    elif m and m != "unknown":
        # Subagent LM call with a model id that does not contain our markers (still count).
        counts["mini"] += 1


def make_subcall_counter() -> tuple[dict[str, int], Callable[[int, str, str], None]]:
    counts: dict[str, int] = {"mini": 0, "nano": 0}

    def on_subcall_start(_depth: int, model: str, _preview: str) -> None:
        _increment_subcall_bucket(counts, model)

    return counts, on_subcall_start


def reset_subcall_counts(counts: dict[str, int]) -> None:
    counts["mini"] = 0
    counts["nano"] = 0


def symlink_latest(baseline_name: str, max_depth: int, results_file: Path) -> None:
    base_dir = results_dir(baseline_name, max_depth)
    latest = base_dir / "latest.jsonl"
    rel = os.path.relpath(results_file.resolve(), base_dir.resolve())
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(rel)


def main() -> None:
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY2"):
        raise ValueError("OPENAI_API_KEY2 is required to run this benchmark runner.")
    if args.max_depth < 1:
        raise ValueError("--max_depth must be >= 1")

    data_path = args.data_path or default_data_path(allow_two_episodes=args.allow_two_episodes)
    ensure_validation_dataset(data_path)
    allowed_episode_counts = {2} if args.allow_two_episodes else {1}
    frozen_ids = replicate_example_ids(allow_two_episodes=args.allow_two_episodes)
    examples = load_replicate_examples(
        data_path,
        allowed_episode_counts=allowed_episode_counts,
        example_ids=frozen_ids,
    )

    baselines = [b for b in REPLICATE_BASELINES if args.baseline is None or b[0] == args.baseline]
    print(
        f"replicate_rlm_benchmark (Oolong Real): max_depth={args.max_depth}, "
        f"data_path={data_path}, {len(frozen_ids)} examples, "
        f"baselines: {[b[0] for b in baselines]}"
    )

    for baseline_name, root_model, sub_model, num_trials in baselines:
        results_path = resolve_output_file(
            baseline_name,
            args.max_depth,
            resume=args.resume,
            timestamped=args.timestamped,
        )
        completed = load_completed_keys(results_path) if args.resume else set()
        results_path.parent.mkdir(parents=True, exist_ok=True)
        open_mode = "a" if (args.resume or args.timestamped) else "w"

        subcalls, on_subcall_start = make_subcall_counter()

        backend_kwargs = ob.get_backend_kwargs(root_model)
        sub_backend_kwargs = ob.get_backend_kwargs(sub_model)
        logger = RLMLogger(log_dir=str(_BENCH_DIR / "logs"))
        rlm = RLM(
            backend="openai",
            backend_kwargs=backend_kwargs,
            subagent_backend_kwargs=sub_backend_kwargs,
            environment="local",
            max_depth=args.max_depth,
            compaction=True,
            verbose=True,
            logger=logger,
            custom_system_prompt=ob.get_custom_system_prompt(args.system_prompt),
            on_subcall_start=on_subcall_start,
        )
        try:
            with open(results_path, open_mode, encoding="utf-8") as out_f:
                for ex in examples:
                    task_id = ex.example_id
                    query_text = ex.question

                    for trial in range(1, num_trials + 1):
                        key = (task_id, baseline_name, trial)
                        if key in completed:
                            print(f"skip {baseline_name} task={task_id} trial={trial}")
                            continue
                        reset_subcall_counts(subcalls)
                        print(f"run {baseline_name} task={task_id} trial={trial}")
                        context_payload = ob.build_context_payload(ex)
                        result = rlm.completion(context_payload, root_prompt=ob.build_prompt())
                        row: dict[str, Any] = {
                            "task_id": task_id,
                            "query": query_text,
                            "baseline": baseline_name,
                            "trial": trial,
                            "ground_truth": ex.answer,
                            "response": result.response,
                            "success": None,
                            "subagent_calls_mini": subcalls["mini"],
                            "subagent_calls_nano": subcalls["nano"],
                            "total_time": result.execution_time,
                            "input_tokens": result.usage_summary.total_input_tokens,
                        }
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out_f.flush()
                        completed.add(key)
        finally:
            rlm.close()

        symlink_latest(baseline_name, args.max_depth, results_path)
        print(f"wrote {results_path}")


if __name__ == "__main__":
    main()
