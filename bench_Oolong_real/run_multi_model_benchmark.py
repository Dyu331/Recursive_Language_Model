"""
Deterministic multi-baseline Oolong Real benchmark: fixed task IDs, 2 trials each, JSONL output.
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
_MULTI_ROOT = _BENCH_DIR / "multi_model_benchmarks"

FROZEN_EXAMPLE_IDS: tuple[str, ...] = (
    "3952f2d5-082f-14b2-5ec4-d9cbedd2f865",
    "17ea4835-8da0-9866-8d3c-753836fa2bcc",
    "46d3403c-75c1-b801-afdc-b2612651e0e3",
    "4abcd845-62d0-843d-9817-ed85290787dd",
    "9c5a1ad1-70a0-9ead-7f30-6ba609e00c1f",
)

BASELINES: tuple[tuple[str, str, str], ...] = (
    ("mini-root_mini-sub", "gpt-5.4-mini", "gpt-5.4-mini"),
    ("mini-root_nano-sub", "gpt-5.4-mini", "gpt-5.4-nano"),
    ("flagship-root_nano-sub", "gpt-5.4", "gpt-5.4-nano"),
)

DYNAMIC_SELECTION_BASELINES: tuple[tuple[str, str, str], ...] = (
    ("dynamic_selection_mini_root", "gpt-5.4-mini", "gpt-5.4-mini"),
    ("dynamic_selection_flagship_root", "gpt-5.4", "gpt-5.4-mini"),
)

_ALL_BASELINE_NAMES: list[str] = [b[0] for b in BASELINES] + [
    b[0] for b in DYNAMIC_SELECTION_BASELINES
]

_PROMPT_TO_BASELINES: dict[str, tuple[tuple[str, str, str], ...]] = {
    "default": BASELINES,
    "subagent_encouraging": BASELINES,
    "subagent_confidence_selfeval": BASELINES,
    "dynamic_model_picker": DYNAMIC_SELECTION_BASELINES,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-model Oolong Real benchmark runner")
    p.add_argument(
        "--baseline",
        default=None,
        choices=_ALL_BASELINE_NAMES,
        help="Run only this baseline (default: all active for the chosen --system_prompt)",
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
            "Write to a new results_YYYYMMDD_HHMMSS.jsonl and point latest.jsonl at it; "
            "default overwrites multi_model_benchmarks/<baseline>/results.jsonl"
        ),
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


def load_frozen_examples() -> list[ob.OolongRealExample]:
    single_path = _BENCH_DIR / "data" / "validation_single_episode.jsonl"
    two_path = _BENCH_DIR / "data" / "validation_two_episode.jsonl"
    single = ob.load_examples_from_jsonl(str(single_path), {1})
    double = ob.load_examples_from_jsonl(str(two_path), {2})
    by_id: dict[str, ob.OolongRealExample] = {ex.example_id: ex for ex in single}
    for ex in double:
        by_id[ex.example_id] = ex
    out: list[ob.OolongRealExample] = []
    for eid in FROZEN_EXAMPLE_IDS:
        if eid not in by_id:
            raise ValueError(f"Frozen example_id not found in validation JSONL: {eid}")
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


def results_dir(system_prompt: str, baseline_name: str) -> Path:
    if system_prompt == "subagent_confidence_selfeval":
        return _MULTI_ROOT / f"confidence_selfeval_{baseline_name}"
    return _MULTI_ROOT / baseline_name


def resolve_output_file(
    system_prompt: str, baseline_name: str, *, resume: bool, timestamped: bool
) -> Path:
    base_dir = results_dir(system_prompt, baseline_name)
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


_MINI_MODEL = "gpt-5.4-mini"
_NANO_MODEL = "gpt-5.4-nano"


def make_subcall_counter() -> tuple[dict[str, int], Callable[[int, str, str], None]]:
    counts: dict[str, int] = {_MINI_MODEL: 0, _NANO_MODEL: 0}

    def on_subcall_start(_depth: int, model: str, _preview: str) -> None:
        if model in counts:
            counts[model] += 1

    return counts, on_subcall_start


def symlink_latest(system_prompt: str, baseline_name: str, results_file: Path) -> None:
    base_dir = results_dir(system_prompt, baseline_name)
    latest = base_dir / "latest.jsonl"
    rel = os.path.relpath(results_file.resolve(), base_dir.resolve())
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(rel)


def main() -> None:
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY2"):
        raise ValueError("OPENAI_API_KEY2 is required to run this benchmark runner.")

    examples = load_frozen_examples()

    active_suite = _PROMPT_TO_BASELINES[args.system_prompt]
    active_names = [b[0] for b in active_suite]
    print(f"system_prompt: {args.system_prompt} → baselines: {active_names}")

    if args.baseline is not None and args.baseline not in active_names:
        valid = ", ".join(active_names)
        raise ValueError(
            f"--baseline '{args.baseline}' is not valid when --system_prompt is "
            f"'{args.system_prompt}'. Valid choices: {valid}"
        )

    baselines = [b for b in active_suite if args.baseline is None or b[0] == args.baseline]

    for baseline_name, root_model, sub_model in baselines:
        results_path = resolve_output_file(
            args.system_prompt, baseline_name, resume=args.resume, timestamped=args.timestamped
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
            max_depth=2,
            compaction=True,
            verbose=True,
            logger=logger,
            custom_system_prompt=ob.get_custom_system_prompt(args.system_prompt),
            on_subcall_start=on_subcall_start,
        )
        try:
            with open(results_path, open_mode, encoding="utf-8") as out_f:
                for ex in examples:
                    for trial in (1, 2):
                        key = (ex.example_id, baseline_name, trial)
                        if key in completed:
                            print(f"skip {baseline_name} task={ex.example_id} trial={trial}")
                            continue
                        subcalls[_MINI_MODEL] = 0
                        subcalls[_NANO_MODEL] = 0
                        print(f"run {baseline_name} task={ex.example_id} trial={trial}")
                        payload = ob.build_context_payload(ex)
                        result = rlm.completion(payload, root_prompt=ob.build_prompt())
                        row: dict[str, Any] = {
                            "task_id": ex.example_id,
                            "query": ex.question,
                            "baseline": baseline_name,
                            "trial": trial,
                            "ground_truth": ex.answer,
                            "response": result.response,
                            "success": None,
                            "subagent_calls_mini": subcalls[_MINI_MODEL],
                            "subagent_calls_nano": subcalls[_NANO_MODEL],
                            "total_time": result.execution_time,
                            "input_tokens": result.usage_summary.total_input_tokens,
                        }
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out_f.flush()
                        completed.add(key)
        finally:
            rlm.close()

        symlink_latest(args.system_prompt, baseline_name, results_path)
        print(f"wrote {results_path}")


if __name__ == "__main__":
    main()
