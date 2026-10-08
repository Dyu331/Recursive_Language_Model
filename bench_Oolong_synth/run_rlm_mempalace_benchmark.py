"""
Oolong-synth RLM benchmark with ephemeral MemPalace (atomic-record ingest).

Iterates the 50-record validation slice and runs each query through RLM with
``search_memories`` over a by-record Memory Palace. Because the slice has very few distinct
``context_window_text`` blobs (2 in the shipped validation JSONL), we **reuse** the palace
across consecutive queries that share a context (hash-based) and only rebuild when the hash
changes. Every query gets a fresh RLM (new REPL + new message history) regardless—palace
state is long-term, RLM state is short-term.

Results schema mirrors ``bench_Oolong_real/run_rlm_mempalace_benchmark.py`` where it makes
sense, plus synth-specific fields (``task``, ``task_group``, ``answer_type``,
``context_window_id``, ``context_len``, ``mempalace_build_seconds`` (wall time for
``build_ephemeral_palace_tools`` on that step, or ``0`` when the palace was reused).
``subagent_calls_*`` uses the same mini/nano bucket rules as the real replicate runner.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BENCH_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

import argparse
import hashlib
import json
import os
import resource
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import run_benchmark as ob
from dotenv import load_dotenv

from rlm import RLM
from rlm.logger.rlm_logger import RLMLogger

load_dotenv()


RUNNER_ID = "run_rlm_mempalace_benchmark"
BENCHMARK_KIND = "oolong_synth_rlm_mempalace"
SUITE_VALIDATION_50 = "validation_50"

_MEMPALACE_ROOT = _BENCH_DIR / "rlm_mempalace_benchmarks"

# (baseline_name, root_model, sub_model, num_trials). Mirrors the real-benchmark slate so
# condition columns line up cross-benchmark; trim via --baseline.
REPLICATE_BASELINES: tuple[tuple[str, str, str, int], ...] = (
    ("flagship-root_mini-sub", "gpt-5.4", "gpt-5.4-mini", 1),
    ("flagship-root_mini-sub_gpt5", "gpt-5", "gpt-5-mini", 1),
    ("mini-root_mini-sub", "gpt-5.4-mini", "gpt-5.4-mini", 2),
    # ("flagship-root_nano-sub", "gpt-5.4", "gpt-5.4-nano", 1),
    # ("mini-root_nano-sub", "gpt-5.4-mini", "gpt-5.4-nano", 1),
)
_REPLICATE_BASELINE_NAMES: tuple[str, ...] = tuple(b[0] for b in REPLICATE_BASELINES)


_SUBCALL_NANO_MARKERS: tuple[str, ...] = ("gpt-5.4-nano",)
_SUBCALL_MINI_MARKERS: tuple[str, ...] = ("gpt-5.4-mini", "gpt-5-mini")


def _increment_subcall_bucket(counts: dict[str, int], model: str) -> None:
    if any(marker in model for marker in _SUBCALL_NANO_MARKERS):
        counts["nano"] += 1
    elif any(marker in model for marker in _SUBCALL_MINI_MARKERS):
        counts["mini"] += 1
    elif model and model != "unknown":
        counts["mini"] += 1


def make_subcall_counter() -> tuple[dict[str, int], Callable[[int, str, str], None]]:
    counts: dict[str, int] = {"mini": 0, "nano": 0}

    def on_subcall_start(_depth: int, model: str, _preview: str) -> None:
        _increment_subcall_bucket(counts, model)

    return counts, on_subcall_start


def reset_subcall_counts(counts: dict[str, int]) -> None:
    counts["mini"] = 0
    counts["nano"] = 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Oolong-synth RLM + ephemeral MemPalace (atomic-record ingest)"
    )
    p.add_argument(
        "--baseline",
        default=None,
        choices=_REPLICATE_BASELINE_NAMES,
        help="Run only this baseline (default: all configured baselines)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip trials whose (task_id, baseline, trial, condition, suite) already appear "
            "in the output JSONL for that baseline"
        ),
    )
    p.add_argument(
        "--timestamped",
        action="store_true",
        help=(
            "Write to results_YYYYMMDD_HHMMSS.jsonl and point latest.jsonl at it; "
            "default overwrites rlm_mempalace_benchmarks/<baseline>[_depthN]/results.jsonl"
        ),
    )
    p.add_argument(
        "--data_path",
        default=None,
        help=f"Validation JSONL path (default: {ob.DEFAULT_DATA_PATH})",
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
            "parallel_subagent",
        ],
        default="default",
    )
    p.add_argument(
        "--palace-poc-strict",
        action="store_true",
        help="Omit context_window_text from context; answer only via search_memories.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N records (default: all in the file).",
    )
    return p.parse_args(argv)


def baseline_results_dir_name(baseline_name: str, max_depth: int) -> str:
    if max_depth <= 0:
        raise ValueError("max_depth must be >= 1")
    if max_depth == 1:
        return baseline_name
    return f"{baseline_name}_depth{max_depth}"


def results_dir(baseline_name: str, max_depth: int) -> Path:
    return _MEMPALACE_ROOT / baseline_results_dir_name(baseline_name, max_depth)


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


def symlink_latest(baseline_name: str, max_depth: int, results_file: Path) -> None:
    base_dir = results_dir(baseline_name, max_depth)
    latest = base_dir / "latest.jsonl"
    rel = os.path.relpath(results_file.resolve(), base_dir.resolve())
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(rel)


def load_completed_keys(jsonl_path: Path) -> set[tuple[str, str, int, str, str]]:
    done: set[tuple[str, str, int, str, str]] = set()
    if not jsonl_path.is_file():
        return done
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cond = str(row.get("condition", ""))
            suite = str(row.get("suite", ""))
            done.add((str(row["task_id"]), str(row["baseline"]), int(row["trial"]), cond, suite))
    return done


def load_all_synth_examples(data_path: str) -> list[ob.OolongSynthExample]:
    out: list[ob.OolongSynthExample] = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("Expected dict records in Oolong-synth JSONL")
            out.append(ob.normalize_example(record))
    return out


def _ctx_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _raise_soft_nofile_limit() -> None:
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(max(soft, 8192), hard)
        if target > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ValueError, OSError):
        pass


def main() -> None:
    args = parse_args()
    _raise_soft_nofile_limit()
    if not os.getenv("OPENAI_API_KEY2"):
        raise ValueError("OPENAI_API_KEY2 is required to run this benchmark runner.")
    if args.max_depth < 1:
        raise ValueError("--max_depth must be >= 1")

    palace_poc_strict = args.palace_poc_strict
    condition = "palace_poc_strict_by_record" if palace_poc_strict else "palace_poc_by_record"
    suite_tag = SUITE_VALIDATION_50

    data_path = args.data_path or ob.DEFAULT_DATA_PATH
    ob.ensure_data(data_path)
    examples = load_all_synth_examples(data_path)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be > 0")
        examples = examples[: args.limit]
    if not examples:
        raise ValueError(f"No Oolong-synth examples loaded from '{data_path}'")

    baselines = [b for b in REPLICATE_BASELINES if args.baseline is None or b[0] == args.baseline]
    print(
        f"{RUNNER_ID} (Oolong-synth): max_depth={args.max_depth}, mempalace=True, "
        f"ingest=by_record, strict={palace_poc_strict}, condition={condition}, "
        f"suite={suite_tag}, data_path={data_path}, {len(examples)} examples, "
        f"baselines={[b[0] for b in baselines]}"
    )

    from benchmark_tools.ephemeral_mempalace_poc import build_ephemeral_palace_tools

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

        last_hash: str | None = None
        last_cleanup: Callable[[], None] | None = None
        last_custom_tools: dict[str, Any] | None = None
        last_n_drawers: int = 0

        try:
            with open(results_path, open_mode, encoding="utf-8") as out_f:
                for ex in examples:
                    task_id = ex.example_id
                    query_text = ex.question
                    labeled_text = ex.context_window_text_with_labels
                    if not labeled_text:
                        raise ValueError(
                            f"task_id={ex.example_id}: 'context_window_text_with_labels' is "
                            "required for by_record MemPalace ingest (label-as-room)"
                        )
                    ctx_hash = _ctx_hash(labeled_text)
                    mempalace_build_seconds = 0.0

                    if ctx_hash != last_hash:
                        if last_cleanup is not None:
                            last_cleanup()
                        # Drop handles before rebuilding so a failed build cannot leave
                        # last_hash pointing at a context whose palace was already deleted.
                        last_cleanup = None
                        last_custom_tools = None
                        last_n_drawers = 0
                        last_hash = None
                        t_build0 = time.perf_counter()
                        custom_tools, cleanup, n_drawers = build_ephemeral_palace_tools(
                            labeled_text,
                            task_id=f"{ex.context_window_id or 'synth'}:{ctx_hash[:8]}",
                            metadata_prefix="oolong_synth",
                            ingest="by_record",
                        )
                        mempalace_build_seconds = time.perf_counter() - t_build0
                        last_hash = ctx_hash
                        last_cleanup = cleanup
                        last_custom_tools = custom_tools
                        last_n_drawers = n_drawers
                        print(
                            f"palace built for context_window_id={ex.context_window_id} "
                            f"hash={ctx_hash[:8]} drawers={n_drawers} "
                            f"build_s={mempalace_build_seconds:.3f}"
                        )

                    for trial in range(1, num_trials + 1):
                        key = (task_id, baseline_name, trial, condition, suite_tag)
                        if key in completed:
                            print(
                                f"skip {baseline_name} task={task_id} trial={trial} "
                                f"cond={condition} suite={suite_tag}"
                            )
                            continue
                        reset_subcall_counts(subcalls)
                        print(
                            f"run {baseline_name} task={task_id} trial={trial} "
                            f"cond={condition} suite={suite_tag}"
                        )

                        rlm_one = RLM(
                            backend="openai",
                            backend_kwargs=backend_kwargs,
                            subagent_backend_kwargs=sub_backend_kwargs,
                            environment="local",
                            max_depth=args.max_depth,
                            compaction=True,
                            verbose=True,
                            logger=logger,
                            custom_system_prompt=ob.get_custom_system_prompt(args.system_prompt),
                            custom_tools=last_custom_tools,
                            on_subcall_start=on_subcall_start,
                        )
                        try:
                            context_payload = ob.build_context_payload(
                                ex, palace_poc_strict=palace_poc_strict
                            )
                            result = rlm_one.completion(
                                context_payload,
                                root_prompt=ob.build_prompt(
                                    palace_poc=True,
                                    palace_poc_strict=palace_poc_strict,
                                    n_palace_drawers=last_n_drawers,
                                ),
                            )
                        finally:
                            rlm_one.close()

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
                            "mempalace": True,
                            "mempalace_build_seconds": mempalace_build_seconds,
                            "ingest": "by_record",
                            "condition": condition,
                            "suite": suite_tag,
                            "task": ex.task,
                            "task_group": ex.task_group,
                            "answer_type": ex.answer_type,
                            "context_window_id": ex.context_window_id,
                            "context_len": ex.context_len,
                            "context_hash": ctx_hash[:16],
                            "runner": RUNNER_ID,
                            "benchmark_kind": BENCHMARK_KIND,
                        }
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out_f.flush()
                        completed.add(key)
        finally:
            if last_cleanup is not None:
                last_cleanup()

        symlink_latest(baseline_name, args.max_depth, results_path)
        print(f"wrote {results_path}")


if __name__ == "__main__":
    main()
