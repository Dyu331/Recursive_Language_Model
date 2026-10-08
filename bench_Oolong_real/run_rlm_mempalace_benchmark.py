"""
Replicate Oolong Real RLM benchmark with ephemeral MemPalace: same frozen example IDs
and baselines as run_replicate_rlm_benchmark.py.

`subagent_calls_*` uses `run_replicate_rlm_benchmark.make_subcall_counter` (same nano/mini
bucket rules as bench_BrowseComp-Plus/run_replicate_rlm_benchmark.py) plus RLM’s
socket `depth>=1` LM accounting in `rlm.core.lm_handler`.

Results mirror bench_BrowseComp-Plus/replicate_rlm_benchmarks: one folder per baseline
under rlm_mempalace_benchmarks/<baseline>[_depthN]/results.jsonl.

MemPalace is always enabled (default ingest: by_block). Per-trial palace cleanup runs in
finally so temp stores are removed even if completion raises.
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
import json
import os
import resource
from datetime import datetime
from typing import Any, Literal

import run_benchmark as ob
from dotenv import load_dotenv

from rlm import RLM
from rlm.logger.rlm_logger import RLMLogger

load_dotenv()

# region agent log
def _agent_debug_log(hypothesis_id: str, location: str, message: str, **extra: Any) -> None:
    import json
    import resource
    import time

    try:
        n_fds = len(os.listdir("/dev/fd"))
    except OSError:
        n_fds = -1
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except OSError:
        soft, hard = -1, -1
    payload: dict[str, Any] = {
        "sessionId": "1f32bd",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": {"n_fds": n_fds, "rlimit_soft": soft, "rlimit_hard": hard, **extra},
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(
            "/Users/dannyyu/Desktop/rlm/.cursor/debug-1f32bd.log",
            "a",
            encoding="utf-8",
        ) as _df:
            _df.write(json.dumps(payload) + "\n")
    except OSError:
        pass


# endregion

_MEMPALACE_ROOT = _BENCH_DIR / "rlm_mempalace_benchmarks"

RUNNER_ID = "run_rlm_mempalace_benchmark"
BENCHMARK_KIND = "oolong_replicate_rlm_mempalace"

SUITE_SINGLE_50 = "single_50"
SUITE_TWO_50 = "two_50"
SUITE_MIXED_25_25 = "mixed_25_25"
MIXED_SUITE_N = 25

import run_replicate_rlm_benchmark as rep  # noqa: E402

REPLICATE_BASELINES = rep.REPLICATE_BASELINES
_REPLICATE_BASELINE_NAMES = rep._REPLICATE_BASELINE_NAMES
baseline_results_dir_name = rep.baseline_results_dir_name
default_data_path = rep.default_data_path
ensure_validation_dataset = rep.ensure_validation_dataset
load_replicate_examples = rep.load_replicate_examples
make_subcall_counter = rep.make_subcall_counter
replicate_example_ids = rep.replicate_example_ids
reset_subcall_counts = rep.reset_subcall_counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Oolong Real replicate benchmark: RLM + ephemeral MemPalace (frozen suite)"
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
        help=(
            "Validation JSONL path (not allowed with --mixed_25_episodes; otherwise default "
            "single- or two-episode file from --allow_two_episodes)"
        ),
    )
    _episode_group = p.add_mutually_exclusive_group()
    _episode_group.add_argument(
        "--allow_two_episodes",
        action="store_true",
        help="Use 2-episode examples and REPLICATE_EXAMPLE_IDS_TWO (default: 1-episode).",
    )
    _episode_group.add_argument(
        "--mixed_25_episodes",
        action="store_true",
        help=(
            f"First {MIXED_SUITE_N} single-episode + first {MIXED_SUITE_N} two-episode replicate "
            "IDs, in that order (default validation JSONLs only)."
        ),
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
    p.add_argument(
        "--palace-poc-strict",
        action="store_true",
        help="Omit context_window_text from context; answer only via search_memories.",
    )
    p.add_argument(
        "--palace-poc-by-speaker",
        action="store_true",
        help="Group transcript by speaker for palace rooms (default: by_block).",
    )
    p.add_argument(
        "--palace-poc-by-block",
        action="store_true",
        help=(
            "Temporal block_NNN rooms (default when neither ingest flag is set). "
            "Mutually exclusive with --palace-poc-by-speaker."
        ),
    )
    return p.parse_args(argv)


def validate_mixed_no_custom_data_path(ns: argparse.Namespace) -> None:
    if ns.mixed_25_episodes and ns.data_path is not None:
        raise ValueError(
            "--data_path is not supported with --mixed_25_episodes; use the default "
            "validation_single_episode.jsonl and validation_two_episode.jsonl under "
            "bench_Oolong_real/data/."
        )


def resolve_palace_ingest_and_prompt_flags(
    ns: argparse.Namespace,
) -> tuple[Literal["by_block", "by_speaker"], bool, bool]:
    """Return ingest mode and flags for build_prompt / build_ephemeral_palace_tools."""
    if ns.palace_poc_by_speaker and ns.palace_poc_by_block:
        raise ValueError(
            "--palace-poc-by-speaker and --palace-poc-by-block are mutually exclusive."
        )
    if ns.palace_poc_by_speaker:
        return "by_speaker", True, False
    return "by_block", False, True


def condition_slug(
    *, palace_poc_strict: bool, palace_poc_by_speaker: bool, palace_poc_by_block: bool
) -> str:
    if palace_poc_strict and palace_poc_by_speaker:
        return "palace_poc_strict_by_speaker"
    if palace_poc_strict and palace_poc_by_block:
        return "palace_poc_strict_by_block"
    if palace_poc_by_speaker:
        return "palace_poc_by_speaker"
    if palace_poc_by_block:
        return "palace_poc_by_block"
    return "palace_poc_by_block"


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


def main() -> None:
    args = parse_args()
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(max(soft, 8192), hard)
        if target > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ValueError, OSError):
        pass
    if not os.getenv("OPENAI_API_KEY2"):
        raise ValueError("OPENAI_API_KEY2 is required to run this benchmark runner.")
    if args.max_depth < 1:
        raise ValueError("--max_depth must be >= 1")

    validate_mixed_no_custom_data_path(args)

    ingest_mode, palace_poc_by_speaker, palace_poc_by_block = (
        resolve_palace_ingest_and_prompt_flags(args)
    )
    palace_poc_strict = args.palace_poc_strict
    cond = condition_slug(
        palace_poc_strict=palace_poc_strict,
        palace_poc_by_speaker=palace_poc_by_speaker,
        palace_poc_by_block=palace_poc_by_block,
    )

    if args.mixed_25_episodes:
        single_path = default_data_path(allow_two_episodes=False)
        two_path = default_data_path(allow_two_episodes=True)
        ensure_validation_dataset(single_path)
        ensure_validation_dataset(two_path)
        ids_single = rep.REPLICATE_EXAMPLE_IDS_SINGLE[:MIXED_SUITE_N]
        ids_two = rep.REPLICATE_EXAMPLE_IDS_TWO[:MIXED_SUITE_N]
        examples = load_replicate_examples(
            single_path, allowed_episode_counts={1}, example_ids=ids_single
        ) + load_replicate_examples(two_path, allowed_episode_counts={2}, example_ids=ids_two)
        suite_tag = SUITE_MIXED_25_25
        data_path_summary = f"{single_path} + {two_path}"
    else:
        data_path = args.data_path or default_data_path(allow_two_episodes=args.allow_two_episodes)
        ensure_validation_dataset(data_path)
        allowed_episode_counts = {2} if args.allow_two_episodes else {1}
        frozen_ids = replicate_example_ids(allow_two_episodes=args.allow_two_episodes)
        examples = load_replicate_examples(
            data_path,
            allowed_episode_counts=allowed_episode_counts,
            example_ids=frozen_ids,
        )
        suite_tag = SUITE_TWO_50 if args.allow_two_episodes else SUITE_SINGLE_50
        data_path_summary = data_path

    baselines = [b for b in REPLICATE_BASELINES if args.baseline is None or b[0] == args.baseline]
    print(
        f"{RUNNER_ID} (Oolong Real): max_depth={args.max_depth}, mempalace=True, "
        f"ingest={ingest_mode}, strict={palace_poc_strict}, condition={cond}, suite={suite_tag}, "
        f"data_path={data_path_summary}, {len(examples)} examples, baselines={[b[0] for b in baselines]}"
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

        with open(results_path, open_mode, encoding="utf-8") as out_f:
            for ex in examples:
                task_id = ex.example_id
                query_text = ex.question

                for trial in range(1, num_trials + 1):
                    key = (task_id, baseline_name, trial, cond, suite_tag)
                    if key in completed:
                        print(
                            f"skip {baseline_name} task={task_id} trial={trial} "
                            f"cond={cond} suite={suite_tag}"
                        )
                        continue
                    reset_subcall_counts(subcalls)
                    print(
                        f"run {baseline_name} task={task_id} trial={trial} "
                        f"cond={cond} suite={suite_tag}"
                    )

                    # region agent log
                    _agent_debug_log(
                        "H1_fd_across_trials",
                        "run_rlm_mempalace_benchmark.py:loop",
                        "trial_start",
                        baseline=baseline_name,
                        task_id=task_id,
                        trial=trial,
                    )
                    # endregion

                    from benchmark_tools.ephemeral_mempalace_poc import build_ephemeral_palace_tools

                    custom_tools, cleanup, n_drawers = build_ephemeral_palace_tools(
                        ex.context_window_text,
                        task_id=ex.example_id,
                        metadata_prefix="oolong",
                        verbose=False,
                        ingest=ingest_mode,
                    )
                    # region agent log
                    _agent_debug_log(
                        "H2_chroma_index",
                        "run_rlm_mempalace_benchmark.py:loop",
                        "after_palace_build",
                        baseline=baseline_name,
                        task_id=task_id,
                        trial=trial,
                        n_drawers=n_drawers,
                    )
                    # endregion
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
                        custom_tools=custom_tools,
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
                                n_palace_drawers=n_drawers,
                                palace_poc_strict=palace_poc_strict,
                                palace_poc_by_speaker=palace_poc_by_speaker,
                                palace_poc_by_block=palace_poc_by_block,
                            ),
                        )
                    finally:
                        rlm_one.close()
                        cleanup()
                        # region agent log
                        _agent_debug_log(
                            "H1_fd_across_trials",
                            "run_rlm_mempalace_benchmark.py:loop",
                            "after_teardown",
                            baseline=baseline_name,
                            task_id=task_id,
                            trial=trial,
                        )
                        # endregion

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
                        "ingest": ingest_mode,
                        "condition": cond,
                        "suite": suite_tag,
                        "num_episode": len(ex.episodes),
                        "runner": RUNNER_ID,
                        "benchmark_kind": BENCHMARK_KIND,
                    }
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_f.flush()
                    completed.add(key)

        symlink_latest(baseline_name, args.max_depth, results_path)
        print(f"wrote {results_path}")


if __name__ == "__main__":
    main()
