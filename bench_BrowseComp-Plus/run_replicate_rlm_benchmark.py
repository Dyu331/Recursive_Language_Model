"""
Replicate BrowseComp-Plus RLM benchmark: 50 frozen query IDs, multiple baselines,
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

import run_benchmark as bc
from dotenv import load_dotenv

from rlm import RLM
from rlm.logger.rlm_logger import RLMLogger

load_dotenv()

_BENCH_DIR = Path(__file__).resolve().parent
_REPLICATE_ROOT = _BENCH_DIR / "replicate_rlm_benchmarks"

# First 50 query_id values in browsecomp_plus_decrypted.jsonl file order (deterministic).
REPLICATE_QUERY_IDS: tuple[str, ...] = (
    "769",
    "770",
    "771",
    "772",
    "773",
    "774",
    "775",
    "776",
    "778",
    "781",
    "783",
    "784",
    "785",
    "786",
    "787",
    "788",
    "790",
    "791",
    "792",
    "793",
    "794",
    "796",
    "797",
    "798",
    "800",
    "801",
    "802",
    "804",
    "805",
    "806",
    "809",
    "810",
    "811",
    "814",
    "815",
    "816",
    "819",
    "820",
    "821",
    "822",
    "823",
    "826",
    "827",
    "828",
    "830",
    "832",
    "833",
    "834",
    "835",
    "836",
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
    p = argparse.ArgumentParser(description="Replicate RLM BrowseComp-Plus benchmark runner")
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
        "--decrypted_path",
        default=str(_BENCH_DIR / "data" / "browsecomp_plus_decrypted.jsonl"),
    )
    p.add_argument(
        "--corpus_path",
        default=str(_BENCH_DIR / "corpus.jsonl"),
    )
    p.add_argument("--num_docs", type=int, default=1000)
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
    return p.parse_args()


def baseline_results_dir_name(baseline_name: str, max_depth: int) -> str:
    if max_depth <= 0:
        raise ValueError("max_depth must be >= 1")
    if max_depth == 1:
        return baseline_name
    return f"{baseline_name}_depth{max_depth}"


def load_replicate_query_records(decrypted_path: str) -> list[dict[str, Any]]:
    return [
        bc.load_query_record(decrypted_path, query_index=0, query_id=qid)
        for qid in REPLICATE_QUERY_IDS
    ]


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


# Subagent models used across baselines (gpt-5.4-* and gpt-5-mini share the "mini" bucket).
# Keep bucket rules in sync with bench_Oolong_real/run_replicate_rlm_benchmark.py.
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


def gold_doc_from_record(query_record: dict[str, Any]) -> bc.CorpusDoc:
    gold_docs_raw = query_record.get("gold_docs")
    if not isinstance(gold_docs_raw, list) or not gold_docs_raw:
        raise ValueError("Expected non-empty list field `gold_docs` in decrypted record")
    d = gold_docs_raw[0]
    if not isinstance(d, dict):
        raise ValueError("Expected gold_docs[0] to be a dict")
    return bc.CorpusDoc(
        docid=str(d.get("docid", "")),
        title=str(d.get("title", "")),
        text=str(d.get("text", "")),
    )


def build_corpus_for_query(
    query_record: dict[str, Any],
    *,
    corpus_path: str,
    num_docs: int,
) -> list[bc.CorpusDoc]:
    bc.ensure_corpus(
        corpus_path,
        smoke_test=False,
        smoke_test_limit=5,
        min_docs_required=num_docs,
    )
    raw_docs = bc.load_jsonl(corpus_path, limit=num_docs)
    corpus_docs = bc.normalize_corpus_docs(raw_docs)
    gold_doc = gold_doc_from_record(query_record)
    corpus_docs, _injected = bc.ensure_gold_doc_in_context(
        corpus_docs,
        gold_doc=gold_doc,
        total_docs=num_docs,
    )
    return corpus_docs


def main() -> None:
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY2"):
        raise ValueError("OPENAI_API_KEY2 is required to run this benchmark runner.")
    if args.max_depth < 1:
        raise ValueError("--max_depth must be >= 1")

    bc.ensure_decrypted_dataset(args.decrypted_path)
    query_records = load_replicate_query_records(args.decrypted_path)

    baselines = [b for b in REPLICATE_BASELINES if args.baseline is None or b[0] == args.baseline]
    print(
        f"replicate_rlm_benchmark: max_depth={args.max_depth}, "
        f"{len(REPLICATE_QUERY_IDS)} queries, baselines: {[b[0] for b in baselines]}"
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

        backend_kwargs = bc.get_backend_kwargs(root_model)
        sub_backend_kwargs = bc.get_backend_kwargs(sub_model)
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
            custom_system_prompt=bc.get_custom_system_prompt(args.system_prompt),
            on_subcall_start=on_subcall_start,
        )
        try:
            with open(results_path, open_mode, encoding="utf-8") as out_f:
                for query_record in query_records:
                    task_id = str(query_record.get("query_id", ""))
                    query_text = query_record.get("query")
                    if not isinstance(query_text, str):
                        query_text = str(query_text)
                    ground_truth = query_record.get("answer")
                    if ground_truth is not None and not isinstance(ground_truth, str):
                        ground_truth = str(ground_truth)

                    corpus_docs = build_corpus_for_query(
                        query_record,
                        corpus_path=args.corpus_path,
                        num_docs=args.num_docs,
                    )
                    context_payload = bc.build_context_payload(query_record, corpus_docs)

                    for trial in range(1, num_trials + 1):
                        key = (task_id, baseline_name, trial)
                        if key in completed:
                            print(f"skip {baseline_name} task={task_id} trial={trial}")
                            continue
                        reset_subcall_counts(subcalls)
                        print(f"run {baseline_name} task={task_id} trial={trial}")
                        result = rlm.completion(context_payload, root_prompt=bc.build_prompt())
                        row: dict[str, Any] = {
                            "task_id": task_id,
                            "query": query_text,
                            "baseline": baseline_name,
                            "trial": trial,
                            "ground_truth": ground_truth,
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
