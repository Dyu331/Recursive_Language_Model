import sys
from pathlib import Path

# Allow `python bench_Oolong_synth/run_benchmark.py` without reinstall: repo root must be on
# sys.path so that `benchmark_tools` at project root is importable later (MemPalace integration
# will be added in a follow-up).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, cast

from dotenv import load_dotenv

from rlm import RLM
from rlm.clients import get_client
from rlm.logger.rlm_logger import RLMLogger
from rlm.utils.dynamic_model_picker_prompt import (
    RLM_SYSTEM_PROMPT as DYNAMIC_MODEL_PICKER_PROMPT,
)
from rlm.utils.parallel_subagent_prompt import (
    RLM_SYSTEM_PROMPT as PARALLEL_SUBAGENT_PROMPT,
)
from rlm.utils.subagent_confidence_selfeval_prompt import (
    RLM_SYSTEM_PROMPT as SUBAGENT_CONFIDENCE_SELFEVAL_PROMPT,
)
from rlm.utils.subagent_encouraging_prompt import (
    RLM_SYSTEM_PROMPT as SUBAGENT_ENCOURAGING_PROMPT,
)

load_dotenv()


DEFAULT_DATA_PATH = "bench_Oolong_synth/data/oolong_synth_validation_trec_coarse_131k.jsonl"


@dataclass(frozen=True)
class OolongSynthExample:
    example_id: str
    context_window_id: str
    dataset: str
    context_len: int
    context_window_text: str
    context_window_text_with_labels: str
    question: str
    answer: str
    task_group: str
    task: str
    answer_type: str
    input_subset: str
    num_labels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default=DEFAULT_DATA_PATH,
        help="Path to the Oolong-synth validation JSONL slice.",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run a single example with a short root prompt, useful as a wiring check.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--query_index",
        type=int,
        default=0,
        help="0-indexed record offset into the Oolong-synth JSONL.",
    )
    group.add_argument(
        "--query_id",
        default=None,
        help="Select the record with matching `id` (string match).",
    )
    parser.add_argument(
        "--model_name",
        default=os.getenv("OOLONG_SYNTH_MODEL", "gpt-5.4-mini"),
    )
    parser.add_argument(
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
    parser.add_argument(
        "--mode",
        choices=["rlm", "default"],
        default="rlm",
    )
    parser.add_argument(
        "--palace-poc",
        action="store_true",
        help=(
            "Index context_window_text in an ephemeral MemPalace with atomic-record ingest "
            "and add search_memories() to the REPL. Lenient: full stitched context stays in "
            "context['context_window_text']."
        ),
    )
    parser.add_argument(
        "--palace-poc-strict",
        action="store_true",
        help=(
            "Same as --palace-poc but omit context_window_text from context; answer using "
            "search_memories only. Implies --palace-poc."
        ),
    )
    return parser.parse_args()


def ensure_data(data_path: str) -> None:
    if os.path.exists(data_path):
        return
    raise FileNotFoundError(
        f"Missing Oolong-synth dataset at '{data_path}'. "
        "Place the validation JSONL slice there (e.g. "
        "oolong_synth_validation_trec_coarse_131k.jsonl)."
    )


def require_str(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected non-empty string field `{key}` in Oolong-synth record")
    return value


def require_int(record: dict[str, Any], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Expected int field `{key}` in Oolong-synth record")
    return value


def normalize_example(record: dict[str, Any]) -> OolongSynthExample:
    return OolongSynthExample(
        example_id=str(record.get("id", "")),
        context_window_id=str(record.get("context_window_id", "")),
        dataset=require_str(record, "dataset"),
        context_len=require_int(record, "context_len"),
        context_window_text=require_str(record, "context_window_text"),
        context_window_text_with_labels=str(record.get("context_window_text_with_labels") or ""),
        question=require_str(record, "question"),
        answer=str(record.get("answer", "")),
        task_group=require_str(record, "task_group"),
        task=require_str(record, "task"),
        answer_type=require_str(record, "answer_type"),
        input_subset=require_str(record, "input_subset"),
        num_labels=require_int(record, "num_labels"),
    )


def load_query_record(
    data_path: str, *, query_index: int, query_id: str | None
) -> OolongSynthExample:
    if query_id is None and query_index < 0:
        raise ValueError("query_index must be >= 0")

    sample_ids: list[str] = []
    with open(data_path, encoding="utf-8") as f:
        i = -1
        for line in f:
            if not line.strip():
                continue
            i += 1
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("Expected dict records in Oolong-synth JSONL")
            row_id = str(record.get("id", ""))
            if row_id and len(sample_ids) < 5:
                sample_ids.append(row_id)
            if query_id is not None:
                if row_id == query_id:
                    return normalize_example(record)
            else:
                if i == query_index:
                    return normalize_example(record)

    if query_id is not None:
        suffix = f" Example id values in this file: {', '.join(sample_ids)}." if sample_ids else ""
        raise ValueError(f"No record found with id='{query_id}' in '{data_path}'.{suffix}")
    raise ValueError(f"No record found at query_index={query_index} in '{data_path}'")


def build_context_payload(
    example: OolongSynthExample, *, palace_poc_strict: bool = False
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "example_id": example.example_id,
        "context_window_id": example.context_window_id,
        "dataset": example.dataset,
        "context_len": example.context_len,
        "task_group": example.task_group,
        "task": example.task,
        "answer_type": example.answer_type,
        "input_subset": example.input_subset,
        "num_labels": example.num_labels,
        "question": example.question,
    }
    if palace_poc_strict:
        payload["palace_poc_strict"] = True
    else:
        payload["context_window_text"] = example.context_window_text
    return payload


def build_prompt(
    *,
    palace_poc: bool = False,
    palace_poc_strict: bool = False,
    n_palace_drawers: int = 0,
) -> str:
    if palace_poc_strict:
        from benchmark_tools.ephemeral_mempalace_poc import palace_poc_prompt_hint_by_record

        return (
            palace_poc_prompt_hint_by_record(n_palace_drawers, strict=True)
            + "You are running inside an RLM REPL. The variable `context` contains the question "
            "in context['question'] and metadata (example_id, dataset, task_group, task, "
            "answer_type, input_subset, num_labels) but does NOT contain the raw stitched "
            "context.\n\n"
            "Task: Answer the question using ONLY information retrieved via `search_memories`. "
            "Use metadata filters (user, date_from/date_to, line_start/line_end) in mode='exact' "
            "when applicable; use mode='semantic' for meaning-based retrieval. Classify or "
            "aggregate per-record with `llm_query_batched` / `rlm_query_batched`—every sub-call "
            "must include the retrieved record text verbatim. Return the final answer with "
            "FINAL_VAR('answer')."
        )

    base = (
        "You are running inside an RLM REPL. The variable `context` is available and contains: "
        "(1) an Oolong-synth question in context['question'] and (2) a long stitched context of "
        "many short independent records in context['context_window_text'] (one record per line, "
        "typically with fields like date, user, and instance). Additional metadata includes "
        "context['example_id'], context['dataset'], context['task_group'], context['task'], "
        "context['answer_type'], context['input_subset'], and context['num_labels'].\n\n"
        "Task: Answer the question using ONLY the provided stitched context. These are aggregation "
        "questions over per-record labels (counts, temporal slices, user-conditioned comparisons), "
        "so be exact rather than estimating.\n\n"
        "Use the REPL to parse lines, filter by user/date, classify each record into one of the "
        "num_labels categories, and aggregate. Prefer `llm_query_batched` or `rlm_query_batched` to "
        "classify batches of records in parallel when per-record reasoning is needed; each sub-call "
        "must include the record text verbatim in the prompt. Do not combine broad parsing, "
        "semantic classification, and final aggregation in the same iteration.\n\n"
        "Return the final answer with FINAL_VAR('answer')."
    )
    if palace_poc:
        from benchmark_tools.ephemeral_mempalace_poc import palace_poc_prompt_hint_by_record

        return palace_poc_prompt_hint_by_record(n_palace_drawers, strict=False) + base
    return base


def build_baseline_prompt(example: OolongSynthExample) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Answer the user's question using only the provided Oolong-synth context. "
                "Do not use external knowledge. Return only the final answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {example.question}\n\n"
                f"Context:\n{example.context_window_text}\n\n"
                "Return only the final answer."
            ),
        },
    ]


def get_custom_system_prompt(system_prompt: str) -> str | None:
    if system_prompt == "default":
        return None
    if system_prompt == "subagent_encouraging":
        return SUBAGENT_ENCOURAGING_PROMPT
    if system_prompt == "subagent_confidence_selfeval":
        return SUBAGENT_CONFIDENCE_SELFEVAL_PROMPT
    if system_prompt == "dynamic_model_picker":
        return DYNAMIC_MODEL_PICKER_PROMPT
    if system_prompt == "parallel_subagent":
        return PARALLEL_SUBAGENT_PROMPT
    raise ValueError(f"Unsupported system_prompt='{system_prompt}'")


def get_backend_kwargs(model_name: str) -> dict[str, Any]:
    return {
        "api_key": os.getenv("OPENAI_API_KEY2"),
        "model_name": model_name,
    }


def main() -> None:
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY2"):
        raise ValueError("OPENAI_API_KEY2 is required to run this benchmark runner.")

    palace_poc = args.palace_poc or args.palace_poc_strict
    palace_poc_strict = args.palace_poc_strict
    if palace_poc and args.mode == "default":
        raise ValueError("--palace-poc / --palace-poc-strict require --mode rlm")

    ensure_data(args.data_path)

    example = load_query_record(
        args.data_path,
        query_index=args.query_index,
        query_id=args.query_id,
    )

    if args.smoke_test:
        print("Smoke test example_id:", example.example_id)
        print("Smoke test task_group:", example.task_group)
        print("Smoke test context_chars:", len(example.context_window_text))

    backend_kwargs = get_backend_kwargs(args.model_name)
    subagent_backend_kwargs = get_backend_kwargs("gpt-5.4-mini")

    logger: RLMLogger | None = None
    rlm: RLM | None = None
    root_client = None
    palace_cleanup = None
    n_drawers = 0
    custom_tools = None

    if args.mode == "default":
        root_client = cast(Any, get_client("openai", backend_kwargs))
    else:
        logger = RLMLogger(log_dir="./bench_Oolong_synth/logs")
        if palace_poc:
            from benchmark_tools.ephemeral_mempalace_poc import build_ephemeral_palace_tools

            labeled_text = example.context_window_text_with_labels
            if not labeled_text:
                raise ValueError(
                    "--palace-poc / --palace-poc-strict require "
                    "'context_window_text_with_labels' on each Oolong-synth record"
                )
            custom_tools, palace_cleanup, n_drawers = build_ephemeral_palace_tools(
                labeled_text,
                task_id=example.example_id,
                metadata_prefix="oolong_synth",
                ingest="by_record",
            )
        rlm = RLM(
            backend="openai",
            backend_kwargs=backend_kwargs,
            subagent_backend_kwargs=subagent_backend_kwargs,
            environment="local",
            max_depth=2,
            compaction=True,
            verbose=True,
            logger=logger,
            custom_system_prompt=get_custom_system_prompt(args.system_prompt),
            custom_tools=custom_tools,
        )

    mode_label = args.mode
    if palace_poc:
        mode_label += "_palace_poc_strict" if palace_poc_strict else "_palace_poc"
    print("mode:", mode_label)
    print("model_name:", args.model_name)
    print("system_prompt:", args.system_prompt)
    print("example_id:", example.example_id)
    print("task_group:", example.task_group)
    print("task:", example.task)
    print("context_chars:", len(example.context_window_text))
    print("question:", example.question)
    if palace_poc:
        print("n_palace_drawers:", n_drawers)

    try:
        if args.mode == "default":
            assert root_client is not None
            response = root_client.completion(build_baseline_prompt(example))
        else:
            assert rlm is not None
            context_payload = build_context_payload(example, palace_poc_strict=palace_poc_strict)
            result = rlm.completion(
                context_payload,
                root_prompt=build_prompt(
                    palace_poc=palace_poc,
                    palace_poc_strict=palace_poc_strict,
                    n_palace_drawers=n_drawers,
                ),
            )
            response = result.response

        print("Response:", response)
        print("Ground truth example_id:", example.example_id)
        print("Ground truth answer:", example.answer)
        if logger is not None and logger.log_file_path:
            print("Log file:", logger.log_file_path)
    finally:
        if rlm is not None:
            rlm.close()
        if palace_cleanup is not None:
            palace_cleanup()


if __name__ == "__main__":
    main()
