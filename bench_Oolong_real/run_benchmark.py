import sys
from pathlib import Path

# Allow `python bench_Oolong_real/run_benchmark.py` without reinstall: repo root must be on sys.path
# (script dir is bench_Oolong_real/, so `benchmark_tools` at project root is not found otherwise).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Literal, cast

from dotenv import load_dotenv

from rlm import RLM
from rlm.clients import get_client
from rlm.logger.rlm_logger import RLMLogger
from rlm.utils.dynamic_model_picker_prompt import (
    RLM_SYSTEM_PROMPT as DYNAMIC_MODEL_PICKER_PROMPT,
)
from rlm.utils.subagent_confidence_selfeval_prompt import (
    RLM_SYSTEM_PROMPT as SUBAGENT_CONFIDENCE_SELFEVAL_PROMPT,
)
from rlm.utils.subagent_encouraging_prompt import RLM_SYSTEM_PROMPT as SUBAGENT_ENCOURAGING_PROMPT

load_dotenv()


DEFAULT_SINGLE_EPISODE_DATA_PATH = "bench_Oolong_real/data/validation_single_episode.jsonl"
DEFAULT_TWO_EPISODE_DATA_PATH = "bench_Oolong_real/data/validation_two_episode.jsonl"


@dataclass(frozen=True)
class OolongRealExample:
    example_id: str
    context_window_id: str
    context_window_text: str
    question: str
    answer: str
    question_type: str
    episodes: tuple[int, ...]
    campaign: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--query_index", type=int, default=None)
    parser.add_argument("--example_id", default=None)
    parser.add_argument("--campaign", default=None)
    parser.add_argument("--question_type", default=None)
    parser.add_argument(
        "--allow_two_episodes",
        action="store_true",
        help="Use only 2-episode examples instead of the default 1-episode examples.",
    )
    parser.add_argument(
        "--model_name",
        default=os.getenv("OOLONG_REAL_MODEL", "gpt-5.4-mini"),
    )
    parser.add_argument(
        "--system_prompt",
        choices=[
            "default",
            "subagent_encouraging",
            "subagent_confidence_selfeval",
            "dynamic_model_picker",
        ],
        default="default",
    )
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument(
        "--palace-poc",
        action="store_true",
        help=(
            "Index context_window_text in an ephemeral MemPalace and add search_memories() "
            "to the REPL (requires mempalace-poc extra). Lenient: full transcript stays in context."
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
    parser.add_argument(
        "--palace-poc-verbose",
        action="store_true",
        help=(
            "With --palace-poc / --palace-poc-strict: print wing/room structure and per-drawer "
            "text previews to stdout right after indexing (before RLM runs)."
        ),
    )
    parser.add_argument(
        "--palace-poc-by-speaker",
        action="store_true",
        help=(
            "With palace PoC: grouped by-speaker drawers (dominant room, _mixed on ties), "
            "_preamble + list_taxonomy(). Requires --palace-poc or --palace-poc-strict."
        ),
    )
    parser.add_argument(
        "--palace-poc-by-block",
        action="store_true",
        help=(
            "With palace PoC: temporal block_001… rooms (overlapping line windows), "
            "list_taxonomy(). Mutually exclusive with --palace-poc-by-speaker."
        ),
    )
    return parser.parse_args()


def require_str(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected non-empty string field `{key}` in Oolong Real record")
    return value


def normalize_episodes(record: dict[str, Any]) -> tuple[int, ...]:
    value = record.get("episodes")
    if not isinstance(value, list) or not value:
        raise ValueError("Expected non-empty list field `episodes` in Oolong Real record")
    normalized: list[int] = []
    for episode in value:
        if not isinstance(episode, int):
            raise ValueError("Expected all `episodes` entries to be ints in Oolong Real record")
        normalized.append(episode)
    return tuple(normalized)


def normalize_example(record: dict[str, Any]) -> OolongRealExample:
    example = OolongRealExample(
        example_id=require_str(record, "id"),
        context_window_id=require_str(record, "context_window_id"),
        context_window_text=require_str(record, "context_window_text"),
        question=require_str(record, "question"),
        answer=require_str(record, "answer"),
        question_type=require_str(record, "question_type"),
        episodes=normalize_episodes(record),
        campaign=require_str(record, "campaign"),
    )
    return example


def load_examples_from_jsonl(
    data_path: str, allowed_episode_counts: set[int]
) -> list[OolongRealExample]:
    examples: list[OolongRealExample] = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("Expected dict records in local Oolong Real JSONL")
            example = normalize_example(record)
            if len(example.episodes) in allowed_episode_counts:
                examples.append(example)
    return examples


def load_examples(
    *,
    limit: int,
    query_index: int | None,
    example_id: str | None,
    campaign: str | None,
    question_type: str | None,
    data_path: str,
    allowed_episode_counts: set[int],
) -> list[OolongRealExample]:
    if limit <= 0:
        raise ValueError("limit must be > 0")
    if query_index is not None and query_index < 0:
        raise ValueError("query_index must be >= 0")
    if query_index is not None and example_id is not None:
        raise ValueError("query_index and example_id are mutually exclusive")

    all_examples = load_examples_from_jsonl(data_path, allowed_episode_counts)
    matches: list[OolongRealExample] = []
    for example in all_examples:
        if example_id is not None and example.example_id != example_id:
            continue
        if campaign is not None and example.campaign != campaign:
            continue
        if question_type is not None and example.question_type != question_type:
            continue
        matches.append(example)
        if example_id is not None:
            return matches
    if query_index is not None:
        if query_index >= len(matches):
            raise ValueError(
                f"No matching validation example found at query_index={query_index} for allowed episode counts {sorted(allowed_episode_counts)}"
            )
        return [matches[query_index]]
    if len(matches) >= limit:
        return matches[:limit]

    if example_id is not None:
        raise ValueError(
            f"No validation example found with example_id='{example_id}' for allowed episode counts {sorted(allowed_episode_counts)}"
        )
    return matches


def build_context_payload(
    example: OolongRealExample, *, palace_poc_strict: bool = False
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "example_id": example.example_id,
        "context_window_id": example.context_window_id,
        "campaign": example.campaign,
        "episodes": list(example.episodes),
        "question_type": example.question_type,
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
    n_palace_drawers: int = 0,
    palace_poc_strict: bool = False,
    palace_poc_by_speaker: bool = False,
    palace_poc_by_block: bool = False,
) -> str:
    if palace_poc_strict:
        from benchmark_tools.ephemeral_mempalace_poc import palace_poc_prompt_hint_strict

        prefix = palace_poc_prompt_hint_strict(
            n_palace_drawers,
            palace_poc_by_speaker=palace_poc_by_speaker,
            palace_poc_by_block=palace_poc_by_block,
        )
        return (
            prefix
            + "You are running inside an RLM REPL. The variable `context` contains the question in "
            "context['question'] and metadata (example_id, campaign, episodes, question_type) but "
            "does NOT contain the raw transcript.\n\n"
            "Task: Answer using ONLY information retrieved via `search_memories(query)` from the indexed "
            "transcript. Use the REPL to search, then aggregate evidence. You may use llm_query / rlm_query "
            "on retrieved text, but each call must **include that text in the prompt** (verbatim excerpts or "
            "a variable holding them)—subagents do not see the palace or empty context. "
            "Return the final answer with FINAL_VAR('answer')."
        )

    base = (
        "You are running inside an RLM REPL. The variable `context` is available and contains: "
        "(1) an Oolong Real question in context['question'] and (2) a long Dungeons and Dragons transcript "
        "in context['context_window_text']. Additional metadata includes context['example_id'], "
        "context['campaign'], context['episodes'], and context['question_type'].\n\n"
        "Task: Answer the question using ONLY the provided transcript. The transcript may be long, so inspect it "
        "carefully and use the REPL to search, count, or aggregate evidence before answering. Additionally, "
        "feel free to use batched llm or rlm queries to divide up the long context and process them in parallel. \n\n"
        "In a REPL block, print the keys of `context`, then compute an answer. Return the final answer with "
        "FINAL_VAR('answer')."
    )
    if palace_poc:
        from benchmark_tools.ephemeral_mempalace_poc import palace_poc_prompt_hint

        return (
            palace_poc_prompt_hint(
                n_palace_drawers,
                palace_poc_by_speaker=palace_poc_by_speaker,
                palace_poc_by_block=palace_poc_by_block,
            )
            + base
        )
    return base


def build_baseline_prompt(example: OolongRealExample) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Answer the user's question using only the provided Dungeons and Dragons transcript. "
                "Do not use external knowledge. Return only the final answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {example.question}\n\n"
                f"Transcript:\n{example.context_window_text}\n\n"
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
    palace_poc_by_speaker = args.palace_poc_by_speaker
    palace_poc_by_block = args.palace_poc_by_block
    if palace_poc_by_speaker and palace_poc_by_block:
        raise ValueError(
            "--palace-poc-by-speaker and --palace-poc-by-block are mutually exclusive."
        )
    if palace_poc_by_speaker and not palace_poc:
        raise ValueError("--palace-poc-by-speaker requires --palace-poc or --palace-poc-strict.")
    if palace_poc_by_block and not palace_poc:
        raise ValueError("--palace-poc-by-block requires --palace-poc or --palace-poc-strict.")
    if palace_poc and args.baseline:
        raise ValueError("--palace-poc / --palace-poc-strict cannot be combined with --baseline.")

    allowed_episode_counts = {2} if args.allow_two_episodes else {1}
    data_path = (
        DEFAULT_TWO_EPISODE_DATA_PATH
        if args.allow_two_episodes
        else DEFAULT_SINGLE_EPISODE_DATA_PATH
    )

    examples = load_examples(
        data_path=data_path,
        limit=args.limit,
        query_index=args.query_index,
        example_id=args.example_id,
        campaign=args.campaign,
        question_type=args.question_type,
        allowed_episode_counts=allowed_episode_counts,
    )
    if not examples:
        raise ValueError(
            f"No matching validation examples found for allowed episode counts {sorted(allowed_episode_counts)}"
        )

    print(f"Loaded {len(examples)} Oolong Real validation examples from {data_path}")

    backend_kwargs = get_backend_kwargs(args.model_name)
    subagent_backend_kwargs = get_backend_kwargs("gpt-5.4-mini")
    logger: RLMLogger | None = None
    rlm: RLM | None = None
    baseline_client = None

    if args.baseline:
        baseline_client = cast(Any, get_client("openai", backend_kwargs))
    elif not palace_poc:
        logger = RLMLogger(log_dir="./bench_Oolong_real/logs")
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
        )
    else:
        logger = RLMLogger(log_dir="./bench_Oolong_real/logs")

    try:
        for example in examples:
            print("---")
            print("example_id:", example.example_id)
            print("campaign:", example.campaign)
            print("episodes:", list(example.episodes))
            print("question_type:", example.question_type)
            print("context_chars:", len(example.context_window_text))
            print("question:", example.question)
            mode = "baseline" if args.baseline else "rlm"
            if palace_poc:
                mode += "_palace_poc_strict" if palace_poc_strict else "_palace_poc"
                if palace_poc_by_speaker:
                    mode += "_by_speaker"
                if palace_poc_by_block:
                    mode += "_by_block"
            print("mode:", mode)
            print("model_name:", args.model_name)
            print("system_prompt:", args.system_prompt)

            if args.baseline:
                response = baseline_client.completion(build_baseline_prompt(example))
            elif palace_poc:
                from benchmark_tools.ephemeral_mempalace_poc import build_ephemeral_palace_tools

                ingest_mode: Literal["sliding", "by_speaker", "by_block"] = "sliding"
                if palace_poc_by_speaker:
                    ingest_mode = "by_speaker"
                elif palace_poc_by_block:
                    ingest_mode = "by_block"
                custom_tools, cleanup, n_drawers = build_ephemeral_palace_tools(
                    example.context_window_text,
                    task_id=example.example_id,
                    metadata_prefix="oolong",
                    verbose=args.palace_poc_verbose,
                    ingest=ingest_mode,
                )
                rlm_one = RLM(
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
                try:
                    context_payload = build_context_payload(
                        example, palace_poc_strict=palace_poc_strict
                    )
                    result = rlm_one.completion(
                        context_payload,
                        root_prompt=build_prompt(
                            palace_poc=True,
                            n_palace_drawers=n_drawers,
                            palace_poc_strict=palace_poc_strict,
                            palace_poc_by_speaker=palace_poc_by_speaker,
                            palace_poc_by_block=palace_poc_by_block,
                        ),
                    )
                    response = result.response
                finally:
                    rlm_one.close()
                    cleanup()
            else:
                context_payload = build_context_payload(example)
                assert rlm is not None
                result = rlm.completion(context_payload, root_prompt=build_prompt())
                response = result.response

            print("Response:", response)
            print("Ground truth example_id:", example.example_id)
            print("Ground truth answer:", example.answer)
            if logger is not None and logger.log_file_path:
                print("Log file:", logger.log_file_path)
    finally:
        if rlm is not None:
            rlm.close()


if __name__ == "__main__":
    main()
