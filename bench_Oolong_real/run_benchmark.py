import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, cast

from rlm import RLM
from rlm.clients import get_client
from rlm.logger.rlm_logger import RLMLogger
from rlm.utils.subagent_encouraging_prompt import RLM_SYSTEM_PROMPT as SUBAGENT_ENCOURAGING_PROMPT
from dotenv import load_dotenv

load_dotenv()


DEFAULT_DATA_PATH = "bench_Oolong_real/data/validation_single_episode.jsonl"


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
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--query_index", type=int, default=None)
    parser.add_argument("--example_id", default=None)
    parser.add_argument("--campaign", default=None)
    parser.add_argument("--question_type", default=None)
    parser.add_argument(
        "--model_name",
        default=os.getenv("OOLONG_REAL_MODEL", "gpt-5-mini"),
    )
    parser.add_argument(
        "--system_prompt",
        choices=["default", "subagent_encouraging"],
        default="default",
    )
    parser.add_argument("--baseline", action="store_true")
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
    if len(example.episodes) != 1:
        raise ValueError("Expected only single-episode examples in local Oolong Real dataset")
    return example


def load_examples_from_jsonl(data_path: str) -> list[OolongRealExample]:
    examples: list[OolongRealExample] = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("Expected dict records in local Oolong Real JSONL")
            examples.append(normalize_example(record))
    return examples


def load_examples(
    *,
    data_path: str,
    limit: int,
    query_index: int | None,
    example_id: str | None,
    campaign: str | None,
    question_type: str | None,
) -> list[OolongRealExample]:
    if limit <= 0:
        raise ValueError("limit must be > 0")
    if query_index is not None and query_index < 0:
        raise ValueError("query_index must be >= 0")
    if query_index is not None and example_id is not None:
        raise ValueError("query_index and example_id are mutually exclusive")

    all_examples = load_examples_from_jsonl(data_path)
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
                f"No matching single-episode validation example found at query_index={query_index}"
            )
        return [matches[query_index]]
    if len(matches) >= limit:
        return matches[:limit]

    if example_id is not None:
        raise ValueError(
            f"No single-episode validation example found with example_id='{example_id}'"
        )
    return matches


def build_context_payload(example: OolongRealExample) -> dict[str, Any]:
    return {
        "example_id": example.example_id,
        "context_window_id": example.context_window_id,
        "campaign": example.campaign,
        "episodes": list(example.episodes),
        "question_type": example.question_type,
        "question": example.question,
        "context_window_text": example.context_window_text,
    }


def build_prompt() -> str:
    return (
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

    examples = load_examples(
        data_path=args.data_path,
        limit=args.limit,
        query_index=args.query_index,
        example_id=args.example_id,
        campaign=args.campaign,
        question_type=args.question_type,
    )
    if not examples:
        raise ValueError("No matching single-episode validation examples found")

    print(f"Loaded {len(examples)} Oolong Real validation examples from {args.data_path}")

    backend_kwargs = get_backend_kwargs(args.model_name)
    subagent_backend_kwargs = get_backend_kwargs("gpt-5-nano")
    logger: RLMLogger | None = None
    rlm: RLM | None = None
    baseline_client = None

    if args.baseline:
        baseline_client = cast(Any, get_client("openai", backend_kwargs))
    else:
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

    try:
        for example in examples:
            print("---")
            print("example_id:", example.example_id)
            print("campaign:", example.campaign)
            print("episodes:", list(example.episodes))
            print("question_type:", example.question_type)
            print("context_chars:", len(example.context_window_text))
            print("question:", example.question)
            print("mode:", "baseline" if args.baseline else "rlm")
            print("model_name:", args.model_name)
            print("system_prompt:", args.system_prompt)

            if args.baseline:
                response = baseline_client.completion(build_baseline_prompt(example))
            else:
                context_payload = build_context_payload(example)
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
