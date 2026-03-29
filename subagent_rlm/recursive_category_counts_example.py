import json
import os
import sys

from dotenv import load_dotenv

from rlm import RLM
from rlm.utils.subagent_encouraging_prompt import (
    RLM_SYSTEM_PROMPT as SUBAGENT_ENCOURAGING_SYSTEM_PROMPT,
)
from rlm.logger import RLMLogger

load_dotenv()


CATEGORIES = ["fruits", "countries", "animals"]

PROMPT_MODE_DEFAULT = "default"
PROMPT_MODE_RECURSIVE = "recursive"


def print_separator(char: str = "─", width: int = 80) -> None:
    print(char * width)


def extract_subcalls(result) -> list[dict]:
    if not result.metadata:
        return []
    subcalls: list[dict] = []
    for iteration in result.metadata.get("iterations", []):
        for code_block in iteration.get("code_blocks", []):
            repl_result = code_block.get("result", {})
            subcalls.extend(repl_result.get("rlm_calls", []))
    return subcalls


def get_prompt_mode() -> str:
    prompt_mode = os.getenv("RLM_PROMPT_MODE", PROMPT_MODE_RECURSIVE).strip().lower()
    if prompt_mode not in {PROMPT_MODE_DEFAULT, PROMPT_MODE_RECURSIVE}:
        raise ValueError(
            "RLM_PROMPT_MODE must be 'default' or 'recursive'"
        )
    return prompt_mode


def build_default_prompt() -> str:
    return (
        "You are running inside an RLM REPL. Your goal is to build a nested Python dictionary with "
        "three top-level keys: fruits, countries, animals. For each category, produce exactly 50 distinct "
        "names and map each name to the number of letter 'r' or 'R' characters in that name. "
        "Use the REPL to organize the work, validate the final structure, store the nested dictionary in "
        "a variable called result, and return it with FINAL_VAR('result')."
    )


def build_recursive_prompt() -> str:
    return (
        "You are running inside an RLM REPL. Your goal is to build a nested Python dictionary with "
        "three top-level keys: fruits, countries, animals. For each category, produce exactly 50 distinct "
        "names and map each name to the number of letter 'r' or 'R' characters in that name.\n\n"
        "These three category subtasks are independent and each requires deeper semantic work, so you should "
        "strongly prefer a single rlm_query_batched() call with one child prompt per category instead of three "
        "separate sequential rlm_query() calls. Use the parent REPL to aggregate the three child results, validate "
        "the final nested dictionary, store it in a variable called result, and return it with FINAL_VAR('result')."
    )


def build_recursive_system_prompt() -> str:
    return (
        SUBAGENT_ENCOURAGING_SYSTEM_PROMPT
        + "\n\nWhen a task naturally decomposes into multiple similar categories, you should strongly prefer "
        "delegating each category to a child via a single rlm_query_batched() call instead of solving everything "
        "with llm_query() or issuing multiple sequential rlm_query() calls."
    )


def build_prompt(prompt_mode: str) -> str:
    if prompt_mode == PROMPT_MODE_DEFAULT:
        return build_default_prompt()
    return build_recursive_prompt()


def build_system_prompt(prompt_mode: str) -> str | None:
    if prompt_mode == PROMPT_MODE_DEFAULT:
        return None
    return build_recursive_system_prompt()


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set. Set it and re-run.")
        sys.exit(1)

    model = os.getenv("RLM_MODEL", "gpt-4o")
    prompt_mode = get_prompt_mode()
    system_prompt = build_system_prompt(prompt_mode)

    print_separator("=")
    print("  Recursive Category Counts Example")
    print(
        f"  Model: {model}  |  max_depth=2  |  max_iterations=15  |  prompt_mode={prompt_mode}"
    )
    print_separator("=")
    print()

    logger = RLMLogger(log_dir="./subagent_rlm/log")

    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": model, "api_key": api_key},
        environment="local",
        max_depth=2,
        max_iterations=15,
        custom_system_prompt=system_prompt,
        logger=logger,
        verbose=True,
    )

    prompt = build_prompt(prompt_mode)
    print(prompt)
    print()
    print_separator()

    result = rlm.completion(prompt)

    print_separator("=")
    print("  RESULT")
    print_separator("=")
    print(result.response)
    print()

    try:
        parsed = json.loads(result.response.replace("'", '"'))
        print("Top-level keys:", list(parsed.keys()))
        for category in CATEGORIES:
            value = parsed.get(category, {})
            size = len(value) if isinstance(value, dict) else "not-a-dict"
            print(f"{category}: {size}")
    except Exception:
        print("Could not parse final response as JSON directly.")
    print()

    subcalls = extract_subcalls(result)
    print_separator("=")
    print("  SUB-CALL SUMMARY")
    print_separator("=")
    print(f"Total sub-calls captured: {len(subcalls)}")
    rlm_subcalls = 0
    for index, subcall in enumerate(subcalls, start=1):
        is_recursive = subcall.get("metadata") is not None
        if is_recursive:
            rlm_subcalls += 1
        call_type = "RLM" if is_recursive else "LLM"
        preview = str(subcall.get("response", ""))[:120]
        print(f"{index}. {call_type} sub-call | model={subcall.get('root_model', '?')} | response={preview}")
    print(f"Recursive sub-calls with metadata: {rlm_subcalls}")
    print()

    if logger.log_file_path:
        print("Log file:", logger.log_file_path)


if __name__ == "__main__":
    main()
