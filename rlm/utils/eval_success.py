"""
Benchmark-agnostic LLM judge for filling `success` in results JSONL rows.

Each line must be a JSON object with `ground_truth` and `response`. Rows with
`success: null` are evaluated in-place; rows with a boolean `success` are left
unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from rlm.clients import get_client
from rlm.clients.base_lm import BaseLM

_JUDGE_PROMPT = """You are grading whether a model's response matches a ground truth answer.
Be LENIENT: treat the response as correct if it conveys the same answer,
even if formatting, capitalisation, punctuation, or minor phrasing differ.
Mark it incorrect only if the answer is substantively wrong or a refusal.

Ground truth: {ground_truth}
Model response: {response}

Reply with exactly one word: "yes" or "no".
"""


def judge_success(ground_truth: str, response: str, client: BaseLM) -> bool:
    prompt = _JUDGE_PROMPT.format(ground_truth=ground_truth, response=response)
    raw = client.completion(prompt)
    return _parse_yes_no(raw)


def _parse_yes_no(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        raise ValueError("empty judge response")
    first = t.split()[0]
    first = first.strip(".,;:!?\"'")
    if first in ("yes", "y", "true", "correct"):
        return True
    if first in ("no", "n", "false", "incorrect"):
        return False
    raise ValueError(f"expected yes/no, got: {text!r}")


def discover_results_jsonl(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("results.jsonl") if p.is_file())


def process_jsonl_file(path: Path, client: BaseLM) -> tuple[int, int, int]:
    """
    Returns (evaluated_count, skipped_count, warn_count).
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    out_rows: list[str] = []
    evaluated = 0
    skipped = 0
    warned = 0

    for line in lines:
        if not line.strip():
            out_rows.append(line)
            continue
        row: dict[str, Any] = json.loads(line)
        success = row.get("success")
        if success is not None:
            skipped += 1
            out_rows.append(json.dumps(row, ensure_ascii=False))
            continue

        gt = row.get("ground_truth")
        resp = row.get("response")
        if gt is None or resp is None:
            warned += 1
            print(
                f"warn {path}: skip row missing ground_truth or response "
                f"(keys present: {sorted(row.keys())})"
            )
            out_rows.append(json.dumps(row, ensure_ascii=False))
            continue

        gt_s = gt if isinstance(gt, str) else str(gt)
        resp_s = resp if isinstance(resp, str) else str(resp)
        row["success"] = judge_success(gt_s, resp_s, client)
        evaluated += 1
        out_rows.append(json.dumps(row, ensure_ascii=False))

    text = "\n".join(out_rows)
    if lines and not text.endswith("\n"):
        text += "\n"

    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=".results.",
        suffix=".jsonl.tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
            tmp_f.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

    return evaluated, skipped, warned


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fill success in results.jsonl using an LLM judge (rows with success=null only)."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--dir",
        type=Path,
        help="Directory; every results.jsonl under it is processed recursively",
    )
    g.add_argument("--file", type=Path, help="Single results.jsonl path")
    p.add_argument(
        "--model",
        default="gpt-5.4-nano",
        help="OpenAI model name for the judge (default: gpt-5.4-nano)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY2")
    if not api_key:
        raise ValueError("OPENAI_API_KEY2 is required for eval_success.")

    client = get_client(
        "openai",
        {"api_key": api_key, "model_name": args.model},
    )

    if args.file is not None:
        paths = [args.file.resolve()]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"not a file: {path}")
    else:
        root = args.dir.resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"not a directory: {root}")
        paths = discover_results_jsonl(root)
        if not paths:
            print(f"no results.jsonl found under {root}")
            return

    for path in paths:
        ev, sk, wn = process_jsonl_file(path, client)
        print(f"{path}: evaluated={ev} skipped={sk} warnings={wn}")


if __name__ == "__main__":
    main()
