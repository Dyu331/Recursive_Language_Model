"""
Benchmark-agnostic LLM judge for filling `success` and `format_error` in results JSONL rows.

Each line must be a JSON object with `ground_truth` and `response`. Rows with
`success: null` are evaluated in-place (both fields). Rows with boolean `success`
and boolean `format_error` are skipped unless `--force-reeval` is set (then every
row with `ground_truth` and `response` is re-judged and both fields overwritten).

With `--backfill-format-error`, rows that already have boolean `success` but are
missing or non-boolean `format_error` get one judge call: `success` is preserved
and `format_error` is filled using the same normalization rules as full eval.

For evaluated rows, `format_error` is true when the response is process/meta text
that defers a final answer (e.g. awaiting REPL output), not when the model gives a
terminal refusal or uncertainty (INSUFFICIENT_EVIDENCE, could not tell, etc.).

After parsing the judge JSON, labels are normalized so they cannot contradict:
if `success` is true, `format_error` is forced to false; else if `format_error`
is true, `success` is forced to false (for full eval; backfill keeps stored
`success` and only applies the rule that true success forces format_error false).

Serialized rows place `format_error` as the last key after `success`.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from json import JSONDecoder
from pathlib import Path
from typing import Any

from rlm.clients import get_client
from rlm.clients.base_lm import BaseLM

_JUDGE_PROMPT = """You are grading benchmark rows. Reply with a single JSON object only, no other text.
Use this exact shape: {{"success": <true or false>, "format_error": <true or false>}}

Field "success" (lenient match to ground truth):
- true if the model response conveys the same answer as ground truth, even if formatting, capitalisation, or punctuation differ.
- If ground_truth looks like a Python list of quoted strings (e.g. "['less common than']" or "['a', 'b']"), treat each listed string as an acceptable answer: success is true if the response clearly states the same claim as any one of them, even if the model adds a prefix like "Answer:" or rephrases with extra labels (e.g. "numeric value is less common than abbreviation" matches "less common than" when that is the comparative claim).
- false if the answer is substantively wrong, or the model gives a terminal refusal/uncertainty as its final stance.

Field "format_error":
- true ONLY if the response is NOT a final answer to the task, but process/meta text that defers answering — for example: waiting on REPL or "next step" before answering, describing inspection/narrowing without stating the benchmark answer, phrases like "Awaiting REPL output from the inspection step", or deferring semantic work to a future rlm_query step.
- false for any terminal outcome, including: a wrong or right substantive answer; token-style refusals; or substantive "no answer" conclusions such as INSUFFICIENT, INSUFFICIENT_EVIDENCE, NOT_SUPPORTED, "could not determine", "cannot tell from the corpus", "unclear from the provided documents", or similar. Those are final outcomes, not format errors.

Ground truth: {ground_truth}
Model response: {response}
"""


def judge_row(ground_truth: str, response: str, client: BaseLM) -> tuple[bool, bool]:
    prompt = _JUDGE_PROMPT.format(ground_truth=ground_truth, response=response)
    raw = client.completion(prompt)
    return _normalize_judge_labels(*_parse_judge_json_raw(raw))


def _normalize_judge_labels(success: bool, format_error: bool) -> tuple[bool, bool]:
    if success:
        return True, False
    if format_error:
        return False, True
    return False, False


def _strip_code_fence(s: str) -> str:
    t = s.strip()
    if not t.startswith("```"):
        return t
    lines = t.split("\n")
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_judge_json_raw(text: str) -> tuple[bool, bool]:
    """Parse judge JSON; return (success, format_error) without normalizing."""
    t = _strip_code_fence(text)
    if not t:
        raise ValueError("empty judge response")
    start = t.find("{")
    if start == -1:
        raise ValueError(f"judge response has no JSON object: {text!r}")
    decoder = JSONDecoder()
    try:
        obj, _end = decoder.raw_decode(t[start:])
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid judge JSON: {e}; text={text!r}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"judge JSON must be an object, got {type(obj)}: {text!r}")
    if "success" not in obj or "format_error" not in obj:
        raise ValueError(f"judge JSON must include success and format_error: {obj!r}")
    su = obj["success"]
    fe = obj["format_error"]
    if not isinstance(su, bool) or not isinstance(fe, bool):
        raise ValueError(
            f"success and format_error must be booleans, got success={su!r} format_error={fe!r}"
        )
    return su, fe


def _parse_judge_json(text: str) -> tuple[bool, bool]:
    return _normalize_judge_labels(*_parse_judge_json_raw(text))


def _needs_format_backfill(row: dict[str, Any], *, enabled: bool) -> bool:
    return bool(
        enabled
        and isinstance(row.get("success"), bool)
        and not isinstance(row.get("format_error"), bool)
    )


def serialize_eval_row(row: dict[str, Any], success: bool, format_error: bool) -> str:
    """Emit JSON with original key order preserved, then success, then format_error last."""
    ordered: dict[str, Any] = {}
    for k, v in row.items():
        if k in ("success", "format_error"):
            continue
        ordered[k] = v
    ordered["success"] = success
    ordered["format_error"] = format_error
    return json.dumps(ordered, ensure_ascii=False)


def discover_results_jsonl(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("results.jsonl") if p.is_file())


def process_jsonl_file(
    path: Path,
    client: BaseLM,
    *,
    backfill_format_error: bool = False,
    force_reeval: bool = False,
) -> tuple[int, int, int, int, int]:
    """
    Returns (evaluated_count, skipped_count, warn_count, format_error_count, backfill_count).
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    out_rows: list[str] = []
    evaluated = 0
    skipped = 0
    warned = 0
    format_errors = 0
    backfilled = 0

    for line in lines:
        if not line.strip():
            out_rows.append(line)
            continue
        row: dict[str, Any] = json.loads(line)
        success = row.get("success")
        format_error = row.get("format_error")

        if isinstance(success, bool) and isinstance(format_error, bool) and not force_reeval:
            skipped += 1
            out_rows.append(json.dumps(row, ensure_ascii=False))
            continue

        gt = row.get("ground_truth")
        resp = row.get("response")

        if _needs_format_backfill(row, enabled=backfill_format_error) and not force_reeval:
            if gt is None or resp is None:
                warned += 1
                print(
                    f"warn {path}: backfill skip row missing ground_truth or response "
                    f"(keys present: {sorted(row.keys())})"
                )
                out_rows.append(json.dumps(row, ensure_ascii=False))
                continue
            gt_s = gt if isinstance(gt, str) else str(gt)
            resp_s = resp if isinstance(resp, str) else str(resp)
            prompt = _JUDGE_PROMPT.format(ground_truth=gt_s, response=resp_s)
            raw = client.completion(prompt)
            _, fe_raw = _parse_judge_json_raw(raw)
            preserved = bool(row["success"])
            suc, fe = _normalize_judge_labels(preserved, fe_raw)
            if fe:
                format_errors += 1
            backfilled += 1
            out_rows.append(serialize_eval_row(row, suc, fe))
            continue

        if success is not None and not force_reeval:
            if backfill_format_error and not isinstance(success, bool):
                warned += 1
                print(
                    f"warn {path}: skip row with non-bool success={success!r} "
                    f"(keys present: {sorted(row.keys())})"
                )
            skipped += 1
            out_rows.append(json.dumps(row, ensure_ascii=False))
            continue

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
        suc, fe = judge_row(gt_s, resp_s, client)
        if fe:
            format_errors += 1
        evaluated += 1
        out_rows.append(serialize_eval_row(row, suc, fe))

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

    return evaluated, skipped, warned, format_errors, backfilled


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fill success and format_error in results.jsonl using an LLM judge. "
            "By default only rows with success=null are evaluated (rows with both "
            "boolean success and format_error are skipped). Use --force-reeval to "
            "re-judge every row that has ground_truth and response. Use "
            "--backfill-format-error to add format_error where success is already set."
        )
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
    p.add_argument(
        "--backfill-format-error",
        action="store_true",
        help=(
            "For rows with boolean success but missing or non-boolean format_error, "
            "call the judge once and set format_error only (success unchanged)."
        ),
    )
    p.add_argument(
        "--force-reeval",
        action="store_true",
        help=(
            "Re-run the judge on every row with ground_truth and response, overwriting "
            "success and format_error even when they are already booleans."
        ),
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

    total_fe = 0
    total_bf = 0
    for path in paths:
        ev, sk, wn, fe, bf = process_jsonl_file(
            path,
            client,
            backfill_format_error=args.backfill_format_error,
            force_reeval=args.force_reeval,
        )
        total_fe += fe
        total_bf += bf
        print(
            f"{path}: evaluated={ev} skipped={sk} warnings={wn} format_errors={fe} backfilled={bf}"
        )
    if len(paths) > 1:
        print(f"total format_errors={total_fe} total backfilled={total_bf}")


if __name__ == "__main__":
    main()
