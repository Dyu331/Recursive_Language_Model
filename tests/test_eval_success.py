"""Tests for rlm.utils.eval_success judge JSON parsing and normalization."""

import json
from unittest.mock import MagicMock

import pytest

from rlm.utils import eval_success as es


def test_normalize_success_wins() -> None:
    assert es._normalize_judge_labels(True, True) == (True, False)
    assert es._normalize_judge_labels(True, False) == (True, False)


def test_normalize_format_error_forces_success_false() -> None:
    assert es._normalize_judge_labels(False, True) == (False, True)
    assert es._normalize_judge_labels(False, False) == (False, False)


def test_parse_judge_json_plain_object() -> None:
    assert es._parse_judge_json('{"success": true, "format_error": false}') == (True, False)
    assert es._parse_judge_json('{"format_error": true, "success": false}') == (False, True)


def test_parse_judge_json_contradiction_success_priority() -> None:
    assert es._parse_judge_json('{"success": true, "format_error": true}') == (True, False)


def test_parse_judge_json_raw_then_normalize_matches_parse_judge_json() -> None:
    raw = '{"success": true, "format_error": true}'
    assert es._normalize_judge_labels(*es._parse_judge_json_raw(raw)) == es._parse_judge_json(raw)


def test_parse_judge_json_raw_is_unnormalized_contradiction() -> None:
    raw = '{"success": true, "format_error": true}'
    assert es._parse_judge_json_raw(raw) == (True, True)


def test_needs_format_backfill() -> None:
    assert not es._needs_format_backfill({"success": True}, enabled=False)
    assert not es._needs_format_backfill({"success": True, "format_error": False}, enabled=True)
    assert es._needs_format_backfill({"success": True}, enabled=True)
    assert es._needs_format_backfill({"success": True, "format_error": None}, enabled=True)
    assert not es._needs_format_backfill({"success": None}, enabled=True)


def test_parse_judge_json_markdown_fence() -> None:
    raw = """```json
{"success": false, "format_error": true}
```
"""
    assert es._parse_judge_json(raw) == (False, True)


def test_parse_judge_json_leading_junk() -> None:
    assert es._parse_judge_json('here: {"success": false, "format_error": false}') == (
        False,
        False,
    )


def test_parse_judge_json_invalid_raises() -> None:
    with pytest.raises(ValueError, match="no JSON"):
        es._parse_judge_json("no braces")
    with pytest.raises(ValueError, match="invalid judge JSON"):
        es._parse_judge_json("{not json")
    with pytest.raises(ValueError, match="must include"):
        es._parse_judge_json('{"success": true}')
    with pytest.raises(ValueError, match="must be booleans"):
        es._parse_judge_json('{"success": "yes", "format_error": false}')


def test_serialize_eval_row_order() -> None:
    row = {"task_id": "1", "response": "x", "ground_truth": "y", "trial": 1}
    s = es.serialize_eval_row(row, True, False)
    data = json.loads(s)
    keys = list(data.keys())
    assert keys[-2] == "success"
    assert keys[-1] == "format_error"
    assert data["success"] is True
    assert data["format_error"] is False


def test_process_jsonl_file_skips_fully_graded_rows(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    row = {
        "ground_truth": "a",
        "response": "b",
        "success": False,
        "format_error": False,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    client = MagicMock()
    ev, sk, wn, fe, bf = es.process_jsonl_file(path, client)
    assert (ev, sk, wn, fe, bf) == (0, 1, 0, 0, 0)
    client.completion.assert_not_called()


def test_process_jsonl_file_force_reeval_overwrites_booleans(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    row = {
        "ground_truth": "a",
        "response": "b",
        "success": False,
        "format_error": False,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    client = MagicMock()
    client.completion.return_value = '{"success": true, "format_error": false}'
    ev, sk, wn, fe, bf = es.process_jsonl_file(path, client, force_reeval=True)
    assert (ev, sk, wn, fe, bf) == (1, 0, 0, 0, 0)
    client.completion.assert_called_once()
    out = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert out["success"] is True
    assert out["format_error"] is False
