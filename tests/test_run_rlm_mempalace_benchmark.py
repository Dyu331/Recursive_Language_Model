"""Unit tests for bench_Oolong_real/run_rlm_mempalace_benchmark.py CLI helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_BENCH = _REPO / "bench_Oolong_real"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import run_rlm_mempalace_benchmark as mp  # noqa: E402


def test_default_ingest_is_by_block() -> None:
    ns = mp.parse_args([])
    ingest, by_speaker, by_block = mp.resolve_palace_ingest_and_prompt_flags(ns)
    assert ingest == "by_block"
    assert by_speaker is False
    assert by_block is True


def test_by_speaker_override() -> None:
    ns = mp.parse_args(["--palace-poc-by-speaker"])
    ingest, by_speaker, by_block = mp.resolve_palace_ingest_and_prompt_flags(ns)
    assert ingest == "by_speaker"
    assert by_speaker is True
    assert by_block is False


def test_by_speaker_and_by_block_mutually_exclusive() -> None:
    ns = mp.parse_args(["--palace-poc-by-speaker", "--palace-poc-by-block"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        mp.resolve_palace_ingest_and_prompt_flags(ns)


def test_condition_slug_lenient_by_block() -> None:
    assert (
        mp.condition_slug(
            palace_poc_strict=False,
            palace_poc_by_speaker=False,
            palace_poc_by_block=True,
        )
        == "palace_poc_by_block"
    )


def test_condition_slug_strict_by_block() -> None:
    assert (
        mp.condition_slug(
            palace_poc_strict=True,
            palace_poc_by_speaker=False,
            palace_poc_by_block=True,
        )
        == "palace_poc_strict_by_block"
    )


def test_allow_two_and_mixed_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        mp.parse_args(["--allow_two_episodes", "--mixed_25_episodes"])


def test_mixed_rejects_data_path() -> None:
    ns = mp.parse_args(["--mixed_25_episodes"])
    ns.data_path = "/tmp/custom.jsonl"
    with pytest.raises(ValueError, match="--data_path"):
        mp.validate_mixed_no_custom_data_path(ns)


def test_mixed_id_slices_length() -> None:
    assert len(mp.rep.REPLICATE_EXAMPLE_IDS_SINGLE[: mp.MIXED_SUITE_N]) == mp.MIXED_SUITE_N
    assert len(mp.rep.REPLICATE_EXAMPLE_IDS_TWO[: mp.MIXED_SUITE_N]) == mp.MIXED_SUITE_N


def test_load_completed_keys_includes_suite(tmp_path: Path) -> None:
    p = tmp_path / "r.jsonl"
    p.write_text(
        '{"task_id":"a","baseline":"b","trial":1,"condition":"c","suite":"single_50"}\n',
        encoding="utf-8",
    )
    assert mp.load_completed_keys(p) == {("a", "b", 1, "c", "single_50")}
