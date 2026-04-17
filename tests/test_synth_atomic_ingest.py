"""Deterministic tests for Oolong-synth atomic-record ingest helpers (no Chroma)."""

from __future__ import annotations

import json

import pytest

from benchmark_tools.ephemeral_mempalace_poc import (
    UNCLASSIFIED_ROOM,
    _exact_search_over_chunks_with_filters,
    _merge_multiroom_semantic_hits,
    assert_synth_preamble_count,
    chunks_for_atomic_ingest,
    count_core_records,
    format_taxonomy_json,
    parse_atomic_records,
    slugify_label,
)

FIXTURE_TEXT = "\n".join(
    [
        "The following lines contain 6 general-knowledge questions, one per line.",
        "",
        "You will be asked to answer questions about the aggregate label statistics.",
        "",
        "Date: Sep 06, 2023 || User: 14512 || Instance: What is a tonne ? || Label: numeric value",
        "Date: Jun 21, 2023 || User: 16295 || Instance: Where is the Orange Bowl ? || Label: location",
        "Date: Jan 14, 2024 || User: 98142 || Instance: What king is satirized ? || Label: human",
        "Date: Sep 11, 2024 || User: 40405 || Instance: How long to type a screenplay ? || Label: numeric value",
        "Date: Jan 09, 2024 || User: 14512 || Instance: What tokens are in Monopoly ? || Label: entity",
        "Date: Feb 26, 2024 || User: 80488 || Instance: What President bred mules ? || Label: human",
        "",
        "",
    ]
)


def test_parse_atomic_records_assigns_label_room_and_flags_core() -> None:
    records = parse_atomic_records(FIXTURE_TEXT)
    core = [r for r in records if r.get("is_record")]
    unclassified = [r for r in records if not r.get("is_record")]
    assert len(core) == 6
    assert len(unclassified) == 2  # two non-empty preamble instruction lines
    assert all(r["room"] == UNCLASSIFIED_ROOM for r in unclassified)

    first = core[0]
    assert first["line_index"] == 5
    assert first["user_id"] == "14512"
    assert first["date_iso"] == "2023-09-06"
    assert first["instance"] == "What is a tonne ?"
    assert first["label"] == "numeric value"
    assert first["room"] == slugify_label("numeric value") == "numeric_value"

    rooms_seen = {r["room"] for r in core}
    assert rooms_seen == {"numeric_value", "location", "human", "entity"}


def test_assert_synth_preamble_count_matches_core_only() -> None:
    records = parse_atomic_records(FIXTURE_TEXT)
    assert_synth_preamble_count(FIXTURE_TEXT, count_core_records(records))


def test_assert_synth_preamble_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="preamble claims 6"):
        assert_synth_preamble_count(FIXTURE_TEXT, 5)


def test_assert_synth_preamble_count_missing_preamble_is_noop() -> None:
    assert_synth_preamble_count("no preamble here", 0)


def test_chunks_for_atomic_ingest_shape_and_rooms() -> None:
    records = parse_atomic_records(FIXTURE_TEXT)
    chunks = chunks_for_atomic_ingest(records, wing="benchmark")
    assert len(chunks) == len(records)
    for i, chunk in enumerate(chunks):
        assert chunk["wing"] == "benchmark"
        assert chunk["chunk_index"] == i
        assert isinstance(chunk["line_index"], str) and chunk["line_index"].isdigit()
        if records[i].get("is_record"):
            assert chunk["room"] == records[i]["room"]
            assert chunk["label"] == records[i]["label"]
            assert chunk["user_id"] == records[i]["user_id"]
        else:
            assert chunk["room"] == UNCLASSIFIED_ROOM
            assert "label" not in chunk


def test_taxonomy_json_counts_match_label_rooms() -> None:
    records = parse_atomic_records(FIXTURE_TEXT)
    chunks = chunks_for_atomic_ingest(records, wing="benchmark")
    data = json.loads(format_taxonomy_json(chunks))
    wing = data["wings"][0]
    room_counts = {r["room"]: r["drawers"] for r in wing["rooms"]}
    assert room_counts == {
        "entity": 1,
        "human": 2,
        "location": 1,
        "numeric_value": 2,
        UNCLASSIFIED_ROOM: 2,
    }
    room_order = [r["room"] for r in wing["rooms"]]
    assert room_order[-1] == UNCLASSIFIED_ROOM


def _build_chunks() -> list[dict[str, object]]:
    records = parse_atomic_records(FIXTURE_TEXT)
    return chunks_for_atomic_ingest(records, wing="benchmark")


def test_exact_search_room_filter_location_only() -> None:
    chunks = _build_chunks()
    out = _exact_search_over_chunks_with_filters(
        chunks,
        "instance",
        wing=None,
        room="location",
        n_results=10,
    )
    assert len(out["results"]) == 1
    assert out["results"][0]["room"] == "location"
    assert out["results"][0]["label"] == "location"


def test_exact_search_no_room_filter_searches_all_rooms() -> None:
    chunks = _build_chunks()
    out = _exact_search_over_chunks_with_filters(
        chunks,
        "instance",
        wing=None,
        room=None,
        n_results=10,
    )
    # All 6 core records carry "Instance:" substring; unclassified preamble lines do not.
    assert len(out["results"]) == 6
    assert {h["room"] for h in out["results"]} >= {
        "numeric_value",
        "location",
        "human",
        "entity",
    }


def test_exact_search_filter_by_user_and_date_range() -> None:
    chunks = _build_chunks()
    out = _exact_search_over_chunks_with_filters(
        chunks,
        "instance",
        wing=None,
        room=None,
        n_results=10,
        user="14512",
    )
    assert {h["user_id"] for h in out["results"]} == {"14512"}

    out2 = _exact_search_over_chunks_with_filters(
        chunks,
        "instance",
        wing=None,
        room=None,
        n_results=10,
        date_from="2024-01-01",
        date_to="2024-02-01",
    )
    assert len(out2["results"]) == 2


def test_merge_multiroom_semantic_hits_dedupes_and_sorts() -> None:
    per_room_outs = [
        {
            "results": [
                {"text": "hit A", "room": "location", "line_index": "000006", "distance": 0.10},
                {"text": "hit B", "room": "location", "line_index": "000010", "distance": 0.50},
            ]
        },
        {
            "results": [
                # duplicate line_index with previous room → should be dropped
                {"text": "hit B dup", "room": "human", "line_index": "000010", "distance": 0.40},
                {"text": "hit C", "room": "human", "line_index": "000007", "distance": 0.05},
            ]
        },
    ]
    merged = _merge_multiroom_semantic_hits(per_room_outs, query="q", n_results=5)
    texts = [h["text"] for h in merged["results"]]
    assert texts == ["hit C", "hit A", "hit B"]  # sorted by best score (lowest distance)
    assert merged["filters"] == {"wing": None, "room": None}


def test_merge_multiroom_propagates_error() -> None:
    per_room_outs = [{"results": []}, {"error": "boom", "results": []}]
    merged = _merge_multiroom_semantic_hits(per_room_outs, query="q", n_results=3)
    assert merged.get("error") == "boom"
