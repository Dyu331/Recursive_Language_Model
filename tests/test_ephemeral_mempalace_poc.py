"""Tests for benchmark MemPalace PoC (optional mempalace dependency)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmark_tools.ephemeral_mempalace_poc import (
    DEFAULT_CHUNK_SIZE,
    END_OF_EPISODE,
    LINE_INDEX_PAD_WIDTH,
    PREAMBLE_SPEAKER,
    START_OF_EPISODE,
    _chunk_text,
    _drawer_id,
    _exact_search_over_chunks,
    _hit_semantic_score,
    build_grouped_speaker_blocks_from_transcript,
    build_speaker_blocks_from_transcript,
    build_temporal_turn_blocks_from_transcript,
    chunks_for_grouped_speaker_ingest,
    chunks_for_speaker_ingest,
    chunks_for_temporal_turn_ingest,
    collect_speaker_slugs_from_labeled_text,
    extract_preamble_and_episode,
    format_palace_lobby_preview,
    format_taxonomy_json,
    group_dense_window_lines_into_interaction_drawers,
    merge_micro_blocks_to_grouped_drawers,
    ordered_speakers_from_labeled_lines,
    parse_transcript_speaker_blocks,
    parse_transcript_speaker_blocks_with_lines,
    slugify_speaker,
)


def test_hit_semantic_score_maps_chroma_cosine_distance() -> None:
    """Chroma cosine distance is in [0, 2]; score uses 1 - d/2 (not 1 - d, which is 0 for d > 1)."""
    hit: dict[str, Any] = {"similarity": 0.0, "distance": 0.92, "effective_distance": 0.88}
    assert _hit_semantic_score(hit) == pytest.approx(1.0 - 0.88 / 2.0)


def test_hit_semantic_score_distance_over_one_nonzero() -> None:
    hit: dict[str, Any] = {"similarity": 0.0, "distance": 1.3425}
    assert _hit_semantic_score(hit) == pytest.approx(1.0 - 1.3425 / 2.0)


def test_chunk_text_empty() -> None:
    assert _chunk_text("") == []
    assert _chunk_text("   \n\n  ") == []


def test_chunk_text_single_short_below_min() -> None:
    assert _chunk_text("x" * 30) == []


def test_chunk_text_preserves_order() -> None:
    body = "a" * 100 + "\n\n" + "b" * 100
    chunks = _chunk_text(body, chunk_size=80, chunk_overlap=10)
    assert len(chunks) >= 2
    assert all("chunk_index" in c and "content" in c for c in chunks)


def test_format_palace_lobby_preview_truncates_long_drawer() -> None:
    tid = "unit-test-task"
    chunks = [
        {"chunk_index": 0, "content": "A" * 80},
        {"chunk_index": 1, "content": "hello\n\nworld"},
    ]
    out = format_palace_lobby_preview(chunks, task_id=tid, preview_chars=50)
    assert f"task_id={tid!r}" in out
    assert "Wing: benchmark" in out
    assert "Room: corpus (2 drawers)" in out
    assert "[0]" in out and "[1]" in out
    assert _drawer_id(tid, 0) in out
    line0 = [ln for ln in out.splitlines() if ln.strip().startswith("[0]")][0]
    preview_part = line0.split(_drawer_id(tid, 0), 1)[1].strip()
    assert preview_part == "A" * 50 + "..."
    line1 = [ln for ln in out.splitlines() if ln.strip().startswith("[1]")][0]
    assert "hello world" in line1


def test_slugify_speaker_preamble_reserved() -> None:
    assert slugify_speaker(PREAMBLE_SPEAKER) == "_preamble"
    assert slugify_speaker("Matt") == "matt"
    assert slugify_speaker("Chris perkins") == "chris_perkins"


def test_parse_transcript_speaker_blocks_continuation_and_prefix() -> None:
    episode = (
        "MATT: He looks at you and says...\n"
        "(The group laughs)\n"
        "...I'm not sure about that.\n"
        "LAURA: Hi there."
    )
    blocks = parse_transcript_speaker_blocks(episode)
    assert len(blocks) == 2
    assert blocks[0]["speaker"] == "MATT"
    assert "He looks at you" in blocks[0]["text"]
    assert "(The group laughs)" in blocks[0]["text"]
    assert "...I'm not sure" in blocks[0]["text"]
    assert blocks[1]["speaker"] == "LAURA"
    assert "Hi there" in blocks[1]["text"]


def test_parse_transcript_prefix_lines_attach_to_first_speaker() -> None:
    episode = "Orphan stage line\nMATT: Start.\nLAURA: End."
    blocks = parse_transcript_speaker_blocks(episode)
    assert blocks[0]["speaker"] == "MATT"
    assert "Orphan stage line" in blocks[0]["text"]
    assert "Start." in blocks[0]["text"]


def test_chunks_for_speaker_ingest_same_room_all_subchunks() -> None:
    body = "word " * 6000
    blocks = [{"speaker": "Matt", "text": body}]
    chunks = chunks_for_speaker_ingest(blocks, wing="benchmark", chunk_size=200, chunk_overlap=20)
    assert len(chunks) >= 2
    for c in chunks:
        assert c["room"] == "matt"
        assert c["wing"] == "benchmark"
        assert c["speaker"] == "Matt"


def test_format_taxonomy_json_preamble_first() -> None:
    chunks = [
        {"wing": "benchmark", "room": "zebra", "chunk_index": 0},
        {"wing": "benchmark", "room": "_preamble", "chunk_index": 1},
        {"wing": "benchmark", "room": "amy", "chunk_index": 2},
    ]
    data = json.loads(format_taxonomy_json(chunks))
    rooms = data["wings"][0]["rooms"]
    assert [r["room"] for r in rooms] == ["_preamble", "amy", "zebra"]


def test_format_palace_lobby_preview_preamble_room_first() -> None:
    chunks = [
        {"chunk_index": 0, "content": "zebra text", "wing": "benchmark", "room": "zebra"},
        {"chunk_index": 1, "content": "pre", "wing": "benchmark", "room": "_preamble"},
    ]
    out = format_palace_lobby_preview(chunks, task_id="t1")
    assert out.index("Room: _preamble") < out.index("Room: zebra")


def test_extract_preamble_and_episode_prefers_real_markers_over_instruction_echo() -> None:
    """Boilerplate quotes both markers; longest inner span must win."""
    full = (
        "Intro delimited by "
        + START_OF_EPISODE
        + " and "
        + END_OF_EPISODE
        + ". Mapping here.\n\n"
        + START_OF_EPISODE
        + "\nMatt: Real dialogue line.\n"
        + END_OF_EPISODE
    )
    pre, ep = extract_preamble_and_episode(full)
    assert "Mapping here" in pre
    assert "delimited by" in pre
    assert "Matt: Real dialogue line." in ep
    assert len(ep) > 5


def test_build_speaker_blocks_includes_preamble() -> None:
    full = (
        "Instructions here.\nMatt plays DM.\n\n[START OF EPISODE]\nMATT: Hello.\n[END OF EPISODE]"
    )
    blocks = build_speaker_blocks_from_transcript(full)
    assert blocks[0]["speaker"] == PREAMBLE_SPEAKER
    assert "Matt plays DM" in blocks[0]["text"]
    assert blocks[1]["speaker"] == "MATT"


def test_format_palace_lobby_preview_groups_wing_room() -> None:
    chunks = [
        {"chunk_index": 0, "content": "alpha", "wing": "east", "room": "library"},
        {"chunk_index": 1, "content": "beta", "wing": "west", "room": "kitchen"},
    ]
    out = format_palace_lobby_preview(chunks, task_id="g1", wing="ignored", room="ignored")
    assert "Wing: east" in out and "Room: library (1 drawers)" in out
    assert "Wing: west" in out and "Room: kitchen (1 drawers)" in out
    assert out.index("Wing: east") < out.index("Wing: west")


def test_build_ephemeral_palace_tools_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stub Chroma/MemPalace backend so CI/sandbox does not download ONNX models."""
    pytest.importorskip("mempalace")
    bucket: list[str] = []

    class FakeCol:
        def upsert(
            self,
            documents: list[str],
            ids: list[str] | None = None,
            metadatas: list[dict[str, object]] | None = None,
        ) -> None:
            bucket.extend(documents)

    def fake_search(
        query: str,
        palace_path: str,
        wing: str | None = None,
        room: str | None = None,
        n_results: int = 8,
    ) -> dict[str, object]:
        words = [w.lower() for w in query.replace("-", " ").split() if len(w) > 2]
        results: list[dict[str, str]] = []
        for doc in bucket:
            low = doc.lower()
            if words and all(w in low for w in words):
                results.append(
                    {"text": doc, "wing": "benchmark", "room": "corpus", "similarity": "0.99"}
                )
            if len(results) >= n_results:
                break
        return {"results": results}

    monkeypatch.setattr("mempalace.palace.get_collection", lambda *_a, **_k: FakeCol())
    monkeypatch.setattr("mempalace.searcher.search_memories", fake_search)

    class FakePersistentClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._col = FakeCol()

        def get_or_create_collection(
            self, name: str, metadata: dict[str, object] | None = None
        ) -> FakeCol:
            return self._col

        def reset(self) -> None:
            return None

    monkeypatch.setattr("chromadb.PersistentClient", FakePersistentClient)

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    from benchmark_tools.ephemeral_mempalace_poc import build_ephemeral_palace_tools

    text = (
        "The magic number for this unit test is forty-two. " * 20 + "\n\n"
        "Second paragraph about dragons and dice."
    )
    tools, cleanup, n_drawers = build_ephemeral_palace_tools(
        text,
        task_id="test-task-id",
        metadata_prefix="pytest",
        chunk_size=min(200, DEFAULT_CHUNK_SIZE),
        chunk_overlap=20,
    )
    try:
        assert n_drawers >= 1
        assert "search_memories" in tools
        entry = tools["search_memories"]
        fn = entry["tool"] if isinstance(entry, dict) and "tool" in entry else entry
        out = fn("magic number forty-two", n_results=3)
        assert "forty-two" in out or "42" in out
    finally:
        cleanup()


def test_build_ephemeral_palace_tools_verbose_prints_lobby(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Avoid Chroma/ONNX download: stub palace collection and searcher."""
    pytest.importorskip("mempalace")

    class FakeCol:
        def upsert(self, *_a: object, **_k: object) -> None:
            return None

    monkeypatch.setattr(
        "mempalace.palace.get_collection",
        lambda *_a, **_k: FakeCol(),
    )
    monkeypatch.setattr(
        "mempalace.searcher.search_memories",
        lambda *_a: {"results": []},
    )

    class FakePersistentClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._col = FakeCol()

        def get_or_create_collection(
            self, name: str, metadata: dict[str, object] | None = None
        ) -> FakeCol:
            return self._col

        def reset(self) -> None:
            return None

    monkeypatch.setattr("chromadb.PersistentClient", FakePersistentClient)

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    from benchmark_tools.ephemeral_mempalace_poc import build_ephemeral_palace_tools

    text = "Verbose lobby preview line one. " * 5 + "\n\n" + "Second block for chunking."
    tools, cleanup, n_drawers = build_ephemeral_palace_tools(
        text,
        task_id="verbose-cap-task",
        metadata_prefix="pytest",
        chunk_size=120,
        chunk_overlap=10,
        verbose=True,
        preview_chars=40,
    )
    try:
        assert n_drawers >= 1
        captured = capsys.readouterr()
        assert "Memory Palace lobby" in captured.out
        assert "verbose-cap-task" in captured.out
        assert "Wing: benchmark" in captured.out
        assert "search_memories" in tools
    finally:
        cleanup()


def test_build_ephemeral_palace_tools_by_speaker_has_list_taxonomy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("mempalace")

    class FakeCol:
        def upsert(self, *_a: object, **_k: object) -> None:
            return None

    monkeypatch.setattr("mempalace.palace.get_collection", lambda *_a, **_k: FakeCol())
    monkeypatch.setattr(
        "mempalace.searcher.search_memories",
        lambda *_a: {"results": []},
    )

    class FakePersistentClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._col = FakeCol()

        def get_or_create_collection(
            self, name: str, metadata: dict[str, object] | None = None
        ) -> FakeCol:
            return self._col

        def reset(self) -> None:
            return None

    monkeypatch.setattr("chromadb.PersistentClient", FakePersistentClient)

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    from benchmark_tools.ephemeral_mempalace_poc import build_ephemeral_palace_tools

    matt_long = "MATT: " + ("word " * 60) + "\n"
    text = (
        "Preamble only.\n[START OF EPISODE]\n" + matt_long + "LAURA: Short reply.\n[END OF EPISODE]"
    )
    tools, cleanup, _n = build_ephemeral_palace_tools(
        text,
        task_id="by-speaker-task",
        metadata_prefix="pytest",
        chunk_size=500,
        chunk_overlap=50,
        ingest="by_speaker",
    )
    try:
        assert "list_taxonomy" in tools
        assert "search_memories" in tools
        tax_fn = tools["list_taxonomy"]["tool"]
        out = json.loads(tax_fn())
        rooms = out["wings"][0]["rooms"]
        assert rooms[0]["room"] == "_preamble"
        room_slugs = {r["room"] for r in rooms}
        # Short interjections can roll into the dominant speaker's room; require preamble + ≥1 speaker room.
        assert len(room_slugs) >= 2
        assert "matt" in room_slugs or "laura" in room_slugs
    finally:
        cleanup()


def test_parse_transcript_speaker_blocks_with_lines_matches_plain_text() -> None:
    ep = "MATT: one\nLAURA: two\n"
    plain = parse_transcript_speaker_blocks(ep)
    lined = parse_transcript_speaker_blocks_with_lines(ep)
    assert len(plain) == len(lined)
    for a, b in zip(plain, lined, strict=True):
        assert a["speaker"] == b["speaker"] and a["text"] == b["text"]
        assert b["episode_line_start"] <= b["episode_line_end"]


def test_merge_micro_dominant_room_and_tie_mixed() -> None:
    micro = [
        {"speaker": "MATT", "text": "a" * 400, "episode_line_start": 1, "episode_line_end": 1},
        {"speaker": "TRAVIS", "text": "hi", "episode_line_start": 2, "episode_line_end": 2},
    ]
    g = merge_micro_blocks_to_grouped_drawers(micro, min_drawer_chars=10_000)
    assert len(g) == 1
    assert g[0]["room"] == "matt"
    assert "TRAVIS" in g[0]["speakers"]
    micro_tie = [
        {"speaker": "A", "text": "xx", "episode_line_start": 1, "episode_line_end": 1},
        {"speaker": "B", "text": "yy", "episode_line_start": 2, "episode_line_end": 2},
    ]
    g2 = merge_micro_blocks_to_grouped_drawers(micro_tie, min_drawer_chars=10_000)
    assert g2[0]["room"] == "_mixed"


def test_chunks_for_grouped_zero_pad_line_metadata() -> None:
    blocks = [
        {
            "text": "MATT: Hello.",
            "speakers": ["MATT"],
            "room": "matt",
            "episode_line_start": 2,
            "episode_line_end": 100,
            "line_ranges": [[2, 2], [99, 100]],
        }
    ]
    chunks = chunks_for_grouped_speaker_ingest(
        blocks, wing="benchmark", chunk_size=500, chunk_overlap=0
    )
    assert len(chunks) == 1
    assert chunks[0]["line_start"] == f"{2:0{LINE_INDEX_PAD_WIDTH}d}"
    assert chunks[0]["line_end"] == f"{100:0{LINE_INDEX_PAD_WIDTH}d}"


def test_format_taxonomy_block_numeric_order() -> None:
    chunks = [
        {"wing": "benchmark", "room": "block_010", "chunk_index": 0},
        {"wing": "benchmark", "room": "block_002", "chunk_index": 1},
        {"wing": "benchmark", "room": "_preamble", "chunk_index": 2},
    ]
    data = json.loads(format_taxonomy_json(chunks))
    rooms = [r["room"] for r in data["wings"][0]["rooms"]]
    assert rooms == ["_preamble", "block_002", "block_010"]


def test_build_temporal_turn_blocks_has_block_rooms() -> None:
    body = "\n".join(f"MATT: line {i} " + "x" * 30 for i in range(120))
    full = f"Pre.\n{START_OF_EPISODE}\n{body}\n{END_OF_EPISODE}"
    turns = build_temporal_turn_blocks_from_transcript(
        full,
        lines_per_block=50,
        overlap_lines=10,
        min_drawer_chars=200_000,
    )
    rooms = {t["room"] for t in turns}
    assert "_preamble" in rooms
    assert "block_001" in rooms
    assert any(r.startswith("block_") for r in rooms)
    # Preamble + one interaction drawer per window (threshold huge → never split mid-window).
    assert len(turns) == 4  # preamble + three 50-line windows (step 40 over 120 lines)
    assert turns[1]["speaker"] == "MATT"
    assert turns[1]["speakers"] == ["MATT"]
    assert turns[1]["episode_line_start"] == 1
    assert turns[1]["episode_line_end"] == 50
    assert turns[1]["room"] == "block_001"


def test_ordered_speakers_first_appearance() -> None:
    lines = ["MATT: a", "TRAVIS: b", "MATT: c"]
    assert ordered_speakers_from_labeled_lines(lines) == ["MATT", "TRAVIS"]


def test_interaction_grouping_keeps_multi_speaker_until_min_chars() -> None:
    rows = [
        (1, "MATT: one"),
        (2, "TRAVIS: two"),
        (3, "MATT: three"),
        (4, "TRAVIS: four"),
    ]
    d = group_dense_window_lines_into_interaction_drawers(rows, min_drawer_chars=500)
    assert len(d) == 1
    assert d[0]["speakers"] == ["MATT", "TRAVIS"]
    assert "TRAVIS: two" in d[0]["text"] and "MATT: one" in d[0]["text"]


def test_interaction_grouping_flushes_when_min_reached_then_continues() -> None:
    # Each line alone meets min_drawer_chars; trigger line repeats at start of next drawer.
    rows = [
        (1, "MATT: " + "x" * 44),
        (2, "TRAVIS: " + "y" * 42),
        (3, "MATT: " + "z" * 44),
    ]
    d = group_dense_window_lines_into_interaction_drawers(rows, min_drawer_chars=50)
    assert len(d) == 3
    assert d[0]["episode_line_end"] == 1
    assert d[0]["text"] == rows[0][1]
    assert d[1]["text"].startswith(rows[0][1] + "\n")
    assert rows[1][1] in d[1]["text"]
    assert d[1]["speakers"] == ["MATT", "TRAVIS"]
    assert d[2]["text"].startswith(rows[1][1] + "\n")
    assert d[2]["speakers"] == ["TRAVIS", "MATT"]


def test_temporal_chunks_carry_padded_line_metadata() -> None:
    turns = [
        {"speaker": PREAMBLE_SPEAKER, "text": "Map.", "room": "_preamble"},
        {
            "speaker": "MATT",
            "text": "a.\nb.",
            "room": "block_001",
            "speakers": ["MATT", "TRAVIS"],
            "episode_line_start": 1,
            "episode_line_end": 2,
        },
    ]
    chunks = chunks_for_temporal_turn_ingest(
        turns, wing="benchmark", chunk_size=4000, chunk_overlap=0
    )
    assert len(chunks) == 2
    assert chunks[1]["room"] == "block_001"
    assert chunks[1]["line_start"] == f"{1:0{LINE_INDEX_PAD_WIDTH}d}"
    assert chunks[1]["line_end"] == f"{2:0{LINE_INDEX_PAD_WIDTH}d}"
    assert chunks[1]["speakers"] == ["MATT", "TRAVIS"]
    assert chunks[1]["speaker_slugs"] == "matt,travis"


def test_collect_speaker_slugs_from_labeled_text() -> None:
    t = "MATT: Hello.\nLAURA: Hi.\n"
    assert collect_speaker_slugs_from_labeled_text(t) == {"matt", "laura"}


def test_build_grouped_speaker_produces_two_rooms_when_flush() -> None:
    matt_long = "MATT: " + ("word " * 60) + "\nLAURA: Short.\n"
    full = f"I.\n{START_OF_EPISODE}\n{matt_long}{END_OF_EPISODE}"
    grouped = build_grouped_speaker_blocks_from_transcript(full, min_drawer_chars=250)
    rooms = {b["room"] for b in grouped}
    assert "_preamble" in rooms
    assert "matt" in rooms
    assert "laura" in rooms


def test_grouped_search_tool_merges_secondary_room(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("mempalace")

    class FakeCol:
        def upsert(self, *_a: object, **_k: object) -> None:
            return None

    def fake_search(
        query: str,
        palace_path: str,
        wing: str | None = None,
        room: str | None = None,
        n_results: int = 8,
    ) -> dict[str, object]:
        if room == "travis":
            return {
                "query": query,
                "results": [
                    {
                        "text": "TRAVIS: I roll.",
                        "wing": "benchmark",
                        "room": "travis",
                        "similarity": 0.5,
                    }
                ],
            }
        return {
            "query": query,
            "results": [
                {
                    "text": "MATT: Room desc.\nTRAVIS: I roll stealth.",
                    "wing": "benchmark",
                    "room": "matt",
                    "similarity": 0.95,
                },
                {
                    "text": "MATT: Unrelated monologue.",
                    "wing": "benchmark",
                    "room": "matt",
                    "similarity": 0.1,
                },
            ],
        }

    monkeypatch.setattr("mempalace.palace.get_collection", lambda *_a, **_k: FakeCol())
    monkeypatch.setattr("mempalace.searcher.search_memories", fake_search)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    from benchmark_tools.ephemeral_mempalace_poc import build_ephemeral_palace_tools

    matt_long = "MATT: " + ("word " * 60) + "\nTRAVIS: Short interjection.\n"
    text = f"P.\n{START_OF_EPISODE}\n{matt_long}{END_OF_EPISODE}"
    tools, cleanup, _ = build_ephemeral_palace_tools(
        text,
        task_id="merge-search",
        metadata_prefix="pytest",
        chunk_size=800,
        chunk_overlap=0,
        ingest="by_speaker",
        min_drawer_chars=250,
    )
    try:
        fn = tools["search_memories"]["tool"]
        out = fn("roll stealth", room="travis", n_results=4)
        assert "stealth" in out
        assert out.index("0.95") < out.index("0.5")
    finally:
        cleanup()


def test_exact_search_over_chunks_case_insensitive_and_filters() -> None:
    chunks: list[dict[str, Any]] = [
        {"content": "Alpha beta gamma", "wing": "east", "room": "a", "chunk_index": 2},
        {"content": "Beta only here", "wing": "east", "room": "b", "chunk_index": 0},
        {"content": "no match", "wing": "west", "room": "a", "chunk_index": 1},
    ]
    out = _exact_search_over_chunks(chunks, "BETA", wing="east", room=None, n_results=10)
    assert len(out["results"]) == 2
    rooms = {str(h["room"]) for h in out["results"]}
    assert rooms == {"a", "b"}

    out_room = _exact_search_over_chunks(chunks, "beta", wing=None, room="a", n_results=10)
    assert len(out_room["results"]) == 1
    assert out_room["results"][0]["room"] == "a"

    capped = _exact_search_over_chunks(chunks, "beta", wing=None, room=None, n_results=1)
    assert len(capped["results"]) == 1


def test_exact_search_over_chunks_empty_query() -> None:
    out = _exact_search_over_chunks([], "   ", wing=None, room=None, n_results=5)
    assert "error" in out


def test_build_ephemeral_palace_tools_exact_mode_no_vector_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("mempalace")

    class FakeCol:
        def upsert(self, *_a: object, **_k: object) -> None:
            return None

    def boom_search(*_a: object, **_k: object) -> dict[str, object]:
        raise AssertionError("semantic search_memories must not run in exact mode")

    monkeypatch.setattr("mempalace.palace.get_collection", lambda *_a, **_k: FakeCol())
    monkeypatch.setattr("mempalace.searcher.search_memories", boom_search)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    from benchmark_tools.ephemeral_mempalace_poc import build_ephemeral_palace_tools

    token = "ZZ_UNIQUE_EXACT_TOKEN_ZZ"
    text = ("Intro. " * 30) + token + (" tail. " * 30)
    tools, cleanup, _n = build_ephemeral_palace_tools(
        text,
        task_id="exact-mode-task",
        metadata_prefix="pytest",
        chunk_size=200,
        chunk_overlap=0,
    )
    try:
        fn = tools["search_memories"]["tool"]
        out = fn(token, n_results=3, mode="exact")
        assert token in out
        assert "mode=exact" in out
    finally:
        cleanup()


def test_build_ephemeral_palace_tools_search_memories_invalid_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("mempalace")

    class FakeCol:
        def upsert(self, *_a: object, **_k: object) -> None:
            return None

    monkeypatch.setattr("mempalace.palace.get_collection", lambda *_a, **_k: FakeCol())
    monkeypatch.setattr(
        "mempalace.searcher.search_memories",
        lambda *_a, **_k: {"results": []},
    )
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    from benchmark_tools.ephemeral_mempalace_poc import build_ephemeral_palace_tools

    tools, cleanup, _ = build_ephemeral_palace_tools(
        "hello world " * 20,
        task_id="bad-mode",
        metadata_prefix="pytest",
        chunk_size=100,
        chunk_overlap=0,
    )
    try:
        fn = tools["search_memories"]["tool"]
        bad_mode: str = "bogus"
        with pytest.raises(ValueError, match="semantic"):
            fn("x", mode=bad_mode)
    finally:
        cleanup()
