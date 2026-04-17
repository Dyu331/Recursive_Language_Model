"""
Ephemeral MemPalace PoC: index arbitrary benchmark text in a temp Chroma palace,
expose ``search_memories`` for LocalREPL ``custom_tools``.

Requires: ``uv pip install -e ".[mempalace-poc]"`` or ``pip install mempalace`` (with compatible chromadb).
"""

from __future__ import annotations

import hashlib
import json
import numbers
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any, Literal

DEFAULT_CHUNK_SIZE = 4000
DEFAULT_CHUNK_OVERLAP = 400
MIN_CHUNK_SIZE = 50
DEFAULT_MIN_DRAWER_CHARS = 350
DEFAULT_TEMPORAL_LINES_PER_BLOCK = 75
DEFAULT_TEMPORAL_BLOCK_OVERLAP = 7
LINE_INDEX_PAD_WIDTH = 6
UPSERT_BATCH_SIZE = 256

START_OF_EPISODE = "[START OF EPISODE]"
END_OF_EPISODE = "[END OF EPISODE]"
PREAMBLE_SPEAKER = "_preamble"
MIXED_ROOM_SLUG = "_mixed"
_SPEAKER_LINE_RE = re.compile(r"^([^:\n]+):\s*(.*)$")

ATOMIC_RECORDS_ROOM = "records"
UNCLASSIFIED_ROOM = "unclassified"
_SYNTH_RECORD_RE = re.compile(
    r"^Date:\s*(?P<date>[^|]+?)\s*\|\|\s*User:\s*(?P<user>[^|]+?)\s*\|\|\s*Instance:\s*"
    r"(?P<instance>.*?)(?:\s*\|\|\s*Label:\s*(?P<label>.*?))?\s*$"
)
_SYNTH_PREAMBLE_COUNT_RE = re.compile(r"The following lines contain\s+(\d+)\s+", re.IGNORECASE)


def slugify_label(label: str) -> str:
    """Stable Chroma ``room`` slug for TREC-style labels (lowercase, ``_`` separators)."""
    s = label.strip().lower()
    out = re.sub(r"[^a-z0-9]+", "_", s)
    out = re.sub(r"_+", "_", out).strip("_")
    return out if out else UNCLASSIFIED_ROOM


def extract_preamble_and_episode(full_text: str) -> tuple[str, str]:
    """
    Split Oolong-style ``context_window_text`` into preamble (instructions + mapping) and episode body.

    Preamble is everything before the chosen ``[START OF EPISODE]``. Episode is between that
    marker and the next ``[END OF EPISODE]`` after it, or from START to EOF if no END exists.

    Instructions often **repeat** the marker strings (e.g. "delimited by [START OF EPISODE] and
    [END OF EPISODE]"). Using ``find`` once would pair those mentions and yield a tiny fake
    episode. We therefore consider **every** occurrence of ``[START OF EPISODE]`` and pick
    the pair whose episode body is **longest** (the real transcript dwarfs the instructional
    fragment).
    """
    text = full_text.strip()
    if not text:
        return "", ""

    sm, em = START_OF_EPISODE, END_OF_EPISODE
    starts: list[int] = []
    search_from = 0
    while True:
        i = text.find(sm, search_from)
        if i == -1:
            break
        starts.append(i)
        search_from = i + 1

    if not starts:
        return "", text

    best_episode = ""
    best_start_idx = starts[0]
    for idx_s in starts:
        body_start = idx_s + len(sm)
        idx_e = text.find(em, body_start)
        if idx_e == -1:
            episode = text[body_start:].strip()
        else:
            episode = text[body_start:idx_e].strip()
        if len(episode) > len(best_episode):
            best_episode = episode
            best_start_idx = idx_s

    preamble = text[:best_start_idx].strip()
    return preamble, best_episode


def slugify_speaker(speaker: str) -> str:
    """Stable Chroma ``room`` slug; reserved preamble label stays ``_preamble``."""
    if speaker == PREAMBLE_SPEAKER:
        return "_preamble"
    s = speaker.strip().lower()
    out = re.sub(r"[^a-z0-9]+", "_", s)
    out = re.sub(r"_+", "_", out).strip("_")
    return out if out else "unknown"


def parse_transcript_speaker_blocks(episode_body: str) -> list[dict[str, str]]:
    """
    Parse ``Name: line`` dialogue. Non-matching lines are continuations of the current speaker.
    Lines before the first ``Name:`` are prepended to the first speaker's block.
    """
    blocks: list[dict[str, str]] = []
    current: str | None = None
    buf: list[str] = []
    prefix: list[str] = []
    first_speaker_seen = False

    def flush() -> None:
        nonlocal current, buf
        if current is None:
            return
        joined = "\n".join(buf).strip()
        if joined:
            blocks.append({"speaker": current, "text": joined})
        buf = []

    for raw_line in episode_body.splitlines():
        if not raw_line.strip():
            if current is not None:
                buf.append("")
            continue

        m = _SPEAKER_LINE_RE.match(raw_line)
        if m:
            name = m.group(1).strip()
            rest = m.group(2)
            flush()
            current = name
            buf = []
            if not first_speaker_seen:
                first_speaker_seen = True
                buf.extend(prefix)
                prefix.clear()
            if rest:
                buf.append(rest)
        elif current is None:
            low = raw_line.strip().lower()
            if low in (START_OF_EPISODE.lower(), END_OF_EPISODE.lower()):
                continue
            prefix.append(raw_line)
        else:
            buf.append(raw_line)

    flush()
    return blocks


def build_speaker_blocks_from_transcript(full_text: str) -> list[dict[str, str]]:
    """Preamble as ``_preamble`` block plus episode speaker blocks."""
    preamble, episode = extract_preamble_and_episode(full_text)
    out: list[dict[str, str]] = []
    if preamble:
        out.append({"speaker": PREAMBLE_SPEAKER, "text": preamble})
    out.extend(parse_transcript_speaker_blocks(episode))
    return out


def _pad_line_index(n: int) -> str:
    """Zero-pad episode line numbers for Chroma string ordering."""
    return f"{n:0{LINE_INDEX_PAD_WIDTH}d}"


def parse_transcript_speaker_blocks_with_lines(episode_body: str) -> list[dict[str, Any]]:
    """
    Like ``parse_transcript_speaker_blocks`` but each block includes 1-based inclusive
    ``episode_line_start`` / ``episode_line_end`` into ``episode_body.splitlines()``.
    """
    blocks: list[dict[str, Any]] = []
    current: str | None = None
    buf: list[str] = []
    prefix: list[str] = []
    first_speaker_seen = False
    block_start_line: int | None = None
    all_lines = episode_body.splitlines()

    def flush(end_line: int | None) -> None:
        nonlocal current, buf, block_start_line
        if current is None:
            return
        joined = "\n".join(buf).strip()
        if joined and block_start_line is not None and end_line is not None:
            blocks.append(
                {
                    "speaker": current,
                    "text": joined,
                    "episode_line_start": block_start_line,
                    "episode_line_end": end_line,
                }
            )
        buf = []
        block_start_line = None

    line_no = 0
    for raw_line in all_lines:
        line_no += 1
        if not raw_line.strip():
            if current is not None:
                buf.append("")
            continue

        m = _SPEAKER_LINE_RE.match(raw_line)
        if m:
            name = m.group(1).strip()
            rest = m.group(2)
            if current is not None:
                flush(line_no - 1)
            current = name
            buf = []
            block_start_line = line_no
            if not first_speaker_seen:
                first_speaker_seen = True
                buf.extend(prefix)
                prefix.clear()
            if rest:
                buf.append(rest)
        elif current is None:
            low = raw_line.strip().lower()
            if low in (START_OF_EPISODE.lower(), END_OF_EPISODE.lower()):
                continue
            prefix.append(raw_line)
        else:
            buf.append(raw_line)

    if current is not None:
        flush(line_no)
    return blocks


def _joined_segment_len(segments: list[dict[str, Any]]) -> int:
    return sum(len(str(s.get("text", ""))) for s in segments)


def merge_micro_blocks_to_grouped_drawers(
    micro_blocks: list[dict[str, Any]],
    *,
    min_drawer_chars: int,
) -> list[dict[str, Any]]:
    """
    Buffer short cross-talk into beefy drawers; assign dominant-speaker ``room`` slug
    (tie → ``_mixed``). Preserves ``Name:`` labels in merged ``text``.
    """
    if not micro_blocks:
        return []

    grouped: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal segments
        if not segments:
            return
        speakers_order: list[str] = []
        seen: set[str] = set()
        char_totals: dict[str, int] = defaultdict(int)
        line_ranges: list[list[int]] = []
        text_lines: list[str] = []
        for seg in segments:
            sp = str(seg["speaker"])
            t = str(seg["text"])
            char_totals[sp] += len(t)
            if sp not in seen:
                seen.add(sp)
                speakers_order.append(sp)
            ls, le = int(seg["episode_line_start"]), int(seg["episode_line_end"])
            line_ranges.append([ls, le])
            body_lines = t.splitlines() or [""]
            text_lines.append(f"{sp}: {body_lines[0]}")
            text_lines.extend(body_lines[1:])
        full_text = "\n".join(text_lines).strip()
        line_start = min(r[0] for r in line_ranges)
        line_end = max(r[1] for r in line_ranges)
        max_chars = max(char_totals.values()) if char_totals else 0
        winners = [s for s, c in char_totals.items() if c == max_chars]
        if len(winners) == 1:
            room_slug = slugify_speaker(winners[0])
        else:
            room_slug = MIXED_ROOM_SLUG
        grouped.append(
            {
                "text": full_text,
                "speakers": speakers_order,
                "char_totals": dict(char_totals),
                "room": room_slug,
                "episode_line_start": line_start,
                "episode_line_end": line_end,
                "line_ranges": line_ranges,
            }
        )
        segments = []

    for mb in micro_blocks:
        if not segments:
            segments.append(mb)
            continue
        if mb["speaker"] == segments[-1]["speaker"]:
            prev = segments[-1]
            merged_text = f"{prev['text'].rstrip()}\n{mb['text'].lstrip()}"
            segments[-1] = {
                "speaker": prev["speaker"],
                "text": merged_text.strip(),
                "episode_line_start": int(prev["episode_line_start"]),
                "episode_line_end": int(mb["episode_line_end"]),
            }
            continue
        if _joined_segment_len(segments) < min_drawer_chars:
            segments.append(mb)
        else:
            flush()
            segments.append(mb)
    flush()
    return grouped


def build_grouped_speaker_blocks_from_transcript(
    full_text: str,
    *,
    min_drawer_chars: int = DEFAULT_MIN_DRAWER_CHARS,
) -> list[dict[str, Any]]:
    """Preamble block plus grouped episode drawers (dominant room, metadata speakers)."""
    preamble, episode = extract_preamble_and_episode(full_text)
    out: list[dict[str, Any]] = []
    if preamble:
        out.append(
            {
                "speaker": PREAMBLE_SPEAKER,
                "text": preamble,
                "speakers": [PREAMBLE_SPEAKER],
                "room": "_preamble",
            }
        )
    micro = parse_transcript_speaker_blocks_with_lines(episode)
    out.extend(merge_micro_blocks_to_grouped_drawers(micro, min_drawer_chars=min_drawer_chars))
    return out


def collect_speaker_slugs_from_labeled_text(drawer_text: str) -> set[str]:
    """Recover speaker slugs from ``Name:`` lines inside indexed drawer text."""
    slugs: set[str] = set()
    for block in parse_transcript_speaker_blocks(drawer_text):
        slugs.add(slugify_speaker(block["speaker"]))
    return slugs


def ordered_speakers_from_labeled_lines(lines: Iterable[str]) -> list[str]:
    """First-appearance order of speakers from ``Name:`` lines (continuations ignored)."""
    ordered: list[str] = []
    seen: set[str] = set()
    for raw_line in lines:
        m = _SPEAKER_LINE_RE.match(raw_line)
        if m:
            name = m.group(1).strip()
            if name not in seen:
                seen.add(name)
                ordered.append(name)
    return ordered


def chunks_for_speaker_ingest(
    blocks: list[dict[str, str]],
    *,
    wing: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Chunk each speaker block with ``_chunk_text``; every sub-chunk inherits the same ``wing`` /
    ``room`` (speaker slug) as its parent block. ``chunk_index`` is global across all drawers.
    """
    global_chunks: list[dict[str, Any]] = []
    gidx = 0
    for block in blocks:
        speaker = block["speaker"]
        raw_text = block["text"].strip()
        if not raw_text:
            continue
        room = slugify_speaker(speaker)
        subs = _chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not subs:
            subs = [{"content": raw_text, "chunk_index": 0}]
        for sub in subs:
            global_chunks.append(
                {
                    "content": sub["content"],
                    "wing": wing,
                    "room": room,
                    "speaker": speaker,
                    "chunk_index": gidx,
                }
            )
            gidx += 1
    return global_chunks


def chunks_for_grouped_speaker_ingest(
    blocks: list[dict[str, Any]],
    *,
    wing: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Chunk grouped speaker blocks; each chunk carries dominant ``room``, full ``speakers`` list,
    and optional padded line metadata.
    """
    global_chunks: list[dict[str, Any]] = []
    gidx = 0
    for block in blocks:
        raw_text = str(block.get("text", "")).strip()
        if not raw_text:
            continue
        room = str(block.get("room") or slugify_speaker(str(block.get("speaker", "unknown"))))
        speakers_list = block.get("speakers")
        if not isinstance(speakers_list, list) or not speakers_list:
            sp0 = block.get("speaker")
            speakers_list = [str(sp0)] if sp0 else []
        speaker_slugs = ",".join(slugify_speaker(str(s)) for s in speakers_list)
        primary = speakers_list[0] if speakers_list else ""
        line_start = block.get("episode_line_start")
        line_end = block.get("episode_line_end")
        line_ranges = block.get("line_ranges")
        subs = _chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not subs:
            subs = [{"content": raw_text, "chunk_index": 0}]
        for sub in subs:
            row: dict[str, Any] = {
                "content": sub["content"],
                "wing": wing,
                "room": room,
                "speaker": primary,
                "speakers": speakers_list,
                "speaker_slugs": speaker_slugs,
                "chunk_index": gidx,
            }
            if isinstance(line_start, int) and isinstance(line_end, int):
                row["line_start"] = _pad_line_index(line_start)
                row["line_end"] = _pad_line_index(line_end)
                row["line_ranges"] = json.dumps(line_ranges) if line_ranges else ""
            global_chunks.append(row)
            gidx += 1
    return global_chunks


def group_dense_window_lines_into_interaction_drawers(
    rows: list[tuple[int, str]],
    *,
    min_drawer_chars: int,
) -> list[dict[str, Any]]:
    """
    **Interaction grouping** within one temporal window: append lines in order (full ``Speaker:``
    lines preserved) without flushing on speaker change; flush when joined text length reaches
    ``min_drawer_chars``, or at window end (remainder may be shorter). If a flush is triggered by
    reaching ``min_drawer_chars`` and more window lines follow, the **triggering line** is also
    prepended to the next drawer (drawer overlap). Each drawer lists every speaker that appears
    (``Name:`` lines), in first-appearance order, in ``speakers``; ``speaker`` is the first in that
    list (or ``_unknown`` if there are no labels). ``rows`` are ``(1-based_index_in_episode_dense_lines, raw_line)``.
    """
    blocks: list[dict[str, Any]] = []
    buf: list[tuple[int, str]] = []
    prefix_before_first_name: list[tuple[int, str]] = []
    current_speaker: str | None = None
    first_name_line_seen = False

    def flush_buf(*, force: bool, more_lines_follow: bool) -> None:
        nonlocal buf
        if not buf:
            return
        text = "\n".join(r for _, r in buf).strip()
        if not text:
            buf = []
            return
        size_hit = len(text) >= min_drawer_chars
        if not size_hit and not force:
            return
        lines_only = [r for _, r in buf]
        lo = min(ln for ln, _ in buf)
        hi = max(ln for ln, _ in buf)
        spk_ordered = ordered_speakers_from_labeled_lines(lines_only)
        primary = spk_ordered[0] if spk_ordered else "_unknown"
        trigger = buf[-1]
        blocks.append(
            {
                "speaker": primary,
                "speakers": spk_ordered,
                "text": text,
                "episode_line_start": lo,
                "episode_line_end": hi,
            }
        )
        buf = []
        if size_hit and not force and more_lines_follow:
            buf.append(trigger)

    for idx, (line_no, raw_line) in enumerate(rows):
        if not raw_line.strip():
            continue
        more = idx < len(rows) - 1
        m = _SPEAKER_LINE_RE.match(raw_line)
        if m:
            name = m.group(1).strip()
            current_speaker = name
            if not first_name_line_seen:
                first_name_line_seen = True
                buf.extend(prefix_before_first_name)
                prefix_before_first_name.clear()
            buf.append((line_no, raw_line))
            flush_buf(force=False, more_lines_follow=more)
        elif current_speaker is None:
            low = raw_line.strip().lower()
            if low in (START_OF_EPISODE.lower(), END_OF_EPISODE.lower()):
                continue
            prefix_before_first_name.append((line_no, raw_line))
        else:
            buf.append((line_no, raw_line))
            flush_buf(force=False, more_lines_follow=more)

    flush_buf(force=True, more_lines_follow=False)
    if prefix_before_first_name:
        buf = list(prefix_before_first_name)
        prefix_before_first_name.clear()
        flush_buf(force=True, more_lines_follow=False)

    return blocks


def chunks_for_temporal_turn_ingest(
    turn_blocks: list[dict[str, Any]],
    *,
    wing: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    One or more Chroma rows per interaction-grouped drawer inside ``block_NNN``; optional size
    split with shared ``line_start`` / ``line_end`` and ``speakers`` / ``speaker_slugs`` metadata.
    """
    global_chunks: list[dict[str, Any]] = []
    gidx = 0
    for block in turn_blocks:
        raw_text = str(block.get("text", "")).strip()
        if not raw_text:
            continue
        room = str(block["room"])
        speakers_list = block.get("speakers")
        if not isinstance(speakers_list, list):
            speakers_list = []
        if not speakers_list:
            sp0 = block.get("speaker")
            speakers_list = [str(sp0)] if sp0 else []
        primary = str(block.get("speaker") or (speakers_list[0] if speakers_list else ""))
        speaker_slugs = ",".join(slugify_speaker(str(s)) for s in speakers_list)
        ls_int = block.get("episode_line_start")
        le_int = block.get("episode_line_end")
        line_start: str | None = None
        line_end: str | None = None
        if isinstance(ls_int, int) and isinstance(le_int, int):
            line_start = _pad_line_index(ls_int)
            line_end = _pad_line_index(le_int)
        subs = _chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not subs:
            subs = [{"content": raw_text, "chunk_index": 0}]
        for sub in subs:
            row: dict[str, Any] = {
                "content": sub["content"],
                "wing": wing,
                "room": room,
                "speaker": primary,
                "chunk_index": gidx,
            }
            if speakers_list:
                row["speakers"] = speakers_list
                row["speaker_slugs"] = speaker_slugs
            if line_start and line_end:
                row["line_start"] = line_start
                row["line_end"] = line_end
            global_chunks.append(row)
            gidx += 1
    return global_chunks


def build_temporal_turn_blocks_from_transcript(
    full_text: str,
    *,
    lines_per_block: int = DEFAULT_TEMPORAL_LINES_PER_BLOCK,
    overlap_lines: int = DEFAULT_TEMPORAL_BLOCK_OVERLAP,
    min_drawer_chars: int = DEFAULT_MIN_DRAWER_CHARS,
) -> list[dict[str, Any]]:
    """
    Preamble as ``_preamble``; episode as overlapping dense-line windows (``lines_per_block`` /
    ``overlap_lines``). Each window is ``room = block_NNN`` with **interaction-grouped** drawers:
    lines append in order (``Speaker:`` prefixes kept) until ``min_drawer_chars`` is reached, then
    a new drawer starts (the line that triggered the flush is repeated at the start of the next
    drawer when more lines remain in the window). Speaker may change within a drawer. Each episode drawer has
    ``episode_line_start`` / ``episode_line_end`` and ``speakers`` (all ``Name:`` speakers in that
    drawer, first-appearance order).
    """
    preamble, episode = extract_preamble_and_episode(full_text)
    out: list[dict[str, Any]] = []
    if preamble:
        out.append({"speaker": PREAMBLE_SPEAKER, "text": preamble, "room": "_preamble"})
    dense = [ln for ln in episode.splitlines() if ln.strip()]
    if not dense:
        return out
    step = max(1, lines_per_block - overlap_lines)
    bi = 1
    for start in range(0, len(dense), step):
        window_lines = dense[start : start + lines_per_block]
        if not window_lines:
            break
        room = f"block_{bi:03d}"
        rows = [(start + i + 1, window_lines[i]) for i in range(len(window_lines))]
        for drawer in group_dense_window_lines_into_interaction_drawers(
            rows, min_drawer_chars=min_drawer_chars
        ):
            out.append({**drawer, "room": room})
        bi += 1
        if start + lines_per_block >= len(dense):
            break
    return out


def _room_slug_sort_tuple(room: str) -> tuple[int, int, str]:
    """Sort key fragment for a room slug (preamble, numeric blocks, mixed, then lexical)."""
    if room == "_preamble":
        return (0, 0, room)
    if room.startswith("block_") and room[6:].isdigit():
        return (1, int(room[6:]), "")
    if room == MIXED_ROOM_SLUG:
        return (2, 0, room)
    if room == UNCLASSIFIED_ROOM:
        return (4, 0, room)
    return (3, 0, room)


def _wing_room_sort_key(wing_room: tuple[str, str]) -> tuple[str, int, int, str]:
    w, r = wing_room
    tier, num, rest = _room_slug_sort_tuple(r)
    return (w, tier, num, rest)


def format_taxonomy_json(chunks: list[dict[str, Any]]) -> str:
    """Wing/room drawer counts; ``_preamble`` first, then ``block_NNN`` numerically, then other slugs."""
    by_wing: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    mixed_hint: dict[str, str] = {}
    for item in chunks:
        w = str(item.get("wing", "benchmark"))
        r = str(item.get("room", "corpus"))
        by_wing[w][r] += 1
        if r == MIXED_ROOM_SLUG and w not in mixed_hint:
            sp = item.get("speakers")
            if isinstance(sp, list) and sp:
                mixed_hint[w] = ", ".join(str(x) for x in sp)

    wings_out: list[dict[str, Any]] = []
    for w in sorted(by_wing.keys()):
        rooms_counts = by_wing[w]
        room_slugs = sorted(rooms_counts.keys(), key=_room_slug_sort_tuple)
        rooms_out: list[dict[str, Any]] = []
        for r in room_slugs:
            entry: dict[str, Any] = {"room": r, "drawers": rooms_counts[r]}
            if r == "_preamble":
                entry["description"] = (
                    "Instructions and player-to-character mapping (check here first)."
                )
            elif r == MIXED_ROOM_SLUG:
                hint = mixed_hint.get(w, "")
                entry["description"] = (
                    "Multi-speaker bundle (equal text share); speakers: " + hint
                    if hint
                    else "Multi-speaker bundle (equal text share between speakers)."
                )
            elif r.startswith("block_") and r[6:].isdigit():
                entry["description"] = "Temporal segment (not a single character room)."
            elif r == UNCLASSIFIED_ROOM:
                entry["description"] = (
                    "Non-atomic lines (preamble/trailer instructions, or records with no label)."
                )
            rooms_out.append(entry)
        wings_out.append({"name": w, "rooms": rooms_out})
    return json.dumps({"wings": wings_out}, indent=2)


def _chunk_text(
    content: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Split content into overlapping chunks; prefer paragraph / line boundaries."""
    content = content.strip()
    if not content:
        return []

    chunks: list[dict[str, Any]] = []
    start = 0
    chunk_index = 0

    while start < len(content):
        end = min(start + chunk_size, len(content))

        if end < len(content):
            newline_pos = content.rfind("\n\n", start, end)
            if newline_pos > start + chunk_size // 2:
                end = newline_pos
            else:
                newline_pos = content.rfind("\n", start, end)
                if newline_pos > start + chunk_size // 2:
                    end = newline_pos

        chunk = content[start:end].strip()
        if len(chunk) >= MIN_CHUNK_SIZE:
            chunks.append({"content": chunk, "chunk_index": chunk_index})
            chunk_index += 1

        start = end - chunk_overlap if end < len(content) else end

    return chunks


def _drawer_id(task_id: str, chunk_index: int) -> str:
    digest = hashlib.sha256(f"{task_id}:{chunk_index}".encode()).hexdigest()[:32]
    return f"bpoc_{digest}"


def _preview_one_line(text: str, preview_chars: int) -> str:
    collapsed = " ".join(str(text).split())
    if preview_chars <= 0:
        return ""
    if len(collapsed) <= preview_chars:
        return collapsed
    return collapsed[:preview_chars].rstrip() + "..."


def format_palace_lobby_preview(
    chunks: list[dict[str, Any]],
    *,
    task_id: str,
    wing: str = "benchmark",
    room: str = "corpus",
    preview_chars: int = 50,
) -> str:
    """
    Human-readable wing/room tree with per-drawer text previews (in-memory; no Chroma read).

    Each chunk dict must include ``content`` and ``chunk_index``. Optional keys ``wing`` and
    ``room`` override the defaults for that drawer (for future multi-room palaces).
    """
    if not chunks:
        return f"Memory Palace lobby (task_id={task_id!r})\n(no drawers)\n"

    by_wing_room: dict[tuple[str, str], list[tuple[int, str, str]]] = defaultdict(list)
    for item in chunks:
        w = item["wing"] if "wing" in item and isinstance(item["wing"], str) else wing
        r = item["room"] if "room" in item and isinstance(item["room"], str) else room
        idx = int(item["chunk_index"])
        content = str(item["content"])
        did = _drawer_id(task_id, idx)
        by_wing_room[(w, r)].append((idx, did, content))

    lines: list[str] = [f"Memory Palace lobby (task_id={task_id!r})"]
    for wr in sorted(by_wing_room.keys(), key=_wing_room_sort_key):
        w, r = wr
        drawers = sorted(by_wing_room[wr], key=lambda t: t[0])
        lines.append(f"Wing: {w}")
        lines.append(f"  Room: {r} ({len(drawers)} drawers)")
        for idx, did, content in drawers:
            prev = _preview_one_line(content, preview_chars)
            lines.append(f"    [{idx}] {did}  {prev}")
    lines.append("")
    return "\n".join(lines)


def _search_branch_fetch_size(user_n: int, *, wide: bool = False) -> int:
    """Over-fetch per internal MemPalace call before merge (similarity dilution guard)."""
    base = max(user_n * 2, 16)
    cap = 80 if wide else 64
    return min(base + (8 if wide else 0), cap)


def _chroma_cosine_distance_to_score(d: float) -> float:
    """
    Map Chroma cosine distance ``d`` in ``[0, 2]`` (0 = identical, 2 = opposite) to ``[0, 1]``.

    MemPalace stores ``similarity`` as ``max(0, 1 - effective_dist)``, which is **0** whenever
    ``distance > 1`` even though the hit is a valid (weak) match. Using ``1 - d/2`` matches
    ``(1 + cos_sim) / 2`` when ``d`` is ``1 - cos_sim``.
    """
    x = float(d)
    if x < 0.0:
        return 0.0
    if x > 2.0:
        x = 2.0
    return max(0.0, min(1.0, 1.0 - x / 2.0))


def _hit_semantic_score(hit: dict[str, Any]) -> float:
    """
    Scalar in ``[0, 1]`` for ranking / display (higher = closer to the query).

    Prefer ``effective_distance`` then ``distance`` from MemPalace/Chroma and map with
    :func:`_chroma_cosine_distance_to_score` so ``distance > 1`` does not collapse to zero.
    Fall back to MemPalace's ``similarity`` only when distances are absent.
    """
    for v in (hit.get("effective_distance"), hit.get("distance")):
        if isinstance(v, numbers.Real) and not isinstance(v, bool):
            return _chroma_cosine_distance_to_score(float(v))
    s = hit.get("similarity")
    if isinstance(s, numbers.Real) and not isinstance(s, bool):
        return max(0.0, float(s))
    return 0.0


def _format_hit_similarity(hit: dict[str, Any]) -> str:
    return f"{_hit_semantic_score(hit):.4f}"


def _merge_grouped_by_speaker_search(
    out_room: dict[str, Any],
    out_wing: dict[str, Any],
    *,
    slug: str,
    user_n: int,
) -> dict[str, Any]:
    """Merge primary-room and wing-wide hits; keep wing-wide rows where ``slug`` appears in text."""
    if isinstance(out_room, dict) and out_room.get("error"):
        return out_room
    if isinstance(out_wing, dict) and out_wing.get("error"):
        return out_wing
    seen: set[tuple[str, str, str]] = set()
    merged: list[dict[str, Any]] = []
    for out, apply_cast_filter in ((out_room, False), (out_wing, True)):
        if not isinstance(out, dict):
            continue
        for h in out.get("results", []):
            if not isinstance(h, dict):
                continue
            text = str(h.get("text", ""))
            meta_r = str(h.get("room", ""))
            meta_w = str(h.get("wing", ""))
            if apply_cast_filter:
                if slug != meta_r and slug not in collect_speaker_slugs_from_labeled_text(text):
                    continue
            key = (text[:480], meta_r, meta_w)
            if key in seen:
                continue
            seen.add(key)
            merged.append(h)
    merged.sort(key=lambda x: -_hit_semantic_score(x))
    return {
        "query": out_room.get("query", "") if isinstance(out_room, dict) else "",
        "filters": {"wing": None, "room": f"{slug}+cast"},
        "results": merged[:user_n],
    }


def _merge_multiroom_semantic_hits(
    per_room_outs: list[dict[str, Any]],
    *,
    query: str,
    n_results: int,
) -> dict[str, Any]:
    """Merge per-room semantic search outputs into one hit list.

    Deduplicates by ``line_index`` (fall back to ``(room, chunk_index)`` when line_index is
    absent), sorts by :func:`_hit_semantic_score` desc, truncates to ``n_results``. Any per-room
    dict carrying an ``error`` key short-circuits and is returned as-is so the caller surfaces
    the Chroma failure loudly instead of returning a silently partial answer.
    """
    for out in per_room_outs:
        if isinstance(out, dict) and out.get("error"):
            return out
    seen: set[tuple[str, ...]] = set()
    merged: list[dict[str, Any]] = []
    for out in per_room_outs:
        if not isinstance(out, dict):
            continue
        for h in out.get("results", []):
            if not isinstance(h, dict):
                continue
            li = h.get("line_index")
            if isinstance(li, str) and li:
                key: tuple[str, ...] = ("li", li)
            else:
                key = ("ri", str(h.get("room", "")), str(h.get("chunk_index", "")))
            if key in seen:
                continue
            seen.add(key)
            merged.append(h)
    merged.sort(key=lambda x: -_hit_semantic_score(x))
    return {
        "query": query,
        "filters": {"wing": None, "room": None},
        "results": merged[:n_results],
    }


def _exact_search_over_chunks(
    chunks: list[dict[str, Any]],
    query: str,
    *,
    wing: str | None,
    room: str | None,
    n_results: int,
) -> dict[str, Any]:
    """
    Case-insensitive substring match over indexed chunk ``content`` (no embeddings).

    ``wing`` / ``room`` of ``None`` mean no filter on that axis (same idea as MemPalace search).
    """
    needle = query.strip()
    if not needle:
        return {"error": "exact mode requires a non-empty query string", "results": []}
    low = needle.lower()
    hits: list[dict[str, Any]] = []
    for item in chunks:
        content = str(item.get("content", ""))
        if low not in content.lower():
            continue
        w = str(item.get("wing", ""))
        r = str(item.get("room", ""))
        if wing is not None and w != wing:
            continue
        if room is not None and r != room:
            continue
        ci = int(item.get("chunk_index", 0))
        hits.append(
            {
                "text": content,
                "wing": w,
                "room": r,
                "similarity": 1.0,
                "distance": 0.0,
                "chunk_index": ci,
                "search_mode": "exact",
            }
        )
    hits.sort(key=lambda h: (h["wing"], h["room"], h["chunk_index"]))
    return {"query": needle, "filters": {"wing": wing, "room": room}, "results": hits[:n_results]}


def _format_search_hits(out: dict[str, Any]) -> str:
    if isinstance(out, dict) and out.get("error"):
        return json.dumps(out, indent=2)
    if not isinstance(out, dict):
        return str(out)
    lines: list[str] = []
    for i, hit in enumerate(out.get("results", []), 1):
        t = hit.get("text", "")
        meta_w = hit.get("wing", "")
        meta_r = hit.get("room", "")
        sm = (
            hit.get("search_mode")
            if hit.get("search_mode") in ("semantic", "exact")
            else "semantic"
        )
        sim = _format_hit_similarity(hit)
        lines.append(
            f"--- hit {i} (mode={sm} wing={meta_w} room={meta_r} similarity={sim}) ---\n{t}"
        )
    return "\n\n".join(lines) if lines else "(no results)"


def _search_memories_reusing_drawers_col(
    drawers_col: Any,
    palace_path: str,
    query: str,
    **kwargs: Any,
) -> Any:
    """Run ``mempalace.searcher.search_memories`` using an existing drawers collection.

    Avoids ``chromadb.PersistentClient`` construction on every search (which leaks file
    descriptors when the soft NOFILE limit is low, e.g. macOS defaults).
    """
    import mempalace.searcher as _mp_s
    from mempalace.searcher import search_memories

    real_get = _mp_s.get_collection
    real_closets = _mp_s.get_closets_collection

    def shim_get(path: str, collection_name: str = "mempalace_drawers", create: bool = True):
        if path == palace_path and collection_name == "mempalace_drawers":
            return drawers_col
        return real_get(path, collection_name, create)

    def shim_closets(path: str, create: bool = True):
        if path == palace_path:
            raise FileNotFoundError("ephemeral palace has no closets collection")
        return real_closets(path, create=create)

    _mp_s.get_collection = shim_get
    _mp_s.get_closets_collection = shim_closets
    try:
        return search_memories(query, palace_path, **kwargs)
    finally:
        _mp_s.get_collection = real_get
        _mp_s.get_closets_collection = real_closets


def _parse_synth_date_iso(raw: str) -> str:
    """Convert an Oolong-synth ``Date: Sep 06, 2023`` string to ISO ``2023-09-06``.

    Fail loudly on any parse error: the record anchor is supposed to be strict, and silently
    dropping a bad date would corrupt per-date filter results downstream.
    """
    return datetime.strptime(raw.strip(), "%b %d, %Y").strftime("%Y-%m-%d")


def parse_atomic_records(text: str) -> list[dict[str, Any]]:
    """Split Oolong-synth ``context_window_text`` (+ optional ``|| Label:``) into atomic drawers.

    Emits one drawer per non-empty line so that **every** source line is indexed:

    - Lines matching ``_SYNTH_RECORD_RE`` with a non-empty ``Label:`` segment → ``room`` is the
      slugified label (e.g. ``location``, ``numeric_value``). These are the core atomic records.
    - Anchor-matching lines without a ``Label:`` segment → ``room = UNCLASSIFIED_ROOM``; they
      still carry ``date_iso`` / ``user_id`` / ``line_index`` metadata.
    - Non-empty lines that do not match the anchor at all (preamble, trailer, malformed) →
      ``room = UNCLASSIFIED_ROOM`` with only ``line_index`` and the verbatim line as ``document``.

    ``line_index`` is the 1-based index into ``text.splitlines()`` of the source line so drawers
    preserve deterministic ordering; ``is_record`` flags core atomic records for preamble-count
    integrity checks.
    """
    records: list[dict[str, Any]] = []
    for i, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue
        m = _SYNTH_RECORD_RE.match(raw_line)
        if not m:
            records.append(
                {
                    "line_index": i,
                    "room": UNCLASSIFIED_ROOM,
                    "label": "",
                    "document": raw_line,
                    "is_record": False,
                }
            )
            continue
        date_raw = m.group("date").strip()
        user_id = m.group("user").strip()
        instance = m.group("instance").strip()
        label_raw = (m.group("label") or "").strip()
        date_iso = _parse_synth_date_iso(date_raw)
        room = slugify_label(label_raw) if label_raw else UNCLASSIFIED_ROOM
        records.append(
            {
                "line_index": i,
                "date_raw": date_raw,
                "date_iso": date_iso,
                "user_id": user_id,
                "instance": instance,
                "label": label_raw,
                "room": room,
                "document": raw_line,
                "is_record": True,
            }
        )
    return records


def assert_synth_preamble_count(text: str, n_records: int) -> None:
    """Verify the parsed **core record** count matches the preamble count (e.g. ``3182``).

    Loud failure: silently accepting a mismatch would corrupt aggregate answers. When the
    preamble does not announce a count at all, skip the check. ``n_records`` must count only
    anchor-matching core records (``is_record=True``), not unclassified instruction drawers.
    """
    m = _SYNTH_PREAMBLE_COUNT_RE.search(text)
    if not m:
        return
    declared = int(m.group(1))
    if declared != n_records:
        raise ValueError(f"synth ingest saw {n_records} records, preamble claims {declared}")


def count_core_records(records: list[dict[str, Any]]) -> int:
    """Number of anchor-matching records (excludes unclassified preamble/trailer drawers)."""
    return sum(1 for r in records if r.get("is_record"))


def chunks_for_atomic_ingest(
    records: list[dict[str, Any]],
    *,
    wing: str,
) -> list[dict[str, Any]]:
    """One Chroma chunk per parsed line; ``room`` comes from the record's label slug.

    ``chunk_index`` is a dense global index across all rooms so ``_drawer_id(task_id,
    chunk_index)`` stays unique. Core atomic records carry full metadata (``user_id``,
    ``date_iso``, ``date_raw``, ``label``); unclassified non-anchor lines carry only
    ``line_index`` plus the raw document so the agent can still keyword-search instructions.
    """
    out: list[dict[str, Any]] = []
    for idx, rec in enumerate(records):
        chunk: dict[str, Any] = {
            "content": str(rec["document"]),
            "wing": wing,
            "room": str(rec.get("room") or UNCLASSIFIED_ROOM),
            "chunk_index": idx,
            "line_index": _pad_line_index(int(rec["line_index"])),
        }
        if rec.get("is_record"):
            chunk["user_id"] = str(rec["user_id"])
            chunk["date_iso"] = str(rec["date_iso"])
            chunk["date_raw"] = str(rec["date_raw"])
            label = str(rec.get("label") or "")
            if label:
                chunk["label"] = label
        out.append(chunk)
    return out


def _record_metadata_matches(
    item: dict[str, Any],
    *,
    user: str | None,
    date_from: str | None,
    date_to: str | None,
    line_start: int | None,
    line_end: int | None,
) -> bool:
    """Post-filter predicate for by_record drawers; missing metadata means no match."""
    if user is not None:
        if str(item.get("user_id", "")) != user:
            return False
    if date_from is not None or date_to is not None:
        d = str(item.get("date_iso", ""))
        if not d:
            return False
        if date_from is not None and d < date_from:
            return False
        if date_to is not None and d > date_to:
            return False
    if line_start is not None or line_end is not None:
        li_raw = item.get("line_index")
        try:
            li = int(li_raw) if li_raw is not None else None
        except (TypeError, ValueError):
            li = None
        if li is None:
            return False
        if line_start is not None and li < line_start:
            return False
        if line_end is not None and li > line_end:
            return False
    return True


def _exact_search_over_chunks_with_filters(
    chunks: list[dict[str, Any]],
    query: str,
    *,
    wing: str | None,
    room: str | None,
    n_results: int,
    user: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    line_start: int | None = None,
    line_end: int | None = None,
) -> dict[str, Any]:
    """Like :func:`_exact_search_over_chunks` but with atomic-record metadata post-filters.

    When all metadata kwargs are ``None`` this still runs the full filter-aware scan so hit
    payloads consistently carry atomic-record metadata (``label``, ``line_index``, ...) for
    by_record ingest; other ingests carry none of those keys and the extra loop cost is O(n).
    """
    needle = query.strip()
    if not needle:
        return {"error": "exact mode requires a non-empty query string", "results": []}
    low = needle.lower()
    hits: list[dict[str, Any]] = []
    for item in chunks:
        content = str(item.get("content", ""))
        if low not in content.lower():
            continue
        w = str(item.get("wing", ""))
        r = str(item.get("room", ""))
        if wing is not None and w != wing:
            continue
        if room is not None and r != room:
            continue
        if not _record_metadata_matches(
            item,
            user=user,
            date_from=date_from,
            date_to=date_to,
            line_start=line_start,
            line_end=line_end,
        ):
            continue
        ci = int(item.get("chunk_index", 0))
        hit: dict[str, Any] = {
            "text": content,
            "wing": w,
            "room": r,
            "similarity": 1.0,
            "distance": 0.0,
            "chunk_index": ci,
            "search_mode": "exact",
        }
        for meta_key in ("line_index", "user_id", "date_iso", "date_raw", "label"):
            v = item.get(meta_key)
            if isinstance(v, str) and v:
                hit[meta_key] = v
        hits.append(hit)
    hits.sort(key=lambda h: (h["wing"], h["room"], h["chunk_index"]))
    filters_out: dict[str, Any] = {"wing": wing, "room": room}
    if user is not None:
        filters_out["user"] = user
    if date_from is not None:
        filters_out["date_from"] = date_from
    if date_to is not None:
        filters_out["date_to"] = date_to
    if line_start is not None:
        filters_out["line_start"] = line_start
    if line_end is not None:
        filters_out["line_end"] = line_end
    return {"query": needle, "filters": filters_out, "results": hits[:n_results]}


def build_ephemeral_palace_tools(
    text: str,
    *,
    task_id: str,
    metadata_prefix: str = "benchmark",
    wing: str = "benchmark",
    room: str = "corpus",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    verbose: bool = True,
    preview_chars: int = 50,
    ingest: Literal["sliding", "by_speaker", "by_block", "by_record"] = "sliding",
    min_drawer_chars: int = DEFAULT_MIN_DRAWER_CHARS,
    temporal_lines_per_block: int = DEFAULT_TEMPORAL_LINES_PER_BLOCK,
    temporal_block_overlap: int = DEFAULT_TEMPORAL_BLOCK_OVERLAP,
) -> tuple[dict[str, Any], Callable[[], None], int]:
    """
    Create a temporary palace, upsert chunked text, return custom_tools and cleanup.

    ``ingest='by_speaker'`` uses grouped speaker drawers (dominant ``room``, optional line metadata).
    ``ingest='by_block'`` uses temporal ``block_NNN`` rooms with **interaction grouping** inside
    each window: lines append (speaker labels preserved) until ``min_drawer_chars``, then a new
    drawer (the overflow line repeats at the next drawer head when more lines follow). ``speakers_json`` lists everyone in that drawer. ``line_start`` / ``line_end`` are
    episode dense-line indices. Drawers split further only if over ``chunk_size``. Both register
    ``list_taxonomy``.

    If ``verbose`` is True, prints a lobby summary (wings, rooms, per-drawer previews) to
    stdout immediately after indexing, before returning.

    Returns:
        Tuple of (custom_tools dict for RLM, cleanup callable, number of drawers indexed).
    """
    try:
        import chromadb
        from mempalace.backends.chroma import ChromaCollection, _fix_blob_seq_ids
    except ImportError as e:
        raise ImportError(
            "MemPalace PoC requires the mempalace package. Install with:\n"
            '  uv pip install -e ".[mempalace-poc]"\n'
            "or: pip install mempalace  (with chromadb compatible to MemPalace)"
        ) from e

    tmpdir = tempfile.mkdtemp(prefix="rlm_mempalace_poc_")
    source_file = f"{metadata_prefix}:{task_id}"
    filed_at = datetime.now(UTC).isoformat()

    _fix_blob_seq_ids(tmpdir)
    chroma_settings = chromadb.Settings(allow_reset=True, persist_directory=tmpdir)
    chroma_client = chromadb.PersistentClient(path=tmpdir, settings=chroma_settings)
    raw_drawers = chroma_client.get_or_create_collection(
        "mempalace_drawers", metadata={"hnsw:space": "cosine"}
    )
    col = ChromaCollection(raw_drawers)
    if ingest == "by_speaker":
        blocks = build_grouped_speaker_blocks_from_transcript(
            text, min_drawer_chars=min_drawer_chars
        )
        chunks = chunks_for_grouped_speaker_ingest(
            blocks, wing=wing, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    elif ingest == "by_block":
        turn_blocks = build_temporal_turn_blocks_from_transcript(
            text,
            lines_per_block=temporal_lines_per_block,
            overlap_lines=temporal_block_overlap,
            min_drawer_chars=min_drawer_chars,
        )
        chunks = chunks_for_temporal_turn_ingest(
            turn_blocks, wing=wing, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    elif ingest == "by_record":
        records = parse_atomic_records(text)
        n_core = count_core_records(records)
        assert_synth_preamble_count(text, n_core)
        if n_core == 0:
            raise ValueError(
                "by_record ingest saw no atomic records (no line matched the "
                "'Date: ... || User: ... || Instance: ...' anchor)"
            )
        chunks = chunks_for_atomic_ingest(records, wing=wing)
    else:
        chunks = _chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for item in chunks:
            item["wing"] = wing
            item["room"] = room
    if not chunks:
        raise ValueError(
            "No non-empty chunks produced from input text (content too short or whitespace only)"
        )

    agent = "benchmark_poc"
    docs_buf: list[str] = []
    ids_buf: list[str] = []
    metas_buf: list[dict[str, Any]] = []
    for item in chunks:
        idx = int(item["chunk_index"])
        drawer_id = _drawer_id(task_id, idx)
        meta_room = str(item.get("room", room))
        meta_wing = str(item.get("wing", wing))
        metadata: dict[str, Any] = {
            "wing": meta_wing,
            "room": meta_room,
            "source_file": source_file,
            "chunk_index": idx,
            "added_by": agent,
            "filed_at": filed_at,
        }
        sp = item.get("speaker")
        if isinstance(sp, str) and sp:
            metadata["speaker"] = sp
        sp_list = item.get("speakers")
        if isinstance(sp_list, list) and sp_list:
            metadata["speakers_json"] = json.dumps(sp_list)
        slugs = item.get("speaker_slugs")
        if isinstance(slugs, str) and slugs:
            metadata["speaker_slugs"] = slugs
        ls = item.get("line_start")
        le = item.get("line_end")
        lr = item.get("line_ranges")
        if isinstance(ls, str) and ls and isinstance(le, str) and le:
            metadata["line_start"] = ls
            metadata["line_end"] = le
        if isinstance(lr, str) and lr:
            metadata["line_ranges"] = lr
        li = item.get("line_index")
        if isinstance(li, str) and li:
            metadata["line_index"] = li
        uid = item.get("user_id")
        if isinstance(uid, str) and uid:
            metadata["user_id"] = uid
        diso = item.get("date_iso")
        if isinstance(diso, str) and diso:
            metadata["date_iso"] = diso
        draw = item.get("date_raw")
        if isinstance(draw, str) and draw:
            metadata["date_raw"] = draw
        lbl = item.get("label")
        if isinstance(lbl, str) and lbl:
            metadata["label"] = lbl
        docs_buf.append(item["content"])
        ids_buf.append(drawer_id)
        metas_buf.append(metadata)

    # Batched upsert: identical per-id content/metadata as the row-by-row path, but one
    # Chroma call per batch slashes per-call overhead (embedding + SQLite commit). Batch size
    # is bounded so we never exceed Chroma's per-request max_batch_size limit.
    batch_size = UPSERT_BATCH_SIZE
    max_batch = getattr(chroma_client, "get_max_batch_size", lambda: None)()
    if isinstance(max_batch, int) and max_batch > 0:
        batch_size = min(batch_size, max_batch)
    for i in range(0, len(ids_buf), batch_size):
        col.upsert(
            documents=docs_buf[i : i + batch_size],
            ids=ids_buf[i : i + batch_size],
            metadatas=metas_buf[i : i + batch_size],
        )

    n_drawers = len(chunks)
    palace_path = tmpdir
    default_wing = wing
    grouped_or_search = ingest == "by_speaker"
    active_rooms_sorted: list[str] = sorted(
        {str(c.get("room", "")) for c in chunks if c.get("room")},
        key=_room_slug_sort_tuple,
    )

    if verbose:
        lobby = format_palace_lobby_preview(
            chunks,
            task_id=task_id,
            wing=wing,
            room=room,
            preview_chars=preview_chars,
        )
        print(lobby, end="", flush=True)

    def search_memories_tool(
        query: str,
        wing: str | None = None,
        room: str | None = None,
        n_results: int = 8,
        mode: Literal["semantic", "exact"] = "semantic",
        user: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> str:
        """Search the ephemeral palace: vector ``semantic`` or literal ``exact`` substring match.

        ``user`` / ``date_from`` / ``date_to`` / ``line_start`` / ``line_end`` are supported
        only by ``by_record`` ingest in ``mode='exact'``; they filter atomic-record drawers by
        metadata (``user_id``, ISO ``date_iso``, 1-based ``line_index``).
        """
        if mode not in ("semantic", "exact"):
            raise ValueError("mode must be 'semantic' or 'exact'")
        metadata_filter_set = any(
            v is not None for v in (user, date_from, date_to, line_start, line_end)
        )
        if metadata_filter_set and (mode != "exact" or ingest != "by_record"):
            raise ValueError(
                "user/date_from/date_to/line_start/line_end filters require mode='exact' "
                "and by_record ingest"
            )
        if mode == "exact":
            out = _exact_search_over_chunks_with_filters(
                chunks,
                query,
                wing=wing,
                room=room,
                n_results=n_results,
                user=user,
                date_from=date_from,
                date_to=date_to,
                line_start=line_start,
                line_end=line_end,
            )
            return _format_search_hits(out)
        eff_wing = wing if wing is not None else default_wing
        if grouped_or_search and room and room not in ("_preamble", MIXED_ROOM_SLUG):
            k_room = _search_branch_fetch_size(n_results, wide=False)
            k_wide = _search_branch_fetch_size(n_results, wide=True)
            out1 = _search_memories_reusing_drawers_col(
                col, palace_path, query, wing=eff_wing, room=room, n_results=k_room
            )
            out2 = _search_memories_reusing_drawers_col(
                col, palace_path, query, wing=eff_wing, room=None, n_results=k_wide
            )
            merged = _merge_grouped_by_speaker_search(out1, out2, slug=room, user_n=n_results)
            return _format_search_hits(merged)
        if ingest == "by_record" and room is None and len(active_rooms_sorted) > 1:
            nr = len(active_rooms_sorted)
            base_k = max(1, n_results // nr)
            remainder = max(0, n_results - base_k * nr)
            per_room_outs: list[dict[str, Any]] = []
            for ri, r_slug in enumerate(active_rooms_sorted):
                k_r = base_k + (1 if ri < remainder else 0)
                per_room_outs.append(
                    _search_memories_reusing_drawers_col(
                        col,
                        palace_path,
                        query,
                        wing=eff_wing,
                        room=r_slug,
                        n_results=max(k_r, 2),
                    )
                )
            merged = _merge_multiroom_semantic_hits(
                per_room_outs, query=query, n_results=n_results
            )
            return _format_search_hits(merged)
        out = _search_memories_reusing_drawers_col(
            col,
            palace_path,
            query,
            wing=wing,
            room=room,
            n_results=n_results,
        )
        return _format_search_hits(out)

    search_desc = (
        "Search the indexed benchmark transcript/corpus for this task only. "
        "``mode='semantic'`` (default): natural-language vector search via MemPalace/Chroma. "
        "``mode='exact'``: case-insensitive substring over chunk text (no embeddings); use for "
        "proper nouns, spellings, or quoted fragments. "
        "Optional ``wing=`` / ``room=`` filter metadata (``None`` = no filter on that axis). "
        "Returns hits with ``mode=`` and ``similarity=`` in each header. "
        "The palace only stores the same transcript/context in structured chunks; it does **not** "
        "contain precomputed cumulative answers (totals, counts, etc.). For how-many or "
        "occurrence questions, use `search_memories` to find evidence, then aggregate in the REPL "
        "(Python on retrieved text, and/or `llm_query` / `rlm_query` on assembled evidence)—do not "
        "expect one search to return the final number. "
        "`llm_query` / `rlm_query` only see the string you pass: **paste or assign the actual hit text** "
        "(or a concatenation you built from `search_memories`) into every subagent call—do not invoke "
        "them with empty prompts or instructions alone. "
        "Drawers can **repeat or overlap** the same transcript lines (chunk overlap, grouping splits, "
        "or a repeated boundary line between drawers) so each hit keeps local context—when you merge "
        "several hits or count occurrences, **deduplicate** (e.g. use ``line_start``/``line_end`` when "
        "metadata has them, otherwise normalize overlapping text) and **do not double-count**."
    )
    if ingest == "by_speaker":
        search_desc += (
            " Grouped by-speaker ingest: each room is usually the **dominant** speaker for that "
            "drawer (most text); shorter interjections still appear with ``Name:`` labels. "
            "``room=<slug>`` also returns drawers where that speaker spoke in a **different** "
            "dominant room (one internal merge). "
            "``room='_mixed'`` is for tied multi-speaker bundles. "
            "Use room='_preamble' for instructions and player-to-character mapping if needed by the query."
        )
    elif ingest == "by_block":
        search_desc += (
            " Temporal by-block ingest: room='block_001', block_002, … are overlapping windows of "
            f"non-empty episode lines (default window {DEFAULT_TEMPORAL_LINES_PER_BLOCK} lines). "
            "Inside each room, **interaction grouping** buffers lines in order (any speaker) until "
            f"a size threshold (same min_drawer_chars idea as grouped ingest, default {DEFAULT_MIN_DRAWER_CHARS}), "
            "then starts the next drawer—rapid back-and-forth stays in one drawer. "
            "The line that tripped the threshold is duplicated at the start of the following drawer when more lines remain in that block. "
            "Metadata includes line_start/line_end and speakers_json (all Name: speakers in that drawer). "
            "Use list_taxonomy() for ordering."
        )
    elif ingest == "by_record":
        search_desc += (
            " Atomic-record ingest (Oolong-synth): every drawer is exactly one line of the form "
            "'Date: <Mon DD, YYYY> || User: <user_id> || Instance: <text> || Label: <label>'. "
            "There is **one room per TREC-coarse label slug** (e.g. 'location', 'numeric_value', "
            "'description', ...) plus 'unclassified' for preamble/trailer instructions and any "
            "records without a Label segment; call list_taxonomy() for the exact list and "
            "per-label drawer counts. Drawers do NOT overlap (each source line appears in "
            "exactly one drawer) so occurrence counts need no dedup. "
            "If room is omitted (None), the search audits the **entire palace**—exact mode "
            "scans all drawers, semantic mode merges per-label-room Chroma queries and "
            "dedupes by line_index. Pass room='<label_slug>' to restrict to one label for "
            "precision. Each hit carries metadata fields line_index (1-based into the stitched "
            "context), user_id, date_iso (YYYY-MM-DD), date_raw, and label (when present). "
            "In mode='exact' only, the tool accepts user=<id>, date_from=<YYYY-MM-DD>, "
            "date_to=<YYYY-MM-DD> (inclusive range), and line_start=<int>/line_end=<int> "
            "filters for O(n) metadata-filtered scans without a vector query."
        )

    custom_tools: dict[str, Any] = {
        "search_memories": {
            "tool": search_memories_tool,
            "description": search_desc,
        }
    }

    if ingest in ("by_speaker", "by_block", "by_record"):

        def list_taxonomy_tool() -> str:
            """List wings and rooms with drawer counts; _preamble is always first per wing."""
            return format_taxonomy_json(chunks)

        if ingest == "by_record":
            taxonomy_desc = (
                "Return JSON listing each wing and its rooms with drawer counts. "
                "Rooms correspond to TREC-coarse label buckets (e.g. 'location', "
                "'numeric_value'); 'unclassified' holds preamble/trailer instructions and any "
                "records that had no Label segment. Call this first for per-label totals before "
                "issuing any targeted search."
            )
        else:
            taxonomy_desc = (
                "Return JSON listing each wing and its rooms with drawer counts. "
                "The _preamble room is always listed first and holds the mapping from players to "
                "characters; call this before searching dialogue if you need that mapping."
            )
        custom_tools["list_taxonomy"] = {
            "tool": list_taxonomy_tool,
            "description": taxonomy_desc,
        }

    def cleanup() -> None:
        try:
            chroma_client.reset()
        except Exception:
            pass
        shutil.rmtree(palace_path, ignore_errors=True)

    return custom_tools, cleanup, n_drawers


def palace_poc_prompt_hint(
    n_drawers: int,
    *,
    palace_poc_by_speaker: bool = False,
    palace_poc_by_block: bool = False,
) -> str:
    """Extra root-prompt text when palace PoC is enabled."""
    base = (
        f"The same transcript is also indexed in an ephemeral Memory Palace ({n_drawers} chunks). "
        "Call `search_memories(query)` in a REPL block for vector search by meaning, or "
        "`search_memories(query, mode='exact')` for case-insensitive substring / spelling matches. "
        "You still have the full text in context['context_window_text'] (lenient mode); prefer search "
        "when the transcript is very long.\n\n"
        "The palace holds the transcript in a **structured, chunked** form only; it does **not** "
        "store final aggregate answers (e.g. total counts). For counting or how-many questions, "
        "use `search_memories` to gather evidence, then **aggregate in the REPL** with Python "
        "and/or `llm_query` / `rlm_query` on the combined hits—do not assume the final number "
        "appears verbatim in search results.\n\n"
        "**Sub-LLM / subagent calls (`llm_query`, `rlm_query`):** they only receive the prompt string you "
        "give them—**include the retrieved passage text** (verbatim from `search_memories` or variables "
        "you assembled). Do not call subagents with no transcript excerpt; they cannot search the palace "
        "themselves.\n\n"
        "Indexing **reuses lines across drawers** on purpose (overlap for context). When combining "
        "multiple retrievals or counting, **avoid double-counting** the same "
        "lines.\n\n"
    )
    if palace_poc_by_speaker:
        base += (
            "The palace is indexed **by speaker** (grouped drawers): each ``room`` is usually the "
            "**dominant** speaker for that bundle; other speakers still appear with ``Name:`` lines. "
            "``room='_mixed'`` marks tied multi-speaker bundles. "
            "``search_memories(..., room='<slug>')`` in **semantic** mode also finds drawers where that "
            "character spoke under another dominant room (exact mode filters strictly to that room). "
            "Call `list_taxonomy()` for wings and rooms. **If the query requires knowing player or character names, always check the `_preamble` room** for "
            "character-to-player mappings before relying on dialogue alone.\n\n"
        )
    elif palace_poc_by_block:
        base += (
            "The palace is indexed **by time** in ``block_001``, ``block_002``, … rooms (see "
            "`list_taxonomy()`). Inside each block, drawers use **interaction grouping** (multi-speaker "
            "runs until a size threshold); metadata lists all speakers in a drawer. "
            "Room names are temporal segments, not cast. "
            "**If the query requires knowing the player or character names, always check the `_preamble` room** for character-to-player mappings.\n\n"
        )
    return base


def palace_poc_prompt_hint_strict(
    n_drawers: int,
    *,
    palace_poc_by_speaker: bool = False,
    palace_poc_by_block: bool = False,
) -> str:
    """Root-prompt text for strict palace mode (no full transcript in context)."""
    base = (
        f"The transcript is NOT in context as raw text; it is only available via an indexed Memory "
        f"Palace ({n_drawers} chunks). Always look at preexisting variables inside the REPL environment to understand the question and keywords to look for.Use `search_memories(query)` for semantic retrieval, or "
        "`search_memories(query, mode='exact')` for literal substring search. Use this exact mode when you need to find exact things like specific names or numbers. Be careful with brittle exact matches."
        "Then answer from retrieved content.\n\n"
        "The palace is only the transcript in **structured chunks**; it does **not** contain "
        "precomputed totals or cumulative answers. You should not query for the final answer as if it were stored in the index. For counts or occurrences, search for "
        "relevant passages, then **aggregate in the REPL** (Python over returned text, and/or "
        "`llm_query` / `rlm_query` on evidence you assemble)—do not query for the final answer "
        "as if it were stored in the index.\n\n"
        "**Sub-LLM / subagent calls:** `llm_query` / `rlm_query` have **no** access to the palace or unseen "
        "context—**paste the exact transcript text** you retrieved (or a variable holding it) into each "
        "call. Never call a subagent with only a question and no evidence string.\n\n"
    )
    if palace_poc_by_speaker:
        base += (
            "The palace is indexed **by speaker** (grouped drawers; dominant ``room`` per bundle, "
            "``_mixed`` on ties; semantic `search_memories` expands `room=` to include non-dominant turns; "
            "exact mode does not). "
            "Use `list_taxonomy()` for room names. "
            "**If required, always check the `_preamble` room** for character-to-player mappings before "
            "relying on dialogue alone.\n\n"
        )
    elif palace_poc_by_block:
        base += (
            "The palace is indexed **by time** (`block_NNN` rooms); drawers bundle rapid dialogue "
            "(interaction grouping) and record all involved speakers in metadata. "
            "Use `list_taxonomy()` for order. "
            "**If required, always check the `_preamble` room** for character-to-player mappings.\n\n"
        )
    return base


def palace_poc_prompt_hint_by_record(
    n_drawers: int,
    *,
    strict: bool = False,
) -> str:
    """Root-prompt text for Oolong-synth atomic-record palace ingest.

    Describes the ingest shape, available metadata, both search modes, and per-record
    aggregation strategy. ``strict`` swaps wording for the case where the full stitched
    context is NOT in ``context['context_window_text']``.
    """
    if strict:
        opener = (
            f"The Oolong-synth context is NOT in context as raw text; it is only available via an "
            f"indexed Memory Palace with {n_drawers} drawers, one per atomic record."
        )
    else:
        opener = (
            f"The same Oolong-synth context is also indexed in an ephemeral Memory Palace with "
            f"{n_drawers} drawers, one per atomic record. The raw stitched text is still in "
            f"context['context_window_text']; prefer the palace when the context is long."
        )
    return (
        f"{opener} Each drawer is exactly one line of the form "
        "'Date: <Mon DD, YYYY> || User: <user_id> || Instance: <text> || Label: <label>'. "
        "The palace has **one room per TREC-coarse label slug** (e.g. 'location', "
        "'numeric_value', 'description', ...) plus 'unclassified' for preamble/trailer "
        "instruction lines and any records that lacked a Label segment. Drawers do not overlap, "
        "so occurrence counts do not need dedup.\n\n"
        "Each hit carries metadata: line_index (1-based into the full stitched context), "
        "user_id, date_iso (YYYY-MM-DD), date_raw, and label (when present).\n\n"
        "Recommended workflow:\n"
        "1. Call `list_taxonomy()` FIRST for per-label drawer counts and the full room list. "
        "For questions that reduce to 'how many records have label X', the taxonomy count is "
        "already the answer—no search needed.\n"
        "2. For conditioned questions (e.g. 'location mentions Europe', 'numeric questions "
        "asked by user42'), prefer a **room-scoped** search: "
        "`search_memories(query, room='<label_slug>', mode='semantic')` or `mode='exact'`. "
        "Room-scoped searches are sharper than whole-palace aggregation.\n"
        "3. Omit `room` (or pass `None`) only when you truly want to audit every label at "
        "once—semantic mode merges per-room Chroma queries and dedupes by line_index; exact "
        "mode scans all drawers.\n"
        "4. `mode='exact'` also accepts atomic-record filters: user=<id>, "
        "date_from=<YYYY-MM-DD>, date_to=<YYYY-MM-DD> (inclusive), line_start=<int>, "
        "line_end=<int>. Use a guaranteed substring (e.g. ' || ') when you only want to filter "
        "by metadata.\n\n"
        "Classify or aggregate retrieved records per-line in the REPL—prefer "
        "`llm_query_batched` / `rlm_query_batched` to parallelize across many hits. Every "
        "sub-call must include the actual record text verbatim; subagents cannot see the palace.\n\n"
    )
