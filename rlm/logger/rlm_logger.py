"""
Logger for RLM iterations.

Captures run metadata and iterations in memory so they can be attached to
RLMChatCompletion.metadata. Optionally writes the same data to JSON-lines files.
"""

import json
import os
import uuid
from datetime import datetime
from typing import Any

from rlm.core.types import RLMIteration, RLMMetadata


def _truncate_words(text: str, max_words: int) -> str:
    """Return the first max_words whitespace-separated words, with a suffix if truncated."""
    words = text.split()
    if len(words) <= max_words:
        return text
    omitted = len(words) - max_words
    return " ".join(words[:max_words]) + f"\n... [truncated: {omitted} words omitted]"


def _truncate_repl_streams(obj: Any, max_words: int) -> Any:
    """
    Deep-copy JSON-serializable structures, truncating str values for keys stdout/stderr.
    """
    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        for k, v in obj.items():
            if k in ("stdout", "stderr") and isinstance(v, str):
                out[k] = _truncate_words(v, max_words)
            else:
                out[k] = _truncate_repl_streams(v, max_words)
        return out
    if isinstance(obj, list):
        return [_truncate_repl_streams(item, max_words) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_truncate_repl_streams(item, max_words) for item in obj)
    return obj


class RLMLogger:
    """
    Captures trajectory (run metadata + iterations) for each completion.
    By default only captures in memory; set log_dir to also save to JSON-lines files.

    - log_dir=None: trajectory is available via get_trajectory() and can be
      attached to RLMChatCompletion.metadata (no disk write).
    - log_dir="path": same capture plus appends to a JSONL file per run.

    truncate_repl_output_words: When not None, stdout and stderr strings anywhere in
    logged iteration payloads (including nested subcall metadata) are truncated to
    this many whitespace-separated words, with a suffix noting how many were omitted.
    Pass None to keep full streams. Default is 200.
    """

    def __init__(
        self,
        log_dir: str | None = None,
        file_name: str = "rlm",
        *,
        truncate_repl_output_words: int | None = 200,
    ):
        self._save_to_disk = log_dir is not None
        self.log_dir = log_dir
        self.log_file_path: str | None = None
        self._truncate_repl_output_words = truncate_repl_output_words
        if self._save_to_disk and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_id = str(uuid.uuid4())[:8]
            self.log_file_path = os.path.join(log_dir, f"{file_name}_{timestamp}_{run_id}.jsonl")

        self._run_metadata: dict | None = None
        self._iterations: list[dict] = []
        self._iteration_count = 0
        self._metadata_logged = False

    def log_metadata(self, metadata: RLMMetadata) -> None:
        """Capture run metadata (and optionally write to file)."""
        if self._metadata_logged:
            return

        self._run_metadata = metadata.to_dict()
        self._metadata_logged = True

        if self._save_to_disk and self.log_file_path:
            entry = {
                "type": "metadata",
                "timestamp": datetime.now().isoformat(),
                **self._run_metadata,
            }
            with open(self.log_file_path, "a") as f:
                json.dump(entry, f)
                f.write("\n")

    def log(self, iteration: RLMIteration) -> None:
        """Capture one iteration (and optionally append to file)."""
        self._iteration_count += 1
        entry: dict[str, Any] = {
            "type": "iteration",
            "iteration": self._iteration_count,
            "timestamp": datetime.now().isoformat(),
            **iteration.to_dict(),
        }
        if self._truncate_repl_output_words is not None:
            entry = _truncate_repl_streams(entry, self._truncate_repl_output_words)
        self._iterations.append(entry)

        if self._save_to_disk and self.log_file_path:
            with open(self.log_file_path, "a") as f:
                json.dump(entry, f)
                f.write("\n")

    def clear_iterations(self) -> None:
        """Reset iterations for the next completion (trajectory is per completion)."""
        self._iterations = []
        self._iteration_count = 0

    def get_trajectory(self) -> dict | None:
        """Return captured run_metadata + iterations for the current completion, or None if no metadata yet."""
        if self._run_metadata is None:
            return None
        return {
            "run_metadata": self._run_metadata,
            "iterations": list(self._iterations),
        }

    @property
    def iteration_count(self) -> int:
        return self._iteration_count
