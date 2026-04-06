from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TerminalObservation:
    timestamp_sec: float
    turn_index: int
    incremental_output: str
    current_screen: str
    session_alive: bool
    recent_commands: list[str] = field(default_factory=list)


class ContextStore:
    """Host-side memory for terminal observations and derived summaries."""

    def __init__(self, max_items: int = 60):
        self.max_items = max_items
        self.observations: list[TerminalObservation] = []
        self.recent_commands: list[str] = []
        self.error_summaries: list[str] = []
        self.file_chunk_summaries: list[str] = []
        self.subcall_results: list[str] = []
        self.decision_summaries: list[str] = []

    def append_with_limit(self, bucket: list[Any], value: Any) -> None:
        bucket.append(value)
        if len(bucket) > self.max_items:
            overflow = len(bucket) - self.max_items
            del bucket[:overflow]

    def materialize(self, recent_items: int = 8) -> dict[str, Any]:
        latest = self.observations[-1] if self.observations else None

        return {
            "latest_observation": {
                "turn_index": latest.turn_index if latest else None,
                "timestamp_sec": latest.timestamp_sec if latest else None,
                "session_alive": latest.session_alive if latest else False,
                "incremental_output": latest.incremental_output if latest else "",
                "current_screen": latest.current_screen if latest else "",
            },
            "recent_observations": [
                {
                    "turn_index": item.turn_index,
                    "incremental_output": item.incremental_output,
                    "current_screen": item.current_screen,
                    "session_alive": item.session_alive,
                }
                for item in self.observations[-recent_items:]
            ],
            "recent_commands": self.recent_commands[-recent_items:],
            "error_summaries": self.error_summaries[-recent_items:],
            "file_chunk_summaries": self.file_chunk_summaries[-recent_items:],
            "subcall_results": self.subcall_results[-recent_items:],
            "decision_summaries": self.decision_summaries[-recent_items:],
        }
