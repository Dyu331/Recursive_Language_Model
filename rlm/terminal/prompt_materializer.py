from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from rlm.terminal.context_store import ContextStore
from rlm.terminal.terminal_adapter import TerminalAction


@dataclass
class TerminalDecision:
    analysis: str
    task_complete: bool
    commands: list[TerminalAction]
    error_summary: str | None = None
    file_chunk_summaries: list[str] = field(default_factory=list)
    notes: str | None = None


class TerminalPromptMaterializer:
    """Materializes host-side memory for RLM and parses action decisions."""

    output_schema: dict[str, Any] = {
        "analysis": "string",
        "task_complete": "boolean",
        "commands": [{"keystrokes": "string", "duration_sec": "number"}],
        "error_summary": "string|null",
        "file_chunk_summaries": ["string"],
        "notes": "string|null",
    }

    def build_context_payload(
        self,
        task_description: str,
        context_store: ContextStore,
        turn_index: int,
    ) -> dict[str, Any]:
        return {
            "task_description": task_description,
            "turn_index": turn_index,
            "terminal_memory": context_store.materialize(),
            "reminder": (
                "Task-world operations must remain terminal-first. "
                "Do not use host-local file APIs as task-world APIs."
            ),
        }

    def build_root_prompt(self, task_description: str, turn_index: int) -> str:
        schema_str = json.dumps(self.output_schema, ensure_ascii=False)
        return (
            f"Task: {task_description}\n"
            f"Current turn: {turn_index}\n"
            "Use RLM REPL and recursive rlm_query calls for cognition. "
            "Return only the next terminal action batch in FINAL(JSON).\n\n"
            "Do not claim task completion unless terminal evidence supports it.\n"
            "If no action is needed, return commands=[].\n\n"
            "Return exactly one FINAL(...) containing a JSON object with schema:\n"
            f"{schema_str}\n\n"
            "`commands` are keystrokes for the tmux terminal adapter; include \\n for Enter."
        )

    def parse_decision(self, completion_response: str) -> TerminalDecision:
        payload = self._parse_json_object(completion_response)
        if payload is None:
            return TerminalDecision(
                analysis=completion_response.strip()[:400],
                task_complete=False,
                commands=[],
                error_summary="Could not parse JSON decision from RLM response.",
            )

        commands = self._parse_commands(payload.get("commands", []))
        return TerminalDecision(
            analysis=str(payload.get("analysis", "")),
            task_complete=bool(payload.get("task_complete", False)),
            commands=commands,
            error_summary=self._nullable_string(payload.get("error_summary")),
            file_chunk_summaries=self._parse_string_list(payload.get("file_chunk_summaries", [])),
            notes=self._nullable_string(payload.get("notes")),
        )

    def _parse_json_object(self, text: str) -> dict[str, Any] | None:
        stripped = text.strip()

        direct = self._try_load_json(stripped)
        if direct is not None:
            return direct

        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if match is None:
            return None

        return self._try_load_json(match.group(0))

    def _try_load_json(self, maybe_json: str) -> dict[str, Any] | None:
        try:
            loaded = json.loads(maybe_json)
        except json.JSONDecodeError:
            return None

        if isinstance(loaded, dict):
            return loaded

        return None

    def _parse_commands(self, payload: Any) -> list[TerminalAction]:
        if not isinstance(payload, list):
            return []

        commands: list[TerminalAction] = []
        for item in payload:
            if isinstance(item, str):
                commands.append(TerminalAction(keystrokes=item, duration_sec=0.5))
                continue

            if not isinstance(item, dict):
                continue

            keystrokes = str(item.get("keystrokes", ""))
            duration_value = item.get("duration_sec", item.get("duration", 0.5))
            duration_sec = 0.5
            if isinstance(duration_value, (int, float)):
                duration_sec = float(duration_value)

            commands.append(
                TerminalAction(
                    keystrokes=keystrokes,
                    duration_sec=max(0.0, duration_sec),
                )
            )

        return commands

    def _parse_string_list(self, payload: Any) -> list[str]:
        if not isinstance(payload, list):
            return []

        return [str(item) for item in payload if str(item).strip()]

    def _nullable_string(self, value: Any) -> str | None:
        if value is None:
            return None

        text = str(value).strip()
        if text == "":
            return None

        return text
