from __future__ import annotations

from typing import Any

from rlm.core.types import RLMChatCompletion

from rlm.terminal.context_store import ContextStore, TerminalObservation
from rlm.terminal.prompt_materializer import TerminalDecision
from rlm.terminal.terminal_adapter import TerminalAction


class TerminalObservationIndexer:
    """Indexes terminal observations and RLM artifacts into ContextStore."""

    error_markers: tuple[str, ...] = (
        "traceback",
        "error",
        "exception",
        "failed",
        "no such file",
        "not found",
        "permission denied",
    )

    file_read_prefixes: tuple[str, ...] = (
        "cat ",
        "sed ",
        "head ",
        "tail ",
        "less ",
        "more ",
        "awk ",
    )

    def ingest_observation(self, observation: TerminalObservation, store: ContextStore) -> None:
        store.append_with_limit(store.observations, observation)

        error_summary = self._summarize_error_signal(observation.incremental_output)
        if error_summary is not None:
            store.append_with_limit(store.error_summaries, error_summary)

    def ingest_actions(self, actions: list[TerminalAction], store: ContextStore) -> None:
        for action in actions:
            store.append_with_limit(store.recent_commands, action.keystrokes)
            file_summary = self._summarize_file_read(action.keystrokes)
            if file_summary is not None:
                store.append_with_limit(store.file_chunk_summaries, file_summary)

    def ingest_decision(self, decision: TerminalDecision, store: ContextStore) -> None:
        summary = f"analysis={decision.analysis[:220]} complete={decision.task_complete}"
        store.append_with_limit(store.decision_summaries, summary)

        if decision.error_summary:
            store.append_with_limit(store.error_summaries, decision.error_summary)

        for item in decision.file_chunk_summaries:
            if item.strip():
                store.append_with_limit(store.file_chunk_summaries, item.strip())

    def ingest_completion(self, completion: RLMChatCompletion, store: ContextStore) -> None:
        for item in self._extract_subcall_results(completion.metadata):
            store.append_with_limit(store.subcall_results, item)

    def _summarize_error_signal(self, text: str) -> str | None:
        lowered = text.lower()
        if not any(marker in lowered for marker in self.error_markers):
            return None

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        lines = lines[-6:]
        joined = " | ".join(lines)
        return f"Terminal error signal: {joined[:500]}"

    def _summarize_file_read(self, keystrokes: str) -> str | None:
        stripped = keystrokes.strip()
        if stripped.startswith(self.file_read_prefixes):
            return f"Observed file-read command: {stripped[:240]}"
        return None

    def _extract_subcall_results(self, metadata: dict[str, Any] | None) -> list[str]:
        if metadata is None:
            return []

        results: list[str] = []
        iterations = metadata.get("iterations", [])
        if not isinstance(iterations, list):
            return results

        for iteration in iterations:
            if not isinstance(iteration, dict):
                continue
            code_blocks = iteration.get("code_blocks", [])
            if not isinstance(code_blocks, list):
                continue

            for block in code_blocks:
                if not isinstance(block, dict):
                    continue
                result = block.get("result", {})
                if not isinstance(result, dict):
                    continue
                calls = result.get("rlm_calls", [])
                if not isinstance(calls, list):
                    continue

                for call in calls:
                    if not isinstance(call, dict):
                        continue

                    model = str(call.get("root_model", "unknown"))
                    prompt_preview = str(call.get("prompt", ""))[:180]
                    response_preview = str(call.get("response", ""))[:220]
                    results.append(
                        " | ".join(
                            [
                                f"model={model}",
                                f"prompt={prompt_preview}",
                                f"response={response_preview}",
                            ]
                        )
                    )

        return results
