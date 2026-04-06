from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rlm.core.types import RLMChatCompletion
from rlm.terminal.context_store import ContextStore, TerminalObservation
from rlm.terminal.observation_indexer import TerminalObservationIndexer
from rlm.terminal.prompt_materializer import TerminalDecision, TerminalPromptMaterializer
from rlm.terminal.terminal_adapter import TerminalAdapter, TerminalAction

if TYPE_CHECKING:
    from rlm.core.rlm import RLM


@dataclass
class TerminalRunResult:
    task_complete: bool
    turns_executed: int
    last_decision: TerminalDecision | None
    last_completion: RLMChatCompletion | None
    context_snapshot: dict


class TerminalRLMRuntime:
    """
    Connects terminal observations to existing RLM completion loops.

    Per turn:
    1) Observe tmux terminal output/screen.
    2) Index observations into host-side ContextStore.
    3) Call existing RLM.completion(...) with materialized context.
    4) Parse decision JSON into terminal actions.
    5) Execute actions via terminal adapter.
    """

    def __init__(
        self,
        rlm: "RLM",
        adapter: TerminalAdapter,
        context_store: ContextStore | None = None,
        observation_indexer: TerminalObservationIndexer | None = None,
        prompt_materializer: TerminalPromptMaterializer | None = None,
        max_turns: int = 40,
        idle_wait_sec: float = 0.5,
    ):
        if rlm.environment_type != "local":
            raise ValueError(
                "TerminalRLMRuntime requires RLM(environment='local') so host-side "
                "reasoning stays on the existing local REPL path."
            )

        self.rlm = rlm
        self.adapter = adapter
        self.context_store = context_store or ContextStore()
        self.observation_indexer = observation_indexer or TerminalObservationIndexer()
        self.prompt_materializer = prompt_materializer or TerminalPromptMaterializer()
        self.max_turns = max_turns
        self.idle_wait_sec = idle_wait_sec

    def run(self, task_description: str, max_turns: int | None = None) -> TerminalRunResult:
        turn_limit = max_turns if max_turns is not None else self.max_turns

        last_decision: TerminalDecision | None = None
        last_completion: RLMChatCompletion | None = None

        for turn_index in range(1, turn_limit + 1):
            observation = self.collect_observation(turn_index)
            self.observation_indexer.ingest_observation(observation, self.context_store)

            if not observation.session_alive:
                break

            context_payload = self.prompt_materializer.build_context_payload(
                task_description=task_description,
                context_store=self.context_store,
                turn_index=turn_index,
            )
            root_prompt = self.prompt_materializer.build_root_prompt(
                task_description=task_description,
                turn_index=turn_index,
            )

            completion = self.rlm.completion(prompt=context_payload, root_prompt=root_prompt)
            self.observation_indexer.ingest_completion(completion, self.context_store)

            decision = self.prompt_materializer.parse_decision(completion.response)
            self.observation_indexer.ingest_decision(decision, self.context_store)

            self.execute_actions(decision.commands)

            last_decision = decision
            last_completion = completion

            if decision.task_complete:
                return TerminalRunResult(
                    task_complete=True,
                    turns_executed=turn_index,
                    last_decision=decision,
                    last_completion=completion,
                    context_snapshot=self.context_store.materialize(),
                )

        return TerminalRunResult(
            task_complete=False,
            turns_executed=turn_limit,
            last_decision=last_decision,
            last_completion=last_completion,
            context_snapshot=self.context_store.materialize(),
        )

    def collect_observation(self, turn_index: int) -> TerminalObservation:
        incremental_output = self.adapter.get_incremental_output()
        current_screen = self.adapter.capture_screen()
        alive = self.adapter.is_alive()

        return TerminalObservation(
            timestamp_sec=time.time(),
            turn_index=turn_index,
            incremental_output=incremental_output,
            current_screen=current_screen,
            session_alive=alive,
            recent_commands=self.context_store.recent_commands[-8:],
        )

    def execute_actions(self, actions: list[TerminalAction]) -> None:
        if not actions:
            self.adapter.send_keys("", self.idle_wait_sec)
            return

        self.observation_indexer.ingest_actions(actions, self.context_store)
        for action in actions:
            self.adapter.send_keys(action.keystrokes, action.duration_sec)
