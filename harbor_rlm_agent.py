"""
Harbor-compatible RLM agent for Terminal-Bench tasks.

This agent is intentionally thin:
- Harbor provides the task container and verifier lifecycle.
- HarborRLMAgent bridges Harbor's real task environment into RLM.
- RLM does the actual reasoning and task execution through the task-backed REPL.

Usage:
    harbor run -d "terminal-bench@2.0" \
        --agent-import-path harbor_rlm_agent:HarborRLMAgent \
        --model <model_name> \
        --agent-kwarg backend=openai \
        --agent-kwarg 'backend_kwargs={"base_url": "...", "api_key": "..."}' \
        --agent-kwarg rlm_max_depth=2 \
        --agent-kwarg rlm_max_iterations=30
"""

from __future__ import annotations

import json
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.agent.context import AgentContext

from rlm import (
    BudgetExceededError,
    CancellationError,
    ErrorThresholdExceededError,
    RLM,
    TimeoutExceededError,
    TokenLimitExceededError,
)
from rlm.environments.task_backed_repl import HarborExecRunner
from rlm.logger import RLMLogger


REAL_TASK_PROMPT = """\
You are solving a benchmark task inside a real Harbor task container.

Important environment notes:
- The REPL you are using is task-backed, not local-only.
- Python code you execute from the REPL can inspect and modify the real task files.
- Read the real filesystem state instead of assuming files are missing.
- Complete the task end-to-end in the task container, verify your work there, and only then finish.

Task instruction:
{instruction}

Initial shell snapshot:
{terminal_output}
"""

INITIAL_SNAPSHOT_COMMAND = (
    "pwd && "
    "printf '\\n== /app ==\\n' && ls -la /app && "
    "printf '\\n== /app/src ==\\n' && ls -la /app/src && "
    "printf '\\n== /app/data ==\\n' && ls -la /app/data"
)


class HarborRLMAgent(BaseAgent):
    """Harbor agent that directly delegates task execution to RLM."""

    @staticmethod
    def name() -> str:
        return "rlm-agent"

    def version(self) -> str | None:
        return "3.0"

    def __init__(
        self,
        *args,
        backend: str = "litellm",
        backend_kwargs: dict[str, Any] | str | None = None,
        rlm_environment: str = "task",
        task_repl_workdir: str = "/app",
        task_repl_workspace_root: str = "/tmp/rlm_task_repl",
        task_repl_proxy_host: str = "host.docker.internal",
        rlm_max_depth: int | str = 2,
        rlm_max_iterations: int | str = 30,
        rlm_max_timeout_sec: float | str | None = None,
        rlm_max_errors: int | str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if isinstance(backend_kwargs, str):
            backend_kwargs = json.loads(backend_kwargs)
        backend_kwargs = backend_kwargs or {}

        rlm_max_depth = int(rlm_max_depth)
        rlm_max_iterations = int(rlm_max_iterations)
        parsed_max_timeout = (
            float(rlm_max_timeout_sec) if rlm_max_timeout_sec is not None else None
        )
        parsed_max_errors = int(rlm_max_errors) if rlm_max_errors is not None else None

        # Inject model_name into backend_kwargs from harbor's model_name
        if self.model_name and "model_name" not in backend_kwargs:
            backend_kwargs["model_name"] = self.model_name

        # Normalize backend kwargs (openai uses base_url, litellm uses api_base)
        if backend == "openai":
            api_base = backend_kwargs.pop("api_base", None)
            if api_base and "base_url" not in backend_kwargs:
                backend_kwargs["base_url"] = api_base
        elif backend == "litellm":
            base_url = backend_kwargs.pop("base_url", None)
            if base_url and "api_base" not in backend_kwargs:
                backend_kwargs["api_base"] = base_url

        self._rlm_environment = rlm_environment
        self._task_repl_workdir = task_repl_workdir
        self._task_repl_workspace_root = task_repl_workspace_root
        self._task_repl_proxy_host = task_repl_proxy_host

        self._rlm = RLM(
            backend=backend,
            backend_kwargs=backend_kwargs,
            environment=rlm_environment,
            environment_kwargs={
                "workdir": task_repl_workdir,
                "workspace_root": task_repl_workspace_root,
                "proxy_host": task_repl_proxy_host,
            },
            max_depth=rlm_max_depth,
            max_iterations=rlm_max_iterations,
            max_timeout=parsed_max_timeout,
            max_errors=parsed_max_errors,
            verbose=True,
            logger=RLMLogger(log_dir="/home/subc/rlm_tb/rlm/visualizer/public/logs"),
        )

    async def setup(self, environment: BaseEnvironment) -> None:
        """No special setup is needed beyond Harbor's task environment."""
        del environment

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if self._rlm_environment == "task":
            self._rlm.environment_kwargs.update(
                {
                    "runner": HarborExecRunner(environment),
                    "workdir": self._task_repl_workdir,
                    "workspace_root": self._task_repl_workspace_root,
                    "proxy_host": self._task_repl_proxy_host,
                }
            )

        terminal_output = await self._exec(environment, INITIAL_SNAPSHOT_COMMAND)
        prompt = REAL_TASK_PROMPT.format(
            instruction=instruction,
            terminal_output=terminal_output,
        )

        self._write_text_artifact("rlm_prompt.txt", prompt)
        self._write_text_artifact("initial_terminal_output.txt", terminal_output)

        try:
            completion = self._rlm.completion(prompt, root_prompt=instruction)
        except (
            BudgetExceededError,
            CancellationError,
            ErrorThresholdExceededError,
            TimeoutExceededError,
            TokenLimitExceededError,
        ) as exc:
            self._record_failure(context, exc)
            raise
        except Exception as exc:
            self._record_failure(context, exc)
            raise

        usage = completion.usage_summary
        context.n_input_tokens = usage.total_input_tokens
        context.n_output_tokens = usage.total_output_tokens
        context.metadata = {
            "agent": "rlm-agent",
            "environment": self._rlm_environment,
            "root_model": completion.root_model,
            "execution_time_sec": completion.execution_time,
            "final_response_preview": completion.response[:1000],
        }

        self._write_text_artifact("rlm_final_response.txt", completion.response)
        if completion.metadata is not None:
            self._write_text_artifact(
                "rlm_trajectory.json",
                json.dumps(completion.metadata, indent=2),
            )

    async def _exec(
        self,
        environment: BaseEnvironment,
        command: str,
        timeout_sec: int = 120,
    ) -> str:
        """Execute a command and return combined stdout/stderr as a string."""
        try:
            result: ExecResult = await environment.exec(
                command,
                timeout_sec=timeout_sec,
            )
        except Exception as exc:
            return f"[EXEC ERROR] {exc}"

        parts: list[str] = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(f"[STDERR] {result.stderr}")
        if result.return_code != 0:
            parts.append(f"[EXIT CODE] {result.return_code}")
        return "\n".join(parts) if parts else "(empty output)"

    def _record_failure(self, context: AgentContext, exc: Exception) -> None:
        partial_answer = getattr(exc, "partial_answer", None)
        context.metadata = {
            "agent": "rlm-agent",
            "environment": self._rlm_environment,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "partial_answer_preview": partial_answer[:1000] if partial_answer else None,
        }
        self._write_text_artifact("rlm_error.txt", str(exc))
        if partial_answer:
            self._write_text_artifact("rlm_partial_answer.txt", partial_answer)

    def _write_text_artifact(self, filename: str, content: str) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / filename).write_text(content)
