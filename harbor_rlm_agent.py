"""
Harbor-compatible RLM agent for Terminal-Bench tasks.

Default execution path is terminal-first and tmux-based:
- Host-side RLM (local REPL + recursive subcalls) remains the reasoning core.
- Task-world operations happen through a Terminus-style tmux bridge.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from typing import Any, Awaitable, Callable

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
from rlm.terminal import CommandResult, TerminalRLMRuntime, TmuxTerminalAdapter


REAL_TASK_PROMPT = """\
You are solving a benchmark task inside a real Harbor task container.

Important environment notes:
- The task world is terminal-first.
- You must reason via RLM's local REPL and recursive calls.
- All task operations happen in terminal.
- Do not assume host-local filesystem access to task files.
- Complete the task end-to-end in the task container, verify, then finish.

Task instruction:
{instruction}

Initial terminal snapshot:
{terminal_output}
"""

INITIAL_SNAPSHOT_COMMAND = (
    "pwd && "
    "printf '\\n== /app ==\\n' && ls -la /app && "
    "printf '\\n== /app/src ==\\n' && ls -la /app/src && "
    "printf '\\n== /app/data ==\\n' && ls -la /app/data"
)


class HarborShellRunner:
    """Sync wrapper over Harbor async `environment.exec` for terminal adapter usage."""

    def __init__(
        self,
        environment: BaseEnvironment,
        default_cwd: str = "/app",
        default_timeout_sec: int = 120,
    ):
        self._environment = environment
        self._default_cwd = default_cwd
        self._default_timeout_sec = default_timeout_sec

    def run(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> CommandResult:
        effective_cwd = cwd if cwd is not None else self._default_cwd
        effective_timeout = (
            timeout_sec if timeout_sec is not None else self._default_timeout_sec
        )

        result = self._run_coroutine(
            lambda: self._environment.exec(
                command=command,
                cwd=effective_cwd,
                env=env,
                timeout_sec=effective_timeout,
            )
        )
        return CommandResult(
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            return_code=result.return_code,
        )

    @staticmethod
    def _run_coroutine(coroutine_factory: Callable[[], Awaitable[Any]]) -> Any:
        result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def _worker() -> None:
            try:
                result_queue.put((True, asyncio.run(coroutine_factory())))
            except BaseException as exc:  # noqa: BLE001
                result_queue.put((False, exc))

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        success, payload = result_queue.get()
        thread.join()
        if success:
            return payload
        raise payload


class HarborRLMAgent(BaseAgent):
    """Harbor agent that delegates execution to RLM with a tmux terminal runtime."""

    @staticmethod
    def name() -> str:
        return "rlm-agent"

    def version(self) -> str | None:
        return "5.0"

    def __init__(
        self,
        *args,
        backend: str = "litellm",
        backend_kwargs: dict[str, Any] | str | None = None,
        sub_backend: str | None = None,
        sub_backend_kwargs: dict[str, Any] | str | None = None,
        execution_path: str = "terminal",
        rlm_environment: str = "local",
        task_repl_workdir: str = "/app",
        task_repl_workspace_root: str = "/tmp/rlm_task_repl",
        task_repl_proxy_host: str = "host.docker.internal",
        rlm_max_depth: int | str = 2,
        rlm_max_iterations: int | str = 30,
        rlm_subagent_max_iterations: int | str | None = 1,
        rlm_max_timeout_sec: float | str | None = None,
        rlm_max_errors: int | str | None = None,
        terminal_idle_wait_sec: float | str = 0.5,
        tmux_auto_install: bool | str = True,
        tmux_install_timeout_sec: int | str = 240,
        rlm_prompt_profile: str = "default",
        rlm_log_dir: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        parsed_backend_kwargs = self._parse_backend_kwargs(backend_kwargs)
        parsed_sub_backend_kwargs = self._parse_backend_kwargs(sub_backend_kwargs)

        parsed_max_depth = int(rlm_max_depth)
        parsed_max_iterations = int(rlm_max_iterations)
        parsed_subagent_max_iterations = (
            int(rlm_subagent_max_iterations)
            if rlm_subagent_max_iterations is not None
            else None
        )
        parsed_max_timeout = (
            float(rlm_max_timeout_sec) if rlm_max_timeout_sec is not None else None
        )
        parsed_max_errors = int(rlm_max_errors) if rlm_max_errors is not None else None
        parsed_terminal_idle_wait = float(terminal_idle_wait_sec)
        parsed_tmux_auto_install = self._parse_bool(tmux_auto_install)
        parsed_tmux_install_timeout_sec = int(tmux_install_timeout_sec)
        parsed_prompt_profile = rlm_prompt_profile.strip() or "default"

        parsed_backend_kwargs = self._inject_model_name_if_missing(parsed_backend_kwargs)
        parsed_backend_kwargs = self._normalize_backend_kwargs(backend, parsed_backend_kwargs)

        resolved_sub_backend = sub_backend
        if resolved_sub_backend is not None:
            parsed_sub_backend_kwargs = self._inject_model_name_if_missing(
                parsed_sub_backend_kwargs
            )
            parsed_sub_backend_kwargs = self._normalize_backend_kwargs(
                resolved_sub_backend,
                parsed_sub_backend_kwargs,
            )
        else:
            parsed_sub_backend_kwargs = None

        resolved_execution_path = execution_path.strip().lower()
        if resolved_execution_path not in {"terminal", "legacy"}:
            raise ValueError("execution_path must be 'terminal' or 'legacy'")

        self._execution_path = resolved_execution_path
        self._rlm_environment = rlm_environment
        self._task_repl_workdir = task_repl_workdir
        self._task_repl_workspace_root = task_repl_workspace_root
        self._task_repl_proxy_host = task_repl_proxy_host
        # Terminal path uses 1:1 semantics:
        # - rlm_max_iterations controls outer terminal turns.
        # - each turn runs exactly one internal RLM iteration.
        self._terminal_max_turns = parsed_max_iterations
        self._terminal_idle_wait_sec = parsed_terminal_idle_wait
        self._tmux_auto_install = parsed_tmux_auto_install
        self._tmux_install_timeout_sec = parsed_tmux_install_timeout_sec

        rlm_environment_type = (
            "local" if self._execution_path == "terminal" else self._rlm_environment
        )
        rlm_internal_max_iterations = (
            1 if self._execution_path == "terminal" else parsed_max_iterations
        )
        rlm_environment_kwargs: dict[str, Any] = {}
        if self._execution_path == "legacy":
            rlm_environment_kwargs = {
                "workdir": task_repl_workdir,
                "workspace_root": task_repl_workspace_root,
                "proxy_host": task_repl_proxy_host,
            }

        self._rlm = RLM(
            backend=backend,
            backend_kwargs=parsed_backend_kwargs,
            environment=rlm_environment_type,
            environment_kwargs=rlm_environment_kwargs,
            subagent_max_iterations=parsed_subagent_max_iterations,
            subagent_backend=resolved_sub_backend,
            subagent_backend_kwargs=parsed_sub_backend_kwargs,
            max_depth=parsed_max_depth,
            max_iterations=rlm_internal_max_iterations,
            max_timeout=parsed_max_timeout,
            max_errors=parsed_max_errors,
            prompt_profile=parsed_prompt_profile,
            verbose=True,
            logger=RLMLogger(log_dir=rlm_log_dir),
        )

    async def setup(self, environment: BaseEnvironment) -> None:
        del environment

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if self._execution_path == "terminal":
            await self._run_terminal_path(instruction, environment, context)
            return

        await self._run_legacy_path(instruction, environment, context)

    async def _run_terminal_path(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        runner = HarborShellRunner(
            environment=environment,
            default_cwd=self._task_repl_workdir,
        )
        adapter = TmuxTerminalAdapter(
            runner=runner,
            workdir=self._task_repl_workdir,
            auto_install_tmux=self._tmux_auto_install,
            tmux_install_timeout_sec=self._tmux_install_timeout_sec,
        )

        try:
            adapter.send_keys(f"{INITIAL_SNAPSHOT_COMMAND}\n", duration=1.0)
            terminal_output = adapter.get_incremental_output()

            prompt = REAL_TASK_PROMPT.format(
                instruction=instruction,
                terminal_output=terminal_output,
            )
            self._write_text_artifact("rlm_prompt.txt", prompt)
            self._write_text_artifact("initial_terminal_output.txt", terminal_output)

            runtime = TerminalRLMRuntime(
                rlm=self._rlm,
                adapter=adapter,
                max_turns=self._terminal_max_turns,
                idle_wait_sec=self._terminal_idle_wait_sec,
            )

            runtime_result = runtime.run(task_description=prompt)
            last_completion = runtime_result.last_completion

            if last_completion is not None:
                usage = last_completion.usage_summary
                context.n_input_tokens = usage.total_input_tokens
                context.n_output_tokens = usage.total_output_tokens
                self._write_text_artifact("rlm_final_response.txt", last_completion.response)

                if last_completion.metadata is not None:
                    self._write_text_artifact(
                        "rlm_trajectory.json",
                        json.dumps(last_completion.metadata, ensure_ascii=False, indent=2),
                    )
            else:
                context.n_input_tokens = 0
                context.n_output_tokens = 0

            runtime_payload = {
                "task_complete": runtime_result.task_complete,
                "turns_executed": runtime_result.turns_executed,
                "last_decision": {
                    "analysis": runtime_result.last_decision.analysis,
                    "task_complete": runtime_result.last_decision.task_complete,
                    "commands": [
                        {
                            "keystrokes": item.keystrokes,
                            "duration_sec": item.duration_sec,
                        }
                        for item in runtime_result.last_decision.commands
                    ],
                    "error_summary": runtime_result.last_decision.error_summary,
                    "file_chunk_summaries": runtime_result.last_decision.file_chunk_summaries,
                    "notes": runtime_result.last_decision.notes,
                }
                if runtime_result.last_decision is not None
                else None,
                "context_snapshot": runtime_result.context_snapshot,
            }
            self._write_text_artifact(
                "terminal_runtime_result.json",
                json.dumps(runtime_payload, ensure_ascii=False, indent=2),
            )

            context.metadata = {
                "agent": "rlm-agent",
                "execution_path": "terminal",
                "environment": "local",
                "task_complete": runtime_result.task_complete,
                "turns_executed": runtime_result.turns_executed,
                "root_model": last_completion.root_model if last_completion else "unknown",
                "execution_time_sec": last_completion.execution_time if last_completion else None,
                "final_response_preview": (
                    last_completion.response[:1000]
                    if last_completion
                    else "no completion produced"
                ),
            }
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
        finally:
            try:
                adapter.close()
            except Exception:
                pass

    async def _run_legacy_path(
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
            "execution_path": "legacy",
            "environment": self._rlm_environment,
            "root_model": completion.root_model,
            "execution_time_sec": completion.execution_time,
            "final_response_preview": completion.response[:1000],
        }

        self._write_text_artifact("rlm_final_response.txt", completion.response)
        if completion.metadata is not None:
            self._write_text_artifact(
                "rlm_trajectory.json",
                json.dumps(completion.metadata, ensure_ascii=False, indent=2),
            )

    async def _exec(
        self,
        environment: BaseEnvironment,
        command: str,
        timeout_sec: int = 120,
    ) -> str:
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
            "execution_path": self._execution_path,
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

    def _parse_backend_kwargs(self, value: dict[str, Any] | str | None) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, str):
            if value.strip() == "":
                return {}
            parsed = json.loads(value)
            if not isinstance(parsed, dict):
                raise ValueError("backend kwargs JSON must decode to an object")
            return parsed
        return dict(value)

    def _inject_model_name_if_missing(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(kwargs)
        if self.model_name and "model_name" not in enriched:
            enriched["model_name"] = self.model_name
        return enriched

    def _normalize_backend_kwargs(self, backend: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(kwargs)
        if backend == "openai":
            api_base = normalized.pop("api_base", None)
            if api_base and "base_url" not in normalized:
                normalized["base_url"] = api_base
        elif backend == "litellm":
            base_url = normalized.pop("base_url", None)
            if base_url and "api_base" not in normalized:
                normalized["api_base"] = base_url
        return normalized

    def _parse_bool(self, value: bool | str) -> bool:
        if isinstance(value, bool):
            return value
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"Cannot parse boolean value: {value}")
