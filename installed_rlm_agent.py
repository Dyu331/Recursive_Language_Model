"""
Harbor-compatible RLM agent that runs fully through TaskBackedREPL.

This execution path keeps RLM as the core controller while executing REPL code
inside the task container. Before starting RLM, the agent ensures `python3`
exists in the task runtime and attempts installation when missing.
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
- The REPL is task-backed and executes in the real task runtime.
- Treat task files as container-side resources.
- Read actual state from terminal/shell commands and task-backed REPL code.
- Complete the task end-to-end in the task container, verify, then finish.
- Finalization is strict: once verification passes, immediately submit using
  FINAL_VAR(final_answer) (or FINAL(...)) in the very next response.
- Do not output empty responses.
- Do not keep chatting after verification is complete.
- Always set a variable named `final_answer` before calling FINAL_VAR(final_answer).

Task instruction:
{instruction}

Initial shell snapshot:
{terminal_output}
"""

INITIAL_SNAPSHOT_COMMAND = (
    "pwd && "
    "printf '\\n== /app ==\\n' && ls -la /app && "
    "printf '\\n== /app/repo ==\\n' && ls -la /app/repo"
)


class InstalledRLMAgent(BaseAgent):
    """RLM agent that runs in task-backed mode with runtime dependency checks."""

    @staticmethod
    def name() -> str:
        return "installed-rlm-agent"

    def version(self) -> str | None:
        return "1.0"

    def __init__(
        self,
        *args,
        backend: str = "litellm",
        backend_kwargs: dict[str, Any] | str | None = None,
        sub_backend: str | None = None,
        sub_backend_kwargs: dict[str, Any] | str | None = None,
        task_repl_workdir: str = "/app",
        task_repl_workspace_root: str = "/tmp/rlm_task_repl",
        task_repl_proxy_host: str = "host.docker.internal",
        task_repl_proxy_bind_host: str = "0.0.0.0",
        python_auto_install: bool | str = True,
        python_install_timeout_sec: int | str = 300,
        rlm_max_depth: int | str = 2,
        rlm_max_iterations: int | str = 30,
        rlm_subagent_max_iterations: int | str | None = 1,
        rlm_max_timeout_sec: float | str | None = None,
        rlm_max_errors: int | str | None = None,
        rlm_prompt_profile: str = "default",
        rlm_log_dir: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        parsed_backend_kwargs = self.parse_backend_kwargs(backend_kwargs)
        parsed_sub_backend_kwargs = self.parse_backend_kwargs(sub_backend_kwargs)

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
        parsed_python_auto_install = self.parse_bool(python_auto_install)
        parsed_python_install_timeout_sec = int(python_install_timeout_sec)
        parsed_prompt_profile = rlm_prompt_profile.strip() or "default"

        parsed_backend_kwargs = self.inject_model_name_if_missing(parsed_backend_kwargs)
        parsed_backend_kwargs = self.normalize_backend_kwargs(backend, parsed_backend_kwargs)

        resolved_sub_backend = sub_backend
        if resolved_sub_backend is not None:
            parsed_sub_backend_kwargs = self.inject_model_name_if_missing(
                parsed_sub_backend_kwargs
            )
            parsed_sub_backend_kwargs = self.normalize_backend_kwargs(
                resolved_sub_backend,
                parsed_sub_backend_kwargs,
            )
        else:
            parsed_sub_backend_kwargs = None

        self.task_repl_workdir = task_repl_workdir
        self.task_repl_workspace_root = task_repl_workspace_root
        self.task_repl_proxy_host = task_repl_proxy_host
        self.task_repl_proxy_bind_host = task_repl_proxy_bind_host
        self.python_auto_install = parsed_python_auto_install
        self.python_install_timeout_sec = parsed_python_install_timeout_sec

        self.rlm = RLM(
            backend=backend,
            backend_kwargs=parsed_backend_kwargs,
            environment="task",
            environment_kwargs={
                "workdir": task_repl_workdir,
                "workspace_root": task_repl_workspace_root,
                "proxy_host": task_repl_proxy_host,
                "proxy_bind_host": task_repl_proxy_bind_host,
            },
            subagent_max_iterations=parsed_subagent_max_iterations,
            subagent_backend=resolved_sub_backend,
            subagent_backend_kwargs=parsed_sub_backend_kwargs,
            max_depth=parsed_max_depth,
            max_iterations=parsed_max_iterations,
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
        try:
            await self.ensure_python_available(environment)
            resolved_proxy_host = await self.resolve_task_repl_proxy_host(environment)

            self.rlm.environment_kwargs.update(
                {
                    "runner": HarborExecRunner(environment),
                    "workdir": self.task_repl_workdir,
                    "workspace_root": self.task_repl_workspace_root,
                    "proxy_host": resolved_proxy_host,
                    "proxy_bind_host": self.task_repl_proxy_bind_host,
                }
            )

            terminal_output = await self.exec_capture(environment, INITIAL_SNAPSHOT_COMMAND)
            prompt = REAL_TASK_PROMPT.format(
                instruction=instruction,
                terminal_output=terminal_output,
            )

            self.write_text_artifact("rlm_prompt.txt", prompt)
            self.write_text_artifact("initial_terminal_output.txt", terminal_output)

            completion = self.rlm.completion(prompt, root_prompt=instruction)
        except (
            BudgetExceededError,
            CancellationError,
            ErrorThresholdExceededError,
            TimeoutExceededError,
            TokenLimitExceededError,
        ) as exc:
            self.record_failure(context, exc)
            raise
        except Exception as exc:
            self.record_failure(context, exc)
            raise

        usage = completion.usage_summary
        context.n_input_tokens = usage.total_input_tokens
        context.n_output_tokens = usage.total_output_tokens
        context.metadata = {
            "agent": "installed-rlm-agent",
            "execution_path": "task-backed-repl",
            "environment": "task",
            "root_model": completion.root_model,
            "execution_time_sec": completion.execution_time,
            "final_response_preview": completion.response[:1000],
        }

        self.write_text_artifact("rlm_final_response.txt", completion.response)
        if completion.metadata is not None:
            self.write_text_artifact(
                "rlm_trajectory.json",
                json.dumps(completion.metadata, ensure_ascii=False, indent=2),
            )

    async def resolve_task_repl_proxy_host(self, environment: BaseEnvironment) -> str:
        configured = self.task_repl_proxy_host.strip()
        candidates: list[str] = []

        if configured and configured.lower() != "auto":
            candidates.append(configured)

        for fallback in ["host.docker.internal", "gateway.docker.internal"]:
            if fallback not in candidates:
                candidates.append(fallback)

        gateway = await self.detect_task_gateway_ip(environment)
        if gateway and gateway not in candidates:
            candidates.append(gateway)

        probe_results: list[str] = []
        for candidate in candidates:
            if await self.task_can_resolve_host(environment, candidate):
                self.write_text_artifact(
                    "task_repl_proxy_resolution.txt",
                    "\n".join(
                        [
                            f"configured_host={self.task_repl_proxy_host}",
                            f"resolved_host={candidate}",
                            f"proxy_bind_host={self.task_repl_proxy_bind_host}",
                            "probe_results:",
                            *probe_results,
                        ]
                    ),
                )
                return candidate
            probe_results.append(f"{candidate}=unresolved")

        probe_summary = ", ".join(candidates) if candidates else "(none)"
        self.write_text_artifact(
            "task_repl_proxy_resolution.txt",
            "\n".join(
                [
                    f"configured_host={self.task_repl_proxy_host}",
                    f"resolved_host=(none)",
                    f"proxy_bind_host={self.task_repl_proxy_bind_host}",
                    "probe_results:",
                    *probe_results,
                ]
            ),
        )
        raise RuntimeError(
            "TaskBackedREPL could not find a proxy host reachable from the task runtime. "
            f"Tried: {probe_summary}. "
            "Set task_repl_proxy_host explicitly or ensure host.docker.internal resolves."
        )

    async def detect_task_gateway_ip(self, environment: BaseEnvironment) -> str | None:
        command = """python3 - <<'PY'
from pathlib import Path

route_path = Path('/proc/net/route')
if not route_path.exists():
    raise SystemExit(0)

for line in route_path.read_text().splitlines()[1:]:
    fields = line.split()
    if len(fields) < 3 or fields[1] != '00000000':
        continue
    raw = fields[2]
    octets = [str(int(raw[i : i + 2], 16)) for i in range(6, -2, -2)]
    print(".".join(octets))
    break
PY"""
        result = await self.exec_raw(environment, command, timeout_sec=30)
        if result.return_code != 0:
            return None
        gateway = (result.stdout or "").strip().splitlines()
        if not gateway:
            return None
        return gateway[-1].strip() or None

    async def task_can_resolve_host(self, environment: BaseEnvironment, host: str) -> bool:
        command = f"""python3 - <<'PY'
import socket

host = {host!r}
try:
    socket.getaddrinfo(host, None)
except OSError:
    raise SystemExit(1)
PY"""
        result = await self.exec_raw(environment, command, timeout_sec=30)
        return result.return_code == 0

    async def ensure_python_available(self, environment: BaseEnvironment) -> None:
        check = await self.exec_raw(environment, "python3 --version", timeout_sec=30)
        if check.return_code == 0:
            self.write_text_artifact(
                "dependency_check.txt",
                f"python3 already available: {(check.stdout or check.stderr).strip()}",
            )
            return

        if not self.python_auto_install:
            detail = (check.stderr or check.stdout).strip() or "python3 not found"
            raise RuntimeError(
                "TaskBackedREPL requires python3 in the task runtime. "
                f"Auto-install is disabled and check failed: {detail}"
            )

        manager = await self.detect_package_manager(environment)
        install_command = self.python_install_command(manager)
        if install_command is None:
            raise RuntimeError(
                "TaskBackedREPL requires python3, but no supported package manager "
                "was detected for automatic installation."
            )

        install_result = await self.exec_raw(
            environment,
            install_command,
            timeout_sec=self.python_install_timeout_sec,
        )
        if install_result.return_code != 0:
            detail = (install_result.stderr or install_result.stdout).strip()
            raise RuntimeError(
                "Failed to install python3 in task runtime. "
                f"Package manager={manager}. Error: {detail}"
            )

        verify = await self.exec_raw(environment, "python3 --version", timeout_sec=30)
        if verify.return_code != 0:
            detail = (verify.stderr or verify.stdout).strip() or "python3 still unavailable"
            raise RuntimeError(
                "python3 installation command completed, but verification failed: "
                f"{detail}"
            )

        self.write_text_artifact(
            "dependency_check.txt",
            (
                f"python3 auto-installed via {manager}\n"
                f"version: {(verify.stdout or verify.stderr).strip()}\n"
            ),
        )

    async def detect_package_manager(self, environment: BaseEnvironment) -> str | None:
        managers = [
            "apt-get",
            "dnf",
            "yum",
            "apk",
            "pacman",
            "zypper",
            "pkg",
            "brew",
        ]
        for manager in managers:
            result = await self.exec_raw(
                environment,
                f"command -v {manager} >/dev/null 2>&1",
                timeout_sec=20,
            )
            if result.return_code == 0:
                return manager
        return None

    def python_install_command(self, manager: str | None) -> str | None:
        commands: dict[str, str] = {
            "apt-get": (
                "DEBIAN_FRONTEND=noninteractive apt-get update && "
                "DEBIAN_FRONTEND=noninteractive apt-get install -y python3"
            ),
            "dnf": "dnf install -y python3",
            "yum": "yum install -y python3",
            "apk": "apk add --no-cache python3",
            "pacman": "pacman -S --noconfirm python",
            "zypper": "zypper install -y -n python3",
            "pkg": "ASSUME_ALWAYS_YES=yes pkg install -y python3",
            "brew": "brew install python",
        }
        if manager is None:
            return None
        return commands.get(manager)

    async def exec_raw(
        self,
        environment: BaseEnvironment,
        command: str,
        timeout_sec: int = 120,
    ) -> ExecResult:
        return await environment.exec(
            command=command,
            cwd=self.task_repl_workdir,
            timeout_sec=timeout_sec,
        )

    async def exec_capture(
        self,
        environment: BaseEnvironment,
        command: str,
        timeout_sec: int = 120,
    ) -> str:
        try:
            result = await self.exec_raw(environment, command, timeout_sec=timeout_sec)
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

    def record_failure(self, context: AgentContext, exc: Exception) -> None:
        partial_answer = getattr(exc, "partial_answer", None)
        context.metadata = {
            "agent": "installed-rlm-agent",
            "execution_path": "task-backed-repl",
            "environment": "task",
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "partial_answer_preview": partial_answer[:1000] if partial_answer else None,
        }
        self.write_text_artifact("rlm_error.txt", str(exc))
        if partial_answer:
            self.write_text_artifact("rlm_partial_answer.txt", partial_answer)

    def write_text_artifact(self, filename: str, content: str) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / filename).write_text(content)

    def parse_backend_kwargs(self, value: dict[str, Any] | str | None) -> dict[str, Any]:
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

    def inject_model_name_if_missing(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(kwargs)
        if self.model_name and "model_name" not in enriched:
            enriched["model_name"] = self.model_name
        return enriched

    def normalize_backend_kwargs(self, backend: str, kwargs: dict[str, Any]) -> dict[str, Any]:
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

    def parse_bool(self, value: bool | str) -> bool:
        if isinstance(value, bool):
            return value
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"Cannot parse boolean value: {value}")
