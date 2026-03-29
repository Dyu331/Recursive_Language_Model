from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import queue
import shlex
import textwrap
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Protocol

from rlm.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from rlm.core.types import REPLResult, RLMChatCompletion
from rlm.environments.base_env import NonIsolatedEnv


@dataclass
class ShellCommandResult:
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0


class TaskShellRunner(Protocol):
    def run(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ) -> ShellCommandResult: ...


class LLMProxyHandler(BaseHTTPRequestHandler):
    lm_handler_address: tuple[str, int] | None = None
    pending_calls: list[RLMChatCompletion] = []
    lock: threading.Lock = threading.Lock()
    depth: int = 1

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))

        if self.path == "/llm_query":
            result = self._handle_single(body)
        elif self.path == "/llm_query_batched":
            result = self._handle_batched(body)
        else:
            self._respond(404, {"error": "Not found"})
            return

        self._respond(200, result)

    def _respond(self, status: int, data: dict[str, Any]):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _handle_single(self, body: dict[str, Any]) -> dict[str, Any]:
        if not self.lm_handler_address:
            return {"error": "No LM handler configured"}

        request = LMRequest(
            prompt=body.get("prompt"),
            model=body.get("model"),
            depth=self.depth,
        )
        response = send_lm_request(self.lm_handler_address, request)

        if not response.success:
            return {"error": response.error}

        with self.lock:
            self.pending_calls.append(response.chat_completion)

        return {"response": response.chat_completion.response}

    def _handle_batched(self, body: dict[str, Any]) -> dict[str, Any]:
        if not self.lm_handler_address:
            return {"error": "No LM handler configured"}

        prompts = body.get("prompts", [])
        responses = send_lm_request_batched(
            self.lm_handler_address,
            prompts,
            model=body.get("model"),
            depth=self.depth,
        )

        results = []
        for response in responses:
            if not response.success:
                results.append(f"Error: {response.error}")
            else:
                with self.lock:
                    self.pending_calls.append(response.chat_completion)
                results.append(response.chat_completion.response)

        return {"responses": results}


def _build_exec_script(
    code: str,
    workspace_dir: str,
    proxy_host: str,
    proxy_port: int,
    depth: int = 1,
) -> str:
    code_b64 = base64.b64encode(code.encode()).decode()

    return textwrap.dedent(
        f"""
import base64
import io
import json
import os
import sys
import traceback
import urllib.error
import urllib.request

try:
    import dill as serializer
except ImportError:
    import pickle as serializer

WORKSPACE = {workspace_dir!r}
STATE = os.path.join(WORKSPACE, "state.pkl")
PROXY = "http://{proxy_host}:{proxy_port}"
CODE_B64 = {code_b64!r}

os.makedirs(WORKSPACE, exist_ok=True)

def _post_json(path, payload):
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{{PROXY}}{{path}}",
        data=data,
        headers={{"Content-Type": "application/json"}},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode())

def llm_query(prompt, model=None):
    try:
        data = _post_json("/llm_query", {{"prompt": prompt, "model": model, "depth": {depth}}})
        return data.get("response") or f"Error: {{data.get('error')}}"
    except Exception as exc:
        return f"Error: {{exc}}"

def llm_query_batched(prompts, model=None):
    try:
        data = _post_json(
            "/llm_query_batched",
            {{"prompts": prompts, "model": model, "depth": {depth}}},
        )
        responses = data.get("responses")
        if responses is None:
            return [f"Error: {{data.get('error')}}"] * len(prompts)
        return responses
    except Exception as exc:
        return [f"Error: {{exc}}"] * len(prompts)

def load_state():
    if os.path.exists(STATE):
        try:
            with open(STATE, "rb") as handle:
                return serializer.load(handle)
        except Exception:
            pass
    return {{}}

def save_state(state):
    clean = {{}}
    for key, value in state.items():
        if key.startswith("_"):
            continue
        try:
            serializer.dumps(value)
        except Exception:
            continue
        clean[key] = value

    with open(STATE, "wb") as handle:
        serializer.dump(clean, handle)

_locals = load_state()
_final_answer = None

def FINAL_VAR(name):
    global _final_answer
    key = str(name).strip().strip("\\"'")
    if key in _locals:
        _final_answer = str(_locals[key])
        return _final_answer
    available = [k for k in _locals.keys() if not k.startswith("_")]
    if available:
        return (
            f"Error: Variable '{{key}}' not found. Available variables: {{available}}. "
            "You must create and assign a variable BEFORE calling FINAL_VAR on it."
        )
    return (
        f"Error: Variable '{{key}}' not found. No variables have been created yet. "
        "You must create and assign a variable in a REPL block BEFORE calling FINAL_VAR on it."
    )

def SHOW_VARS():
    available = {{k: type(v).__name__ for k, v in _locals.items() if not k.startswith("_")}}
    if not available:
        return "No variables created yet. Use ```repl``` blocks to create variables."
    return f"Available variables: {{available}}"

_globals = {{
    "__builtins__": __builtins__,
    "__name__": "__main__",
    "llm_query": llm_query,
    "llm_query_batched": llm_query_batched,
    "FINAL_VAR": FINAL_VAR,
    "SHOW_VARS": SHOW_VARS,
}}

stdout_buf = io.StringIO()
stderr_buf = io.StringIO()
old_stdout, old_stderr = sys.stdout, sys.stderr

try:
    sys.stdout, sys.stderr = stdout_buf, stderr_buf
    combined = {{**_globals, **_locals}}
    exec(base64.b64decode(CODE_B64).decode(), combined, combined)
    for key, value in combined.items():
        if key not in _globals and not key.startswith("_"):
            _locals[key] = value
except Exception:
    traceback.print_exc(file=stderr_buf)
finally:
    sys.stdout, sys.stderr = old_stdout, old_stderr

if "context_0" in _locals:
    _locals["context"] = _locals["context_0"]
if "history_0" in _locals:
    _locals["history"] = _locals["history_0"]

save_state(_locals)

print(
    json.dumps(
        {{
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "locals": {{k: repr(v) for k, v in _locals.items() if not k.startswith("_")}},
            "final_answer": _final_answer,
        }},
        ensure_ascii=False,
    )
)
"""
    )


def _build_python_command(script: str) -> str:
    return "python3 - <<'__RLM_TASK_REPL_PY__'\n" + script + "\n__RLM_TASK_REPL_PY__"


class HarborExecRunner:
    def __init__(self, environment: Any):
        self._environment = environment

    def run(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ) -> ShellCommandResult:
        timeout = None if timeout_sec is None else max(1, math.ceil(timeout_sec))
        result = self._run_coroutine(
            self._environment.exec(
                command=command,
                cwd=cwd,
                env=env,
                timeout_sec=timeout,
            )
        )
        return ShellCommandResult(
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            return_code=result.return_code,
        )

    @staticmethod
    def _run_coroutine(coroutine: Any) -> Any:
        result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def _worker():
            try:
                result_queue.put((True, asyncio.run(coroutine)))
            except BaseException as exc:  # noqa: BLE001
                result_queue.put((False, exc))

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        success, payload = result_queue.get()
        thread.join()
        if success:
            return payload
        raise payload


class TmuxShellRunner:
    def __init__(self, session: Any, capture_entire: bool = True):
        self._session = session
        self._capture_entire = capture_entire

    def run(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ) -> ShellCommandResult:
        marker = uuid.uuid4().hex
        wrapped = self._build_wrapped_command(
            command=command,
            cwd=cwd,
            env=env,
            marker=marker,
        )
        timeout = 300.0 if timeout_sec is None else float(timeout_sec)
        self._session.send_keys([wrapped, "Enter"], block=True, max_timeout_sec=timeout)
        pane = self._session.capture_pane(capture_entire=self._capture_entire)
        return self._extract_result(pane, marker)

    @staticmethod
    def _build_wrapped_command(
        command: str,
        cwd: str | None,
        env: dict[str, str] | None,
        marker: str,
    ) -> str:
        env_prefix = ""
        if env:
            env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " "

        cwd_prefix = f"cd {shlex.quote(cwd)} && " if cwd else ""
        escaped_command = command

        return textwrap.dedent(
            f"""
            __rlm_out=$(mktemp /tmp/rlm_task_stdout.{marker}.XXXXXX)
            __rlm_err=$(mktemp /tmp/rlm_task_stderr.{marker}.XXXXXX)
            (
              {cwd_prefix}{env_prefix}{escaped_command}
            ) >"$__rlm_out" 2>"$__rlm_err"
            __rlm_rc=$?
            printf '__RLM_TASK_RESULT_BEGIN__{marker}\\n'
            RLM_OUT="$__rlm_out" RLM_ERR="$__rlm_err" RLM_RC="$__rlm_rc" python3 - <<'__RLM_TASK_RESULT_PY__'
import base64
import json
import os
from pathlib import Path

payload = {{
    "stdout": Path(os.environ["RLM_OUT"]).read_text(encoding="utf-8", errors="replace"),
    "stderr": Path(os.environ["RLM_ERR"]).read_text(encoding="utf-8", errors="replace"),
    "return_code": int(os.environ["RLM_RC"]),
}}
print(base64.b64encode(json.dumps(payload).encode()).decode(), end="")
__RLM_TASK_RESULT_PY__
            printf '\\n__RLM_TASK_RESULT_END__{marker}\\n'
            rm -f "$__rlm_out" "$__rlm_err"
            """
        ).strip()

    @staticmethod
    def _extract_result(pane: str, marker: str) -> ShellCommandResult:
        start = f"__RLM_TASK_RESULT_BEGIN__{marker}"
        end = f"__RLM_TASK_RESULT_END__{marker}"

        start_index = pane.rfind(start)
        end_index = pane.rfind(end)
        if start_index == -1 or end_index == -1 or end_index <= start_index:
            return ShellCommandResult(
                stdout="",
                stderr="Failed to capture tmux command result payload.",
                return_code=1,
            )

        payload = pane[start_index + len(start) : end_index]
        b64 = "".join(payload.split())
        try:
            data = json.loads(base64.b64decode(b64).decode())
        except Exception as exc:  # noqa: BLE001
            return ShellCommandResult(
                stdout="",
                stderr=f"Failed to decode tmux command result payload: {exc}",
                return_code=1,
            )

        return ShellCommandResult(
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            return_code=int(data.get("return_code", 1)),
        )


class TaskBackedREPL(NonIsolatedEnv):
    def __init__(
        self,
        runner: TaskShellRunner,
        lm_handler_address: tuple[str, int] | None = None,
        context_payload: dict | list | str | None = None,
        setup_code: str | None = None,
        persistent: bool = False,
        depth: int = 1,
        workdir: str = "/app",
        workspace_root: str = "/tmp/rlm_task_repl",
        proxy_host: str = "host.docker.internal",
        **kwargs,
    ):
        if persistent:
            raise NotImplementedError(
                "Persistent REPLs are currently not supported for environment: TaskBackedREPL"
            )
        super().__init__(persistent=persistent, depth=depth, **kwargs)

        self.runner = runner
        self.lm_handler_address = lm_handler_address
        self.workdir = workdir
        self.workspace_root = workspace_root.rstrip("/") or "/tmp/rlm_task_repl"
        self.remote_workspace = f"{self.workspace_root}/{uuid.uuid4().hex}"
        self.proxy_host = proxy_host

        self.proxy_server: HTTPServer | None = None
        self.proxy_thread: threading.Thread | None = None
        self.proxy_port = 0
        self.pending_calls: list[RLMChatCompletion] = []
        self._calls_lock = threading.Lock()
        self._locals: dict[str, str] = {}

        self.setup()

        if context_payload is not None:
            self.load_context(context_payload)
        if setup_code:
            self.execute_code(setup_code)

    def setup(self):
        handler = type(
            "Handler",
            (LLMProxyHandler,),
            {
                "lm_handler_address": self.lm_handler_address,
                "pending_calls": self.pending_calls,
                "lock": self._calls_lock,
                "depth": self.depth,
            },
        )
        self.proxy_server = HTTPServer(("127.0.0.1", 0), handler)
        self.proxy_port = self.proxy_server.server_address[1]
        self.proxy_thread = threading.Thread(target=self.proxy_server.serve_forever, daemon=True)
        self.proxy_thread.start()

        self.runner.run(
            f"mkdir -p {shlex.quote(self.remote_workspace)}",
            cwd=self.workdir,
            timeout_sec=30,
        )

    def _write_remote_file(self, path: str, content: str) -> None:
        content_b64 = base64.b64encode(content.encode()).decode()
        script = textwrap.dedent(
            f"""
            import base64
            from pathlib import Path

            path = Path({path!r})
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(base64.b64decode({content_b64!r}))
            """
        )
        result = self.runner.run(
            _build_python_command(script),
            cwd=self.workdir,
            timeout_sec=30,
        )
        if result.return_code != 0:
            raise RuntimeError(result.stderr or "Failed to write remote file")

    def load_context(self, context_payload: dict | list | str):
        if isinstance(context_payload, str):
            context_path = f"{self.remote_workspace}/context_0.txt"
            self._write_remote_file(context_path, context_payload)
            load_code = textwrap.dedent(
                f"""
                with open({context_path!r}, "r", encoding="utf-8", errors="replace") as handle:
                    context_0 = handle.read()
                context = context_0
                history_0 = []
                history = history_0
                """
            )
        else:
            context_path = f"{self.remote_workspace}/context_0.json"
            self._write_remote_file(context_path, json.dumps(context_payload))
            load_code = textwrap.dedent(
                f"""
                import json

                with open({context_path!r}, "r", encoding="utf-8") as handle:
                    context_0 = json.load(handle)
                context = context_0
                history_0 = []
                history = history_0
                """
            )

        self.execute_code(load_code)

    def execute_code(self, code: str) -> REPLResult:
        start = time.perf_counter()

        with self._calls_lock:
            self.pending_calls.clear()

        script = _build_exec_script(
            code=code,
            workspace_dir=self.remote_workspace,
            proxy_host=self.proxy_host,
            proxy_port=self.proxy_port,
            depth=self.depth,
        )
        result = self.runner.run(
            _build_python_command(script),
            cwd=self.workdir,
            timeout_sec=300,
        )

        with self._calls_lock:
            calls = self.pending_calls.copy()
            self.pending_calls.clear()

        if result.return_code != 0:
            return REPLResult(
                stdout=result.stdout,
                stderr=result.stderr,
                locals=self._locals.copy(),
                execution_time=time.perf_counter() - start,
                rlm_calls=calls,
            )

        try:
            data = json.loads(result.stdout.strip().splitlines()[-1]) if result.stdout.strip() else {}
            self._locals = data.get("locals", {})
            return REPLResult(
                stdout=data.get("stdout", ""),
                stderr=data.get("stderr", ""),
                locals=self._locals.copy(),
                execution_time=time.perf_counter() - start,
                rlm_calls=calls,
                final_answer=data.get("final_answer"),
            )
        except json.JSONDecodeError as exc:
            stderr = result.stderr
            if stderr:
                stderr += "\n"
            stderr += f"Failed to parse task-backed REPL payload: {exc}"
            return REPLResult(
                stdout=result.stdout,
                stderr=stderr,
                locals=self._locals.copy(),
                execution_time=time.perf_counter() - start,
                rlm_calls=calls,
            )

    def cleanup(self):
        try:
            self.runner.run(
                f"rm -rf {shlex.quote(self.remote_workspace)}",
                cwd=self.workdir,
                timeout_sec=30,
            )
        except Exception:
            pass

        if self.proxy_server is not None:
            self.proxy_server.shutdown()
            self.proxy_server = None
        if self.proxy_thread is not None:
            self.proxy_thread.join(timeout=1)
            self.proxy_thread = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def __del__(self):
        self.cleanup()
