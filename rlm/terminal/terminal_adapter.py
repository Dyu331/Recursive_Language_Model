from __future__ import annotations

import shlex
import time
import uuid
from dataclasses import dataclass
from typing import Protocol


@dataclass
class CommandResult:
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0


class ShellRunner(Protocol):
    """Minimal shell execution interface for terminal adapters."""

    def run(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> CommandResult: ...


class TerminalAdapter(Protocol):
    """Terminus-2 style terminal bridge surface."""

    def send_keys(self, keystrokes: str, duration: float = 0.0) -> None: ...

    def get_incremental_output(self) -> str: ...

    def capture_screen(self) -> str: ...

    def is_alive(self) -> bool: ...


@dataclass
class TerminalAction:
    keystrokes: str
    duration_sec: float = 0.5


class TmuxTerminalAdapter:
    """
    Terminus-2 compatible adapter using tmux as a hard dependency.

    Assumption: tmux is provided by the external runtime stack. If unavailable,
    this adapter fails fast and does not provide no-tmux fallback.
    """

    special_keys: frozenset[str] = frozenset(
        {
            "Enter",
            "Tab",
            "Space",
            "BSpace",
            "Escape",
            "Up",
            "Down",
            "Left",
            "Right",
            "Home",
            "End",
            "PageUp",
            "PageDown",
            "C-c",
            "C-d",
            "C-z",
            "C-l",
            "C-m",
            "C-j",
            "^M",
            "^J",
        }
    )

    def __init__(
        self,
        runner: ShellRunner,
        workdir: str = "/app",
        session_name: str | None = None,
        shell_command: str = "bash --login",
        pane_width: int = 160,
        pane_height: int = 40,
        history_limit: int = 50_000,
        startup_wait_sec: float = 0.2,
        auto_install_tmux: bool = True,
        tmux_install_timeout_sec: int = 240,
    ):
        self.runner = runner
        self.workdir = workdir
        self.session_name = session_name or f"rlm_terminal_{uuid.uuid4().hex[:8]}"
        self.shell_command = shell_command
        self.pane_width = pane_width
        self.pane_height = pane_height
        self.history_limit = history_limit
        self.startup_wait_sec = startup_wait_sec
        self.auto_install_tmux = auto_install_tmux
        self.tmux_install_timeout_sec = tmux_install_timeout_sec
        self.previous_buffer: str | None = None
        self.tmux_binary = "tmux"

        self._ensure_tmux_available()
        self._start_session()

    def _ensure_tmux_available(self) -> None:
        binary, detail = self._resolve_tmux_binary()
        if binary is not None:
            self.tmux_binary = binary
            return

        if not self.auto_install_tmux:
            raise RuntimeError(
                "TmuxTerminalAdapter requires tmux in the terminal runtime stack. "
                f"Fast fail because tmux is unavailable: {detail}"
            )

        self._attempt_tmux_installation()
        binary, detail = self._resolve_tmux_binary()
        if binary is not None:
            self.tmux_binary = binary
            return

        raise RuntimeError(
            "TmuxTerminalAdapter requires tmux in the terminal runtime stack. "
            "Attempted automatic installation, but tmux is still unavailable: "
            f"{detail}"
        )

    def _resolve_tmux_binary(self) -> tuple[str | None, str]:
        candidates = [
            ("tmux", "tmux -V"),
            ("/usr/local/bin/tmux", "/usr/local/bin/tmux -V"),
        ]
        details: list[str] = []
        for binary, check_command in candidates:
            result = self.runner.run(
                check_command,
                cwd=self.workdir,
                timeout_sec=30,
            )
            if result.return_code == 0:
                return binary, (result.stdout or result.stderr or "").strip()

            output = (result.stderr or result.stdout).strip()
            if output:
                details.append(output)

        detail = "; ".join(details) if details else "tmux command not found"
        return None, detail

    def _attempt_tmux_installation(self) -> None:
        manager = self._detect_package_manager()
        install_commands = self._get_tmux_install_commands(manager)

        last_detail = "unable to detect a supported package manager"
        for command in install_commands:
            result = self.runner.run(
                command,
                cwd=self.workdir,
                timeout_sec=self.tmux_install_timeout_sec,
            )
            if result.return_code == 0:
                return

            detail = (result.stderr or result.stdout).strip()
            if detail:
                last_detail = detail

        raise RuntimeError(
            "Failed to automatically install tmux. "
            f"Last installation error: {last_detail}"
        )

    def _detect_package_manager(self) -> str | None:
        candidates = [
            "apt-get",
            "dnf",
            "yum",
            "apk",
            "pacman",
            "brew",
            "pkg",
            "zypper",
        ]
        for manager in candidates:
            result = self.runner.run(
                f"command -v {manager} >/dev/null 2>&1",
                cwd=self.workdir,
                timeout_sec=20,
            )
            if result.return_code == 0:
                return manager
        return None

    def _get_tmux_install_commands(self, manager: str | None) -> list[str]:
        install_by_manager = {
            "apt-get": (
                "DEBIAN_FRONTEND=noninteractive apt-get update && "
                "DEBIAN_FRONTEND=noninteractive apt-get install -y tmux"
            ),
            "dnf": "dnf install -y tmux",
            "yum": "yum install -y tmux",
            "apk": "apk add --no-cache tmux",
            "pacman": "pacman -S --noconfirm tmux",
            "brew": "brew install tmux",
            "pkg": "ASSUME_ALWAYS_YES=yes pkg install -y tmux",
            "zypper": "zypper install -y -n tmux",
        }
        if manager is None:
            return []
        command = install_by_manager.get(manager)
        return [command] if command is not None else []

    def _start_session(self) -> None:
        if self.is_alive():
            return

        launch_shell = f"cd {shlex.quote(self.workdir)} && {self.shell_command}"
        create_command = (
            f"tmux has-session -t {shlex.quote(self.session_name)} 2>/dev/null || "
            f"tmux new-session -x {self.pane_width} -y {self.pane_height} -d "
            f"-s {shlex.quote(self.session_name)} {shlex.quote(launch_shell)}"
        )
        create_result = self.runner.run(create_command, cwd=self.workdir, timeout_sec=30)
        if create_result.return_code != 0:
            detail = (create_result.stderr or create_result.stdout).strip()
            raise RuntimeError(f"Failed to start tmux session '{self.session_name}': {detail}")

        self._run_tmux(
            ["set-option", "-t", self.session_name, "history-limit", str(self.history_limit)],
            check=False,
            timeout_sec=30,
        )

        if self.startup_wait_sec > 0:
            time.sleep(self.startup_wait_sec)

    def close(self) -> None:
        self._run_tmux(["kill-session", "-t", self.session_name], check=False, timeout_sec=10)

    def send_keys(self, keystrokes: str, duration: float = 0.0) -> None:
        if not self.is_alive():
            raise RuntimeError("tmux session is not alive")

        if keystrokes == "":
            if duration > 0:
                time.sleep(duration)
            return

        if keystrokes in self.special_keys:
            self._run_tmux(["send-keys", "-t", self.session_name, keystrokes])
        else:
            self._send_literal_keystrokes(keystrokes)

        if duration > 0:
            time.sleep(duration)

    def _send_literal_keystrokes(self, keystrokes: str) -> None:
        chunks = keystrokes.splitlines(keepends=True)
        if not chunks:
            chunks = [keystrokes]

        for chunk in chunks:
            ends_with_newline = chunk.endswith("\n") or chunk.endswith("\r")
            literal = chunk.rstrip("\r\n")

            if literal:
                self._run_tmux(["send-keys", "-t", self.session_name, "-l", literal])

            if ends_with_newline:
                self._run_tmux(["send-keys", "-t", self.session_name, "Enter"])

    def get_incremental_output(self) -> str:
        current_buffer = self._capture_pane(capture_entire=True)

        if self.previous_buffer is None:
            self.previous_buffer = current_buffer
            return f"Current Terminal Screen:\n{self.capture_screen()}"

        new_content = self._find_new_content(current_buffer)
        self.previous_buffer = current_buffer

        if new_content is not None and new_content.strip():
            return f"New Terminal Output:\n{new_content}"

        return f"Current Terminal Screen:\n{self.capture_screen()}"

    def _find_new_content(self, current_buffer: str) -> str | None:
        if self.previous_buffer is None:
            return None

        previous = self.previous_buffer.strip()
        if previous == "":
            return current_buffer

        if previous in current_buffer:
            idx = current_buffer.index(previous) + len(previous)
            return current_buffer[idx:]

        return None

    def capture_screen(self) -> str:
        return self._capture_pane(capture_entire=False)

    def _capture_pane(self, capture_entire: bool = False) -> str:
        args = ["capture-pane", "-p", "-t", self.session_name]
        if capture_entire:
            args = ["capture-pane", "-p", "-S", "-", "-t", self.session_name]

        result = self._run_tmux(args, check=False)
        if result.return_code != 0:
            return ""

        return result.stdout

    def is_alive(self) -> bool:
        result = self._run_tmux(
            ["has-session", "-t", self.session_name],
            check=False,
            timeout_sec=10,
        )
        return result.return_code == 0

    def _run_tmux(
        self,
        args: list[str],
        check: bool = True,
        timeout_sec: int | None = None,
    ) -> CommandResult:
        command = " ".join(shlex.quote(part) for part in [self.tmux_binary, *args])
        result = self.runner.run(
            command=command,
            cwd=self.workdir,
            timeout_sec=timeout_sec,
        )

        if check and result.return_code != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"tmux command failed: {command}\n{detail}")

        return result
