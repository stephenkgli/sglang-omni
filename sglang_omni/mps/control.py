# SPDX-License-Identifier: Apache-2.0
"""Subprocess-backed MpsControlClient talking to nvidia-cuda-mps-control.

Pure I/O; all lifecycle decisions live in :class:`sglang_omni.mps.manager.
MpsManager`. Covered by the GPU CI smoke rather than unit tests.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

_CONTROL_BINARY = "nvidia-cuda-mps-control"
_QUERY_TIMEOUT_SECONDS = 10
_PID_LINE = re.compile(r"^\d+$")


class SubprocessMpsControlClient:
    def _control_env(self, pipe_dir: Path) -> dict[str, str]:
        env = os.environ.copy()
        env["CUDA_MPS_PIPE_DIRECTORY"] = str(pipe_dir)
        return env

    def _query(self, pipe_dir: Path, command: str) -> str:
        result = subprocess.run(
            [_CONTROL_BINARY],
            input=command + "\n",
            capture_output=True,
            text=True,
            timeout=_QUERY_TIMEOUT_SECONDS,
            env=self._control_env(pipe_dir),
        )
        if result.returncode != 0:
            raise OSError(
                f"{_CONTROL_BINARY} {command!r} failed "
                f"(rc={result.returncode}): {result.stderr.strip()}"
            )
        return result.stdout

    def start_daemon(self, pipe_dir: Path, log_dir: Path, gpu_uuid: str) -> int:
        env = self._control_env(pipe_dir)
        env["CUDA_MPS_LOG_DIRECTORY"] = str(log_dir)
        # Note (Jiaxin Deng): scope the daemon by GPU UUID, not ordinal.
        # A daemon started with ordinal visibility remaps client-side
        # ordinals; UUIDs are immune (same contract as examples/mps_dp).
        env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
        subprocess.run(
            [_CONTROL_BINARY, "-d"],
            check=True,
            capture_output=True,
            timeout=_QUERY_TIMEOUT_SECONDS,
            env=env,
        )
        pid_file = pipe_dir / f"{_CONTROL_BINARY}.pid"
        return int(pid_file.read_text().strip())

    def control_responds(self, pipe_dir: Path) -> bool:
        try:
            self._query(pipe_dir, "get_default_active_thread_percentage")
            return True
        except (OSError, subprocess.SubprocessError, ValueError):
            return False

    def _pid_lines(self, output: str) -> list[int]:
        return [int(line) for line in output.split() if _PID_LINE.match(line)]

    def get_server_list(self, pipe_dir: Path) -> list[int]:
        return self._pid_lines(self._query(pipe_dir, "get_server_list"))

    def get_client_list(self, pipe_dir: Path, server_pid: int) -> list[int]:
        return self._pid_lines(self._query(pipe_dir, f"get_client_list {server_pid}"))

    def quit_daemon(self, pipe_dir: Path) -> None:
        self._query(pipe_dir, "quit")

    def pid_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def daemon_owns_pipe(self, pid: int, pipe_dir: Path) -> bool:
        try:
            environ = Path(f"/proc/{pid}/environ").read_bytes()
        except OSError:
            return False
        needle = f"CUDA_MPS_PIPE_DIRECTORY={pipe_dir}".encode()
        return needle in environ.split(b"\0")
