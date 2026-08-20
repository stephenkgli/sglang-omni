# SPDX-License-Identifier: Apache-2.0
"""Lifecycle manager for one private per-GPU CUDA MPS daemon.

The manager owns the daemon for exactly one colocated GPU: private pipe/log
directories under a per-run state dir, attach verification against the control
daemon's client list, and fail-closed teardown. All ``nvidia-cuda-mps-control``
I/O goes through the :class:`MpsControlClient` protocol so the state machine is
fully testable without a GPU.
"""

from __future__ import annotations

import enum
import json
import logging
import shutil
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sglang_omni.mps.state import MpsRunPaths, validate_control_socket

logger = logging.getLogger(__name__)


class MpsError(RuntimeError):
    """Raised when the MPS lifecycle cannot proceed; state dir is preserved."""


class MpsState(enum.Enum):
    IDLE = "idle"
    PREFLIGHT = "preflight"
    STARTING = "starting"
    READY = "ready"
    VERIFYING = "verifying"
    SERVING = "serving"
    DRAINING = "draining"
    STOPPING = "stopping"
    CLEANED = "cleaned"
    FAILED = "failed"


class MpsControlClient(Protocol):
    """Process/daemon I/O surface; the real client wraps subprocess calls."""

    def start_daemon(self, pipe_dir: Path, log_dir: Path, gpu_uuid: str) -> int: ...

    def control_responds(self, pipe_dir: Path) -> bool: ...

    def get_server_list(self, pipe_dir: Path) -> list[int]: ...

    def get_client_list(self, pipe_dir: Path, server_pid: int) -> list[int]: ...

    def quit_daemon(self, pipe_dir: Path) -> None: ...

    def pid_alive(self, pid: int) -> bool: ...

    def daemon_owns_pipe(self, pid: int, pipe_dir: Path) -> bool: ...


@dataclass
class MpsManager:
    paths: MpsRunPaths
    gpu_uuid: str
    client: MpsControlClient
    poll_interval: float = 0.2
    start_timeout: float = 5.0
    verify_timeout: float = 30.0
    drain_timeout: float = 60.0
    stop_timeout: float = 10.0

    def __post_init__(self) -> None:
        self.state = MpsState.IDLE
        self.daemon_pid: int | None = None

    def preflight(self) -> None:
        self.state = MpsState.PREFLIGHT
        validate_control_socket(self.paths.control_socket)
        self._recover_stale_runs()

    def _recover_stale_runs(self) -> None:
        root = self.paths.state_root
        if not root.exists():
            return
        for run_dir in sorted(root.glob("gpu-*/run-*")):
            if run_dir == self.paths.state_dir:
                continue
            self._recover_one_stale_run(run_dir)

    def _recover_one_stale_run(self, run_dir: Path) -> None:
        pipe_dir = run_dir / "mps" / "pipe"
        manifest_path = run_dir / "manifest"
        daemon_pid: int | None = None
        try:
            daemon_pid = int(json.loads(manifest_path.read_text())["daemon_pid"])
        except (OSError, ValueError, KeyError, TypeError):
            # Note (Jiaxin Deng): no readable ownership record. Only reclaim
            # when nothing answers on the stale control socket; a live unknown
            # daemon is not ours to kill.
            if self.client.control_responds(pipe_dir):
                raise MpsError(
                    f"stale MPS state dir {run_dir} has no readable manifest "
                    "but its control socket still answers; refusing to touch "
                    "it. Clean it up manually."
                )
            shutil.rmtree(run_dir)
            return

        owned = self.client.pid_alive(daemon_pid) and self.client.daemon_owns_pipe(
            daemon_pid, pipe_dir
        )
        if not owned:
            # Dead daemon, or the PID was recycled by an unrelated process.
            shutil.rmtree(run_dir)
            return

        clients: set[int] = set()
        for server_pid in self.client.get_server_list(pipe_dir):
            clients.update(self.client.get_client_list(pipe_dir, server_pid))
        if clients:
            raise MpsError(
                f"MPS daemon pid {daemon_pid} from stale run dir {run_dir} is "
                f"alive with attached clients {sorted(clients)}; it may belong "
                "to a concurrent pipeline. Refusing to reclaim it."
            )

        logger.warning(
            "Reclaiming idle orphan MPS daemon pid %d from stale run dir %s",
            daemon_pid,
            run_dir,
        )
        self.client.quit_daemon(pipe_dir)
        self._wait_for(
            lambda: not self.client.pid_alive(daemon_pid),
            self.stop_timeout,
            f"orphan MPS daemon pid {daemon_pid} did not exit after quit",
        )
        shutil.rmtree(run_dir)

    def start(self) -> None:
        self.state = MpsState.STARTING
        self.paths.pipe_dir.mkdir(parents=True, exist_ok=True)
        self.paths.log_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.daemon_pid = self.client.start_daemon(
                self.paths.pipe_dir, self.paths.log_dir, self.gpu_uuid
            )
            self._wait_for(
                lambda: self.client.control_responds(self.paths.pipe_dir),
                self.start_timeout,
                "MPS control daemon did not answer on its control socket",
            )
        except MpsError:
            self._fail()
            raise
        except Exception as exc:
            self._fail()
            raise MpsError(f"failed to start MPS daemon: {exc}") from exc
        self._write_manifest()
        self.state = MpsState.READY

    def env_for_stage(self) -> dict[str, str]:
        return {
            "CUDA_MPS_PIPE_DIRECTORY": str(self.paths.pipe_dir),
            "CUDA_MPS_LOG_DIRECTORY": str(self.paths.log_dir),
            "CUDA_VISIBLE_DEVICES": self.gpu_uuid,
        }

    def verify_attached(self, expected_pids: Iterable[int]) -> None:
        self.state = MpsState.VERIFYING
        expected = set(expected_pids)
        missing = expected

        def attached() -> bool:
            nonlocal missing
            missing = expected - self._attached_client_pids()
            return not missing

        try:
            self._wait_for(
                attached,
                self.verify_timeout,
                # Note (Jiaxin Deng): a process that missed the pipe dir falls
                # back to time-slicing without any error, so absence here must
                # fail startup rather than degrade silently.
                lambda: (
                    f"stage process(es) {sorted(missing)} never attached to the "
                    f"MPS server (pipe dir {self.paths.pipe_dir})"
                ),
            )
        except MpsError:
            self._fail()
            raise
        self.state = MpsState.SERVING

    def probe(self) -> bool:
        """Watchdog check: daemon process alive and control socket answering."""
        if self.daemon_pid is None or not self.client.pid_alive(self.daemon_pid):
            return False
        return self.client.control_responds(self.paths.pipe_dir)

    def stop(self) -> None:
        self.state = MpsState.DRAINING
        try:
            self._wait_for(
                lambda: not self._attached_client_pids(),
                self.drain_timeout,
                "MPS clients still attached; refusing to quit the daemon under them",
            )
            self.state = MpsState.STOPPING
            self.client.quit_daemon(self.paths.pipe_dir)
            self._wait_for(
                lambda: self.daemon_pid is None
                or not self.client.pid_alive(self.daemon_pid),
                self.stop_timeout,
                "MPS daemon did not exit after quit",
            )
        except MpsError:
            self._fail()
            raise
        shutil.rmtree(self.paths.state_dir)
        self.state = MpsState.CLEANED

    def _attached_client_pids(self) -> set[int]:
        pids: set[int] = set()
        for server_pid in self.client.get_server_list(self.paths.pipe_dir):
            pids.update(self.client.get_client_list(self.paths.pipe_dir, server_pid))
        return pids

    def _write_manifest(self) -> None:
        manifest = {
            "run_id": self.paths.run_id,
            "gpu_id": self.paths.gpu_id,
            "gpu_uuid": self.gpu_uuid,
            "daemon_pid": self.daemon_pid,
            "pipe_dir": str(self.paths.pipe_dir),
        }
        self.paths.manifest.write_text(json.dumps(manifest))

    def _wait_for(self, predicate, timeout: float, message) -> None:
        deadline = time.monotonic() + timeout
        while True:
            if predicate():
                return
            if time.monotonic() >= deadline:
                text = message() if callable(message) else message
                raise MpsError(
                    f"{text}. State dir preserved for inspection: "
                    f"{self.paths.state_dir}"
                )
            time.sleep(self.poll_interval)

    def _fail(self) -> None:
        self.state = MpsState.FAILED
