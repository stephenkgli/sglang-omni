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
import os
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

    def kill_pid(self, pid: int, force: bool = False) -> None: ...

    def parent_of(self, pid: int) -> int | None: ...

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
        owner_pid: int | None = None
        try:
            manifest = json.loads(manifest_path.read_text())
            owner_pid = int(manifest["owner_pid"])
            daemon_raw = manifest.get("daemon_pid")
            daemon_pid = int(daemon_raw) if daemon_raw is not None else None
        except (OSError, ValueError, KeyError, TypeError):
            # Note (Jiaxin Deng): no ownership record; a live unknown daemon is
            # not ours to kill, so reclaim only if its control socket is silent.
            if self.client.control_responds(pipe_dir):
                raise MpsError(
                    f"stale MPS state dir {run_dir} has no readable manifest "
                    "but its control socket still answers; refusing to touch "
                    "it. Clean it up manually."
                )
            shutil.rmtree(run_dir)
            return

        if owner_pid is not None and self.client.pid_alive(owner_pid):
            # Note (Jiaxin Deng): the owning runtime is still running (a
            # concurrent pipeline); its run is not stale, leave it alone.
            logger.info(
                "MPS run dir %s belongs to live runtime pid %d; skipping",
                run_dir,
                owner_pid,
            )
            return

        if daemon_pid is None:
            # Owner died between creating the dir and spawning its daemon.
            shutil.rmtree(run_dir)
            return

        owned = self.client.pid_alive(daemon_pid) and self.client.daemon_owns_pipe(
            daemon_pid, pipe_dir
        )
        if not owned:
            # Note (Jiaxin Deng): dead, or the PID was recycled by a stranger.
            shutil.rmtree(run_dir)
            return

        if self._live_clients(pipe_dir):
            # Note (Jiaxin Deng): the manifest proves the daemon is our dead
            # run's, so its clients are that run's orphans; reap, don't block.
            self._reap_orphan_clients(run_dir, pipe_dir, daemon_pid)

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

    def _reap_orphan_clients(
        self,
        run_dir: Path,
        pipe_dir: Path,
        daemon_pid: int,
    ) -> None:
        targets: set[int] = set()
        for force in (False, True):
            # Note (Jiaxin Deng): re-query membership right before signalling
            # so a recycled PID that is no longer an MPS client is never hit.
            targets = self._live_clients(pipe_dir)
            if not targets:
                return
            for pid in sorted(targets):
                logger.warning(
                    "Killing orphaned MPS client pid %d left by stale run %s",
                    pid,
                    run_dir,
                )
                self.client.kill_pid(pid, force=force)
            try:
                self._wait_for(
                    lambda: not self._live_clients(pipe_dir),
                    self.stop_timeout,
                    "orphaned MPS clients survived",
                )
                return
            except MpsError:
                if force:
                    raise MpsError(
                        f"MPS daemon pid {daemon_pid} from stale run dir "
                        f"{run_dir} still has live clients "
                        f"{sorted(targets)} after SIGKILL; refusing to "
                        "reclaim it. Kill them manually."
                    )

    def _live_clients(self, pipe_dir: Path) -> set[int]:
        return {pid for pid in self._clients_of(pipe_dir) if self.client.pid_alive(pid)}

    def _clients_of(self, pipe_dir: Path) -> set[int]:
        pids: set[int] = set()
        for server_pid in self.client.get_server_list(pipe_dir):
            pids.update(self.client.get_client_list(pipe_dir, server_pid))
        return pids

    def start(self) -> None:
        self.state = MpsState.STARTING
        self.paths.pipe_dir.mkdir(parents=True, exist_ok=True)
        self.paths.log_dir.mkdir(parents=True, exist_ok=True)
        # Note (Jiaxin Deng): claim ownership before spawning anything, so a
        # concurrent preflight never sees this dir as an abandoned leftover.
        self._write_manifest()
        try:
            self.daemon_pid = self.client.start_daemon(
                self.paths.pipe_dir, self.paths.log_dir, self.gpu_uuid
            )
            # Note (Jiaxin Deng): record ownership before waiting; a daemon
            # that starts but never answers must stay findable and reapable.
            self._write_manifest()
            self._wait_for(
                lambda: self.client.control_responds(self.paths.pipe_dir),
                self.start_timeout,
                "MPS control daemon did not answer on its control socket",
            )
        except MpsError:
            self._fail()
            self._reap_failed_daemon()
            raise
        except Exception as exc:
            self._fail()
            self._reap_failed_daemon()
            raise MpsError(f"failed to start MPS daemon: {exc}") from exc
        self.state = MpsState.READY

    def _reap_failed_daemon(self) -> None:
        if self.daemon_pid is None or not self.client.pid_alive(self.daemon_pid):
            return
        logger.warning(
            "Killing unresponsive MPS daemon pid %d after failed startup",
            self.daemon_pid,
        )
        self.client.kill_pid(self.daemon_pid)
        try:
            self._wait_for(
                lambda: not self.client.pid_alive(self.daemon_pid),
                self.stop_timeout,
                "unresponsive MPS daemon survived SIGTERM",
            )
        except MpsError:
            self.client.kill_pid(self.daemon_pid, force=True)

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
            attached_pids = self._attached_client_pids()
            missing = {
                pid for pid in expected if not self._tree_attached(pid, attached_pids)
            }
            return not missing

        try:
            self._wait_for(
                attached,
                self.verify_timeout,
                # Note (Jiaxin Deng): a client that missed the pipe dir
                # time-slices with no error, so absence must fail startup.
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
        if self.daemon_pid is None:
            shutil.rmtree(self.paths.state_dir, ignore_errors=True)
            self.state = MpsState.CLEANED
            return
        if self.daemon_pid is not None and not self.client.pid_alive(self.daemon_pid):
            logger.warning(
                "MPS daemon pid %d already dead; removing state dir %s",
                self.daemon_pid,
                self.paths.state_dir,
            )
            shutil.rmtree(self.paths.state_dir, ignore_errors=True)
            self.state = MpsState.CLEANED
            return
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
                lambda: (
                    self.daemon_pid is None
                    or not self.client.pid_alive(self.daemon_pid)
                ),
                self.stop_timeout,
                "MPS daemon did not exit after quit",
            )
        except MpsError:
            self._fail()
            raise
        except OSError as exc:
            self._fail()
            raise MpsError(
                f"MPS control I/O failed during teardown: {exc}. State dir "
                f"preserved for inspection: {self.paths.state_dir}"
            ) from exc
        shutil.rmtree(self.paths.state_dir)
        self.state = MpsState.CLEANED

    def _tree_attached(self, expected_pid: int, attached: set[int]) -> bool:
        # Note (Jiaxin Deng): engines may create their CUDA context in a child
        # process, so an attached descendant counts for its spawned ancestor.
        for pid in attached:
            current: int | None = pid
            for _ in range(32):
                if current == expected_pid:
                    return True
                if current is None or current <= 1:
                    break
                current = self.client.parent_of(current)
        return False

    def _attached_client_pids(self) -> set[int]:
        pids: set[int] = set()
        for server_pid in self.client.get_server_list(self.paths.pipe_dir):
            pids.update(self.client.get_client_list(self.paths.pipe_dir, server_pid))
        return pids

    def _write_manifest(self) -> None:
        manifest = {
            "run_id": self.paths.run_id,
            "owner_pid": os.getpid(),
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
