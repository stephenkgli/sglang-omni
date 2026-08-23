# SPDX-License-Identifier: Apache-2.0
"""Lifecycle manager for the shared per-GPU CUDA MPS daemon.

All pipelines that colocate work on one physical GPU share one daemon: MPS
merges kernels only for clients of the same server, so per-run daemons would
time-slice against each other. The first manager to arrive creates the daemon
and every manager registers itself in an owners directory; the last one to
leave drains the clients and quits the daemon. Recovery after a hard kill
reaps what the dead owners left behind. All ``nvidia-cuda-mps-control`` I/O
goes through the :class:`MpsControlClient` protocol so the state machine is
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

from sglang_omni.mps.state import MpsGpuPaths, state_root_lock, validate_control_socket

try:
    import fcntl
except ImportError:  # non-POSIX unit-test hosts
    fcntl = None

logger = logging.getLogger(__name__)


class MpsError(RuntimeError):
    """Raised when the MPS lifecycle cannot proceed; state dir is preserved."""


class MpsState(enum.Enum):
    IDLE = "idle"
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

    def owner_lease_held(self, lease_file: Path) -> bool: ...


@dataclass
class MpsManager:
    paths: MpsGpuPaths
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
        self._registered = False
        self._owner_lease_fh = None

    @property
    def _owner_file(self) -> Path:
        return self.paths.owners_dir / str(os.getpid())

    def start(self) -> None:
        self.state = MpsState.STARTING
        validate_control_socket(self.paths.control_socket)
        try:
            with state_root_lock(self.paths.state_root, f".lock-{self.gpu_uuid}"):
                self._start_locked()
        except MpsError:
            self._fail()
            raise
        except Exception as exc:
            self._fail()
            raise MpsError(f"failed to start MPS daemon: {exc}") from exc
        self.state = MpsState.READY

    def _start_locked(self) -> None:
        if not self.paths.state_dir.exists():
            self.paths.pipe_dir.mkdir(parents=True, exist_ok=True)
            self.paths.log_dir.mkdir(parents=True, exist_ok=True)
            self.paths.owners_dir.mkdir(parents=True, exist_ok=True)
            self._spawn_fresh_daemon()
            self._acquire_owner_lease()
            return

        daemon_pid = self._manifest_daemon_pid()
        owners = self._owner_pids()
        live_owners = {pid for pid in owners if self._owner_alive(pid)}
        dead_leases = owners - live_owners
        daemon_healthy = (
            daemon_pid is not None
            and self.client.pid_alive(daemon_pid)
            and self.client.daemon_owns_pipe(daemon_pid, self.paths.pipe_dir)
            and self.client.control_responds(self.paths.pipe_dir)
        )
        if daemon_healthy and live_owners:
            # A dead co-owner's lease was released by the kernel; the file is
            # unowned and joining proceeds under the surviving pool.
            self._remove_leases(dead_leases)
            self.daemon_pid = daemon_pid
            self._acquire_owner_lease()
            logger.info(
                "Joining shared MPS daemon pid %d on %s (owners: %s)",
                daemon_pid,
                self.gpu_uuid,
                sorted(live_owners),
            )
            return
        if daemon_healthy and not self._live_clients(self.paths.pipe_dir):
            # The one allowed adoption: provable identity, healthy control,
            # no live owner, empty client list.
            self._remove_leases(dead_leases)
            self.daemon_pid = daemon_pid
            self._acquire_owner_lease()
            logger.warning(
                "Adopting idle MPS daemon pid %d on %s", daemon_pid, self.gpu_uuid
            )
            return
        # Anything else is dirty state from a hard kill; fail with the full
        # picture instead of guessing.
        raise MpsError(self._dirty_state_report(daemon_pid, live_owners, dead_leases))

    def _acquire_owner_lease(self) -> None:
        self._owner_file.write_text("")
        if fcntl is not None:
            # Held for the life of this process; the kernel releases it on any
            # exit, so liveness probing is immune to PID reuse.
            self._owner_lease_fh = open(self._owner_file, "r+")
            fcntl.flock(self._owner_lease_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self._registered = True

    def _owner_alive(self, pid: int) -> bool:
        return self.client.owner_lease_held(self.paths.owners_dir / str(pid))

    def _remove_leases(self, pids: set[int]) -> None:
        for pid in pids:
            (self.paths.owners_dir / str(pid)).unlink(missing_ok=True)

    def _dirty_state_report(
        self,
        daemon_pid: int | None,
        owners: set[int],
        dead_owners: set[int],
    ) -> str:
        daemon_state = "unreadable manifest"
        if daemon_pid is not None:
            daemon_state = f"pid {daemon_pid} " + (
                "alive" if self.client.pid_alive(daemon_pid) else "dead"
            )
        try:
            clients = sorted(self._live_clients(self.paths.pipe_dir))
        except Exception:
            clients = []
        kill_targets = sorted(
            {pid for pid in clients}
            | (
                {daemon_pid}
                if daemon_pid and self.client.pid_alive(daemon_pid)
                else set()
            )
        )
        kill_hint = (
            f"kill the leftover process(es) {kill_targets} and " if kill_targets else ""
        )
        return (
            f"MPS state dir {self.paths.state_dir} holds dirty state from a "
            f"previous run: daemon {daemon_state}; owners "
            f"{sorted(owners) or 'none'} (dead: {sorted(dead_owners) or 'none'}); "
            f"live clients {clients or 'none'}. Refusing to start. "
            f"After confirming nothing on this GPU should be running, "
            f"{kill_hint}remove the directory with: rm -rf {self.paths.state_dir}"
        )

    def _spawn_fresh_daemon(self) -> None:
        self._write_manifest()
        try:
            self.daemon_pid = self.client.start_daemon(
                self.paths.pipe_dir, self.paths.log_dir, self.gpu_uuid
            )
            self._write_manifest()
            self._wait_for(
                lambda: self.client.control_responds(self.paths.pipe_dir),
                self.start_timeout,
                "MPS control daemon did not answer on its control socket",
            )
        except BaseException:
            self._reap_failed_daemon()
            raise

    def _manifest_daemon_pid(self) -> int | None:
        try:
            manifest = json.loads(self.paths.manifest.read_text())
            raw = manifest.get("daemon_pid")
            return int(raw) if raw is not None else None
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _reap_failed_daemon(self) -> None:
        # The one process this manager may signal: the daemon it just spawned.
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

    def _owner_pids(self) -> set[int]:
        if not self.paths.owners_dir.exists():
            return set()
        return {
            int(entry.name)
            for entry in self.paths.owners_dir.iterdir()
            if entry.name.isdigit()
        }

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
                    f"stage process(es) {sorted(missing)} never attached to "
                    f"the MPS server (pipe dir {self.paths.pipe_dir})"
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

    def stop(self) -> None:
        try:
            with state_root_lock(self.paths.state_root, f".lock-{self.gpu_uuid}"):
                self._stop_locked()
        except MpsError:
            self._fail()
            raise
        except Exception as exc:
            self._fail()
            raise MpsError(
                f"MPS control I/O failed during teardown: {exc}. State dir "
                f"preserved for inspection: {self.paths.state_dir}"
            ) from exc

    def _stop_locked(self) -> None:
        if self._registered:
            if self._owner_lease_fh is not None:
                self._owner_lease_fh.close()
                self._owner_lease_fh = None
            self._owner_file.unlink(missing_ok=True)
            self._registered = False
        remaining = {pid for pid in self._owner_pids() if self._owner_alive(pid)}
        if remaining:
            logger.info(
                "Leaving shared MPS daemon on %s to owners %s",
                self.gpu_uuid,
                sorted(remaining),
            )
            self.state = MpsState.CLEANED
            return
        if self.daemon_pid is None:
            shutil.rmtree(self.paths.state_dir, ignore_errors=True)
            self.state = MpsState.CLEANED
            return
        if not self.client.pid_alive(self.daemon_pid):
            logger.warning(
                "MPS daemon pid %d already dead; removing state dir %s",
                self.daemon_pid,
                self.paths.state_dir,
            )
            shutil.rmtree(self.paths.state_dir, ignore_errors=True)
            self.state = MpsState.CLEANED
            return
        self.state = MpsState.DRAINING
        self._wait_for(
            lambda: not self._live_clients(self.paths.pipe_dir),
            self.drain_timeout,
            "MPS clients still attached; refusing to quit the daemon under them",
        )
        self.state = MpsState.STOPPING
        self.client.quit_daemon(self.paths.pipe_dir)
        self._wait_for(
            lambda: not self.client.pid_alive(self.daemon_pid),
            self.stop_timeout,
            "MPS daemon did not exit after quit",
        )
        shutil.rmtree(self.paths.state_dir)
        self.state = MpsState.CLEANED

    def _live_clients(self, pipe_dir: Path) -> set[int]:
        return {pid for pid in self._clients_of(pipe_dir) if self.client.pid_alive(pid)}

    def _clients_of(self, pipe_dir: Path) -> set[int]:
        pids: set[int] = set()
        for server_pid in self.client.get_server_list(pipe_dir):
            pids.update(self.client.get_client_list(pipe_dir, server_pid))
        return pids

    def _attached_client_pids(self) -> set[int]:
        return self._clients_of(self.paths.pipe_dir)

    def _write_manifest(self) -> None:
        manifest = {
            "gpu_uuid": self.gpu_uuid,
            "daemon_pid": self.daemon_pid,
            "creator_pid": os.getpid(),
            "pipe_dir": str(self.paths.pipe_dir),
        }
        staging = self.paths.manifest.with_suffix(".tmp")
        staging.write_text(json.dumps(manifest))
        os.replace(staging, self.paths.manifest)

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
