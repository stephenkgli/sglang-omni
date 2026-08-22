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
        self.paths.pipe_dir.mkdir(parents=True, exist_ok=True)
        self.paths.log_dir.mkdir(parents=True, exist_ok=True)
        self.paths.owners_dir.mkdir(parents=True, exist_ok=True)
        self._prune_dead_owners()

        daemon_pid = self._manifest_daemon_pid()
        alive = (
            daemon_pid is not None
            and self.client.pid_alive(daemon_pid)
            and self.client.daemon_owns_pipe(daemon_pid, self.paths.pipe_dir)
        )
        if alive and self._owner_pids():
            self.daemon_pid = daemon_pid
            logger.info(
                "Joining shared MPS daemon pid %d on %s (owners: %s)",
                daemon_pid,
                self.gpu_uuid,
                sorted(self._owner_pids()),
            )
            # Note (Jiaxin Deng): a co-owner may have been hard-killed while
            # others live on; its clients are the ones no live owner claims.
            protected = self._registered_pids_of_live_owners()
            if self._unclaimed_clients(protected):
                self._reap_orphan_clients(
                    self.paths.state_dir,
                    self.paths.pipe_dir,
                    daemon_pid,
                    protected=protected,
                )
        elif alive:
            # Every owner died (hard kill); the daemon and possibly its
            # clients are orphans of our namespace. Reap the clients, keep
            # the healthy daemon.
            if self._live_clients(self.paths.pipe_dir):
                self._reap_orphan_clients(
                    self.paths.state_dir,
                    self.paths.pipe_dir,
                    daemon_pid,
                    protected=set(),
                )
            self.daemon_pid = daemon_pid
            logger.warning(
                "Adopting orphan MPS daemon pid %d on %s left by dead owners",
                daemon_pid,
                self.gpu_uuid,
            )
        else:
            self._spawn_fresh_daemon()
        self._owner_file.write_text(json.dumps({"pids": []}))
        self._registered = True

    def _spawn_fresh_daemon(self) -> None:
        # A stale control socket from a dead daemon confuses the new one.
        shutil.rmtree(self.paths.pipe_dir, ignore_errors=True)
        self.paths.pipe_dir.mkdir(parents=True, exist_ok=True)
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
        except BaseException:
            self._reap_failed_daemon()
            raise

    def _manifest_daemon_pid(self) -> int | None:
        try:
            manifest = json.loads(self.paths.manifest.read_text())
            raw = manifest.get("daemon_pid")
            if raw is not None:
                return int(raw)
        except (OSError, ValueError, KeyError, TypeError):
            pass
        # Note (Jiaxin Deng): the daemon's own PID file survives a torn or
        # missing manifest; trust it only with the environ ownership proof.
        try:
            recovered = int(
                (self.paths.pipe_dir / "nvidia-cuda-mps-control.pid")
                .read_text()
                .strip()
            )
            if self.client.pid_alive(recovered) and self.client.daemon_owns_pipe(
                recovered, self.paths.pipe_dir
            ):
                return recovered
        except (OSError, ValueError):
            pass
        if self.client.control_responds(self.paths.pipe_dir):
            raise MpsError(
                f"MPS state dir {self.paths.state_dir} has no readable "
                "manifest but its control socket still answers; refusing to "
                "touch it. Clean it up manually."
            )
        return None

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

    def _reap_orphan_clients(
        self,
        state_dir: Path,
        pipe_dir: Path,
        daemon_pid: int,
        protected: set[int],
    ) -> None:
        targets: set[int] = set()
        for force in (False, True):
            # Note (Jiaxin Deng): re-query membership right before signalling
            # so a recycled PID that is no longer an MPS client is never hit.
            targets = self._unclaimed_clients(protected)
            if not targets:
                return
            for pid in sorted(targets):
                logger.warning(
                    "Killing orphaned MPS client pid %d left under %s",
                    pid,
                    state_dir,
                )
                self.client.kill_pid(pid, force=force)
            try:
                self._wait_for(
                    lambda: not self._unclaimed_clients(protected),
                    self.stop_timeout,
                    "orphaned MPS clients survived",
                )
                return
            except MpsError:
                if force:
                    raise MpsError(
                        f"MPS daemon pid {daemon_pid} under {state_dir} "
                        f"still has live clients {sorted(targets)} after "
                        "SIGKILL; refusing to proceed. Kill them manually."
                    )

    def _owner_pids(self) -> set[int]:
        if not self.paths.owners_dir.exists():
            return set()
        return {
            int(entry.name)
            for entry in self.paths.owners_dir.iterdir()
            if entry.name.isdigit()
        }

    def register_clients(self, expected_pids: set[int]) -> None:
        with state_root_lock(self.paths.state_root, f".lock-{self.gpu_uuid}"):
            self._owner_file.write_text(json.dumps({"pids": sorted(expected_pids)}))

    def _registered_pids_of_live_owners(self) -> set[int]:
        pids: set[int] = set()
        for owner in self._owner_pids():
            if owner != os.getpid() and not self.client.pid_alive(owner):
                continue
            try:
                recorded = json.loads((self.paths.owners_dir / str(owner)).read_text())
                pids.update(int(pid) for pid in recorded.get("pids", []))
            except (OSError, ValueError, TypeError, AttributeError):
                continue
        return pids

    def _unclaimed_clients(self, protected: set[int]) -> set[int]:
        return {
            pid
            for pid in self._live_clients(self.paths.pipe_dir)
            if not any(self._tree_attached(p, {pid}) for p in protected)
        }

    def _prune_dead_owners(self) -> None:
        for pid in self._owner_pids():
            if pid != os.getpid() and not self.client.pid_alive(pid):
                (self.paths.owners_dir / str(pid)).unlink(missing_ok=True)

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
        self.register_clients(expected)
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
            self._owner_file.unlink(missing_ok=True)
            self._registered = False
        self._prune_dead_owners()
        remaining = self._owner_pids()
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
