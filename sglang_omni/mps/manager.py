# SPDX-License-Identifier: Apache-2.0
"""Ownership-based lifecycle for one shared per-GPU CUDA MPS daemon.

The manager has no lifecycle state of its own. A successful :meth:`acquire`
returns the only cleanup authority, an :class:`MpsLease`; every later operation
requires that token. Existing state is joined only when the native daemon
identity is provable and every published owner lease is held. Anything
ambiguous is preserved for an operator instead of being repaired in place.
"""

from __future__ import annotations

import fcntl
import logging
import os
import shlex
import shutil
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from sglang_omni.mps.state import MpsGpuPaths, state_root_lock, validate_control_socket

logger = logging.getLogger(__name__)


class MpsError(RuntimeError):
    """Raised when the MPS lifecycle cannot proceed safely."""


class MpsDirtyStateError(MpsError):
    """Cleanup persisted dirty state and released its owner lock."""


class MpsControlError(MpsError):
    """Raised when a strict MPS control or process query fails."""


_OWNER_ACTIVE = "active"
_OWNER_RETAINED = "retained"
_OWNER_STATUSES = {_OWNER_ACTIVE, _OWNER_RETAINED}


@dataclass(frozen=True, order=True)
class MpsClientRef:
    """One CUDA client as identified by the MPS server that owns it."""

    server_pid: int
    client_pid: int


@dataclass
class MpsLease:
    """All authority and runtime-local evidence owned by one acquisition."""

    daemon_pid: int
    owner_fd: int
    attached_clients: set[MpsClientRef] = field(default_factory=set)
    attachment_verified: bool = False


class MpsControlClient(Protocol):
    """Strict domain I/O used by :class:`MpsManager`."""

    def start_daemon(self, pipe_dir: Path, log_dir: Path, gpu_uuid: str) -> None: ...

    def read_daemon_identity(self, pipe_dir: Path) -> int: ...

    def snapshot(self, pipe_dir: Path) -> set[MpsClientRef]: ...

    def quit_daemon(self, pipe_dir: Path) -> None: ...

    def daemon_process_alive(self, pid: int) -> bool: ...

    def parent_of(self, pid: int) -> int | None: ...

    def owner_lease_held(self, lease_file: Path) -> bool: ...


@dataclass
class _ExistingState:
    daemon_pid: int | None = None
    owners: dict[int, bool] = field(default_factory=dict)
    owner_statuses: dict[int, str] = field(default_factory=dict)
    clients: set[MpsClientRef] | None = None
    errors: list[str] = field(default_factory=list)


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

    @property
    def _owner_file(self) -> Path:
        return self.paths.owners_dir / str(os.getpid())

    def acquire(self) -> MpsLease:
        """Create or join the daemon and return the sole cleanup token."""

        validate_control_socket(self.paths.control_socket)
        try:
            with state_root_lock(self.paths.state_root, f".lock-{self.gpu_uuid}"):
                if not self.paths.state_dir.exists():
                    return self._create_locked()
                return self._join_locked()
        except MpsError:
            raise
        except Exception as exc:
            raise MpsError(
                f"failed to acquire MPS on {self.gpu_uuid}: {exc}. State dir "
                f"preserved for inspection: {self.paths.state_dir}"
            ) from exc

    def _create_locked(self) -> MpsLease:
        self.paths.pipe_dir.mkdir(parents=True)
        self.paths.log_dir.mkdir()
        self.paths.owners_dir.mkdir()
        owner_fd = self._publish_owner()
        lease: MpsLease | None = None
        try:
            self.client.start_daemon(
                self.paths.pipe_dir, self.paths.log_dir, self.gpu_uuid
            )
            lease = MpsLease(
                daemon_pid=self.client.read_daemon_identity(self.paths.pipe_dir),
                owner_fd=owner_fd,
            )
            clients = self._wait_for_snapshot(
                self.start_timeout,
                "MPS control daemon did not answer on its control socket",
            )
        except BaseException as startup_error:
            cleanup_error: MpsError | None = None
            if lease is None:
                try:
                    lease = MpsLease(
                        daemon_pid=self.client.read_daemon_identity(
                            self.paths.pipe_dir
                        ),
                        owner_fd=owner_fd,
                    )
                except MpsControlError as identity_error:
                    cleanup_error = self._persist_unidentified_dirty(
                        owner_fd,
                        identity_error,
                    )
            if lease is not None:
                cleanup_error = self._rollback_create(lease)
            if cleanup_error is not None:
                raise startup_error from cleanup_error
            raise
        assert lease is not None
        if clients:
            error = MpsError(
                "a newly created private MPS daemon already reports clients "
                f"{sorted(clients)}; refusing ambiguous ownership and persisting "
                f"{self.paths.state_dir}"
            )
            raise self._persist_dirty_locked(lease, error, clients=clients) from error
        return lease

    def _join_locked(self) -> MpsLease:
        state = self._inspect_existing_state()
        if (
            not state.errors
            and state.daemon_pid is not None
            and state.clients is not None
            and state.owners
            and all(state.owners.values())
            and set(state.owner_statuses.values()) == {_OWNER_ACTIVE}
        ):
            owner_fd = self._publish_owner()
            logger.info(
                "Joining shared MPS daemon pid %d on %s (owners: %s)",
                state.daemon_pid,
                self.gpu_uuid,
                sorted(state.owners),
            )
            return MpsLease(daemon_pid=state.daemon_pid, owner_fd=owner_fd)
        raise MpsError(self._dirty_state_report(state))

    def _publish_owner(self) -> int:
        try:
            owner_fd = os.open(
                self._owner_file,
                os.O_CREAT | os.O_EXCL | os.O_RDWR,
                0o600,
            )
        except FileExistsError as exc:
            raise MpsError(
                f"owner lease {self._owner_file} already exists; refusing to "
                "replace ambiguous state"
            ) from exc
        try:
            fcntl.flock(owner_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._write_owner_status(owner_fd, _OWNER_ACTIVE)
        except BaseException:
            os.close(owner_fd)
            self._owner_file.unlink(missing_ok=True)
            raise
        return owner_fd

    @staticmethod
    def _write_owner_status(owner_fd: int, status: str) -> None:
        if status not in _OWNER_STATUSES:
            raise ValueError(f"invalid owner status {status!r}")
        value = f"{status}\n".encode()
        if os.pwrite(owner_fd, value, 0) != len(value):
            raise OSError("short write while updating MPS owner status")
        os.ftruncate(owner_fd, len(value))
        os.fsync(owner_fd)

    @staticmethod
    def _read_owner_status(owner_file: Path) -> str:
        try:
            status = owner_file.read_text().strip()
        except OSError as exc:
            raise MpsError(
                f"cannot read owner lease status {owner_file}: {exc}"
            ) from exc
        if status not in _OWNER_STATUSES:
            raise MpsError(
                f"owner lease {owner_file} has invalid status {status!r}"
            )
        return status

    def _owner_files(self) -> dict[int, Path]:
        if not self.paths.owners_dir.is_dir():
            raise MpsError(f"owner lease directory is missing: {self.paths.owners_dir}")
        owners: dict[int, Path] = {}
        for entry in self.paths.owners_dir.iterdir():
            if not entry.name.isdigit() or int(entry.name) <= 0 or not entry.is_file():
                raise MpsError(f"malformed owner lease entry: {entry}")
            owners[int(entry.name)] = entry
        return owners

    def _inspect_existing_state(self) -> _ExistingState:
        state = _ExistingState()
        try:
            state.daemon_pid = self.client.read_daemon_identity(self.paths.pipe_dir)
        except MpsControlError as exc:
            state.errors.append(f"daemon identity: {exc}")
        try:
            state.clients = self.client.snapshot(self.paths.pipe_dir)
        except MpsControlError as exc:
            state.errors.append(f"control snapshot: {exc}")
        try:
            owner_files = self._owner_files()
        except MpsError as exc:
            state.errors.append(str(exc))
        else:
            for pid, owner_file in owner_files.items():
                try:
                    state.owners[pid] = self.client.owner_lease_held(owner_file)
                    state.owner_statuses[pid] = self._read_owner_status(owner_file)
                except MpsControlError as exc:
                    state.errors.append(f"owner lease {pid}: {exc}")
                except MpsError as exc:
                    state.errors.append(str(exc))
        return state

    def _dirty_state_report(self, state: _ExistingState) -> str:
        daemon = (
            f"pid {state.daemon_pid} with verified native identity"
            if state.daemon_pid is not None
            else "identity unverified"
        )
        owners = {
            pid: {
                "lock": "held" if held else "dead",
                "status": state.owner_statuses.get(pid, "unknown"),
            }
            for pid, held in sorted(state.owners.items())
        }
        clients = sorted(state.clients) if state.clients is not None else "unavailable"
        details = f"; query errors: {state.errors}" if state.errors else ""
        return (
            f"MPS state dir {self.paths.state_dir} holds dirty state from a previous "
            f"run: daemon {daemon}; owner leases {owners or 'none'}; clients "
            f"{clients}{details}. Refusing to start and preserving all evidence. "
            f"{self._cleanup_guidance(state.clients, owned_clients=set())}"
        )

    def _cleanup_guidance(
        self,
        clients: set[MpsClientRef] | None,
        *,
        owned_clients: set[MpsClientRef] | None = None,
    ) -> str:
        control = (
            f"CUDA_MPS_PIPE_DIRECTORY={shlex.quote(str(self.paths.pipe_dir))} "
            "nvidia-cuda-mps-control"
        )

        def command(value: str) -> str:
            return f"printf '%s\\n' {shlex.quote(value)} | {control}"

        actionable_clients = (
            clients
            if clients is None or owned_clients is None
            else clients & owned_clients
        )
        foreign_clients = (
            set()
            if clients is None or owned_clients is None
            else clients - owned_clients
        )

        if actionable_clients is None:
            client_steps = (
                "The control snapshot is unavailable, so no safe client command "
                "can be generated. Restore control access before signaling any "
                "possible CUDA client."
            )
            if owned_clients:
                client_steps += (
                    " The last verified refs for this owner were "
                    f"{sorted(owned_clients)}; revalidate them against a fresh "
                    "snapshot before issuing terminate_client."
                )
        elif actionable_clients:
            client_steps = (
                "Run only these commands for clients proven to belong to this "
                "owner before any forced OS signal:\n  "
                + "\n  ".join(
                    command(
                        f"terminate_client {client.server_pid} {client.client_pid}"
                    )
                    for client in sorted(actionable_clients)
                )
            )
        else:
            client_steps = "No current client is proven to belong to this owner."

        if foreign_clients:
            client_steps += (
                f" Other observed refs are not proven to belong to this lease: "
                f"{sorted(foreign_clients)}. Do not terminate them from this report."
            )

        current_clients = "unavailable" if clients is None else repr(sorted(clients))
        prefix = (
            f"Current MPS client refs: {current_clients}. After confirming no "
            "workload owned by this serve should remain, clean up in this order. "
            f"{client_steps}\n"
        )
        return prefix + (
            "Stop every remaining workload process, repeat the snapshot and client "
            "termination if needed, and only after every owner lease is unlocked "
            "and a fresh snapshot is empty run:\n  "
            f"{command('quit')}\n  "
            f"rm -rf {shlex.quote(str(self.paths.state_dir))}"
        )

    def _rollback_create(self, lease: MpsLease) -> MpsError | None:
        try:
            self._release_locked(lease)
            return None
        except BaseException as exc:
            if lease.owner_fd >= 0:
                dirty_error: MpsError = self._persist_dirty_locked(lease, exc)
            elif isinstance(exc, MpsError):
                dirty_error = exc
            else:
                dirty_error = MpsError(f"MPS startup rollback failed: {exc}")
            logger.error(
                "MPS startup rollback persisted dirty state at %s: %s",
                self.paths.state_dir,
                dirty_error,
            )
            return dirty_error

    def env_for_stage(self) -> dict[str, str]:
        return {
            "CUDA_MPS_PIPE_DIRECTORY": str(self.paths.pipe_dir),
            "CUDA_MPS_LOG_DIRECTORY": str(self.paths.log_dir),
            "CUDA_VISIBLE_DEVICES": self.gpu_uuid,
        }

    def verify(
        self, lease: MpsLease, expected_pids: Iterable[int]
    ) -> set[MpsClientRef]:
        """Gate startup on attachment and retain the exact owned client refs."""

        self._require_live_lease(lease)
        expected = set(expected_pids)
        missing = set(expected)
        last_error: MpsControlError | None = None
        deadline = time.monotonic() + self.verify_timeout
        while True:
            try:
                snapshot = self.client.snapshot(self.paths.pipe_dir)
                attached = self._clients_belonging_to(snapshot, expected)
                lease.attached_clients.update(attached)
                attached_roots = {
                    root
                    for root in expected
                    if any(
                        self._client_belongs_to(client.client_pid, root)
                        for client in snapshot
                    )
                }
                missing = expected - attached_roots
                last_error = None
                if not missing:
                    lease.attachment_verified = True
                    return set(lease.attached_clients)
            except MpsControlError as exc:
                last_error = exc
            if time.monotonic() >= deadline:
                detail = f"; last control error: {last_error}" if last_error else ""
                raise MpsError(
                    f"stage process(es) {sorted(missing)} never attached to the MPS "
                    f"server (pipe dir {self.paths.pipe_dir}){detail}. State dir "
                    f"preserved for inspection: {self.paths.state_dir}"
                )
            time.sleep(self.poll_interval)

    def probe(self, lease: MpsLease) -> bool:
        """Check daemon identity, control health, and every retained client ref."""

        self._require_live_lease(lease)
        try:
            daemon_pid = self.client.read_daemon_identity(self.paths.pipe_dir)
            if daemon_pid != lease.daemon_pid:
                return False
            return lease.attached_clients <= self.client.snapshot(self.paths.pipe_dir)
        except MpsControlError:
            return False

    def release(
        self,
        lease: MpsLease,
        *,
        ownership_complete: bool = True,
    ) -> None:
        """Release exactly one acquired lease, quitting only as the last owner."""

        self._require_live_lease(lease)
        try:
            with state_root_lock(self.paths.state_root, f".lock-{self.gpu_uuid}"):
                try:
                    self._release_locked(
                        lease,
                        ownership_complete=ownership_complete,
                    )
                except BaseException as exc:
                    if lease.owner_fd >= 0:
                        raise self._persist_dirty_locked(lease, exc) from exc
                    raise
        except MpsDirtyStateError:
            raise
        except MpsError:
            raise
        except Exception as exc:
            if lease.owner_fd >= 0:
                owner_pid = os.getpid()
                self._abandon_owner(lease)
                raise MpsDirtyStateError(
                    f"MPS cleanup could not persist a retained status under the "
                    f"GPU lock for {self.gpu_uuid}: {exc}. Owner PID {owner_pid} "
                    f"marker {self._owner_file} was left in place with an "
                    f"unconfirmed status and its lock is released; state "
                    f"directory {self.paths.state_dir} is preserved. "
                    f"{self._cleanup_guidance(None, owned_clients=lease.attached_clients)}"
                ) from exc
            raise MpsError(
                f"MPS control I/O failed during release: {exc}. State dir "
                f"preserved for inspection: {self.paths.state_dir}"
            ) from exc

    def _release_locked(
        self,
        lease: MpsLease,
        *,
        ownership_complete: bool = True,
    ) -> None:
        self._wait_for_owned_clients_to_detach(lease)

        try:
            daemon_pid = self.client.read_daemon_identity(self.paths.pipe_dir)
            snapshot = self.client.snapshot(self.paths.pipe_dir)
        except MpsControlError:
            raise
        if daemon_pid != lease.daemon_pid:
            raise MpsError(
                f"MPS daemon identity changed from {lease.daemon_pid} to {daemon_pid}; "
                "owner lease and shared state preserved"
            )

        remaining_files = {
            pid: path
            for pid, path in self._owner_files().items()
            if path != self._owner_file
        }
        remaining = {
            pid: {
                "held": self.client.owner_lease_held(path),
                "status": self._read_owner_status(path),
            }
            for pid, path in remaining_files.items()
        }

        if remaining:
            if snapshot and (
                not lease.attachment_verified or not ownership_complete
            ):
                raise MpsError(
                    "MPS client ownership is incomplete at shutdown; refusing "
                    "to release this owner while a shared daemon still has "
                    f"clients. State preserved: "
                    f"{self.paths.state_dir}"
                )
            self._drop_owner(lease)
            dirty_owners = {
                pid
                for pid, state in remaining.items()
                if not state["held"] or state["status"] != _OWNER_ACTIVE
            }
            logger.log(
                logging.WARNING if dirty_owners else logging.INFO,
                "Leaving shared MPS daemon on %s to owners %s%s",
                self.gpu_uuid,
                sorted(remaining),
                f"; dirty owner markers: {sorted(dirty_owners)}"
                if dirty_owners
                else "",
            )
            return

        if snapshot:
            snapshot = self._wait_for_no_clients()
        if snapshot:
            raise MpsError(
                f"MPS clients {sorted(snapshot)} remain while releasing the last "
                f"owner; refusing to release its lease or quit daemon "
                f"{lease.daemon_pid}. State preserved: {self.paths.state_dir}"
            )

        try:
            self.client.quit_daemon(self.paths.pipe_dir)
        except MpsControlError:
            # The local control command may lose its response after the daemon
            # has already exited. A dead, previously verified native PID is a
            # sufficient commit point; a surviving daemon keeps the lease.
            if self.client.daemon_process_alive(lease.daemon_pid):
                raise
        else:
            self._wait_for(
                lambda: not self.client.daemon_process_alive(lease.daemon_pid),
                self.stop_timeout,
                "MPS daemon did not exit after quit",
            )
        self._drop_owner(lease)
        shutil.rmtree(self.paths.state_dir)

    def _persist_dirty_locked(
        self,
        lease: MpsLease,
        error: BaseException,
        *,
        clients: set[MpsClientRef] | None = None,
    ) -> MpsDirtyStateError:
        owner_pid = os.getpid()
        status_error: BaseException | None = None
        try:
            self._mark_retained_locked(lease)
        except BaseException as exc:
            status_error = exc

        observed_daemon_pid: int | None = None
        query_error: MpsControlError | None = None
        try:
            observed_daemon_pid = self.client.read_daemon_identity(
                self.paths.pipe_dir
            )
            if clients is None:
                clients = self.client.snapshot(self.paths.pipe_dir)
        except MpsControlError as exc:
            query_error = exc

        guidance = self._cleanup_guidance(
            clients,
            owned_clients=lease.attached_clients,
        )
        self._abandon_owner(lease)
        status = (
            "retained"
            if status_error is None
            else f"unconfirmed because the retained-status write failed: {status_error}"
        )
        observed = (
            str(observed_daemon_pid)
            if observed_daemon_pid is not None
            else f"unavailable ({query_error})"
        )
        snapshot = "unavailable" if clients is None else repr(sorted(clients))
        return MpsDirtyStateError(
            f"MPS cleanup persisted dirty state for GPU {self.gpu_uuid}: {error}. "
            f"Owner PID {owner_pid} marker {self._owner_file} is {status} and its "
            f"lock is released; state directory {self.paths.state_dir} is preserved. "
            f"Expected daemon PID {lease.daemon_pid}; observed daemon PID {observed}; "
            f"last verified owned clients {sorted(lease.attached_clients)}; current "
            f"snapshot {snapshot}. {guidance}"
        )

    def _persist_unidentified_dirty(
        self,
        owner_fd: int,
        error: BaseException,
    ) -> MpsDirtyStateError:
        owner_pid = os.getpid()
        status_error = self._abandon_owner_fd(owner_fd)
        status = (
            "retained"
            if status_error is None
            else f"unconfirmed because the retained-status write failed: {status_error}"
        )
        return MpsDirtyStateError(
            f"MPS startup persisted dirty state for GPU {self.gpu_uuid}: {error}. "
            f"Owner PID {owner_pid} marker {self._owner_file} is {status} and its "
            f"lock is released; state directory {self.paths.state_dir} is preserved. "
            "Daemon identity and client snapshot are unavailable. "
            f"{self._cleanup_guidance(None, owned_clients=set())}"
        )

    def _wait_for_owned_clients_to_detach(self, lease: MpsLease) -> None:
        deadline = time.monotonic() + self.drain_timeout
        while True:
            snapshot = self.client.snapshot(self.paths.pipe_dir)
            remaining = lease.attached_clients & snapshot
            if not remaining:
                return
            if time.monotonic() >= deadline:
                raise MpsError(
                    f"owned MPS clients {sorted(remaining)} are still attached; "
                    "refusing clean release. State dir preserved for inspection: "
                    f"{self.paths.state_dir}"
                )
            time.sleep(self.poll_interval)

    def _wait_for_no_clients(self) -> set[MpsClientRef]:
        deadline = time.monotonic() + self.drain_timeout
        while True:
            snapshot = self.client.snapshot(self.paths.pipe_dir)
            if not snapshot or time.monotonic() >= deadline:
                return snapshot
            time.sleep(self.poll_interval)

    def _clients_belonging_to(
        self,
        clients: set[MpsClientRef],
        roots: set[int],
    ) -> set[MpsClientRef]:
        if not roots:
            return set()
        return {
            client
            for client in clients
            if any(self._client_belongs_to(client.client_pid, root) for root in roots)
        }

    def _client_belongs_to(self, client_pid: int, root_pid: int) -> bool:
        current: int | None = client_pid
        for _ in range(32):
            if current == root_pid:
                return True
            if current is None or current <= 1:
                return False
            current = self.client.parent_of(current)
        return False

    def _require_live_lease(self, lease: MpsLease) -> None:
        try:
            fd_stat = os.fstat(lease.owner_fd)
            owner_stat = self._owner_file.stat()
        except (OSError, ValueError):
            raise MpsError("operation requires this manager's live MPS lease") from None
        if (fd_stat.st_dev, fd_stat.st_ino) != (owner_stat.st_dev, owner_stat.st_ino):
            raise MpsError("operation requires this manager's live MPS lease")

    def _mark_retained_locked(self, lease: MpsLease) -> None:
        self._require_live_lease(lease)
        self._write_owner_status(lease.owner_fd, _OWNER_RETAINED)

    def _drop_owner(self, lease: MpsLease) -> None:
        owner_fd = lease.owner_fd
        lease.owner_fd = -1
        self._discard_owner_fd(owner_fd)

    def _abandon_owner(self, lease: MpsLease) -> None:
        owner_fd = lease.owner_fd
        lease.owner_fd = -1
        os.close(owner_fd)

    def _abandon_owner_fd(self, owner_fd: int) -> BaseException | None:
        status_error: BaseException | None = None
        try:
            self._write_owner_status(owner_fd, _OWNER_RETAINED)
        except BaseException as exc:
            status_error = exc
            logger.error(
                "Could not write retained status to %s; the unlocked owner marker "
                "will still block future acquisition: %s",
                self._owner_file,
                exc,
            )
        os.close(owner_fd)
        return status_error

    def _discard_owner_fd(self, owner_fd: int) -> None:
        os.close(owner_fd)
        self._owner_file.unlink(missing_ok=True)

    def _wait_for_snapshot(self, timeout: float, message: str) -> set[MpsClientRef]:
        deadline = time.monotonic() + timeout
        last_error: MpsControlError | None = None
        while True:
            try:
                return self.client.snapshot(self.paths.pipe_dir)
            except MpsControlError as exc:
                last_error = exc
            if time.monotonic() >= deadline:
                detail = f": {last_error}" if last_error else ""
                raise MpsError(f"{message}{detail}")
            time.sleep(self.poll_interval)

    def _wait_for(self, predicate, timeout: float, message: str) -> None:
        deadline = time.monotonic() + timeout
        while True:
            if predicate():
                return
            if time.monotonic() >= deadline:
                raise MpsError(
                    f"{message}. State dir preserved for inspection: "
                    f"{self.paths.state_dir}"
                )
            time.sleep(self.poll_interval)
