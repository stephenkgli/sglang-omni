# SPDX-License-Identifier: Apache-2.0
"""Layout of the per-run MPS launcher state directory.

Mirrors the directory structure created by examples/mps_dp/launch.sh:
``<state_root>/gpu-<gpu_id>/<run_id>/`` containing the launch manifest,
replica table, attach report, and the private MPS pipe and log directories.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

try:
    import fcntl
except ImportError:  # non-POSIX unit-test hosts
    fcntl = None

RUN_ID_PATTERN = re.compile(r"run-[A-Za-z0-9_-]+")
# AF_UNIX sun_path is 108 bytes including the terminator on Linux.
SUN_PATH_LIMIT = 107


def validate_control_socket(control_socket: Path) -> None:
    socket_bytes = len(str(control_socket).encode())
    if socket_bytes > SUN_PATH_LIMIT:
        # Note (Jiaxin Deng): over the limit the daemon starts, fails to bind,
        # and exits reporting only "Cannot find MPS control daemon process".
        raise ValueError(
            f"MPS control socket path is {socket_bytes} bytes, over the "
            f"{SUN_PATH_LIMIT}-byte AF_UNIX sun_path limit: "
            f"{control_socket}. Use a shorter state root."
        )


@dataclass(frozen=True)
class MpsRunPaths:
    state_root: Path
    gpu_id: int
    run_id: str

    def __post_init__(self) -> None:
        if not RUN_ID_PATTERN.fullmatch(self.run_id):
            raise ValueError("run_id must be a safe run-<suffix> path component")

    @property
    def state_dir(self) -> Path:
        return self.state_root / f"gpu-{self.gpu_id}" / self.run_id

    @property
    def pipe_dir(self) -> Path:
        return self.state_dir / "mps" / "pipe"

    @property
    def log_dir(self) -> Path:
        return self.state_dir / "mps" / "log"

    @property
    def control_socket(self) -> Path:
        return self.pipe_dir / "control"

    @property
    def manifest(self) -> Path:
        return self.state_dir / "manifest"

    @property
    def replicas_tsv(self) -> Path:
        return self.state_dir / "replicas.tsv"

    @property
    def attach_report(self) -> Path:
        return self.state_dir / "mps_attach.txt"

    @property
    def control_errors(self) -> Path:
        return self.state_dir / "mps_ctl.err"


GPU_DIR_PATTERN = re.compile(r"(GPU|MIG)-[0-9a-fA-F-]+")


@dataclass(frozen=True)
class MpsGpuPaths:
    """Layout of the shared per-physical-GPU MPS state directory.

    One daemon serves every pipeline that colocates work on this GPU, so the
    directory is keyed by the immutable device UUID, not by run or ordinal.
    """

    state_root: Path
    gpu_uuid: str

    def __post_init__(self) -> None:
        if not GPU_DIR_PATTERN.fullmatch(self.gpu_uuid):
            raise ValueError(f"unexpected GPU uuid {self.gpu_uuid!r}")

    @property
    def state_dir(self) -> Path:
        return self.state_root / self.gpu_uuid

    @property
    def pipe_dir(self) -> Path:
        return self.state_dir / "pipe"

    @property
    def log_dir(self) -> Path:
        return self.state_dir / "log"

    @property
    def owners_dir(self) -> Path:
        return self.state_dir / "owners"

    @property
    def manifest(self) -> Path:
        return self.state_dir / "manifest"

    @property
    def control_socket(self) -> Path:
        return self.pipe_dir / "control"


@contextmanager
def state_root_lock(root: Path, lock_name: str = ".lock"):
    """Serialize daemon create/join/leave for one GPU across processes.

    No-op where flock is unavailable (non-POSIX unit-test hosts).
    """
    if fcntl is None:
        yield
        return
    root.mkdir(parents=True, exist_ok=True)
    with open(root / lock_name, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
