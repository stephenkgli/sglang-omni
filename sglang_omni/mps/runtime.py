# SPDX-License-Identifier: Apache-2.0
"""Pipeline-level MPS orchestration: one MpsManager per colocated GPU.

Bridges the activation decision (:mod:`sglang_omni.mps.decision`) to per-GPU
daemon lifecycles (:mod:`sglang_omni.mps.manager`) and hands the runner the
per-process env, the attach-verification gate, the watchdog probe, and
teardown.
"""

from __future__ import annotations

import contextlib
import getpass
import logging
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Protocol

from sglang_omni.mps.decision import MpsGpuPlan, plan_mps_gpus
from sglang_omni.mps.manager import MpsControlClient, MpsError, MpsManager
from sglang_omni.mps.state import MpsRunPaths

logger = logging.getLogger(__name__)


class MpsDeviceInfo(Protocol):
    def gpu_uuid(self, gpu_id: int) -> str: ...

    def unsupported_reason(self, gpu_id: int) -> str | None: ...


@contextlib.contextmanager
def _state_root_lock(root: Path):
    # Note (Jiaxin Deng): serializes preflight recovery and run-dir creation
    # across concurrent serves on one host; no-op where flock is unavailable.
    try:
        import fcntl
    except ImportError:
        yield
        return
    lock_path = root / ".lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _default_state_root() -> Path:
    # Note (Jiaxin Deng): keep this short. The control socket lives under it
    # and must fit the 107-byte AF_UNIX sun_path budget.
    return Path(tempfile.gettempdir()) / f"sglang-omni-mps-{getpass.getuser()}"


class MpsPipelineRuntime:
    def __init__(
        self,
        managers: dict[int, MpsManager],
        plans: dict[int, MpsGpuPlan],
        mode: str = "auto",
    ):
        self.managers = managers
        self._mode = mode
        self._client_gpu: dict[str, int] = {
            name: gpu_id
            for gpu_id, plan in plans.items()
            for name in plan.client_process_names
        }

    @classmethod
    def create(
        cls,
        *,
        mode: str,
        process_specs,
        device_info: MpsDeviceInfo,
        client: MpsControlClient,
        state_root: Path | None = None,
        run_id: str | None = None,
    ) -> MpsPipelineRuntime | None:
        if os.environ.get("CUDA_MPS_PIPE_DIRECTORY"):
            raise MpsError(
                "CUDA_MPS_PIPE_DIRECTORY is already set in the environment, "
                "which points every process at an externally managed MPS "
                "daemon. Unset it, or run with mps=off to keep managing MPS "
                "yourself."
            )
        plans = plan_mps_gpus(process_specs, mode)
        if not plans:
            return None

        usable: dict[int, MpsGpuPlan] = {}
        for plan in plans:
            reason = device_info.unsupported_reason(plan.gpu_id)
            if reason is None:
                usable[plan.gpu_id] = plan
                continue
            if mode == "on":
                raise MpsError(
                    f"mps=on but GPU {plan.gpu_id} does not support MPS: {reason}"
                )
            logger.warning(
                "MPS auto: GPU %d colocates %d process(es) but does not "
                "support MPS (%s); running without MPS",
                plan.gpu_id,
                len(plan.client_process_names),
                reason,
            )
        if not usable:
            return None

        root = state_root if state_root is not None else _default_state_root()
        run = run_id if run_id is not None else f"run-{uuid.uuid4().hex[:8]}"
        managers = {
            gpu_id: MpsManager(
                paths=MpsRunPaths(state_root=root, gpu_id=gpu_id, run_id=run),
                gpu_uuid=device_info.gpu_uuid(gpu_id),
                client=client,
            )
            for gpu_id in usable
        }
        return cls(managers, usable, mode=mode)

    def start(self) -> None:
        root = next(iter(self.managers.values())).paths.state_root
        root.mkdir(parents=True, exist_ok=True)
        root.chmod(0o700)
        with _state_root_lock(root):
            for gpu_id, manager in self.managers.items():
                manager.preflight()
                manager.start()
                logger.info(
                    "MPS daemon ready on GPU %d (pipe dir %s)",
                    gpu_id,
                    manager.paths.pipe_dir,
                )
        logger.info(
            "MPS summary: mode=%s %s",
            self._mode,
            {
                gpu_id: {
                    "daemon_pid": manager.daemon_pid,
                    "clients": sorted(self._names_on(gpu_id)),
                }
                for gpu_id, manager in self.managers.items()
            },
        )

    def _names_on(self, gpu_id: int) -> list[str]:
        return [name for name, gpu in self._client_gpu.items() if gpu == gpu_id]

    def env_for_process(self, process_name: str) -> dict[str, str]:
        gpu_id = self._client_gpu.get(process_name)
        if gpu_id is None:
            return {}
        env = self.managers[gpu_id].env_for_stage()
        # Note (Jiaxin Deng): the single-visible-device contract makes the
        # worker normalize its local gpu_id to 0 under the UUID-scoped daemon.
        env["SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS"] = "true"
        return env

    def verify(self, pids_by_process_name: dict[str, int]) -> None:
        for gpu_id, manager in self.managers.items():
            expected = {
                pid
                for name, pid in pids_by_process_name.items()
                if self._client_gpu.get(name) == gpu_id
            }
            if expected:
                manager.verify_attached(expected)

    def probe_failures(self) -> list[int]:
        return [
            gpu_id for gpu_id, manager in self.managers.items() if not manager.probe()
        ]

    def stop(self) -> None:
        errors: list[str] = []
        for manager in self.managers.values():
            try:
                manager.stop()
            except MpsError as exc:
                errors.append(str(exc))
        if errors:
            raise MpsError("; ".join(errors))

    def stop_best_effort(self) -> None:
        try:
            self.stop()
        except MpsError as exc:
            logger.error("MPS teardown incomplete: %s", exc)


def create_for_pipeline(mode: str, process_specs) -> MpsPipelineRuntime | None:
    """Build the orchestrator with production device info and control I/O.

    Imports are deferred so the mps package stays importable on hosts without
    the platform/NVML stack (unit tests inject fakes instead).
    """
    if mode == "off":
        return None
    from sglang_omni.platforms import current_platform

    if not current_platform.is_cuda_alike():
        if mode == "on":
            raise MpsError("mps=on requires an NVIDIA CUDA platform")
        logger.warning("MPS auto: platform is not CUDA; running without MPS")
        return None

    if shutil.which("nvidia-cuda-mps-control") is None:
        if mode == "on":
            raise MpsError("mps=on but nvidia-cuda-mps-control is not on PATH")
        logger.warning(
            "MPS auto: nvidia-cuda-mps-control not found; running without MPS"
        )
        return None

    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        # Note (Jiaxin Deng): a parent CUDA context predates the MPS env and
        # would not attach; refuse instead of serving a half-managed pipeline.
        raise MpsError(
            "CUDA was initialized in the parent process before MPS setup; "
            "this is a runtime bug, please report it"
        )

    from sglang_omni.utils.ipc_weights import get_weight_share_config

    if get_weight_share_config(os.environ) is not None:
        raise MpsError(
            "CUDA IPC weight sharing (launch.sh WEIGHT_SHARE=1) and native "
            "mps cannot be combined; use examples/mps_dp/launch.sh for that "
            "deployment shape"
        )

    from sglang_omni.mps.control import SubprocessMpsControlClient
    from sglang_omni.mps.devices import NvmlDeviceInfo

    return MpsPipelineRuntime.create(
        mode=mode,
        process_specs=process_specs,
        device_info=NvmlDeviceInfo(),
        client=SubprocessMpsControlClient(),
    )
