# SPDX-License-Identifier: Apache-2.0
"""Transactional pipeline ownership of one MPS lease per eligible GPU."""

from __future__ import annotations

import getpass
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Protocol

from sglang_omni.mps.decision import MpsGpuPlan, plan_mps_gpus
from sglang_omni.mps.devices import MpsPhysicalDevice
from sglang_omni.mps.manager import (
    MpsControlClient,
    MpsError,
    MpsLease,
    MpsLeaseRetainedError,
    MpsManager,
)
from sglang_omni.mps.state import MpsGpuPaths

logger = logging.getLogger(__name__)


class MpsDeviceInfo(Protocol):
    def inspect(self, gpu_id: int) -> MpsPhysicalDevice: ...


def _default_state_root() -> Path:
    # Keep this short: the control socket must fit Linux's AF_UNIX path budget.
    override = os.environ.get("SGLANG_OMNI_MPS_STATE_ROOT")
    if override:
        return Path(override)
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
        self._leases: dict[int, MpsLease] = {}
        self._client_gpu: dict[str, int] = {
            name: gpu_id
            for gpu_id, plan in plans.items()
            for name in plan.client_process_names
        }

    @property
    def has_leases(self) -> bool:
        return bool(self._leases)

    @property
    def process_names(self) -> frozenset[str]:
        return frozenset(self._client_gpu)

    @classmethod
    def create(
        cls,
        *,
        mode: str,
        process_specs,
        device_info: MpsDeviceInfo,
        client: MpsControlClient,
        state_root: Path | None = None,
    ) -> MpsPipelineRuntime | None:
        if mode == "off":
            return None
        if os.environ.get("CUDA_MPS_PIPE_DIRECTORY"):
            raise MpsError(
                "CUDA_MPS_PIPE_DIRECTORY is already set in the environment, which "
                "points every process at an externally managed MPS daemon. Unset "
                "it, or run with mps=off to keep managing MPS yourself."
            )
        plans = plan_mps_gpus(process_specs, mode)
        if not plans:
            return None

        usable: dict[int, MpsGpuPlan] = {}
        gpu_uuids: dict[int, str] = {}
        for plan in plans:
            device = device_info.inspect(plan.gpu_id)
            if device.unsupported_reason is None and device.gpu_uuid is not None:
                usable[plan.gpu_id] = plan
                gpu_uuids[plan.gpu_id] = device.gpu_uuid
                continue
            reason = device.unsupported_reason or "physical GPU UUID is unavailable"
            if mode == "on":
                raise MpsError(
                    f"mps=on but GPU {plan.gpu_id} does not support MPS: {reason}"
                )
            logger.warning(
                "MPS auto: GPU %d colocates %d process(es) but does not support "
                "MPS (%s); running without MPS",
                plan.gpu_id,
                len(plan.client_process_names),
                reason,
            )
        if not usable:
            return None

        root = state_root if state_root is not None else _default_state_root()
        managers = {
            gpu_id: MpsManager(
                paths=MpsGpuPaths(
                    state_root=root,
                    gpu_uuid=gpu_uuids[gpu_id],
                ),
                gpu_uuid=gpu_uuids[gpu_id],
                client=client,
            )
            for gpu_id in usable
        }
        return cls(managers, usable, mode=mode)

    def start(self) -> None:
        """Acquire every GPU transactionally, rolling back in reverse order."""

        if self._leases:
            raise MpsError("MPS pipeline runtime is already acquired")
        root = next(iter(self.managers.values())).paths.state_root
        root.mkdir(parents=True, exist_ok=True)
        root.chmod(0o700)
        acquired: list[int] = []
        try:
            for gpu_id, manager in self.managers.items():
                lease = manager.acquire()
                self._leases[gpu_id] = lease
                acquired.append(gpu_id)
                logger.info(
                    "MPS daemon ready on GPU %d (pipe dir %s)",
                    gpu_id,
                    manager.paths.pipe_dir,
                )
        except BaseException:
            for gpu_id in reversed(acquired):
                self._release_one(gpu_id, suppress_errors=True)
            raise
        logger.info(
            "MPS summary: mode=%s %s",
            self._mode,
            {
                gpu_id: {
                    "daemon_pid": self._leases[gpu_id].daemon_pid,
                    "clients": sorted(self._names_on(gpu_id)),
                }
                for gpu_id in self._leases
            },
        )

    def _names_on(self, gpu_id: int) -> list[str]:
        return [name for name, gpu in self._client_gpu.items() if gpu == gpu_id]

    def env_for_process(self, process_name: str) -> dict[str, str]:
        gpu_id = self._client_gpu.get(process_name)
        if gpu_id is None:
            return {}
        env = self.managers[gpu_id].env_for_stage()
        # UUID visibility makes the physical device local ordinal zero.
        env["SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS"] = "true"
        return env

    def verify(self, pids_by_process_name: dict[str, int]) -> None:
        for gpu_id, lease in self._leases.items():
            expected = self._pids_on(gpu_id, pids_by_process_name)
            if expected:
                self.managers[gpu_id].verify(lease, expected)

    def probe_failures(self) -> list[int]:
        return [
            gpu_id
            for gpu_id, lease in self._leases.items()
            if not self.managers[gpu_id].probe(lease)
        ]

    def _pids_on(
        self,
        gpu_id: int,
        pids_by_process_name: dict[str, int],
    ) -> set[int]:
        return {
            pid
            for name, pid in pids_by_process_name.items()
            if self._client_gpu.get(name) == gpu_id
        }

    def stop(
        self,
        preserve_process_names: set[str] | None = None,
        live_pids_by_process_name: dict[str, int] | None = None,
    ) -> None:
        live_pids = live_pids_by_process_name or {}
        preserve_gpus = {
            self._client_gpu[name]
            for name in preserve_process_names or ()
            if name in self._client_gpu
        }
        errors: list[str] = []
        for gpu_id in reversed(list(self._leases)):
            if gpu_id in preserve_gpus:
                process_names = {
                    name
                    for name in preserve_process_names or ()
                    if self._client_gpu.get(name) == gpu_id
                }
                try:
                    report = self.managers[gpu_id].preservation_report(
                        self._leases[gpu_id],
                        process_names,
                        {
                            live_pids[name]
                            for name in process_names
                            if name in live_pids
                        },
                    )
                except MpsError as exc:
                    report = f"could not inspect preserved MPS lease: {exc}"
                logger.error("GPU %d: %s", gpu_id, report)
                errors.append(f"GPU {gpu_id}: {report}")
                continue
            error = self._release_one(gpu_id, suppress_errors=False)
            if error is not None:
                errors.append(f"GPU {gpu_id}: {error}")
        if errors:
            error_type = MpsLeaseRetainedError if self._leases else MpsError
            raise error_type("; ".join(errors))

    def _release_one(self, gpu_id: int, *, suppress_errors: bool) -> str | None:
        lease = self._leases[gpu_id]
        error: str | None = None
        try:
            self.managers[gpu_id].release(lease)
        except MpsError as exc:
            error = str(exc)
            if suppress_errors:
                logger.error("MPS rollback incomplete on GPU %d: %s", gpu_id, exc)
        finally:
            # A released owner fd means the token no longer carries cleanup
            # authority, even when later daemon cleanup failed.
            if lease.owner_fd < 0:
                self._leases.pop(gpu_id, None)
        return error


def create_for_pipeline(mode: str, process_specs) -> MpsPipelineRuntime | None:
    """Build the orchestrator with production device inspection and control I/O."""

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
        logger.warning(
            "CUDA was initialized in the parent before MPS setup; the parent's "
            "own context will run outside MPS"
        )

    from sglang_omni.utils.ipc_weights import get_weight_share_config

    if get_weight_share_config(os.environ) is not None:
        raise MpsError(
            "CUDA IPC weight sharing (launch.sh WEIGHT_SHARE=1) and native mps "
            "cannot be combined; use examples/mps_dp/launch.sh for that deployment "
            "shape"
        )

    from sglang_omni.mps.control import SubprocessMpsControlClient
    from sglang_omni.mps.devices import NvmlDeviceInfo

    return MpsPipelineRuntime.create(
        mode=mode,
        process_specs=process_specs,
        device_info=NvmlDeviceInfo(),
        client=SubprocessMpsControlClient(),
    )
