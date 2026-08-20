# SPDX-License-Identifier: Apache-2.0
"""Pipeline-level MPS orchestration: one MpsManager per colocated GPU.

Bridges the activation decision (:mod:`sglang_omni.mps.decision`) to per-GPU
daemon lifecycles (:mod:`sglang_omni.mps.manager`) and hands the runner the
per-process env, the attach-verification gate, the watchdog probe, and
teardown.
"""

from __future__ import annotations

import getpass
import logging
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


def _default_state_root() -> Path:
    # Note (Jiaxin Deng): keep this short. The control socket lives under it
    # and must fit the 107-byte AF_UNIX sun_path budget.
    return Path(tempfile.gettempdir()) / f"sglang-omni-mps-{getpass.getuser()}"


class MpsPipelineRuntime:
    def __init__(
        self,
        managers: dict[int, MpsManager],
        plans: dict[int, MpsGpuPlan],
    ):
        self.managers = managers
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
        return cls(managers, usable)

    def start(self) -> None:
        for gpu_id, manager in self.managers.items():
            manager.preflight()
            manager.start()
            logger.info(
                "MPS daemon ready on GPU %d (pipe dir %s)",
                gpu_id,
                manager.paths.pipe_dir,
            )

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

    from sglang_omni.mps.control import SubprocessMpsControlClient
    from sglang_omni.mps.devices import NvmlDeviceInfo

    return MpsPipelineRuntime.create(
        mode=mode,
        process_specs=process_specs,
        device_info=NvmlDeviceInfo(),
        client=SubprocessMpsControlClient(),
    )
