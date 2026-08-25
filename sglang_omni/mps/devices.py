# SPDX-License-Identifier: Apache-2.0
"""NVML-backed device info for MPS capability and UUID lookup.

NVML never creates a CUDA context, so querying here keeps the parent process
CUDA-free before stage spawn (a hard requirement for MPS env injection).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_MIN_COMPUTE_CAPABILITY = (7, 0)


@dataclass(frozen=True)
class MpsPhysicalDevice:
    gpu_uuid: str | None
    unsupported_reason: str | None = None


class NvmlDeviceInfo:
    def _handle(self, gpu_id: int):
        import pynvml

        pynvml.nvmlInit()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not visible:
            return pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        entries = [item.strip() for item in visible.split(",")]
        if gpu_id >= len(entries):
            raise ValueError(
                f"gpu_id={gpu_id} is outside CUDA_VISIBLE_DEVICES={entries}"
            )
        entry = entries[gpu_id]
        if entry.startswith("GPU-") or entry.startswith("MIG-"):
            return pynvml.nvmlDeviceGetHandleByUUID(entry.encode())
        return pynvml.nvmlDeviceGetHandleByIndex(int(entry))

    def inspect(self, gpu_id: int) -> MpsPhysicalDevice:
        """Resolve physical identity and MPS support from one NVML handle."""

        try:
            import pynvml
        except ImportError:
            return MpsPhysicalDevice(None, "pynvml is not installed")
        try:
            handle = self._handle(gpu_id)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            gpu_uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
            if gpu_uuid.startswith("MIG-"):
                return MpsPhysicalDevice(
                    gpu_uuid,
                    "MIG devices are not validated for native MPS in SGLang Omni",
                )
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            if (major, minor) < _MIN_COMPUTE_CAPABILITY:
                return MpsPhysicalDevice(
                    gpu_uuid,
                    (
                        f"compute capability {major}.{minor} is pre-Volta; "
                        "per-client isolation requires Volta or newer"
                    ),
                )
            try:
                mig_current, _ = pynvml.nvmlDeviceGetMigMode(handle)
                if mig_current == pynvml.NVML_DEVICE_MIG_ENABLE:
                    return MpsPhysicalDevice(
                        gpu_uuid,
                        (
                            "MIG mode is enabled; native MPS is not validated "
                            "for MIG deployments in SGLang Omni, run with mps=off"
                        ),
                    )
            except pynvml.NVMLError_NotSupported:
                pass
            return MpsPhysicalDevice(gpu_uuid)
        except (pynvml.NVMLError, OSError, ValueError) as exc:
            return MpsPhysicalDevice(None, f"NVML query failed: {exc}")
