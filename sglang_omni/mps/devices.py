# SPDX-License-Identifier: Apache-2.0
"""NVML-backed device info for MPS capability and UUID lookup.

NVML never creates a CUDA context, so querying here keeps the parent process
CUDA-free before stage spawn (a hard requirement for MPS env injection).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_MIN_COMPUTE_CAPABILITY = (7, 0)


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

    def gpu_uuid(self, gpu_id: int) -> str:
        import pynvml

        uuid = pynvml.nvmlDeviceGetUUID(self._handle(gpu_id))
        return uuid.decode() if isinstance(uuid, bytes) else uuid

    def unsupported_reason(self, gpu_id: int) -> str | None:
        try:
            import pynvml
        except ImportError:
            return "pynvml is not installed"
        try:
            handle = self._handle(gpu_id)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            if (major, minor) < _MIN_COMPUTE_CAPABILITY:
                return (
                    f"compute capability {major}.{minor} is pre-Volta; "
                    "per-client isolation requires Volta or newer"
                )
            try:
                mig_current, _ = pynvml.nvmlDeviceGetMigMode(handle)
                if mig_current == pynvml.NVML_DEVICE_MIG_ENABLE:
                    return (
                        "MIG mode is enabled; native MPS is not validated "
                        "for MIG deployments in SGLang Omni, run with mps=off"
                    )
            except pynvml.NVMLError_NotSupported:
                pass
            return None
        except Exception as exc:
            return f"NVML query failed: {exc}"
