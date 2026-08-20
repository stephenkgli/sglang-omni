# SPDX-License-Identifier: Apache-2.0
"""Shared primitives for running pipeline replicas under CUDA MPS.

Single source of truth for knowledge that was previously duplicated across
the launcher scripts in examples/mps_dp and the CI runtime in
tests/utils/tts_mps_runtime.py: GPU NUMA core-block derivation, the launcher
state-directory layout, and the AF_UNIX control-socket path limit.
"""

from sglang_omni.mps.state import (
    RUN_ID_PATTERN,
    SUN_PATH_LIMIT,
    MpsRunPaths,
    validate_control_socket,
)
from sglang_omni.mps.topology import derive_core_blocks, format_cpu_list, parse_cpu_list

__all__ = [
    "RUN_ID_PATTERN",
    "SUN_PATH_LIMIT",
    "MpsRunPaths",
    "validate_control_socket",
    "derive_core_blocks",
    "format_cpu_list",
    "parse_cpu_list",
]
