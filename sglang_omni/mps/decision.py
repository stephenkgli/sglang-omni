# SPDX-License-Identifier: Apache-2.0
"""Per-GPU MPS activation predicate over the resolved process plan.

A GPU gets a private MPS daemon only when every CUDA process on it is a
single-GPU, non-TP client. TP groups are excluded (NCCL x MPS is unvalidated
for our pipelines) and a fused process spanning GPUs cannot be scoped to one
daemon, so its GPUs are excluded too.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MPS_MODES = ("off", "on", "auto")


class MpsDecisionError(ValueError):
    pass


@dataclass(frozen=True)
class MpsGpuPlan:
    gpu_id: int
    client_process_names: tuple[str, ...]


def _process_gpu_ids(process_spec) -> set[int]:
    return {spec.gpu_id for spec in process_spec.stage_specs if spec.gpu_id is not None}


def _is_eligible(process_spec, gpu_ids: set[int]) -> bool:
    if len(gpu_ids) != 1:
        return False
    return all(spec.tp_size <= 1 for spec in process_spec.stage_specs)


def plan_mps_gpus(process_specs, mode: str) -> list[MpsGpuPlan]:
    if mode not in MPS_MODES:
        raise MpsDecisionError(f"invalid mps mode {mode!r}; expected {MPS_MODES}")
    if mode == "off":
        return []

    clients_by_gpu: dict[int, list[str]] = {}
    blocked_gpus: dict[int, list[str]] = {}
    for process_spec in process_specs:
        gpu_ids = _process_gpu_ids(process_spec)
        if not gpu_ids:
            continue
        if _is_eligible(process_spec, gpu_ids):
            (gpu_id,) = gpu_ids
            clients_by_gpu.setdefault(gpu_id, []).append(process_spec.process_name)
        else:
            for gpu_id in gpu_ids:
                blocked_gpus.setdefault(gpu_id, []).append(process_spec.process_name)

    min_clients = 2 if mode == "auto" else 1
    plans: list[MpsGpuPlan] = []
    for gpu_id in sorted(clients_by_gpu):
        names = clients_by_gpu[gpu_id]
        if gpu_id in blocked_gpus:
            logger.warning(
                "MPS (%s): skipping GPU %d; process(es) %s are TP or span GPUs "
                "and cannot attach",
                mode,
                gpu_id,
                blocked_gpus[gpu_id],
            )
            continue
        if len(names) < min_clients:
            continue
        plans.append(MpsGpuPlan(gpu_id=gpu_id, client_process_names=tuple(names)))

    if mode == "on" and not plans:
        raise MpsDecisionError(
            "mps=on but no GPU is eligible for MPS (TP stages, multi-GPU "
            "processes, and CPU-only processes cannot attach)"
        )
    return plans
