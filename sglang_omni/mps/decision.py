# SPDX-License-Identifier: Apache-2.0
"""MPS eligibility over resolved physical worker placement."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MPS_MODES = ("off", "on", "auto")
_CUDA_DEVICE = re.compile(r"cuda:(\d+)", re.IGNORECASE)


class MpsDecisionError(ValueError):
    pass


@dataclass(frozen=True)
class MpsGpuPlan:
    gpu_id: int
    client_process_names: tuple[str, ...]


def _process_gpu_ids(process_spec) -> set[int]:
    return {
        int(gpu_id)
        for stage_spec in process_spec.stage_specs
        if (
            gpu_id := (
                stage_spec.placement_gpu_id
                if stage_spec.placement_gpu_id is not None
                else stage_spec.gpu_id
            )
        )
        is not None
    }


def _explicit_cuda_gpu_ids(value) -> set[int]:
    """Find every explicit local CUDA ordinal in resolved launch values."""

    if isinstance(value, str):
        match = _CUDA_DEVICE.fullmatch(value.strip())
        if match is None:
            return set()
        return {int(match.group(1))}
    if isinstance(value, Mapping):
        values = value.values()
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = value
    else:
        return set()
    return {gpu_id for item in values for gpu_id in _explicit_cuda_gpu_ids(item)}


def _process_explicit_cuda_gpu_ids(process_spec) -> set[int]:
    return {
        gpu_id
        for stage_spec in process_spec.stage_specs
        for values in (
            getattr(stage_spec, "factory_args", {}),
            getattr(stage_spec, "factory_arg_defaults", {}),
            getattr(stage_spec, "comm_config", {}),
        )
        for gpu_id in _explicit_cuda_gpu_ids(values)
    }


def _raw_targets(stage_spec) -> set[str]:
    raw_targets = getattr(stage_spec, "next_stages", None)
    if isinstance(raw_targets, str):
        raw_targets = [raw_targets]
    targets = set(raw_targets or ())
    targets.update(getattr(stage_spec, "stream_targets", ()))
    return targets


def _resolved_targets(stage_spec, target: str) -> set[str]:
    replicas = getattr(stage_spec, "replica_topology", {})
    instances = set(replicas.get(target, (target,)))
    same_process = instances & set(
        getattr(stage_spec, "same_process_targets", ())
    )

    # Stages in one replicated Process share one request binding, so a local
    # target resolves to the concrete instance already colocated with source.
    # Targets in another Process have an independent binding and may resolve
    # to any of their instances.
    return same_process or instances


def _cross_gpu_edges(process_specs) -> list[tuple[int, int, str, str]]:
    edges: set[tuple[int, int, str, str]] = set()
    for process_spec in process_specs:
        for stage_spec in process_spec.stage_specs:
            source_gpu = (
                stage_spec.placement_gpu_id
                if stage_spec.placement_gpu_id is not None
                else stage_spec.gpu_id
            )
            if source_gpu is None:
                continue
            remote = set(getattr(stage_spec, "remote_stage_names", ()))
            stage_gpu_ids = getattr(stage_spec, "stage_gpu_ids", {})
            for raw_target in _raw_targets(stage_spec):
                if raw_target in remote:
                    continue
                for target in _resolved_targets(stage_spec, raw_target):
                    if target in remote:
                        continue
                    for target_gpu in stage_gpu_ids.get(target, ()):
                        if target_gpu != source_gpu:
                            edges.add(
                                (
                                    int(source_gpu),
                                    int(target_gpu),
                                    stage_spec.stage_name,
                                    target,
                                )
                            )
    return sorted(edges)


def plan_mps_gpus(process_specs, mode: str) -> list[MpsGpuPlan]:
    """Plan from the already resolved OS-process and physical-placement graph."""

    if mode not in MPS_MODES:
        raise MpsDecisionError(f"invalid mps mode {mode!r}; expected {MPS_MODES}")
    if mode == "off":
        return []

    process_specs = list(process_specs)
    clients_by_gpu: dict[int, list[str]] = {}
    blocked_gpus: dict[int, list[str]] = {}
    for process_spec in process_specs:
        gpu_ids = _process_gpu_ids(process_spec)
        explicit_gpu_ids = _process_explicit_cuda_gpu_ids(process_spec)
        if not gpu_ids:
            for gpu_id in explicit_gpu_ids:
                blocked_gpus.setdefault(gpu_id, []).append(
                    f"process {process_spec.process_name!r} hides an explicit "
                    f"cuda:{gpu_id} device outside resolved placement"
                )
            continue
        if len(gpu_ids) != 1 or any(
            stage_spec.tp_size > 1 for stage_spec in process_spec.stage_specs
        ):
            reason = (
                f"process {process_spec.process_name!r} spans GPUs or contains TP"
            )
            for gpu_id in gpu_ids | explicit_gpu_ids:
                blocked_gpus.setdefault(gpu_id, []).append(reason)
            continue
        (gpu_id,) = gpu_ids
        clients_by_gpu.setdefault(gpu_id, []).append(process_spec.process_name)

        unnormalizable_gpu_ids = explicit_gpu_ids - {0}
        if unnormalizable_gpu_ids:
            reason = (
                f"process {process_spec.process_name!r} contains explicit CUDA "
                f"device(s) {sorted(unnormalizable_gpu_ids)} that cannot be "
                "normalized"
            )
            for blocked_gpu in gpu_ids | unnormalizable_gpu_ids:
                blocked_gpus.setdefault(blocked_gpu, []).append(reason)

    for source_gpu, target_gpu, source, target in _cross_gpu_edges(process_specs):
        reason = (
            f"local CUDA edge {source!r} (GPU {source_gpu}) -> "
            f"{target!r} (GPU {target_gpu}) is not supported by native MPS"
        )
        for gpu_id in {source_gpu, target_gpu}:
            blocked_gpus.setdefault(gpu_id, []).append(reason)

    min_clients = 2 if mode == "auto" else 1
    plans: list[MpsGpuPlan] = []
    for gpu_id, names in sorted(clients_by_gpu.items()):
        if gpu_id in blocked_gpus:
            logger.warning(
                "MPS (%s): skipping GPU %d: %s",
                mode,
                gpu_id,
                "; ".join(blocked_gpus[gpu_id]),
            )
        elif len(names) >= min_clients:
            plans.append(MpsGpuPlan(gpu_id, tuple(names)))

    if mode == "on" and not plans:
        reasons = sorted(
            {reason for gpu_reasons in blocked_gpus.values() for reason in gpu_reasons}
        )
        detail = f": {'; '.join(reasons)}" if reasons else ""
        raise MpsDecisionError(
            "mps=on but no GPU is eligible for MPS (TP stages, multi-GPU "
            f"processes, and CPU-only processes cannot attach){detail}"
        )
    return plans
