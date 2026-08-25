# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-GPU MPS activation predicate."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from sglang_omni.mps.decision import MpsDecisionError, plan_mps_gpus


@dataclass
class StubStage:
    stage_name: str
    gpu_id: int | None
    tp_size: int = 1
    placement_gpu_id: int | None = None
    factory_args: dict = field(default_factory=dict)
    factory_arg_defaults: dict = field(default_factory=dict)
    comm_config: dict = field(default_factory=dict)
    next_stages: str | list[str] | None = None
    stream_targets: list[str] = field(default_factory=list)
    stage_gpu_ids: dict[str, tuple[int, ...]] = field(default_factory=dict)
    remote_stage_names: set[str] = field(default_factory=set)
    replica_topology: dict[str, list[str]] = field(default_factory=dict)
    same_process_targets: set[str] = field(default_factory=set)


@dataclass
class StubProcess:
    process_name: str
    stage_specs: list[StubStage] = field(default_factory=list)


def proc(name, gpu_id, tp_size=1):
    return StubProcess(name, [StubStage(name, gpu_id, tp_size)])


def test_auto_enables_gpu_with_two_single_gpu_processes():
    plans = plan_mps_gpus([proc("a", 0), proc("b", 0), proc("c", 1)], "auto")
    assert len(plans) == 1
    assert plans[0].gpu_id == 0
    assert set(plans[0].client_process_names) == {"a", "b"}


def test_auto_skips_exclusive_gpus():
    assert plan_mps_gpus([proc("a", 0), proc("b", 1)], "auto") == []


def test_auto_excludes_gpu_hosting_tp_process():
    procs = [proc("a", 0), proc("b", 0), proc("tp", 0, tp_size=2)]
    assert plan_mps_gpus(procs, "auto") == []


def test_auto_ignores_cpu_only_processes():
    cpu = StubProcess("cpu", [StubStage("cpu", None)])
    assert plan_mps_gpus([proc("a", 0), cpu], "auto") == []


def test_multi_gpu_fused_process_disables_its_gpus():
    fused = StubProcess("fused", [StubStage("s1", 0), StubStage("s2", 1)])
    procs = [proc("a", 0), proc("b", 0), fused]
    assert plan_mps_gpus(procs, "auto") == []


def test_fused_single_gpu_process_counts_once():
    fused = StubProcess("fused", [StubStage("s1", 0), StubStage("s2", 0)])
    plans = plan_mps_gpus([proc("a", 0), fused], "auto")
    assert len(plans) == 1
    assert set(plans[0].client_process_names) == {"a", "fused"}


def test_off_returns_nothing():
    assert plan_mps_gpus([proc("a", 0), proc("b", 0)], "off") == []


def test_on_enables_even_a_single_process():
    plans = plan_mps_gpus([proc("a", 0)], "on")
    assert [p.gpu_id for p in plans] == [0]


def test_on_with_no_eligible_gpu_raises():
    with pytest.raises(MpsDecisionError, match="no GPU"):
        plan_mps_gpus([proc("tp", 0, tp_size=2)], "on")


def test_uses_resolved_physical_placement():
    placed = proc("placed", 0)
    placed.stage_specs[0].placement_gpu_id = 3

    assert [plan.gpu_id for plan in plan_mps_gpus([placed], "on")] == [3]


@pytest.mark.parametrize(
    "field_name",
    ["factory_args", "factory_arg_defaults", "comm_config"],
)
def test_explicit_nonzero_cuda_device_blocks_every_affected_gpu(field_name):
    processes = [proc("a", 0), proc("b", 0), proc("c", 1), proc("d", 1)]
    setattr(
        processes[0].stage_specs[0],
        field_name,
        {"nested": [{"device": "cuda:1"}]},
    )

    assert plan_mps_gpus(processes, "auto") == []


def test_cuda_zero_remains_valid_after_single_device_normalization():
    processes = [proc("a", 3), proc("b", 3)]
    processes[0].stage_specs[0].factory_args = {"device": "cuda:0"}

    assert [plan.gpu_id for plan in plan_mps_gpus(processes, "auto")] == [3]


def test_unplaced_cuda_zero_blocks_gpu_zero_mps_activation():
    processes = [proc("a", 0), proc("b", 0), proc("hidden", None)]
    processes[-1].stage_specs[0].factory_args = {
        "nested": [{"device": "cuda:0"}]
    }

    assert plan_mps_gpus(processes, "auto") == []


def test_local_cross_gpu_cuda_edge_blocks_both_endpoint_gpus():
    processes = [proc("a", 0), proc("b", 0), proc("c", 1), proc("d", 1)]
    placements = {
        process.process_name: (process.stage_specs[0].gpu_id,)
        for process in processes
    }
    source = processes[0].stage_specs[0]
    source.next_stages = "c"
    source.stage_gpu_ids = placements

    assert plan_mps_gpus(processes, "auto") == []


def test_remote_cross_gpu_edge_is_not_a_local_cuda_transport():
    processes = [proc("a", 0), proc("b", 0), proc("c", 1), proc("d", 1)]
    placements = {
        process.process_name: (process.stage_specs[0].gpu_id,)
        for process in processes
    }
    source = processes[0].stage_specs[0]
    source.next_stages = "c"
    source.stage_gpu_ids = placements
    source.remote_stage_names = {"c"}

    assert [plan.gpu_id for plan in plan_mps_gpus(processes, "auto")] == [0, 1]


def test_same_process_replica_binding_does_not_invent_cross_gpu_edges():
    topology = {"b": ["b@r0", "b@r1"]}
    placements = {
        "a@r0": (0,),
        "b@r0": (0,),
        "extra@r0": (0,),
        "a@r1": (1,),
        "b@r1": (1,),
        "extra@r1": (1,),
    }

    replica_processes = []
    for replica_id, gpu_id in enumerate((0, 1)):
        source = StubStage(
            f"a@r{replica_id}",
            gpu_id,
            placement_gpu_id=gpu_id,
            next_stages="b",
            stage_gpu_ids=placements,
            replica_topology=topology,
            same_process_targets={f"b@r{replica_id}"},
        )
        target = StubStage(
            f"b@r{replica_id}",
            gpu_id,
            placement_gpu_id=gpu_id,
        )
        replica_processes.extend(
            [
                StubProcess(f"pair@r{replica_id}", [source, target]),
                proc(f"extra@r{replica_id}", gpu_id),
            ]
        )

    assert [
        plan.gpu_id for plan in plan_mps_gpus(replica_processes, "auto")
    ] == [0, 1]
