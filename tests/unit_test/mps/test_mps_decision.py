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
