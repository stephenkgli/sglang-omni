# SPDX-License-Identifier: Apache-2.0
"""Tests for the pipeline-level MPS orchestrator (decision -> managers)."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from sglang_omni.mps.manager import MpsError, MpsState
from sglang_omni.mps.runtime import MpsPipelineRuntime
from tests.unit_test.mps.test_mps_manager import FakeControlClient


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


class FakeDeviceInfo:
    def __init__(self, unsupported: dict[int, str] | None = None):
        self.unsupported = unsupported or {}

    def gpu_uuid(self, gpu_id):
        return f"GPU-aaaaaaaa-bbbb-cccc-dddd-00000000000{gpu_id}"

    def unsupported_reason(self, gpu_id):
        return self.unsupported.get(gpu_id)


@pytest.fixture
def short_root():
    root = Path(tempfile.mkdtemp(prefix="mpsr-"))
    yield root
    shutil.rmtree(root, ignore_errors=True)


COLOCATED = [lambda: [proc("a", 0), proc("b", 0), proc("solo", 1)]][0]


def create(short_root, mode="auto", procs=None, unsupported=None, client=None):
    return MpsPipelineRuntime.create(
        mode=mode,
        process_specs=procs if procs is not None else COLOCATED(),
        device_info=FakeDeviceInfo(unsupported),
        client=client or FakeControlClient(),
        state_root=short_root,
    )


def test_off_creates_nothing(short_root):
    assert create(short_root, mode="off") is None


def test_auto_without_colocation_creates_nothing(short_root):
    assert create(short_root, procs=[proc("a", 0), proc("b", 1)]) is None


def test_env_only_for_client_processes(short_root):
    client = FakeControlClient()
    runtime = create(short_root, client=client)
    runtime.start()

    env = runtime.env_for_process("a")
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-aaaaaaaa-bbbb-cccc-dddd-000000000000"
    assert "CUDA_MPS_PIPE_DIRECTORY" in env
    assert env["SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS"] == "true"
    assert runtime.env_for_process("solo") == {}


def test_unsupported_gpu_under_auto_downgrades_to_off(short_root):
    assert create(short_root, unsupported={0: "MIG enabled"}) is None


def test_unsupported_gpu_under_on_raises(short_root):
    with pytest.raises(MpsError, match="MIG"):
        create(short_root, mode="on", unsupported={0: "MIG enabled"})


def test_verify_routes_pids_to_the_gpu_manager(short_root):
    client = FakeControlClient()
    runtime = create(short_root, client=client)
    runtime.start()

    client.servers = {7000: [11, 12]}
    runtime.verify({"a": 11, "b": 12, "solo": 99})
    assert all(m.state is MpsState.SERVING for m in runtime.managers.values())


def test_stop_cleans_all_state(short_root):
    client = FakeControlClient()
    runtime = create(short_root, client=client)
    runtime.start()
    client.servers = {7000: [11, 12]}
    runtime.verify({"a": 11, "b": 12})

    client.servers = {}
    runtime.stop()
    assert all(m.state is MpsState.CLEANED for m in runtime.managers.values())


def test_global_pipe_dir_export_is_rejected(short_root, monkeypatch):
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
    with pytest.raises(MpsError, match="CUDA_MPS_PIPE_DIRECTORY"):
        create(short_root)
