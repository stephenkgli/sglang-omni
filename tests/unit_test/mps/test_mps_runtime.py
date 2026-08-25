# SPDX-License-Identifier: Apache-2.0
"""Pipeline-level MPS acquisition, routing, and rollback tests."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from sglang_omni.mps.devices import MpsPhysicalDevice
from sglang_omni.mps.manager import MpsClientRef, MpsError, MpsLeaseRetainedError
from sglang_omni.mps.runtime import MpsPipelineRuntime
from tests.unit_test.mps.test_mps_manager import FakeControlClient


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


@dataclass
class StubProcess:
    process_name: str
    stage_specs: list[StubStage] = field(default_factory=list)


def proc(name, gpu_id, tp_size=1):
    return StubProcess(name, [StubStage(name, gpu_id, tp_size, gpu_id)])


class FakeDeviceInfo:
    def __init__(self, unsupported: dict[int, str] | None = None):
        self.unsupported = unsupported or {}

    def inspect(self, gpu_id):
        return MpsPhysicalDevice(
            f"GPU-aaaaaaaa-bbbb-cccc-dddd-00000000000{gpu_id}",
            self.unsupported.get(gpu_id),
        )


@pytest.fixture
def short_root():
    root = Path(tempfile.mkdtemp(prefix="mpsr-", dir="/tmp"))
    yield root
    shutil.rmtree(root, ignore_errors=True)


def colocated():
    return [proc("a", 0), proc("b", 0), proc("solo", 1)]


def create(short_root, mode="auto", procs=None, unsupported=None, client=None):
    return MpsPipelineRuntime.create(
        mode=mode,
        process_specs=procs if procs is not None else colocated(),
        device_info=FakeDeviceInfo(unsupported),
        client=client or FakeControlClient(),
        state_root=short_root,
    )


def detach_all(runtime, client):
    for manager in runtime.managers.values():
        client.set_clients(manager.paths.pipe_dir, {})


def test_off_creates_nothing(short_root):
    assert create(short_root, mode="off") is None


def test_auto_without_colocation_creates_nothing(short_root):
    assert create(short_root, procs=[proc("a", 0), proc("b", 1)]) is None


def test_env_only_for_acquired_client_processes(short_root):
    client = FakeControlClient()
    runtime = create(short_root, client=client)
    runtime.start()

    env = runtime.env_for_process("a")
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-aaaaaaaa-bbbb-cccc-dddd-000000000000"
    assert "CUDA_MPS_PIPE_DIRECTORY" in env
    assert env["SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS"] == "true"
    assert runtime.env_for_process("solo") == {}
    assert runtime.process_names == {"a", "b"}

    detach_all(runtime, client)
    runtime.stop()


def test_unsupported_gpu_under_auto_downgrades_to_off(short_root):
    assert create(short_root, unsupported={0: "MIG enabled"}) is None


def test_unsupported_gpu_under_on_raises(short_root):
    with pytest.raises(MpsError, match="MIG"):
        create(short_root, mode="on", unsupported={0: "MIG enabled"})


def test_verify_routes_pids_and_retains_exact_refs(short_root):
    client = FakeControlClient()
    runtime = create(short_root, client=client)
    runtime.start()
    manager = runtime.managers[0]
    client.set_clients(manager.paths.pipe_dir, {7000: [11, 12, 99]})

    runtime.verify({"a": 11, "b": 12, "solo": 99})

    assert runtime._leases[0].attached_clients == {
        MpsClientRef(7000, 11),
        MpsClientRef(7000, 12),
    }
    detach_all(runtime, client)
    runtime.stop()


def test_stop_releases_all_acquired_leases(short_root):
    client = FakeControlClient()
    runtime = create(short_root, client=client)
    runtime.start()
    manager = runtime.managers[0]
    client.set_clients(manager.paths.pipe_dir, {7000: [11, 12]})
    runtime.verify({"a": 11, "b": 12})
    client.set_clients(manager.paths.pipe_dir, {})

    runtime.stop()

    assert not runtime.has_leases
    assert not manager.paths.state_dir.exists()


def test_global_pipe_dir_export_is_rejected(short_root, monkeypatch):
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
    with pytest.raises(MpsError, match="CUDA_MPS_PIPE_DIRECTORY"):
        create(short_root)


def test_multi_gpu_start_rolls_back_only_successful_acquisitions(short_root):
    client = FakeControlClient()
    runtime = create(
        short_root,
        mode="on",
        procs=[proc("a", 0), proc("b", 1)],
        client=client,
    )
    dirty = runtime.managers[1].paths
    dirty.pipe_dir.mkdir(parents=True)
    dirty.log_dir.mkdir()
    dirty.owners_dir.mkdir()
    (dirty.owners_dir / "777").write_text("")

    with pytest.raises(MpsError, match="dirty state"):
        runtime.start()

    assert not runtime.has_leases
    assert not runtime.managers[0].paths.state_dir.exists()
    assert (dirty.owners_dir / "777").exists()


def test_runtime_stop_attempts_every_acquired_manager(short_root, monkeypatch):
    client = FakeControlClient()
    runtime = create(
        short_root,
        procs=[proc("a", 0), proc("b", 0), proc("c", 1), proc("d", 1)],
        client=client,
    )
    runtime.start()
    detach_all(runtime, client)
    released: list[int] = []

    def fail_release(lease):
        del lease
        released.append(1)
        raise MpsError("boom")

    original_release_zero = runtime.managers[0].release
    original_release_one = runtime.managers[1].release

    def release_zero(lease):
        released.append(0)
        original_release_zero(lease)

    monkeypatch.setattr(runtime.managers[1], "release", fail_release)
    monkeypatch.setattr(runtime.managers[0], "release", release_zero)

    with pytest.raises(MpsError, match="boom"):
        runtime.stop()
    assert released == [1, 0]

    monkeypatch.setattr(runtime.managers[1], "release", original_release_one)
    runtime.stop()


def test_runtime_preserves_only_the_gpu_with_unsafe_processes(short_root):
    client = FakeControlClient()
    runtime = create(
        short_root,
        procs=[proc("a", 0), proc("b", 0), proc("c", 1), proc("d", 1)],
        client=client,
    )
    runtime.start()
    detach_all(runtime, client)
    client.set_clients(runtime.managers[0].paths.pipe_dir, {7000: [10]})
    runtime.verify({"a": 10})

    with pytest.raises(
        MpsLeaseRetainedError,
        match="automatic process signals are disabled",
    ) as exc:
        runtime.stop({"a"})

    assert set(runtime._leases) == {0}
    assert runtime.managers[0].paths.state_dir.exists()
    assert not runtime.managers[1].paths.state_dir.exists()
    assert "terminate_client" in str(exc.value)
    assert "kill -TERM" in str(exc.value)

    client.set_clients(runtime.managers[0].paths.pipe_dir, {})
    runtime.stop()
    assert not runtime.has_leases


def test_preverify_live_root_captures_only_proven_descendant_refs(short_root):
    client = FakeControlClient()
    runtime = create(short_root, client=client)
    runtime.start()
    manager = runtime.managers[0]
    client.set_clients(manager.paths.pipe_dir, {7000: [200], 8000: [909]})
    client.parents[200] = 100

    with pytest.raises(MpsLeaseRetainedError) as exc_info:
        runtime.stop({"a"}, {"a": 100})

    message = str(exc_info.value)
    assert runtime._leases[0].attached_clients == {MpsClientRef(7000, 200)}
    # A teardown-time snapshot makes the proven client actionable, but it is
    # not equivalent to startup verification of the complete attachment set.
    assert not runtime._leases[0].attachment_verified
    assert "terminate_client 7000 200" in message
    assert "terminate_client 8000 909" not in message

    client.set_clients(manager.paths.pipe_dir, {})
    runtime.stop()
