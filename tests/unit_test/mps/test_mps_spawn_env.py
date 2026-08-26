# SPDX-License-Identifier: Apache-2.0
"""Spawn-time env injection and fail-closed process shutdown tests."""

from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from sglang_omni.pipeline.stage_workers import (
    StageGroup,
    StageLaunchConfig,
    StageWorkerProcessSpec,
    _patched_spawn_env,
)


@dataclass
class StubStageSpec:
    stage_name: str = "thinker"
    gpu_id: int | None = 0
    tp_size: int = 1
    tp_rank: int = 0
    placement_gpu_id: int | None = None
    env_defaults: dict = field(default_factory=dict)
    factory_arg_defaults: dict = field(default_factory=dict)
    comm_config: dict = field(default_factory=dict)


@dataclass
class StubProcessSpec:
    process_name: str = "thinker"
    stage_specs: list = field(default_factory=lambda: [StubStageSpec()])


def test_extra_env_visible_during_spawn_and_restored_after():
    spec = StubProcessSpec()
    extra = {
        "CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps/pipe",
        "CUDA_VISIBLE_DEVICES": "GPU-abc",
    }
    assert "CUDA_MPS_PIPE_DIRECTORY" not in os.environ
    before_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")

    with _patched_spawn_env(spec, extra_env=extra):
        assert os.environ["CUDA_MPS_PIPE_DIRECTORY"] == "/tmp/mps/pipe"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-abc"

    assert "CUDA_MPS_PIPE_DIRECTORY" not in os.environ
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == before_cvd


def test_no_extra_env_keeps_existing_behavior():
    spec = StubProcessSpec()
    with _patched_spawn_env(spec):
        assert "CUDA_MPS_PIPE_DIRECTORY" not in os.environ


def test_cpu_stage_keeps_none_gpu_id_under_single_device_marker(monkeypatch):
    import logging

    from sglang_omni.pipeline.stage_workers import _prepare_accelerator_environment

    spec = StubStageSpec(stage_name="preprocessing")
    spec.gpu_id = None
    monkeypatch.setenv("SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS", "true")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc")

    _prepare_accelerator_environment(spec, logging.getLogger("test"))
    assert spec.gpu_id is None


def test_spawned_processes_share_one_explicitly_owned_lifecycle():
    class Queue:
        def close(self):
            return None

        def join_thread(self):
            return None

    class Process:
        pid = 123

        def __init__(self, *, daemon, **kwargs):
            del kwargs
            self.daemon = daemon

        def start(self):
            return None

    class Context:
        def __init__(self):
            self.processes = []

        def Event(self):
            return object()

        def Queue(self):
            return Queue()

        def Process(self, **kwargs):
            process = Process(**kwargs)
            self.processes.append(process)
            return process

    context = Context()
    group = StageGroup(
        "group",
        [
            StageWorkerProcessSpec(
                "protected", [StageLaunchConfig(stage_name="protected")]
            ),
            StageWorkerProcessSpec("ordinary", [StageLaunchConfig("ordinary")]),
        ],
    )

    group.spawn(context)

    assert [process.daemon for process in context.processes] == [True, True]


@pytest.mark.asyncio
async def test_shutdown_terminates_and_reaps_every_direct_survivor():
    events: list[int] = []

    class Process:
        def __init__(self, name, pid):
            self.pid = pid
            self.name = name
            self.alive = True

        def join(self, timeout):
            del timeout

        def is_alive(self):
            return self.alive

        def terminate(self):
            events.append(self.pid)
            self.alive = False

        def kill(self):
            raise AssertionError("terminate should have stopped the process")

    first = Process("first", 123)
    second = Process("second", 456)
    group = StageGroup(
        "group",
        [
            StageWorkerProcessSpec("first", []),
            StageWorkerProcessSpec("second", []),
        ],
    )
    group._processes = [first, second]

    await group.shutdown(join_timeout=0)

    assert events == [123, 456]
    assert group.processes == []


@pytest.mark.asyncio
async def test_shutdown_continues_after_one_direct_process_cleanup_fails():
    terminated: list[int] = []

    class Process:
        def __init__(self, pid, fail=False):
            self.pid = pid
            self.name = str(pid)
            self.fail = fail
            self.alive = True

        def join(self, timeout):
            del timeout

        def is_alive(self):
            return self.alive

        def terminate(self):
            terminated.append(self.pid)
            if self.fail:
                raise RuntimeError("cannot terminate")
            self.alive = False

        def kill(self):
            raise AssertionError("terminate should have stopped the process")

    first = Process(123, fail=True)
    second = Process(456)
    group = StageGroup(
        "group",
        [
            StageWorkerProcessSpec("first", []),
            StageWorkerProcessSpec("second", []),
        ],
    )
    group._processes = [first, second]

    with pytest.raises(RuntimeError, match="cannot terminate"):
        await group.shutdown(join_timeout=0)

    assert terminated == [123, 456]
    assert first.is_alive()
    assert not second.is_alive()


def test_dirty_parent_exits_nonzero_after_reaping_its_direct_worker():
    script = """
import multiprocessing
import sys
import time
from pathlib import Path

from sglang_omni.pipeline.stage_workers import _terminate_process
from tests.unit_test.mps.test_mps_manager import FakeControlClient, make_manager

root = Path(sys.argv[1])
client = FakeControlClient()
manager = make_manager(root, client)
lease = manager.acquire()
client.set_clients(manager.paths.pipe_dir, {7000: [101]})
manager.verify(lease, {101})
ctx = multiprocessing.get_context("spawn")
worker = ctx.Process(target=time.sleep, args=(60,), daemon=True)
worker.start()
_terminate_process(worker)
assert not worker.is_alive()
print("worker-reaped", flush=True)
manager.release(lease)
"""

    root = Path(tempfile.mkdtemp(prefix="mps-sub-", dir="/tmp"))
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, str(root)],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )

        assert result.returncode != 0
        assert "worker-reaped" in result.stdout, result.stderr
        assert "MpsDirtyStateError" in result.stderr
        assert "lock is released" in result.stderr

        owner_file = next(root.glob("*/owners/*"))
        assert owner_file.read_text() == "retained\n"
        owner_fd = os.open(owner_file, os.O_RDWR)
        try:
            fcntl.flock(owner_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(owner_fd)
    finally:
        shutil.rmtree(root, ignore_errors=True)
