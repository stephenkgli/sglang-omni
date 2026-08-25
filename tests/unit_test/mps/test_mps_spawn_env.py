# SPDX-License-Identifier: Apache-2.0
"""Spawn-time env injection and fail-closed process shutdown tests."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pytest

from sglang_omni.pipeline.stage_workers import (
    StageGroup,
    StageLaunchConfig,
    StageProcessTeardownError,
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


def test_protected_process_is_not_a_multiprocessing_daemon():
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

    group.spawn(context, protected_process_names={"protected"})

    assert [process.daemon for process in context.processes] == [False, True]


@pytest.mark.asyncio
async def test_shutdown_preserves_selected_stuck_process():
    class Process:
        def __init__(self, name, *, protected):
            self.pid = 123 if protected else 456
            self.name = name
            self.alive = True
            self.protected = protected

        def join(self, timeout):
            del timeout

        def is_alive(self):
            return self.alive

        def terminate(self):
            if self.protected:
                raise AssertionError("preserved process was signaled")

        def kill(self):
            if self.protected:
                raise AssertionError("preserved process was signaled")
            self.alive = False

    protected = Process("protected", protected=True)
    ordinary = Process("ordinary", protected=False)
    group = StageGroup(
        "group",
        [
            StageWorkerProcessSpec("protected", []),
            StageWorkerProcessSpec("ordinary", []),
        ],
    )
    group._processes = [protected, ordinary]

    with pytest.raises(StageProcessTeardownError, match="still alive") as exc_info:
        await group.shutdown(join_timeout=0, preserve_process_names={"protected"})

    assert exc_info.value.process_names == {"protected"}
    assert protected.is_alive()
    assert not ordinary.is_alive()
    assert group.processes == [protected, ordinary]
