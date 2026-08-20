# SPDX-License-Identifier: Apache-2.0
"""Spawn-time env injection seam used to hand MPS env to stage processes."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from sglang_omni.pipeline.stage_workers import _patched_spawn_env


@dataclass
class StubStageSpec:
    stage_name: str = "thinker"
    gpu_id: int = 0
    tp_size: int = 1
    env_defaults: dict = field(default_factory=dict)


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
