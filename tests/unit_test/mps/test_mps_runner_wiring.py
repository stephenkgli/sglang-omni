# SPDX-License-Identifier: Apache-2.0
"""Runner-level MPS wiring: teardown must run on every startup exit path."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from sglang_omni.config import EndpointsConfig, PipelineConfig, StageConfig
from sglang_omni.pipeline import mp_runner


def noop_factory():  # pragma: no cover - never constructed in these tests
    raise AssertionError("factory must not run")


def _make_config(base_path: Path) -> PipelineConfig:
    return PipelineConfig(
        model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        entry_stage="preprocessing",
        mps="auto",
        stages=[
            StageConfig(
                name="preprocessing",
                process="pipeline",
                factory=f"{__name__}.noop_factory",
                terminal=True,
            )
        ],
        endpoints=EndpointsConfig(base_path=str(base_path)),
    )


class _FakeCoordinator:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def run_completion_loop(self) -> None:
        return None


class _FakeMps:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def env_for_process(self, process_name: str) -> dict[str, str]:
        return {}

    def stop_best_effort(self) -> None:
        self.stopped = True


class _FakeGroup:
    process_specs: list = []
    processes: list = []

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc
        self.channels_closed = False

    def spawn(self, ctx, extra_env_for=None) -> None:
        del ctx, extra_env_for
        raise self._exc

    def close_control_channels(self) -> None:
        self.channels_closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc", [asyncio.CancelledError(), RuntimeError("spawn failed")]
)
async def test_startup_exit_paths_stop_mps(tmp_path, monkeypatch, exc):
    fake_mps = _FakeMps()
    group = _FakeGroup(exc)
    monkeypatch.setattr(mp_runner, "Coordinator", _FakeCoordinator)
    monkeypatch.setattr(mp_runner, "create_for_pipeline", lambda mode, specs: fake_mps)
    monkeypatch.setattr(mp_runner, "_build_stage_groups", lambda *a, **k: [group])

    runner = mp_runner.MultiProcessPipelineRunner(_make_config(tmp_path))
    with pytest.raises(type(exc)):
        await runner.start()

    assert fake_mps.started
    assert fake_mps.stopped
