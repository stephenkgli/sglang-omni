# SPDX-License-Identifier: Apache-2.0
"""Runner-level MPS lifecycle contract tests."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import threading
from pathlib import Path

import pytest

from sglang_omni.config import EndpointsConfig, PipelineConfig, StageConfig
from sglang_omni.mps.manager import MpsDirtyStateError, MpsError
from sglang_omni.pipeline import mp_runner
from sglang_omni.pipeline.stage_workers import StageProcessTeardownError


@pytest.fixture
def short_base():
    path = Path(tempfile.mkdtemp(prefix="runner-", dir="/tmp"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


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
        self.events: list[str] | None = None
        self.registered: dict[str, str] = {}
        self.shutdown_targets: set[str] = set()

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def shutdown_stages(self) -> None:
        if self.events is not None:
            self.events.append("graceful shutdown")
        self.shutdown_targets.update(self.registered)
        return None

    async def fail_pending_requests(self, error) -> None:
        del error

    async def run_completion_loop(self) -> None:
        await asyncio.Event().wait()

    def register_stage(self, name, endpoint) -> None:
        self.registered[name] = endpoint


class _FakeMps:
    process_names = frozenset({"a"})

    def __init__(
        self,
        start_gate: tuple[threading.Event, threading.Event] | None = None,
        *,
        stop_fails: bool = False,
        stop_gate: tuple[threading.Event, threading.Event] | None = None,
    ):
        self._start_gate = start_gate
        self._stop_fails = stop_fails
        self._stop_gate = stop_gate
        self.started = False
        self.stop_calls = 0
        self.incomplete_processes: set[str] = set()
        self.events: list[str] | None = None

    @property
    def has_leases(self) -> bool:
        return self.started

    def start(self) -> None:
        if self._start_gate is not None:
            entered, release = self._start_gate
            entered.set()
            assert release.wait(timeout=5)
        self.started = True

    def verify(self, pids) -> None:
        del pids

    def probe_failures(self) -> list[int]:
        return []

    def env_for_process(self, process_name: str) -> dict[str, str]:
        del process_name
        return {}

    def stop(self, ownership_incomplete_process_names=()) -> None:
        if self.events is not None:
            self.events.append("MPS release")
        self.stop_calls += 1
        self.incomplete_processes = set(ownership_incomplete_process_names)
        if self._stop_gate is not None:
            entered, release = self._stop_gate
            entered.set()
            assert release.wait(timeout=5)
        self.started = False
        if self._stop_fails:
            raise MpsDirtyStateError("dirty state persisted and owner lock released")


class _FailingGroup:
    process_specs: list = []
    processes: list = []
    stage_control_endpoints: dict[str, str] = {}

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc
        self.shutdown_calls = 0

    def spawn(
        self,
        ctx,
        extra_env_for=None,
    ) -> None:
        del ctx, extra_env_for
        raise self._exc

    def alive_process_pids(self) -> dict[str, int]:
        return {}

    def dead_process_names(self) -> set[str]:
        return set()

    async def shutdown(
        self,
        join_timeout=30.0,
    ) -> None:
        del join_timeout
        self.shutdown_calls += 1


class _ReadyGroup:
    process_specs: list = []
    processes: list = []
    stage_control_endpoints: dict[str, str] = {}
    process_count = 0

    def __init__(
        self,
        shutdown_error: Exception | None = None,
        *,
        forced_process_names=(),
    ) -> None:
        self.spawn_calls = 0
        self.shutdown_calls = 0
        self.shutdown_error = shutdown_error
        self.forced_process_names = set(forced_process_names)
        self._alive_pids = (
            {name: 101 for name in shutdown_error.process_names}
            if isinstance(shutdown_error, StageProcessTeardownError)
            else {}
        )
        self.events: list[str] | None = None

    def spawn(
        self,
        ctx,
        extra_env_for=None,
    ) -> None:
        del ctx, extra_env_for
        self.spawn_calls += 1

    async def wait_ready(self, timeout) -> None:
        del timeout

    def any_dead(self) -> bool:
        return False

    def process_pids(self) -> dict[str, int]:
        return {}

    def alive_process_pids(self) -> dict[str, int]:
        return dict(self._alive_pids)

    def dead_process_names(self) -> set[str]:
        return set()

    async def shutdown(
        self,
        join_timeout=30.0,
    ) -> None:
        del join_timeout
        if self.events is not None:
            self.events.append("process shutdown")
        self.shutdown_calls += 1
        if self.shutdown_error is not None:
            raise self.shutdown_error
        return set(self.forced_process_names)


def _patch_runner(monkeypatch, fake_mps, group):
    events: list[str] = []
    coordinator = _FakeCoordinator()
    coordinator.events = events
    fake_mps.events = events
    if hasattr(group, "events"):
        group.events = events
    monkeypatch.setattr(
        mp_runner,
        "Coordinator",
        lambda *args, **kwargs: coordinator,
    )
    monkeypatch.setattr(mp_runner, "create_for_pipeline", lambda mode, specs: fake_mps)
    monkeypatch.setattr(
        mp_runner,
        "_build_stage_groups",
        lambda *args, **kwargs: [group],
    )
    return events


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc", [asyncio.CancelledError(), RuntimeError("spawn failed")]
)
async def test_startup_exit_paths_release_acquired_mps(short_base, monkeypatch, exc):
    fake_mps = _FakeMps()
    group = _FailingGroup(exc)
    _patch_runner(monkeypatch, fake_mps, group)

    runner = mp_runner.MultiProcessPipelineRunner(_make_config(short_base))
    with pytest.raises(type(exc)):
        await runner.start()

    assert fake_mps.stop_calls == 1
    assert not fake_mps.has_leases
    assert group.shutdown_calls == 1


@pytest.mark.asyncio
async def test_startup_failure_remains_primary_when_rollback_fails(
    short_base, monkeypatch
):
    fake_mps = _FakeMps(stop_fails=True)
    group = _FailingGroup(RuntimeError("spawn failed"))
    _patch_runner(monkeypatch, fake_mps, group)
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(short_base))

    with pytest.raises(RuntimeError, match="spawn failed") as exc_info:
        await runner.start()

    assert isinstance(exc_info.value.__cause__, MpsError)


@pytest.mark.asyncio
async def test_later_spawn_failure_gracefully_stops_registered_mps_group(
    short_base,
    monkeypatch,
):
    coordinator = _FakeCoordinator()
    fake_mps = _FakeMps()

    class StartedGroup(_ReadyGroup):
        stage_control_endpoints = {"a": "ipc://a"}

        async def shutdown(
            self,
            join_timeout=30.0,
        ) -> None:
            del join_timeout
            self.shutdown_calls += 1
            if "a" not in coordinator.shutdown_targets:
                raise StageProcessTeardownError(
                    {"a"},
                    "process did not receive graceful shutdown",
                )

    started = StartedGroup()
    failing = _FailingGroup(RuntimeError("later spawn failed"))
    monkeypatch.setattr(mp_runner, "Coordinator", lambda *a, **k: coordinator)
    monkeypatch.setattr(mp_runner, "create_for_pipeline", lambda mode, specs: fake_mps)
    monkeypatch.setattr(
        mp_runner,
        "_build_stage_groups",
        lambda *args, **kwargs: [started, failing],
    )
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(short_base))

    with pytest.raises(RuntimeError, match="later spawn failed"):
        await runner.start()

    assert coordinator.registered == {"a": "ipc://a"}
    assert started.shutdown_calls == 1
    assert fake_mps.stop_calls == 1
    assert not fake_mps.has_leases


@pytest.mark.asyncio
async def test_cancelled_acquire_finishes_before_rollback(short_base, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    fake_mps = _FakeMps((entered, release))
    group = _ReadyGroup()
    _patch_runner(monkeypatch, fake_mps, group)
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(short_base))

    start_task = asyncio.create_task(runner.start())
    assert await asyncio.to_thread(entered.wait, 1)
    start_task.cancel()
    await asyncio.sleep(0)
    assert fake_mps.stop_calls == 0

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await start_task

    assert fake_mps.stop_calls == 1
    assert not fake_mps.has_leases
    assert group.spawn_calls == 0


@pytest.mark.asyncio
async def test_concurrent_stop_runs_cleanup_once(short_base, monkeypatch):
    fake_mps = _FakeMps()
    group = _ReadyGroup()
    events = _patch_runner(monkeypatch, fake_mps, group)
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(short_base))
    await runner.start()

    await asyncio.gather(runner.stop(), runner.stop())

    assert fake_mps.stop_calls == 1
    assert group.shutdown_calls == 1
    assert events == ["graceful shutdown", "process shutdown", "MPS release"]


@pytest.mark.asyncio
async def test_direct_process_cleanup_error_does_not_skip_mps_release(
    short_base, monkeypatch
):
    fake_mps = _FakeMps()
    error = StageProcessTeardownError({"a"}, "stage process is still alive")
    group = _ReadyGroup(error)
    _patch_runner(monkeypatch, fake_mps, group)
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(short_base))
    await runner.start()

    with pytest.raises(StageProcessTeardownError, match="still alive"):
        await runner.stop()

    assert group.shutdown_calls == 1
    assert fake_mps.stop_calls == 1
    assert not fake_mps.has_leases
    assert not hasattr(runner, "has_retained_mps_leases")


@pytest.mark.asyncio
async def test_forced_mps_worker_exit_marks_ownership_incomplete(
    short_base, monkeypatch
):
    fake_mps = _FakeMps()
    group = _ReadyGroup(forced_process_names={"a", "ordinary"})
    _patch_runner(monkeypatch, fake_mps, group)
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(short_base))
    await runner.start()

    await runner.stop()

    assert fake_mps.incomplete_processes == {"a"}


@pytest.mark.asyncio
async def test_dirty_mps_stop_is_terminal_without_a_runner_hold(short_base, monkeypatch):
    fake_mps = _FakeMps(stop_fails=True)
    group = _ReadyGroup()
    _patch_runner(monkeypatch, fake_mps, group)
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(short_base))
    await runner.start()

    with pytest.raises(MpsDirtyStateError, match="owner lock released"):
        await runner.stop()

    assert fake_mps.stop_calls == 1
    assert not fake_mps.has_leases


@pytest.mark.asyncio
async def test_cancelled_stop_keeps_dirty_state_error_as_cause(
    short_base,
    monkeypatch,
):
    entered = threading.Event()
    release = threading.Event()
    fake_mps = _FakeMps(stop_fails=True, stop_gate=(entered, release))
    group = _ReadyGroup()
    _patch_runner(monkeypatch, fake_mps, group)
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(short_base))
    await runner.start()

    stop_task = asyncio.create_task(runner.stop())
    assert await asyncio.to_thread(entered.wait, 1)
    stop_task.cancel()
    await asyncio.sleep(0)
    release.set()

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await stop_task

    assert isinstance(exc_info.value.__cause__, MpsDirtyStateError)
    assert not fake_mps.has_leases
