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
from sglang_omni.mps.manager import MpsError, MpsLeaseRetainedError
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
        self.preserved_processes: set[str] = set()
        self.preserved_live_pids: dict[str, int] = {}
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

    def stop(self, preserve_process_names=(), live_pids_by_process_name=None) -> None:
        if self.events is not None:
            self.events.append("MPS release")
        self.stop_calls += 1
        self.preserved_processes = set(preserve_process_names)
        self.preserved_live_pids = dict(live_pids_by_process_name or {})
        if self._stop_gate is not None:
            entered, release = self._stop_gate
            entered.set()
            assert release.wait(timeout=5)
        if self._stop_fails:
            raise MpsError("cleanup is ambiguous")
        if self.preserved_processes:
            raise MpsLeaseRetainedError(
                "lease preserved; follow operator cleanup guidance"
            )
        self.started = False


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
        protected_process_names=(),
    ) -> None:
        del ctx, extra_env_for, protected_process_names
        raise self._exc

    def alive_process_pids(self) -> dict[str, int]:
        return {}

    def dead_process_names(self) -> set[str]:
        return set()

    async def shutdown(
        self,
        join_timeout=30.0,
        preserve_process_names=(),
    ) -> None:
        del join_timeout, preserve_process_names
        self.shutdown_calls += 1


class _ReadyGroup:
    process_specs: list = []
    processes: list = []
    stage_control_endpoints: dict[str, str] = {}
    process_count = 0

    def __init__(self, shutdown_error: Exception | None = None) -> None:
        self.spawn_calls = 0
        self.shutdown_calls = 0
        self.shutdown_error = shutdown_error
        self.preserve_process_names: set[str] = set()
        self.events: list[str] | None = None

    def spawn(
        self,
        ctx,
        extra_env_for=None,
        protected_process_names=(),
    ) -> None:
        del ctx, extra_env_for, protected_process_names
        self.spawn_calls += 1

    async def wait_ready(self, timeout) -> None:
        del timeout

    def any_dead(self) -> bool:
        return False

    def process_pids(self) -> dict[str, int]:
        return {}

    def alive_process_pids(self) -> dict[str, int]:
        if isinstance(self.shutdown_error, StageProcessTeardownError):
            return {name: 101 for name in self.shutdown_error.process_names}
        return {}

    def dead_process_names(self) -> set[str]:
        return set()

    async def shutdown(
        self,
        join_timeout=30.0,
        preserve_process_names=(),
    ) -> None:
        del join_timeout
        if self.events is not None:
            self.events.append("process shutdown")
        self.shutdown_calls += 1
        self.preserve_process_names = set(preserve_process_names)
        if self.shutdown_error is not None:
            raise self.shutdown_error


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
            preserve_process_names=(),
        ) -> None:
            del join_timeout, preserve_process_names
            self.shutdown_calls += 1
            if "a" not in coordinator.shutdown_targets:
                raise StageProcessTeardownError(
                    {"a"},
                    "protected process did not receive graceful shutdown",
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
    assert fake_mps.preserved_processes == set()
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
async def test_alive_mps_process_preserves_its_lease(short_base, monkeypatch):
    fake_mps = _FakeMps()
    error = StageProcessTeardownError({"a"}, "stage process is still alive")
    group = _ReadyGroup(error)
    _patch_runner(monkeypatch, fake_mps, group)
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(short_base))
    await runner.start()

    with pytest.raises(MpsLeaseRetainedError, match="operator cleanup guidance"):
        await runner.stop()

    assert group.preserve_process_names == {"a"}
    assert fake_mps.preserved_processes == {"a"}
    assert fake_mps.preserved_live_pids == {"a": 101}
    assert fake_mps.has_leases
    assert runner.has_retained_mps_leases

    group.shutdown_error = None
    runner.retry_retained_mps_cleanup()

    assert fake_mps.stop_calls == 2
    assert not runner.has_retained_mps_leases


@pytest.mark.asyncio
async def test_dead_mps_root_is_an_unsafe_gpu_disposition(short_base, monkeypatch):
    fake_mps = _FakeMps()

    class DeadGroup(_ReadyGroup):
        dead = False

        def any_dead(self) -> bool:
            return self.dead

        def dead_process_names(self) -> set[str]:
            return {"a"} if self.dead else set()

    group = DeadGroup()
    _patch_runner(monkeypatch, fake_mps, group)
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(short_base))
    await runner.start()
    group.dead = True

    with pytest.raises(MpsLeaseRetainedError):
        await runner.stop()

    assert fake_mps.preserved_processes == {"a"}
    assert fake_mps.preserved_live_pids == {}
    assert runner.has_retained_mps_leases


@pytest.mark.asyncio
async def test_cancelled_stop_keeps_retained_lease_error_as_cause(
    short_base,
    monkeypatch,
):
    entered = threading.Event()
    release = threading.Event()
    fake_mps = _FakeMps(stop_gate=(entered, release))
    group = _ReadyGroup(
        StageProcessTeardownError({"a"}, "stage process is still alive")
    )
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

    assert isinstance(exc_info.value.__cause__, MpsLeaseRetainedError)
    assert runner.has_retained_mps_leases
