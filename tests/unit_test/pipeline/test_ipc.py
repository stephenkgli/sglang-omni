# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import signal
import threading
from pathlib import Path
from types import FrameType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import sglang_omni.pipeline.mp_runner as mp_runner
import sglang_omni.pipeline.runtime_config as runtime_config
from sglang_omni.config.schema import EndpointsConfig, PipelineConfig, StageConfig
from sglang_omni.mps.manager import MpsLeaseRetainedError
from sglang_omni.profiler.event_recorder import get_recorder
from tests.unit_test.fixtures.pipeline_fakes import FakeMpContext, FakeRelay


def noop_factory():
    return None


def failing_factory():
    raise RuntimeError("factory boom")


class _FakeControlPlane:
    def __init__(self, recv_endpoint: str):
        self.recv_endpoint = recv_endpoint


class _FakeStage:
    name = "preprocessing"

    def __init__(self, recv_endpoint: str):
        self.control_plane = _FakeControlPlane(recv_endpoint)

    async def run(self) -> None:
        await asyncio.Event().wait()


class _FakeCoordinator:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.started = False
        self.stopped = False
        self.registered: dict[str, str] = {}

    async def start(self) -> None:
        self.started = True

    async def run_completion_loop(self) -> None:
        await asyncio.Event().wait()

    def register_stage(self, name: str, endpoint: str) -> None:
        self.registered[name] = endpoint

    async def shutdown_stages(self) -> None:
        return None

    async def stop(self) -> None:
        self.stopped = True


def _make_config(base_path: Path) -> PipelineConfig:
    return PipelineConfig(
        model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        entry_stage="preprocessing",
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


@pytest.fixture(autouse=True)
def _fake_stage_relay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sglang_omni.comm.router.create_relay",
        lambda relay_type, **kwargs: FakeRelay(device=kwargs.get("device", "cpu")),
    )


def test_ipc_runtime_dir_creation_and_close_contracts(tmp_path: Path) -> None:
    """Preserves IPC runtime directory creation, uniqueness, and idempotent cleanup."""
    ipc_config = _make_config(tmp_path)

    runtime_a = runtime_config.create_ipc_runtime_dir(ipc_config)
    runtime_b = runtime_config.create_ipc_runtime_dir(ipc_config)
    assert runtime_a is not None
    assert runtime_b is not None
    assert runtime_a.path != runtime_b.path

    runtime_path = runtime_a.path
    runtime_a.close()
    runtime_a.close()
    runtime_b.close()
    assert not runtime_path.exists()
    assert list(tmp_path.iterdir()) == []


def test_prepare_pipeline_runtime_owns_or_preserves_ipc_runtime_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserves owned IPC cleanup and caller-owned IPC directory preservation."""
    config = _make_config(tmp_path)

    def fail_allocate_endpoints(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("boom")

    monkeypatch.setattr(runtime_config, "allocate_endpoints", fail_allocate_endpoints)

    with pytest.raises(RuntimeError, match="boom"):
        runtime_config.prepare_pipeline_runtime(config)
    assert list(tmp_path.iterdir()) == []

    caller_owned = runtime_config.create_ipc_runtime_dir(config)
    assert caller_owned is not None
    caller_path = caller_owned.path
    with pytest.raises(RuntimeError, match="boom"):
        runtime_config.prepare_pipeline_runtime(config, ipc_runtime_dir=caller_owned)
    assert caller_path.exists()
    caller_owned.close()
    assert list(tmp_path.iterdir()) == []


def test_prepare_pipeline_runtime_returns_managed_ipc_runtime_dir(
    tmp_path: Path,
) -> None:
    """Preserves managed IPC runtime directory ownership in runtime prep."""
    prep = runtime_config.prepare_pipeline_runtime(_make_config(tmp_path))
    runtime_dir = prep.runtime_dir
    assert runtime_dir is not None
    try:
        assert runtime_dir.path.exists()
        assert str(runtime_dir.path) in prep.endpoints["stage_preprocessing"]
    finally:
        runtime_dir.close()

    assert list(tmp_path.iterdir()) == []


def test_ipc_stage_groups_use_unique_endpoints_for_same_model_name(
    tmp_path: Path,
) -> None:
    """Preserves unique IPC endpoints across same-model pipeline instances."""
    config = _make_config(tmp_path)
    prep_a = runtime_config.prepare_pipeline_runtime(config)
    prep_b = runtime_config.prepare_pipeline_runtime(config)
    assert prep_a.runtime_dir is not None
    assert prep_b.runtime_dir is not None

    try:
        groups_a = mp_runner._build_stage_groups(
            config,
            FakeMpContext(),
            stages_cfg=prep_a.stages_cfg,
            endpoints=prep_a.endpoints,
            placement_plan=prep_a.placement_plan,
            process_plan=prep_a.process_plan,
        )
        groups_b = mp_runner._build_stage_groups(
            config,
            FakeMpContext(),
            stages_cfg=prep_b.stages_cfg,
            endpoints=prep_b.endpoints,
            placement_plan=prep_b.placement_plan,
            process_plan=prep_b.process_plan,
        )

        assert prep_a.endpoints["completion"] != prep_b.endpoints["completion"]
        assert groups_a[0].leader_endpoint != groups_b[0].leader_endpoint
    finally:
        prep_a.runtime_dir.close()
        prep_b.runtime_dir.close()

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_mp_runner_cleans_runtime_dir_on_start_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserves IPC runtime directory cleanup when runner startup fails."""

    class FailingCoordinator:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        async def start(self) -> None:
            raise RuntimeError("boom")

        async def stop(self) -> None:
            return None

    monkeypatch.setattr(mp_runner, "Coordinator", FailingCoordinator)
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(tmp_path))

    with pytest.raises(RuntimeError, match="boom"):
        await runner.start()

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_mp_runner_cleans_spawned_groups_when_later_spawn_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserves spawned process cleanup if a later stage group fails to spawn."""

    class FakeProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False
            self.join_count = 0
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

        def terminate(self) -> None:
            self.terminated = True
            self._alive = False

        def kill(self) -> None:
            self.killed = True
            self._alive = False

        def join(self, timeout=None) -> None:
            del timeout
            self.join_count += 1

    class FakeGroup:
        def __init__(self, stage_name: str, *, fail_spawn: bool = False) -> None:
            self.stage_name = stage_name
            self.fail_spawn = fail_spawn
            self.process = FakeProcess() if not fail_spawn else None
            self.channels_closed = False
            self.stage_control_endpoints = {
                stage_name: f"ipc://{stage_name}.sock"
            }

        @property
        def processes(self) -> list[FakeProcess]:
            return [self.process] if self.process is not None else []

        def spawn(self, ctx) -> None:
            del ctx
            if self.fail_spawn:
                raise RuntimeError(f"spawn failed for {self.stage_name}")

        async def wait_ready(self, timeout: float) -> None:
            del timeout

        def alive_process_pids(self) -> dict[str, int]:
            return {}

        def dead_process_names(self) -> set[str]:
            return set()

        def close_control_channels(self) -> None:
            self.channels_closed = True

        async def shutdown(
            self,
            join_timeout: float = 30.0,
            preserve_process_names=(),
        ) -> None:
            del join_timeout, preserve_process_names
            if self.process is not None and self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
            self.close_control_channels()

    first_group = FakeGroup("preprocessing")
    second_group = FakeGroup("thinker", fail_spawn=True)
    monkeypatch.setattr(mp_runner, "Coordinator", _FakeCoordinator)
    monkeypatch.setattr(
        mp_runner,
        "_build_stage_groups",
        lambda *a, **k: [first_group, second_group],
    )

    runner = mp_runner.MultiProcessPipelineRunner(_make_config(tmp_path))
    with pytest.raises(RuntimeError, match="spawn failed"):
        await runner.start()

    assert first_group.process.terminated
    assert first_group.process.join_count >= 1
    assert first_group.channels_closed
    assert second_group.channels_closed
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_mp_runner_startup_failure_includes_child_factory_traceback(
    tmp_path: Path,
) -> None:
    config = PipelineConfig(
        model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        name="x",
        entry_stage="preprocessing",
        stages=[
            StageConfig(
                name="preprocessing",
                process="pipeline",
                factory=f"{__name__}.failing_factory",
                terminal=True,
            )
        ],
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
    )
    runner = mp_runner.MultiProcessPipelineRunner(config)

    with pytest.raises(RuntimeError, match="factory boom"):
        await runner.start(timeout=10.0)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_mp_runner_stop_cleans_runtime_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserves IPC runtime directory cleanup when the runner stops."""

    class FakeCoordinator:
        def __init__(
            self,
            completion_endpoint: str,
            abort_endpoint: str,
            entry_stage: str,
            terminal_stages: list[str] | None = None,
            terminal_stages_resolver=None,
            replica_topology=None,
            logical_process_plan=None,
            max_in_flight=None,
        ) -> None:
            del (
                abort_endpoint,
                entry_stage,
                terminal_stages,
                terminal_stages_resolver,
                replica_topology,
                logical_process_plan,
                max_in_flight,
            )
            self.control_plane = SimpleNamespace(
                completion_endpoint=completion_endpoint
            )

        async def start(self) -> None:
            return None

        async def run_completion_loop(self) -> None:
            await asyncio.Event().wait()

        def register_stage(self, name: str, endpoint: str) -> None:
            del name, endpoint

        async def shutdown_stages(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    class FakeGroup:
        stage_name = "preprocessing"
        leader_endpoint = "ipc://stage.sock"
        tp_size = 1
        process_count = 1
        processes: list[object] = []
        stage_control_endpoints = {"preprocessing": "ipc://stage.sock"}

        def __init__(self) -> None:
            self.shutdown_called = False

        def spawn(self, ctx) -> None:
            del ctx

        async def wait_ready(self, timeout: float) -> None:
            del timeout

        def any_dead(self) -> bool:
            return False

        def dead_summary(self) -> str:
            return "(none)"

        def alive_process_pids(self) -> dict[str, int]:
            return {}

        def dead_process_names(self) -> set[str]:
            return set()

        async def shutdown(
            self,
            join_timeout: float = 30.0,
            preserve_process_names=(),
        ) -> None:
            del join_timeout, preserve_process_names
            self.shutdown_called = True

    group = FakeGroup()
    monkeypatch.setattr(mp_runner, "Coordinator", FakeCoordinator)
    monkeypatch.setattr(mp_runner, "_build_stage_groups", lambda *a, **k: [group])

    runner = mp_runner.MultiProcessPipelineRunner(_make_config(tmp_path))
    await runner.start()
    assert len([path for path in tmp_path.iterdir() if path.is_dir()]) == 1

    await runner.stop()

    assert group.shutdown_called
    assert list(tmp_path.iterdir()) == []


async def _run_launcher_with_fake_runner(
    *,
    config: PipelineConfig,
    serve_mock: AsyncMock | None,
    monkeypatch: pytest.MonkeyPatch,
    stop_error: BaseException | None = None,
    wait_error: Exception | None = None,
    retained_leases: bool = False,
    retained_lease_handler=None,
) -> tuple[object, FastAPI, SimpleNamespace]:
    app = FastAPI()
    profiler_calls = SimpleNamespace(starts=[], stops=[])

    from sglang_omni.serve import launcher

    runner_ref = None

    class FakeRunner:
        def __init__(self, pipeline_config: PipelineConfig) -> None:
            del pipeline_config
            nonlocal runner_ref
            self.coordinator = _FakeCoordinator()
            self.stage_control_endpoints = {
                "preprocessing": "ipc://stage_preprocessing.sock"
            }
            self.started = False
            self.stopped = False
            self.has_retained_mps_leases = retained_leases
            # launcher._run_server reads .prep.placement_plan / .process_plan
            # after start() to log the resolved topology. Provide empty stubs
            # that satisfy _placement_log_summary's attribute access.
            self.prep = SimpleNamespace(
                placement_plan=SimpleNamespace(gpus={}),
                process_plan=SimpleNamespace(
                    groups=(),
                    tp_stage_to_processes={},
                ),
            )
            runner_ref = self

        async def start(self, timeout: float) -> None:
            del timeout
            self.started = True

        async def stop(self) -> None:
            self.stopped = True
            if stop_error is not None:
                raise stop_error

        def retry_retained_mps_cleanup(self) -> None:
            self.has_retained_mps_leases = False

        async def wait_failed(self) -> None:
            if wait_error is not None:
                raise wait_error
            await asyncio.Future()

    class FakeProfilerControl:
        def __init__(self, stage_control_endpoints: dict[str, str]) -> None:
            del stage_control_endpoints

        async def broadcast_start(self, **kwargs) -> None:
            profiler_calls.starts.append(kwargs)

        async def broadcast_stop(self, **kwargs) -> None:
            profiler_calls.stops.append(kwargs)

    monkeypatch.setattr(launcher, "_find_available_port", lambda host, port: port)
    monkeypatch.setattr(launcher, "MultiProcessPipelineRunner", FakeRunner)
    monkeypatch.setattr(launcher, "ProfilerControlClient", FakeProfilerControl)
    monkeypatch.setattr(launcher, "create_app", lambda *a, **k: app)
    if serve_mock is not None:
        monkeypatch.setattr(launcher.uvicorn.Server, "serve", serve_mock)

    await launcher._run_server(
        config,
        port=8000,
        _retained_lease_handler=retained_lease_handler,
    )
    assert runner_ref is not None
    return runner_ref, app, profiler_calls


@pytest.mark.asyncio
async def test_launcher_uses_runner_and_mounts_profiler_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    server_serve = AsyncMock(return_value=None)
    retained_errors: list[MpsLeaseRetainedError] = []

    runner, app, profiler_calls = await _run_launcher_with_fake_runner(
        config=config,
        serve_mock=server_serve,
        monkeypatch=monkeypatch,
        retained_lease_handler=retained_errors.append,
    )

    assert runner.started
    assert runner.stopped
    assert retained_errors == []
    server_serve.assert_awaited_once()
    try:
        with TestClient(app) as client:
            start_resp = client.post(
                "/start_profile",
                json={
                    "enable_torch": False,
                    "event_dir": str(tmp_path / "events"),
                },
            )
            stop_resp = client.post("/stop_profile", json={})
        assert start_resp.status_code == 200
        assert stop_resp.status_code == 200
        assert profiler_calls.starts
        assert profiler_calls.starts[0]["enable_torch"] is False
        assert profiler_calls.starts[0]["event_dir"] == str(tmp_path / "events")
        assert profiler_calls.stops == [{"run_id": None}]
    finally:
        rec = get_recorder()
        if rec.is_active():
            rec.stop()


def test_start_profile_request_only_mode_does_not_require_trace_template(
    tmp_path: Path,
) -> None:
    from sglang_omni.serve import launcher

    class FakeProfilerControl:
        def __init__(self) -> None:
            self.starts: list[dict] = []

        async def broadcast_start(self, **kwargs) -> None:
            self.starts.append(kwargs)

    app = FastAPI()
    ctl = FakeProfilerControl()
    launcher._mount_profiler_routes(app, ctl, profiler_dir=None)
    event_dir = str(tmp_path / "events")

    try:
        with TestClient(app) as client:
            resp = client.post(
                "/start_profile",
                json={"enable_torch": False, "event_dir": event_dir},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["enable_torch"] is False
        assert body["trace_path_template"] == ""
        assert body["event_dir"] == event_dir
        assert ctl.starts
        assert ctl.starts[0]["enable_torch"] is False
        assert ctl.starts[0]["trace_path_template"] == ""
        assert ctl.starts[0]["event_dir"] == event_dir
    finally:
        rec = get_recorder()
        if rec.is_active():
            rec.stop()


def test_start_profile_torch_mode_still_requires_trace_template() -> None:
    from sglang_omni.serve import launcher

    class FakeProfilerControl:
        async def broadcast_start(self, **kwargs) -> None:
            raise AssertionError("start_profile should fail before broadcasting")

    app = FastAPI()
    launcher._mount_profiler_routes(app, FakeProfilerControl(), profiler_dir=None)

    with TestClient(app) as client:
        resp = client.post("/start_profile", json={"enable_torch": True})
    assert resp.status_code == 400
    assert "trace_path_template is required" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_launcher_stops_runner_when_server_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    server_serve = AsyncMock(side_effect=RuntimeError("server failed"))

    with pytest.raises(RuntimeError, match="server failed"):
        await _run_launcher_with_fake_runner(
            config=config,
            serve_mock=server_serve,
            monkeypatch=monkeypatch,
        )

    server_serve.assert_awaited_once()


@pytest.mark.asyncio
async def test_launcher_preserves_runtime_failure_when_cleanup_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)

    async def serve_until_failure_is_observed():
        await asyncio.sleep(0.05)

    server_serve = AsyncMock(side_effect=serve_until_failure_is_observed)
    retained_errors: list[MpsLeaseRetainedError] = []

    with pytest.raises(RuntimeError, match="stage died") as exc_info:
        await _run_launcher_with_fake_runner(
            config=config,
            serve_mock=server_serve,
            monkeypatch=monkeypatch,
            stop_error=RuntimeError("cleanup failed"),
            wait_error=RuntimeError("stage died"),
            retained_lease_handler=retained_errors.append,
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "cleanup failed"
    assert retained_errors == []
    server_serve.assert_awaited_once()


@pytest.mark.asyncio
async def test_cli_holds_owner_when_cleanup_retains_an_mps_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    server_serve = AsyncMock(side_effect=RuntimeError("stage died"))
    retained = MpsLeaseRetainedError(
        "owner PID 123 holds /tmp/state; current client refs: "
        "[MpsClientRef(server_pid=7, client_pid=9)]; terminate_client 7 9"
    )

    class OwnerHeld(RuntimeError):
        pass

    handled: list[MpsLeaseRetainedError] = []

    def hold_owner(error, retry, retry_requested) -> None:
        del retry, retry_requested
        handled.append(error)
        raise OwnerHeld("test stopped the otherwise non-returning CLI hold")

    with pytest.raises(OwnerHeld, match="non-returning CLI hold"):
        await _run_launcher_with_fake_runner(
            config=config,
            serve_mock=server_serve,
            monkeypatch=monkeypatch,
            stop_error=retained,
            retained_leases=True,
            retained_lease_handler=hold_owner,
        )

    assert handled == [retained]


@pytest.mark.asyncio
async def test_pipeline_uvicorn_server_consumes_handled_sigterm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.serve import launcher

    config = _make_config(tmp_path)
    replayed_signals: list[int] = []
    server_ref: launcher.uvicorn.Server | None = None
    original_handler = signal.getsignal(signal.SIGTERM)

    def recording_handler(sig: int, frame: FrameType | None) -> None:
        del frame
        replayed_signals.append(sig)

    async def serve_until_sigterm(
        server: launcher.uvicorn.Server,
        sockets=None,
    ) -> None:
        del sockets
        nonlocal server_ref
        server_ref = server
        signal.raise_signal(signal.SIGTERM)
        assert server.should_exit

    monkeypatch.setattr(launcher.uvicorn.Server, "_serve", serve_until_sigterm)
    signal.signal(signal.SIGTERM, recording_handler)
    try:
        runner, _, _ = await _run_launcher_with_fake_runner(
            config=config,
            serve_mock=None,
            monkeypatch=monkeypatch,
        )
        assert signal.getsignal(signal.SIGTERM) is recording_handler
    finally:
        signal.signal(signal.SIGTERM, original_handler)

    assert isinstance(server_ref, launcher._PipelineUvicornServer)
    assert runner.started
    assert runner.stopped
    assert replayed_signals == []
    assert server_ref._captured_signals == []


def test_cli_cleanup_defers_repeated_sigterm_until_lease_disposition() -> None:
    from sglang_omni.serve import launcher

    delivered: list[int] = []
    original_handler = signal.getsignal(signal.SIGTERM)

    def record(sig: int, frame: FrameType | None) -> None:
        del frame
        delivered.append(sig)

    signal.signal(signal.SIGTERM, record)
    try:
        with launcher._defer_cleanup_signals(True) as retry_requested:
            signal.raise_signal(signal.SIGTERM)
            signal.raise_signal(signal.SIGTERM)
            assert delivered == []
            assert retry_requested.is_set()
        signal.raise_signal(signal.SIGTERM)
    finally:
        signal.signal(signal.SIGTERM, original_handler)

    assert delivered == [signal.SIGTERM]


def test_second_sigterm_triggers_one_explicit_retained_release_recheck() -> None:
    from sglang_omni.serve import launcher

    retained = MpsLeaseRetainedError("owner retained with exact cleanup guidance")
    retries: list[None] = []

    with launcher._defer_cleanup_signals(True) as retry_requested:
        signal.raise_signal(signal.SIGTERM)
        launcher._hold_retained_mps_owner(
            retained,
            lambda: retries.append(None),
            retry_requested,
        )

    assert retries == [None]


def test_launch_server_rejects_non_main_thread_mps_before_acquire(
    tmp_path: Path,
) -> None:
    from sglang_omni.serve import launcher

    config = _make_config(tmp_path).model_copy(update={"mps": "auto"})
    errors: list[BaseException] = []

    def launch() -> None:
        try:
            launcher.launch_server(config)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=launch)
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "main thread" in str(errors[0])
    assert "retained-lease cleanup" in str(errors[0])


def test_retained_owner_hold_uses_injected_wait(caplog) -> None:
    from sglang_omni.serve import launcher

    retained = MpsLeaseRetainedError(
        "owner PID 123; state /tmp/state; current client refs []; cleanup steps"
    )

    class StopHold(RuntimeError):
        pass

    def stop_wait() -> None:
        raise StopHold("test release")

    with pytest.raises(StopHold, match="test release"):
        launcher._hold_retained_mps_owner(
            retained,
            lambda: None,
            threading.Event(),
            wait=stop_wait,
        )

    assert "owner PID 123" in caplog.text


@pytest.mark.asyncio
async def test_launcher_preserves_runner_start_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)

    from sglang_omni.serve import launcher
    stopped: list[None] = []

    class FakeRunner:
        has_retained_mps_leases = False

        def __init__(self, pipeline_config: PipelineConfig) -> None:
            del pipeline_config

        async def start(self, timeout: float) -> None:
            del timeout
            raise RuntimeError("start failed")

        async def stop(self) -> None:
            stopped.append(None)

    monkeypatch.setattr(launcher, "_find_available_port", lambda host, port: port)
    monkeypatch.setattr(launcher, "MultiProcessPipelineRunner", FakeRunner)

    with pytest.raises(RuntimeError, match="start failed"):
        await launcher._run_server(config, port=8000)

    assert stopped == [None]


@pytest.mark.asyncio
async def test_startup_retained_lease_enters_the_same_cli_hold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog,
) -> None:
    config = _make_config(tmp_path)
    from sglang_omni.serve import launcher

    retained = MpsLeaseRetainedError(
        "owner PID 123 holds /tmp/state; current client refs: "
        "[MpsClientRef(server_pid=7, client_pid=9)]; terminate_client 7 9"
    )

    class FakeRunner:
        has_retained_mps_leases = True

        def __init__(self, pipeline_config: PipelineConfig) -> None:
            del pipeline_config

        async def start(self, timeout: float) -> None:
            del timeout
            raise RuntimeError("verify failed") from retained

        async def stop(self) -> None:
            raise retained

        def retry_retained_mps_cleanup(self) -> None:
            self.has_retained_mps_leases = False

    class OwnerHeld(RuntimeError):
        pass

    handled: list[MpsLeaseRetainedError] = []

    def hold_owner(error, retry, retry_requested) -> None:
        del retry, retry_requested
        handled.append(error)
        raise OwnerHeld("startup entered CLI hold")

    monkeypatch.setattr(launcher, "_find_available_port", lambda host, port: port)
    monkeypatch.setattr(launcher, "MultiProcessPipelineRunner", FakeRunner)

    with pytest.raises(OwnerHeld, match="startup entered CLI hold"):
        await launcher._run_server(
            config,
            port=8000,
            _retained_lease_handler=hold_owner,
        )

    assert handled == [retained]
    assert "verify failed" in caplog.text
    assert "terminate_client 7 9" in caplog.text


@pytest.mark.asyncio
async def test_startup_sigterm_cancels_and_cleans_up_without_a_retained_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    from sglang_omni.serve import launcher

    class FakeRunner:
        has_retained_mps_leases = False
        cancelled = False
        stopped = False

        def __init__(self, pipeline_config: PipelineConfig) -> None:
            del pipeline_config

        async def start(self, timeout: float) -> None:
            del timeout
            signal.raise_signal(signal.SIGTERM)
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                type(self).cancelled = True
                raise

        async def stop(self) -> None:
            type(self).stopped = True

    retained_errors: list[MpsLeaseRetainedError] = []
    monkeypatch.setattr(launcher, "_find_available_port", lambda host, port: port)
    monkeypatch.setattr(launcher, "MultiProcessPipelineRunner", FakeRunner)

    with pytest.raises(SystemExit) as exc_info:
        await launcher._run_server(
            config,
            port=8000,
            _retained_lease_handler=retained_errors.append,
        )

    assert exc_info.value.code == 128 + signal.SIGTERM
    assert FakeRunner.cancelled
    assert FakeRunner.stopped
    assert retained_errors == []


@pytest.mark.asyncio
async def test_startup_sigterm_holds_a_retained_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    from sglang_omni.serve import launcher

    retained = MpsLeaseRetainedError(
        "owner PID 123 holds /tmp/state; current client refs: "
        "[MpsClientRef(server_pid=7, client_pid=9)]; terminate_client 7 9"
    )

    class FakeRunner:
        has_retained_mps_leases = False

        def __init__(self, pipeline_config: PipelineConfig) -> None:
            del pipeline_config

        async def start(self, timeout: float) -> None:
            del timeout
            signal.raise_signal(signal.SIGTERM)
            try:
                await asyncio.Future()
            except asyncio.CancelledError as exc:
                self.has_retained_mps_leases = True
                raise exc from retained

        async def stop(self) -> None:
            raise retained

        def retry_retained_mps_cleanup(self) -> None:
            self.has_retained_mps_leases = False

    class OwnerHeld(RuntimeError):
        pass

    handled: list[MpsLeaseRetainedError] = []

    def hold_owner(error, retry, retry_requested) -> None:
        del retry, retry_requested
        handled.append(error)
        raise OwnerHeld("SIGTERM entered retained-owner hold")

    monkeypatch.setattr(launcher, "_find_available_port", lambda host, port: port)
    monkeypatch.setattr(launcher, "MultiProcessPipelineRunner", FakeRunner)

    with pytest.raises(OwnerHeld, match="retained-owner hold"):
        await launcher._run_server(
            config,
            port=8000,
            _retained_lease_handler=hold_owner,
        )

    assert handled == [retained]


@pytest.mark.asyncio
async def test_sigterm_after_start_before_uvicorn_capture_cleans_and_holds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    from sglang_omni.serve import launcher

    retained = MpsLeaseRetainedError(
        "owner PID 123 holds /tmp/state; terminate_client 7 9"
    )

    class FakeRunner:
        started = False
        stopped = False
        has_retained_mps_leases = True

        def __init__(self, pipeline_config: PipelineConfig) -> None:
            del pipeline_config
            self.coordinator = _FakeCoordinator()
            self.stage_control_endpoints = {}
            self.prep = SimpleNamespace(
                placement_plan=SimpleNamespace(gpus={}),
                process_plan=SimpleNamespace(
                    groups=(),
                    tp_stage_to_processes={},
                ),
            )

        async def start(self, timeout: float) -> None:
            del timeout
            type(self).started = True

        async def stop(self) -> None:
            type(self).stopped = True
            raise retained

        def retry_retained_mps_cleanup(self) -> None:
            self.has_retained_mps_leases = False

        async def wait_failed(self) -> None:
            await asyncio.Future()

    def create_app_then_signal(*args, **kwargs):
        del args, kwargs
        signal.raise_signal(signal.SIGTERM)
        return FastAPI()

    async def before_uvicorn_capture(server, runtime_watchers) -> None:
        del server
        for watcher in runtime_watchers:
            watcher.close()
        await asyncio.sleep(0)

    class OwnerHeld(RuntimeError):
        pass

    handled: list[MpsLeaseRetainedError] = []

    def hold_owner(error, retry, retry_requested) -> None:
        del retry, retry_requested
        handled.append(error)
        raise OwnerHeld("post-start SIGTERM entered retained-owner hold")

    monkeypatch.setattr(launcher, "_find_available_port", lambda host, port: port)
    monkeypatch.setattr(launcher, "MultiProcessPipelineRunner", FakeRunner)
    monkeypatch.setattr(launcher, "create_app", create_app_then_signal)
    monkeypatch.setattr(launcher, "_mount_profiler_routes", lambda *args: None)
    monkeypatch.setattr(
        launcher,
        "_serve_with_failure_watch",
        before_uvicorn_capture,
    )

    with pytest.raises(OwnerHeld, match="post-start SIGTERM"):
        await launcher._run_server(
            config,
            port=8000,
            _retained_lease_handler=hold_owner,
        )

    assert FakeRunner.started
    assert FakeRunner.stopped
    assert handled == [retained]
