# SPDX-License-Identifier: Apache-2.0
"""State-machine tests for MpsManager against an in-memory fake control client.

The real ``nvidia-cuda-mps-control`` I/O lives behind the ``MpsControlClient``
protocol; these tests script the fake to cover every lifecycle path without a
GPU. The subprocess-backed client is exercised by the GPU CI smoke instead.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from sglang_omni.mps.manager import MpsError, MpsManager, MpsState
from sglang_omni.mps.state import MpsRunPaths


class FakeControlClient:
    """Scriptable stand-in for the MPS control daemon interface."""

    def __init__(self):
        self.daemon_pid = 4242
        self.control_responsive = True
        self.start_fails = False
        # server pid -> list of client pids
        self.servers: dict[int, list[int]] = {}
        self.alive_pids: set[int] = set()
        # pipe dirs (str) this fake believes each daemon pid owns
        self.pid_pipe_dirs: dict[int, str] = {}
        self.quit_calls: list[str] = []

    def start_daemon(self, pipe_dir, log_dir, gpu_uuid):
        if self.start_fails:
            raise OSError("spawn failed")
        self.alive_pids.add(self.daemon_pid)
        self.pid_pipe_dirs[self.daemon_pid] = str(pipe_dir)
        return self.daemon_pid

    def control_responds(self, pipe_dir):
        return self.control_responsive

    def get_server_list(self, pipe_dir):
        return list(self.servers)

    def get_client_list(self, pipe_dir, server_pid):
        return list(self.servers.get(server_pid, []))

    def quit_daemon(self, pipe_dir):
        self.quit_calls.append(str(pipe_dir))
        for pid, owned in list(self.pid_pipe_dirs.items()):
            if owned == str(pipe_dir):
                self.alive_pids.discard(pid)
        self.servers.clear()

    def pid_alive(self, pid):
        return pid in self.alive_pids

    def daemon_owns_pipe(self, pid, pipe_dir):
        return self.pid_pipe_dirs.get(pid) == str(pipe_dir)


GPU_UUID = "GPU-11111111-2222-3333-4444-555555555555"


@pytest.fixture
def short_root():
    # Note (Jiaxin Deng): pytest's tmp_path exceeds the 107-byte sun_path
    # budget on Windows hosts, so build the state root directly under TEMP.
    root = Path(tempfile.mkdtemp(prefix="mps-"))
    yield root
    shutil.rmtree(root, ignore_errors=True)


def make_manager(tmp_path, client, run_id="run-t1"):
    paths = MpsRunPaths(state_root=tmp_path, gpu_id=0, run_id=run_id)
    return MpsManager(
        paths=paths,
        gpu_uuid=GPU_UUID,
        client=client,
        poll_interval=0.0,
        start_timeout=0.05,
        verify_timeout=0.05,
        drain_timeout=0.05,
        stop_timeout=0.05,
    )


def test_happy_path_reaches_cleaned_and_removes_state(short_root):
    client = FakeControlClient()
    mgr = make_manager(short_root, client)

    mgr.preflight()
    mgr.start()
    assert mgr.state is MpsState.READY
    assert mgr.paths.state_dir.is_dir()

    env = mgr.env_for_stage()
    assert env["CUDA_MPS_PIPE_DIRECTORY"] == str(mgr.paths.pipe_dir)
    assert env["CUDA_MPS_LOG_DIRECTORY"] == str(mgr.paths.log_dir)
    assert env["CUDA_VISIBLE_DEVICES"] == GPU_UUID

    client.servers = {7000: [101, 102]}
    mgr.verify_attached({101, 102})
    assert mgr.state is MpsState.SERVING

    client.servers = {}
    mgr.stop()
    assert mgr.state is MpsState.CLEANED
    assert not mgr.paths.state_dir.exists()
    assert client.quit_calls == [str(mgr.paths.pipe_dir)]


def test_start_writes_ownership_manifest(short_root):
    client = FakeControlClient()
    mgr = make_manager(short_root, client)
    mgr.preflight()
    mgr.start()

    manifest = json.loads(mgr.paths.manifest.read_text())
    assert manifest["daemon_pid"] == client.daemon_pid
    assert manifest["run_id"] == "run-t1"
    assert manifest["pipe_dir"] == str(mgr.paths.pipe_dir)


def start_serving(short_root, client, run_id="run-t1"):
    mgr = make_manager(short_root, client, run_id=run_id)
    mgr.preflight()
    mgr.start()
    client.servers = {7000: [101]}
    mgr.verify_attached({101})
    return mgr


def test_verify_timeout_names_missing_pids_and_fails(short_root):
    client = FakeControlClient()
    mgr = make_manager(short_root, client)
    mgr.preflight()
    mgr.start()
    client.servers = {7000: [101]}

    with pytest.raises(MpsError, match=r"\[202\]"):
        mgr.verify_attached({101, 202})
    assert mgr.state is MpsState.FAILED
    assert mgr.paths.state_dir.is_dir()


def test_stop_refuses_to_quit_under_live_clients(short_root):
    client = FakeControlClient()
    mgr = start_serving(short_root, client)

    with pytest.raises(MpsError, match="still attached"):
        mgr.stop()
    assert mgr.state is MpsState.FAILED
    assert client.quit_calls == []
    assert mgr.paths.state_dir.is_dir()


def test_stop_daemon_refusing_to_exit_preserves_state(short_root):
    client = FakeControlClient()
    mgr = start_serving(short_root, client)
    client.servers = {}
    client.quit_daemon = lambda pipe_dir: None  # daemon ignores quit

    with pytest.raises(MpsError, match="did not exit"):
        mgr.stop()
    assert mgr.state is MpsState.FAILED
    assert mgr.paths.state_dir.is_dir()


def test_start_control_never_responding_fails(short_root):
    client = FakeControlClient()
    client.control_responsive = False
    mgr = make_manager(short_root, client)
    mgr.preflight()

    with pytest.raises(MpsError, match="control"):
        mgr.start()
    assert mgr.state is MpsState.FAILED


def test_probe_detects_daemon_death(short_root):
    client = FakeControlClient()
    mgr = start_serving(short_root, client)
    assert mgr.probe() is True

    client.alive_pids.discard(client.daemon_pid)
    assert mgr.probe() is False


def make_stale_dir(short_root, client, *, daemon_pid, run_id="run-old", clients=()):
    paths = MpsRunPaths(state_root=short_root, gpu_id=0, run_id=run_id)
    paths.pipe_dir.mkdir(parents=True)
    paths.log_dir.mkdir(parents=True)
    paths.manifest.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "gpu_id": 0,
                "gpu_uuid": GPU_UUID,
                "daemon_pid": daemon_pid,
                "pipe_dir": str(paths.pipe_dir),
            }
        )
    )
    if clients:
        client.servers = {9000: list(clients)}
    return paths


def test_preflight_removes_stale_dir_of_dead_daemon(short_root):
    client = FakeControlClient()
    stale = make_stale_dir(short_root, client, daemon_pid=999)  # 999 not alive

    mgr = make_manager(short_root, client)
    mgr.preflight()
    assert not stale.state_dir.exists()


def test_preflight_reclaims_idle_orphan_with_ownership_proof(short_root):
    client = FakeControlClient()
    client.alive_pids.add(999)
    stale = make_stale_dir(short_root, client, daemon_pid=999)
    client.pid_pipe_dirs[999] = str(stale.pipe_dir)

    mgr = make_manager(short_root, client)
    mgr.preflight()
    assert not stale.state_dir.exists()
    assert client.quit_calls == [str(stale.pipe_dir)]


def test_preflight_refuses_orphan_whose_clients_survive_sigkill(short_root):
    client = FakeControlClient()
    client.alive_pids.update({999, 555})
    stale = make_stale_dir(short_root, client, daemon_pid=999, clients=[555])
    client.pid_pipe_dirs[999] = str(stale.pipe_dir)
    client.kill_pid = lambda pid, force=False: None  # unkillable client

    mgr = make_manager(short_root, client)
    with pytest.raises(MpsError, match="[Kk]ill them manually"):
        mgr.preflight()
    assert stale.state_dir.is_dir()
    assert client.quit_calls == []


def test_preflight_treats_recycled_pid_as_dead(short_root):
    client = FakeControlClient()
    client.alive_pids.add(999)  # alive, but owns a different pipe dir
    stale = make_stale_dir(short_root, client, daemon_pid=999)
    client.pid_pipe_dirs[999] = "/somewhere/else"

    mgr = make_manager(short_root, client)
    mgr.preflight()
    assert not stale.state_dir.exists()
    assert client.quit_calls == []


def test_stop_after_daemon_death_cleans_without_quit(short_root):
    client = FakeControlClient()
    mgr = start_serving(short_root, client)
    client.alive_pids.discard(client.daemon_pid)
    client.get_server_list = lambda pipe_dir: (_ for _ in ()).throw(
        OSError("control socket gone")
    )

    mgr.stop()
    assert mgr.state is MpsState.CLEANED
    assert not mgr.paths.state_dir.exists()
    assert client.quit_calls == []


def test_proc_stat_zombie_is_not_alive():
    from sglang_omni.mps.control import _stat_says_alive

    assert _stat_says_alive("430465 (nvidia-cuda-mps) Z 1 430465 0") is False
    assert _stat_says_alive("53748 (nvidia-cuda-mps-control) S 1 0 0") is True
    # Process names may contain parentheses and spaces.
    assert _stat_says_alive("7 (weird) name) Z 1 0") is False


def test_preflight_reaps_orphan_clients_of_owned_daemon(short_root):
    client = FakeControlClient()
    client.alive_pids.update({999, 555, 556})
    stale = make_stale_dir(short_root, client, daemon_pid=999, clients=[555, 556])
    client.pid_pipe_dirs[999] = str(stale.pipe_dir)

    killed = []

    def kill_pid(pid, force=False):
        killed.append((pid, force))
        client.alive_pids.discard(pid)
        client.servers = {
            s: [c for c in cs if c != pid] for s, cs in client.servers.items()
        }

    client.kill_pid = kill_pid

    mgr = make_manager(short_root, client)
    mgr.preflight()
    assert {pid for pid, _ in killed} == {555, 556}
    assert not stale.state_dir.exists()
    assert client.quit_calls == [str(stale.pipe_dir)]


def test_stop_before_start_cleans_quietly(short_root):
    client = FakeControlClient()
    client.get_server_list = lambda pipe_dir: (_ for _ in ()).throw(
        OSError("Cannot find MPS control daemon process")
    )
    mgr = make_manager(short_root, client)
    mgr.preflight()

    mgr.stop()
    assert mgr.state is MpsState.CLEANED


def test_stop_wraps_client_io_errors_as_mps_error(short_root):
    client = FakeControlClient()
    mgr = start_serving(short_root, client)
    client.get_server_list = lambda pipe_dir: (_ for _ in ()).throw(
        OSError("control socket wedged")
    )

    with pytest.raises(MpsError, match="wedged"):
        mgr.stop()
    assert mgr.state is MpsState.FAILED
