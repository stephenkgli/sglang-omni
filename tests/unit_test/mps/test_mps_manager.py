# SPDX-License-Identifier: Apache-2.0
"""State-machine tests for the shared per-GPU MpsManager against a fake client.

The real ``nvidia-cuda-mps-control`` I/O lives behind the ``MpsControlClient``
protocol; these tests script the fake to cover every lifecycle path without a
GPU. The subprocess-backed client is exercised by the GPU CI smoke instead.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from sglang_omni.mps.manager import MpsError, MpsManager, MpsState
from sglang_omni.mps.state import MpsGpuPaths


class FakeControlClient:
    """Scriptable stand-in for the MPS control daemon interface."""

    def __init__(self):
        self.daemon_pid = 4242
        self.control_responsive = True
        self.start_fails = False
        self.start_calls = 0
        # server pid -> list of client pids
        self.servers: dict[int, list[int]] = {}
        self.alive_pids: set[int] = set()
        # pipe dirs (str) this fake believes each daemon pid owns
        self.pid_pipe_dirs: dict[int, str] = {}
        self.parents: dict[int, int] = {}
        self.quit_calls: list[str] = []

    def start_daemon(self, pipe_dir, log_dir, gpu_uuid):
        self.start_calls += 1
        if self.start_fails:
            raise OSError("spawn failed")
        self.alive_pids.add(self.daemon_pid)
        self.pid_pipe_dirs[self.daemon_pid] = str(pipe_dir)
        return self.daemon_pid

    def control_responds(self, pipe_dir):
        if not self.control_responsive:
            return False
        return any(
            self.pid_pipe_dirs.get(pid) == str(pipe_dir) for pid in self.alive_pids
        )

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

    def kill_pid(self, pid, force=False):
        self.alive_pids.discard(pid)

    def parent_of(self, pid):
        return self.parents.get(pid)

    def daemon_owns_pipe(self, pid, pipe_dir):
        return self.pid_pipe_dirs.get(pid) == str(pipe_dir)

    def owner_lease_held(self, lease_file):
        try:
            return int(lease_file.name) in self.alive_pids
        except ValueError:
            return False


GPU_UUID = "GPU-11111111-2222-3333-4444-555555555555"


@pytest.fixture
def short_root():
    # Note (Jiaxin Deng): pytest's tmp_path exceeds the 107-byte sun_path
    # budget on Windows hosts, so build the state root directly under TEMP.
    root = Path(tempfile.mkdtemp(prefix="mps-"))
    yield root
    shutil.rmtree(root, ignore_errors=True)


def make_manager(root, client):
    return MpsManager(
        paths=MpsGpuPaths(state_root=root, gpu_uuid=GPU_UUID),
        gpu_uuid=GPU_UUID,
        client=client,
        poll_interval=0.0,
        start_timeout=0.05,
        verify_timeout=0.05,
        drain_timeout=0.05,
        stop_timeout=0.05,
    )


def seed_shared_dir(root, client, *, daemon_pid, owners=(), clients=()):
    """Simulate the state another (possibly dead) serve left behind."""
    paths = MpsGpuPaths(state_root=root, gpu_uuid=GPU_UUID)
    paths.pipe_dir.mkdir(parents=True)
    paths.log_dir.mkdir(parents=True)
    paths.owners_dir.mkdir(parents=True)
    paths.manifest.write_text(
        json.dumps(
            {
                "gpu_uuid": GPU_UUID,
                "daemon_pid": daemon_pid,
                "creator_pid": 1234,
                "pipe_dir": str(paths.pipe_dir),
            }
        )
    )
    for owner in owners:
        (paths.owners_dir / str(owner)).write_text("")
    if clients:
        client.servers = {9000: list(clients)}
    return paths


def start_serving(root, client):
    mgr = make_manager(root, client)
    mgr.start()
    client.servers = {7000: [101]}
    mgr.verify_attached({101})
    return mgr


# --- create / join / adopt ---


def test_first_owner_creates_daemon_and_registers(short_root):
    client = FakeControlClient()
    mgr = make_manager(short_root, client)
    mgr.start()

    assert mgr.state is MpsState.READY
    assert client.start_calls == 1
    assert (mgr.paths.owners_dir / str(os.getpid())).exists()
    manifest = json.loads(mgr.paths.manifest.read_text())
    assert manifest["daemon_pid"] == client.daemon_pid
    assert manifest["gpu_uuid"] == GPU_UUID


def test_second_owner_joins_existing_daemon(short_root):
    client = FakeControlClient()
    client.alive_pids.update({999, 888})
    paths = seed_shared_dir(short_root, client, daemon_pid=999, owners=[888])
    client.pid_pipe_dirs[999] = str(paths.pipe_dir)

    mgr = make_manager(short_root, client)
    mgr.start()
    assert client.start_calls == 0
    assert mgr.daemon_pid == 999

    mgr.stop()
    assert mgr.state is MpsState.CLEANED
    assert client.quit_calls == []
    assert (paths.owners_dir / "888").exists()
    assert not (paths.owners_dir / str(os.getpid())).exists()


def test_start_control_never_responding_reaps_fresh_daemon(short_root):
    client = FakeControlClient()
    client.control_responsive = False
    killed = []

    def kill_pid(pid, force=False):
        killed.append(pid)
        client.alive_pids.discard(pid)

    client.kill_pid = kill_pid
    mgr = make_manager(short_root, client)

    with pytest.raises(MpsError, match="control"):
        mgr.start()
    assert mgr.state is MpsState.FAILED
    assert killed == [client.daemon_pid]
    assert json.loads(mgr.paths.manifest.read_text())["daemon_pid"]


def test_start_daemon_spawn_failure_fails_closed(short_root):
    client = FakeControlClient()
    client.start_fails = True
    mgr = make_manager(short_root, client)

    with pytest.raises(MpsError, match="spawn failed"):
        mgr.start()
    assert mgr.state is MpsState.FAILED
    assert not (mgr.paths.owners_dir / str(os.getpid())).exists()


# --- verify / watchdog ---


def test_happy_path_reaches_cleaned_and_removes_state(short_root):
    client = FakeControlClient()
    mgr = make_manager(short_root, client)
    mgr.start()

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


def test_verify_timeout_names_missing_pids_and_fails(short_root):
    client = FakeControlClient()
    mgr = make_manager(short_root, client)
    mgr.start()
    client.servers = {7000: [101]}

    with pytest.raises(MpsError, match=r"\[202\]"):
        mgr.verify_attached({101, 202})
    assert mgr.state is MpsState.FAILED
    assert mgr.paths.state_dir.is_dir()


def test_verify_matches_descendants_of_expected_pids(short_root):
    client = FakeControlClient()
    mgr = make_manager(short_root, client)
    mgr.start()
    # Engine child 200 (parent 100) creates the CUDA context, not wrapper 100.
    client.servers = {7000: [200]}
    client.parents = {200: 100}

    mgr.verify_attached({100})
    assert mgr.state is MpsState.SERVING


def test_probe_detects_daemon_death(short_root):
    client = FakeControlClient()
    mgr = start_serving(short_root, client)
    assert mgr.probe() is True

    client.alive_pids.discard(client.daemon_pid)
    assert mgr.probe() is False


def test_proc_stat_zombie_is_not_alive():
    from sglang_omni.mps.control import _stat_says_alive

    assert _stat_says_alive("430465 (nvidia-cuda-mps) Z 1 430465 0") is False
    assert _stat_says_alive("53748 (nvidia-cuda-mps-control) S 1 0 0") is True
    # Process names may contain parentheses and spaces.
    assert _stat_says_alive("7 (weird) name) Z 1 0") is False


# --- teardown ---


def test_last_owner_stop_refuses_to_quit_under_live_clients(short_root):
    client = FakeControlClient()
    mgr = start_serving(short_root, client)
    client.alive_pids.add(101)  # the attached client is a live process

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


def test_stop_after_daemon_death_cleans_without_quit(short_root):
    client = FakeControlClient()
    mgr = start_serving(short_root, client)
    client.alive_pids.discard(client.daemon_pid)

    mgr.stop()
    assert mgr.state is MpsState.CLEANED
    assert not mgr.paths.state_dir.exists()
    assert client.quit_calls == []


def test_stop_before_start_cleans_quietly(short_root):
    client = FakeControlClient()
    client.get_server_list = lambda pipe_dir: (_ for _ in ()).throw(
        OSError("Cannot find MPS control daemon process")
    )
    mgr = make_manager(short_root, client)

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


def test_stop_wraps_arbitrary_control_exceptions(short_root):
    import subprocess as sp

    client = FakeControlClient()
    mgr = start_serving(short_root, client)
    client.get_server_list = lambda pipe_dir: (_ for _ in ()).throw(
        sp.TimeoutExpired(cmd="nvidia-cuda-mps-control", timeout=10)
    )

    with pytest.raises(MpsError):
        mgr.stop()
    assert mgr.state is MpsState.FAILED


def test_dead_owner_leftover_fails_with_cleanup_guidance(short_root):
    client = FakeControlClient()
    paths = seed_shared_dir(short_root, client, daemon_pid=999, owners=[777])

    mgr = make_manager(short_root, client)
    with pytest.raises(MpsError) as excinfo:
        mgr.start()
    message = str(excinfo.value)
    assert str(paths.state_dir) in message
    assert "777" in message
    assert "rm -rf" in message
    assert paths.state_dir.is_dir()  # nothing is deleted for the operator


def test_orphan_daemon_with_no_live_owner_fails(short_root):
    client = FakeControlClient()
    client.alive_pids.update({999, 555})
    paths = seed_shared_dir(
        short_root, client, daemon_pid=999, owners=[777], clients=[555]
    )
    client.pid_pipe_dirs[999] = str(paths.pipe_dir)

    mgr = make_manager(short_root, client)
    with pytest.raises(MpsError) as excinfo:
        mgr.start()
    message = str(excinfo.value)
    assert "999" in message  # the daemon to kill
    assert "555" in message  # the client to kill
    assert client.quit_calls == []
    assert 555 in client.alive_pids  # nothing was signalled


def test_join_removes_dead_co_owner_lease_and_joins(short_root):
    # A dead co-owner's lease is already released by the kernel; the leftover
    # file is unowned, so joining proceeds under the surviving owner's pool.
    client = FakeControlClient()
    client.alive_pids.update({999, 888})
    paths = seed_shared_dir(short_root, client, daemon_pid=999, owners=[888, 777])
    client.pid_pipe_dirs[999] = str(paths.pipe_dir)

    mgr = make_manager(short_root, client)
    mgr.start()
    assert mgr.daemon_pid == 999
    assert not (paths.owners_dir / "777").exists()
    assert (paths.owners_dir / "888").exists()


def test_idle_healthy_daemon_is_adopted(short_root):
    # The one allowed adoption: provable identity, healthy control, no live
    # owner, and an empty client list.
    client = FakeControlClient()
    client.alive_pids.add(999)
    paths = seed_shared_dir(short_root, client, daemon_pid=999, owners=[777])
    client.pid_pipe_dirs[999] = str(paths.pipe_dir)

    mgr = make_manager(short_root, client)
    mgr.start()
    assert mgr.daemon_pid == 999
    assert client.start_calls == 0
    assert not (paths.owners_dir / "777").exists()


def test_torn_manifest_fails_with_guidance(short_root):
    client = FakeControlClient()
    client.alive_pids.add(888)
    paths = seed_shared_dir(short_root, client, daemon_pid=999, owners=[888])
    paths.manifest.write_text("{torn")

    mgr = make_manager(short_root, client)
    with pytest.raises(MpsError, match="rm -rf"):
        mgr.start()
