# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang_omni.mps import (
    MpsRunPaths,
    derive_core_blocks,
    format_cpu_list,
    parse_cpu_list,
    validate_control_socket,
)
from sglang_omni.mps import topology


def test_cpu_list_round_trip() -> None:
    assert parse_cpu_list("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]
    assert format_cpu_list([0, 1, 2, 3, 8, 10, 11]) == "0-3,8,10-11"


def test_parse_cpu_list_rejects_inverted_range() -> None:
    with pytest.raises(ValueError, match="invalid CPU range"):
        parse_cpu_list("5-2")


def _fake_sysfs(tmp_path: Path, cpulist: str) -> tuple[Path, Path]:
    pci_devices = tmp_path / "pci"
    numa_nodes = tmp_path / "nodes"
    device = pci_devices / "0000:08:00.0"
    device.mkdir(parents=True)
    (device / "numa_node").write_text("0\n", encoding="utf-8")
    node = numa_nodes / "node0"
    node.mkdir(parents=True)
    (node / "cpulist").write_text(f"{cpulist}\n", encoding="utf-8")
    return pci_devices, numa_nodes


def test_derive_core_blocks_splits_into_n_blocks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pci_devices, numa_nodes = _fake_sysfs(tmp_path, "0-15")
    monkeypatch.setattr(
        topology.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="00000000:08:00.0\n"),
    )
    monkeypatch.setattr(
        topology.os,
        "sched_getaffinity",
        lambda _pid: set(range(16)),
        raising=False,
    )
    assert derive_core_blocks(
        0,
        n_blocks=3,
        pci_devices_root=pci_devices,
        numa_nodes_root=numa_nodes,
    ) == ("0-3", "4-7", "8-11")


def test_derive_core_blocks_rejects_a_node_below_the_minimum(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pci_devices, numa_nodes = _fake_sysfs(tmp_path, "0-2")
    monkeypatch.setattr(
        topology.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="00000000:08:00.0\n"),
    )
    monkeypatch.setattr(
        topology.os,
        "sched_getaffinity",
        lambda _pid: set(range(3)),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="allowed CPU cores"):
        derive_core_blocks(
            0,
            pci_devices_root=pci_devices,
            numa_nodes_root=numa_nodes,
        )


def test_run_paths_match_launcher_layout() -> None:
    paths = MpsRunPaths(Path("/state"), 3, "run-abc")
    assert paths.state_dir == Path("/state/gpu-3/run-abc")
    assert paths.control_socket == Path("/state/gpu-3/run-abc/mps/pipe/control")
    assert paths.manifest == Path("/state/gpu-3/run-abc/manifest")
    assert paths.replicas_tsv == Path("/state/gpu-3/run-abc/replicas.tsv")
    assert paths.attach_report == Path("/state/gpu-3/run-abc/mps_attach.txt")


def test_run_paths_reject_unsafe_run_id() -> None:
    with pytest.raises(ValueError, match="run-"):
        MpsRunPaths(Path("/state"), 0, "../escape")


def test_validate_control_socket_names_the_sun_path_limit() -> None:
    long_socket = Path("/state") / ("x" * 120) / "mps" / "pipe" / "control"
    with pytest.raises(ValueError, match="sun_path"):
        validate_control_socket(long_socket)
    validate_control_socket(Path("/short/mps/pipe/control"))
