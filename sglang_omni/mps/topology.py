# SPDX-License-Identifier: Apache-2.0
"""CPU topology helpers for pinning MPS replicas to a GPU's NUMA node."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def parse_cpu_list(value: str) -> list[int]:
    cpus: list[int] = []
    for item in value.split(","):
        start, separator, end = item.strip().partition("-")
        if not start:
            continue
        first = int(start)
        last = int(end) if separator else first
        if last < first:
            raise ValueError(f"invalid CPU range {item!r}")
        cpus.extend(range(first, last + 1))
    return cpus


def format_cpu_list(cpus: list[int]) -> str:
    if not cpus:
        raise ValueError("CPU block cannot be empty")
    ranges: list[str] = []
    start = previous = cpus[0]
    for cpu in cpus[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def derive_core_blocks(
    gpu_id: int,
    *,
    n_blocks: int = 2,
    cores_per_block: int = 2,
    host_reserve_ratio: float = 0.25,
    pci_devices_root: Path = Path("/sys/bus/pci/devices"),
    numa_nodes_root: Path = Path("/sys/devices/system/node"),
) -> tuple[str, ...]:
    """Split the GPU's NUMA-local allowed CPUs into pinning blocks.

    Resolves the GPU's NUMA node through its PCI bus id (nvidia-smi ordinals
    do not match /sys/class/drm ordinals), intersects the node's cpulist with
    the process affinity, reserves a slice for host-side work, and splits the
    rest into ``n_blocks`` contiguous blocks in cpuset list format.
    """
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=pci.bus_id",
            "--format=csv,noheader",
            "-i",
            str(gpu_id),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    bus_id = result.stdout.strip().lower()
    try:
        domain, bus, device = bus_id.split(":")
        pci_device = pci_devices_root / f"{int(domain, 16):04x}:{bus}:{device}"
    except ValueError as exc:
        raise RuntimeError(f"invalid PCI bus ID for GPU {gpu_id}: {bus_id!r}") from exc
    if not pci_device.is_dir():
        raise RuntimeError(f"cannot resolve one PCI device for GPU {gpu_id}")
    numa_node = int((pci_device / "numa_node").read_text().strip())
    if numa_node < 0:
        raise RuntimeError(f"GPU {gpu_id} has no usable NUMA node")
    node_cpus = set(
        parse_cpu_list(
            (numa_nodes_root / f"node{numa_node}" / "cpulist").read_text().strip()
        )
    )
    allowed = sorted(node_cpus & set(os.sched_getaffinity(0)))
    minimum_cores = cores_per_block * n_blocks
    if len(allowed) < minimum_cores:
        raise RuntimeError(
            f"need {minimum_cores} allowed CPU cores on the GPU NUMA node "
            f"for {n_blocks} pinning blocks, found {len(allowed)}"
        )
    # Reserve a slice for host-side work, but never below the hard minimum, so
    # a host that does have enough cores is not reported as insufficient.
    usable = allowed[: max(minimum_cores, int(len(allowed) * (1 - host_reserve_ratio)))]
    base = len(usable) // n_blocks
    blocks = [
        usable[index * base : (index + 1) * base] for index in range(n_blocks - 1)
    ]
    blocks.append(usable[(n_blocks - 1) * base :])
    return tuple(format_cpu_list(block) for block in blocks)
