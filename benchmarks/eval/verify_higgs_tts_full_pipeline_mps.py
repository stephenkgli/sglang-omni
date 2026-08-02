# SPDX-License-Identifier: Apache-2.0
"""Verify that every configured Higgs TTS GPU process uses CUDA MPS."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

_STAGE_GROUP_PATTERN = re.compile(
    r"StageGroup (?P<name>[^:]+): spawned (?P<count>[0-9]+) "
    r"process\(es\) \(pids=\[(?P<pids>[0-9, ]*)\]\)"
)


class MpsQueryError(RuntimeError):
    """Raised when the private CUDA MPS control endpoint cannot be queried."""


def parse_stage_group_pids(server_log: str) -> dict[str, list[int]]:
    """Return the last observed PID list for each spawned stage group."""
    groups: dict[str, list[int]] = {}
    for match in _STAGE_GROUP_PATTERN.finditer(server_log):
        pids = [
            int(value.strip())
            for value in match.group("pids").split(",")
            if value.strip()
        ]
        count = int(match.group("count"))
        if len(pids) != count:
            raise ValueError(
                f"stage group {match.group('name')!r} declared {count} process(es) "
                f"but logged pids={pids}"
            )
        groups[match.group("name")] = pids
    return groups


def expected_gpu_processes(
    pipeline_config: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Map configured GPU-owning process instances to their GPU stages."""
    stages = pipeline_config.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("pipeline config must contain a non-empty stages list")

    process_stages: dict[str, list[str]] = {}
    for raw_stage in stages:
        if not isinstance(raw_stage, Mapping):
            raise ValueError(f"invalid stage entry: {raw_stage!r}")
        stage_name = raw_stage.get("name")
        if not isinstance(stage_name, str) or not stage_name:
            raise ValueError(f"stage has an invalid name: {stage_name!r}")
        if raw_stage.get("gpu") is None:
            continue

        process_name = raw_stage.get("process")
        if not isinstance(process_name, str) or not process_name:
            raise ValueError(
                f"GPU stage {stage_name!r} requires an explicit process name"
            )
        tp_size = int(raw_stage.get("tp_size", 1))
        if tp_size != 1:
            raise ValueError(
                f"GPU stage {stage_name!r} uses tp_size={tp_size}; "
                "this full-pipeline MPS verifier currently supports tp_size=1"
            )
        num_replicas = int(raw_stage.get("num_replicas", 1))
        if num_replicas < 1:
            raise ValueError(
                f"GPU stage {stage_name!r} has invalid num_replicas={num_replicas}"
            )

        for replica_id in range(num_replicas):
            if num_replicas == 1:
                process_instance = process_name
                stage_instance = stage_name
            else:
                process_instance = f"{process_name}@r{replica_id}"
                stage_instance = f"{stage_name}@r{replica_id}"
            process_stages.setdefault(process_instance, []).append(stage_instance)

    if not process_stages:
        raise ValueError("pipeline config does not contain any GPU stages")
    return {
        process_name: sorted(stage_names)
        for process_name, stage_names in sorted(process_stages.items())
    }


def _numeric_lines(output: str) -> list[int]:
    return [
        int(line.strip())
        for line in output.splitlines()
        if re.fullmatch(r"[0-9]+", line.strip())
    ]


def _query_mps_control(
    command: str,
    *,
    pipe_directory: Path,
    log_directory: Path,
    timeout_seconds: float,
) -> str:
    env = os.environ.copy()
    env["CUDA_MPS_PIPE_DIRECTORY"] = str(pipe_directory)
    env["CUDA_MPS_LOG_DIRECTORY"] = str(log_directory)
    try:
        result = subprocess.run(
            ["nvidia-cuda-mps-control"],
            input=f"{command}\n",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MpsQueryError(f"MPS query {command!r} failed: {exc}") from exc
    if result.returncode != 0:
        raise MpsQueryError(
            f"MPS query {command!r} exited {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    return result.stdout


def query_mps_server_clients(
    *,
    pipe_directory: Path,
    log_directory: Path,
    timeout_seconds: float,
) -> tuple[dict[int, list[int]], dict[str, str]]:
    """Return private MPS server-to-client PID mappings and raw query output."""
    raw_queries: dict[str, str] = {}
    server_output = _query_mps_control(
        "get_server_list",
        pipe_directory=pipe_directory,
        log_directory=log_directory,
        timeout_seconds=timeout_seconds,
    )
    raw_queries["get_server_list"] = server_output
    servers = _numeric_lines(server_output)
    mappings: dict[int, list[int]] = {}
    for server_pid in servers:
        command = f"get_client_list {server_pid}"
        client_output = _query_mps_control(
            command,
            pipe_directory=pipe_directory,
            log_directory=log_directory,
            timeout_seconds=timeout_seconds,
        )
        raw_queries[command] = client_output
        mappings[server_pid] = _numeric_lines(client_output)
    return mappings, raw_queries


def validate_full_pipeline_attachment(
    *,
    pipeline_config: Mapping[str, Any],
    server_log: str,
    server_clients: Mapping[int, list[int]],
    pid_is_live: Callable[[int], bool] | None = None,
) -> dict[str, object]:
    """Validate an exact all-configured-GPU-process MPS attachment set."""
    gpu_process_stages = expected_gpu_processes(pipeline_config)
    groups = parse_stage_group_pids(server_log)
    expected_pids: list[int] = []
    expected_process_rows: list[dict[str, object]] = []
    violations: list[str] = []

    for process_name, stage_names in gpu_process_stages.items():
        pids = groups.get(process_name, [])
        if len(pids) != 1:
            violations.append(
                f"GPU process {process_name!r} must own exactly one process, "
                f"observed={pids}"
            )
            continue
        expected_pids.append(pids[0])
        expected_process_rows.append(
            {
                "process_name": process_name,
                "pid": pids[0],
                "gpu_stages": stage_names,
            }
        )

    if len(set(expected_pids)) != len(expected_pids):
        violations.append(
            f"configured GPU process groups resolved to duplicate PIDs: {expected_pids}"
        )
    if len(server_clients) != 1:
        violations.append(
            "the private single-GPU MPS daemon must expose exactly one server, "
            f"observed={sorted(server_clients)}"
        )

    client_pids = sorted(
        {client_pid for clients in server_clients.values() for client_pid in clients}
    )
    expected_pid_set = set(expected_pids)
    client_pid_set = set(client_pids)
    missing = sorted(expected_pid_set - client_pid_set)
    unexpected = sorted(client_pid_set - expected_pid_set)
    if missing:
        violations.append(
            f"configured GPU processes missing from MPS clients: {missing}"
        )
    if unexpected:
        violations.append(
            "processes outside the configured GPU process set attached to MPS: "
            f"{unexpected}"
        )

    if pid_is_live is not None:
        dead = sorted(pid for pid in expected_pid_set if not pid_is_live(pid))
        if dead:
            violations.append(
                f"configured GPU processes disappeared before validation: {dead}"
            )

    return {
        "mps_scope": "all_gpu_processes_in_server_tree",
        "configured_gpu_process_stages": gpu_process_stages,
        "expected_gpu_processes": expected_process_rows,
        "stage_group_pids": groups,
        "mps_server_clients": {
            str(server_pid): clients for server_pid, clients in server_clients.items()
        },
        "mps_client_pids": client_pids,
        "exact_full_pipeline_attachment": not violations,
        "violations": violations,
    }


def _pid_is_live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--pipeline-config", type=Path, required=True)
    parser.add_argument("--pipe-directory", type=Path, required=True)
    parser.add_argument("--log-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-timeout-seconds", type=float, default=10.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload: dict[str, object]
    try:
        pipeline_config = json.loads(args.pipeline_config.read_text(encoding="utf-8"))
        server_clients, raw_queries = query_mps_server_clients(
            pipe_directory=args.pipe_directory,
            log_directory=args.log_directory,
            timeout_seconds=args.query_timeout_seconds,
        )
        payload = validate_full_pipeline_attachment(
            pipeline_config=pipeline_config,
            server_log=args.server_log.read_text(encoding="utf-8", errors="replace"),
            server_clients=server_clients,
            pid_is_live=_pid_is_live,
        )
        payload["raw_queries"] = raw_queries
    except (MpsQueryError, OSError, ValueError) as exc:
        payload = {
            "mps_scope": "all_gpu_processes_in_server_tree",
            "exact_full_pipeline_attachment": False,
            "violations": [str(exc)],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not payload["exact_full_pipeline_attachment"]:
        for violation in payload["violations"]:
            print(f"ERROR: {violation}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
