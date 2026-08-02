# SPDX-License-Identifier: Apache-2.0
"""Closed-loop full SeedTTS Higgs-TTS benchmark using /v1/audio/speech.

The default workload loads all 1,088 English samples and cycles through them
with 96 continuously active workers for 110 seconds. The first 20 seconds are
warmup; QPS uses the remaining 90 seconds. Requests use voice cloning,
non-streaming WAV output, speaker Ethan, temperature 0.7, and at most 512 new
tokens.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import datetime
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import aiohttp

from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.tasks.tts import make_tts_send_fn

_CGROUP_CPU_STAT = Path("/sys/fs/cgroup/cpu.stat")
_DEFAULT_MODEL = "bosonai/higgs-tts-3-4b"
_DEFAULT_DATASET = DATASETS["seedtts"]
_DEFAULT_DATASET_REPO_ID = DATASETS["seedtts"]
_DEFAULT_DATASET_REVISION = ""
_DEFAULT_DATASET_PARQUET_SHA256 = ""
_DEFAULT_BASE_URL = "http://127.0.0.1:8901"
_DEFAULT_CONCURRENCY = 96
_DEFAULT_TOTAL_SECONDS = 110.0
_DEFAULT_WARMUP_SECONDS = 20.0
_DEFAULT_MAX_NEW_TOKENS = 512
_DEFAULT_EXPECTED_SAMPLES = 1088
_DEFAULT_EXPECTED_UNIQUE_REFERENCES = 666
_DEFAULT_SPEAKER = "Ethan"
_DEFAULT_TEMPERATURE = 0.7


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for row in sorted(rows, key=lambda item: int(item["sequence"])):
            output_file.write(json.dumps(row) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _combined_digest(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _offline_mode_enabled() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def _resolve_dataset_source(
    args: argparse.Namespace,
) -> tuple[str, Path | None, str | None]:
    if not args.dataset_revision:
        source_path = Path(args.meta).expanduser()
        if not source_path.is_dir():
            if args.dataset_parquet_sha256:
                raise RuntimeError(
                    "--dataset-parquet-sha256 requires a local dataset directory "
                    "or a pinned --dataset-revision"
                )
            return args.meta, None, None
        snapshot_dir = source_path.resolve()
        parquet_path = snapshot_dir / f"data/{args.lang}-00000-of-00001.parquet"
        if not parquet_path.is_file():
            raise RuntimeError(f"dataset parquet is missing: {parquet_path}")
        parquet_sha256 = _sha256_file(parquet_path)
        if (
            args.dataset_parquet_sha256
            and parquet_sha256 != args.dataset_parquet_sha256
        ):
            raise RuntimeError(
                "dataset parquet SHA-256 mismatch: "
                f"{parquet_sha256} != {args.dataset_parquet_sha256}"
            )
        return str(snapshot_dir), parquet_path, parquet_sha256

    source_path = Path(args.meta).expanduser()
    if source_path.is_dir():
        snapshot_dir = source_path.resolve()
    else:
        if args.meta != args.dataset_repo_id:
            raise RuntimeError(
                "a pinned dataset must be either the declared repository id "
                "or an exact local snapshot directory"
            )
        try:
            from huggingface_hub import snapshot_download

            snapshot_dir = Path(
                snapshot_download(
                    repo_id=args.dataset_repo_id,
                    repo_type="dataset",
                    revision=args.dataset_revision,
                    local_files_only=_offline_mode_enabled(),
                )
            ).resolve()
        except Exception as exc:
            raise RuntimeError(
                "failed to resolve the pinned SeedTTS snapshot; set --meta to "
                "its local snapshot directory or set HF_HUB_OFFLINE=0 once "
                "to download it"
            ) from exc

    if snapshot_dir.name != args.dataset_revision:
        raise RuntimeError(
            "dataset snapshot mismatch: "
            f"{snapshot_dir.name} != {args.dataset_revision}"
        )

    parquet_path = snapshot_dir / f"data/{args.lang}-00000-of-00001.parquet"
    if not parquet_path.is_file():
        raise RuntimeError(f"dataset parquet is missing: {parquet_path}")
    parquet_sha256 = _sha256_file(parquet_path)
    if args.dataset_parquet_sha256 and parquet_sha256 != args.dataset_parquet_sha256:
        raise RuntimeError(
            "dataset parquet SHA-256 mismatch: "
            f"{parquet_sha256} != {args.dataset_parquet_sha256}"
        )
    return str(snapshot_dir), parquet_path, parquet_sha256


def _read_cpu_stat() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in _CGROUP_CPU_STAT.read_text(encoding="utf-8").splitlines():
        key, value = line.split()
        values[key] = int(value)
    return values


def _cpu_stat_delta(
    before: dict[str, int],
    after: dict[str, int],
) -> dict[str, int]:
    return {key: value - before.get(key, 0) for key, value in after.items()}


def _cpu_affinity_snapshot() -> dict[str, Any]:
    """Record the client's effective process and per-thread CPU affinity."""

    process_cpus = sorted(os.sched_getaffinity(0))
    thread_affinities: dict[str, list[int]] = {}
    for task_path in sorted(
        Path("/proc/self/task").iterdir(), key=lambda path: int(path.name)
    ):
        try:
            thread_affinities[task_path.name] = sorted(
                os.sched_getaffinity(int(task_path.name))
            )
        except (FileNotFoundError, ProcessLookupError):
            continue

    nodes: set[int] = set()
    physical_cores: set[tuple[int, int]] = set()
    for cpu in process_cpus:
        cpu_path = Path(f"/sys/devices/system/cpu/cpu{cpu}")
        node_paths = sorted(cpu_path.glob("node[0-9]*"))
        if len(node_paths) != 1:
            raise RuntimeError(f"CPU {cpu} has unexpected NUMA node links: {node_paths}")
        nodes.add(int(node_paths[0].name.removeprefix("node")))
        package = int(
            (cpu_path / "topology/physical_package_id").read_text().strip()
        )
        core = int((cpu_path / "topology/core_id").read_text().strip())
        physical_cores.add((package, core))

    unique_thread_affinities = sorted(
        {tuple(cpus) for cpus in thread_affinities.values()}
    )
    return {
        "process_cpus": process_cpus,
        "process_cpu_count": len(process_cpus),
        "numa_nodes": sorted(nodes),
        "physical_core_count": len(physical_cores),
        "thread_count": len(thread_affinities),
        "thread_affinity_variants": [list(cpus) for cpus in unique_thread_affinities],
        "all_threads_match_process": unique_thread_affinities == [tuple(process_cpus)],
    }


async def _capture_cpu_stat_at(deadline: float) -> dict[str, int]:
    await asyncio.sleep(max(deadline - time.perf_counter(), 0.0))
    return _read_cpu_stat()


def _parse_nvidia_timestamp(value: str) -> float:
    parsed = datetime.datetime.strptime(value.strip(), "%Y/%m/%d %H:%M:%S.%f")
    return parsed.timestamp()


def _metric_statistics(values: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def _summarize_nvidia_smi(
    csv_path: Path,
    *,
    start_epoch_s: float,
    duration_s: float,
) -> dict[str, Any]:
    end_epoch_s = start_epoch_s + duration_s
    metric_names = {
        "utilization.gpu [%]": "device_busy_percent",
        "utilization.memory [%]": "memory_controller_busy_percent",
        "memory.used [MiB]": "memory_used_mib",
        "power.draw [W]": "power_draw_w",
        "clocks.current.sm [MHz]": "sm_clock_mhz",
        "clocks.current.memory [MHz]": "memory_clock_mhz",
    }
    values_by_metric: dict[str, list[float]] = {
        output_name: [] for output_name in metric_names.values()
    }
    selected_rows = 0
    with csv_path.open(encoding="utf-8") as input_file:
        for raw_row in csv.DictReader(input_file, skipinitialspace=True):
            row = {key.strip(): value.strip() for key, value in raw_row.items()}
            sample_epoch_s = _parse_nvidia_timestamp(row["timestamp"])
            if not start_epoch_s <= sample_epoch_s < end_epoch_s:
                continue
            selected_rows += 1
            for source_name, output_name in metric_names.items():
                raw_value = row.get(source_name)
                if raw_value and raw_value != "N/A":
                    values_by_metric[output_name].append(
                        float(raw_value.split(maxsplit=1)[0])
                    )

    if selected_rows == 0:
        raise RuntimeError(f"no nvidia-smi samples in measurement window: {csv_path}")
    return {
        "source": str(csv_path),
        "window_start_epoch_s": start_epoch_s,
        "window_duration_s": duration_s,
        "selected_rows": selected_rows,
        "semantics": (
            "device_busy_percent is nvidia-smi GPU busy time, not Nsight "
            "Systems SM Active or achieved occupancy"
        ),
        "metrics": {
            name: _metric_statistics(values)
            for name, values in values_by_metric.items()
            if values
        },
    }


def _result_record(
    result: Any,
    *,
    sequence: int,
    sample_index: int,
    sample_id: str,
    completed_s: float,
) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "sample_index": sample_index,
        "sample_id": sample_id,
        "completed_s": completed_s,
        "is_success": bool(result.is_success),
        "latency_s": float(result.latency_s),
        "engine_time_s": float(result.engine_time_s),
        "audio_duration_s": float(result.audio_duration_s),
        "prompt_tokens": int(result.prompt_tokens),
        "completion_tokens": int(result.completion_tokens),
        "rtf": float(result.rtf),
        "error": result.error,
    }


async def _run_closed_loop(
    session: aiohttp.ClientSession,
    samples: list[Any],
    args: argparse.Namespace,
    output_dir: Path,
    send_fn: Any,
) -> tuple[list[dict[str, Any]], float, dict[str, int] | None]:
    started_epoch_s = time.time_ns() / 1e9
    started = time.perf_counter()
    stop_at = started + args.secs
    measurement_start = started + args.warmup_secs
    measurement_start_epoch_s = started_epoch_s + args.warmup_secs
    (output_dir / "measurement-window-started.epoch").write_text(
        f"{measurement_start_epoch_s:.9f}\n",
        encoding="utf-8",
    )

    cpu_before_task: asyncio.Task[dict[str, int]] | None = None
    cpu_after_task: asyncio.Task[dict[str, int]] | None = None
    if args.collect_cpu_metrics:
        cpu_before_task = asyncio.create_task(_capture_cpu_stat_at(measurement_start))
        cpu_after_task = asyncio.create_task(_capture_cpu_stat_at(stop_at))

    records: list[dict[str, Any]] = []

    async def progress_reporter() -> None:
        while True:
            await asyncio.sleep(5)
            elapsed = time.perf_counter() - started
            successes = sum(record["is_success"] for record in records)
            errors = len(records) - successes
            phase = "warmup" if elapsed < args.warmup_secs else "measure"
            print(
                f"progress phase={phase} elapsed={elapsed:.1f}/{args.secs:.1f}s "
                f"completed={len(records)} success={successes} errors={errors}",
                flush=True,
            )

    async def worker(worker_index: int) -> None:
        sequence = args.sample_offset + worker_index
        while time.perf_counter() < stop_at:
            sample_index = sequence % len(samples)
            sample = samples[sample_index]
            result = await send_fn(session, sample)
            completed_s = time.perf_counter() - started
            records.append(
                _result_record(
                    result,
                    sequence=sequence,
                    sample_index=sample_index,
                    sample_id=sample.sample_id,
                    completed_s=completed_s,
                )
            )
            sequence += args.concurrency

    progress_task = asyncio.create_task(progress_reporter())
    try:
        await asyncio.gather(*(worker(index) for index in range(args.concurrency)))
    finally:
        progress_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await progress_task
    wall_time_s = time.perf_counter() - started

    cpu_stat_delta: dict[str, int] | None = None
    if cpu_before_task is not None and cpu_after_task is not None:
        cpu_stat_before = await cpu_before_task
        cpu_stat_after = await cpu_after_task
        cpu_stat_delta = _cpu_stat_delta(cpu_stat_before, cpu_stat_after)
        _write_json(output_dir / "cpu-stat-before.json", cpu_stat_before)
        _write_json(output_dir / "cpu-stat-after.json", cpu_stat_after)
        _write_json(output_dir / "cpu-stat-delta.json", cpu_stat_delta)

    return records, wall_time_s, cpu_stat_delta


def _summarize(
    records: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    wall_time_s: float,
    cpu_stat_delta: dict[str, int] | None,
) -> dict[str, Any]:
    successful = [record for record in records if record["is_success"]]
    in_window = [
        record
        for record in successful
        if args.warmup_secs <= record["completed_s"] <= args.secs
    ]
    warmup = [record for record in records if record["completed_s"] < args.warmup_secs]
    tail = [record for record in records if record["completed_s"] > args.secs]
    latencies = [record["latency_s"] for record in in_window]
    prompt_tokens = [record["prompt_tokens"] for record in in_window]
    completion_tokens = [record["completion_tokens"] for record in in_window]
    audio_durations = [record["audio_duration_s"] for record in in_window]
    successful_sample_indices = {
        int(record["sample_index"]) for record in successful
    }
    measured_sample_indices = {
        int(record["sample_index"]) for record in in_window
    }
    measurement_seconds = args.secs - args.warmup_secs

    buckets: dict[int, int] = {}
    for record in successful:
        bucket = int(record["completed_s"] // 5) * 5
        buckets[bucket] = buckets.get(bucket, 0) + 1
    bucket_qps = {
        bucket: round(count / 5.0, 2) for bucket, count in sorted(buckets.items())
    }
    steady_bucket_values = [
        value
        for bucket, value in bucket_qps.items()
        if args.warmup_secs <= bucket < args.secs
    ]

    qps_window = round(len(in_window) / measurement_seconds, 4)
    summary: dict[str, Any] = {
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "dataset": args.dataset_repo_id,
        "dataset_revision": args.dataset_revision or None,
        "dataset_path": args.resolved_meta,
        "dataset_parquet_sha256": args.resolved_dataset_parquet_sha256,
        "dataset_split": args.lang,
        "request_api": "/v1/audio/speech",
        "request_input": "SeedTTS target_text",
        "reference_payload": "references[0].audio_path + references[0].text",
        "voice_clone": args.voice_clone,
        "speaker": args.speaker,
        "temperature": args.temperature,
        "stream": False,
        "concurrency": args.concurrency,
        "secs": args.secs,
        "warmup_secs": args.warmup_secs,
        "measurement_seconds": measurement_seconds,
        "max_new_tokens": args.max_new_tokens,
        "sample_mode": "worker-strided-cycle",
        "sample_offset": args.sample_offset,
        "dataset_samples": args.expected_samples,
        "unique_samples_total": len(successful_sample_indices),
        "unique_samples_window": len(measured_sample_indices),
        "full_dataset_covered_total": (
            len(successful_sample_indices) == args.expected_samples
        ),
        "full_dataset_covered_window": (
            len(measured_sample_indices) == args.expected_samples
        ),
        "n_completions_total": len(successful),
        "n_completions_window": len(in_window),
        "warmup_completions": len(warmup),
        "tail_completions_excluded": len(tail),
        "errors": sum(not record["is_success"] for record in records),
        "qps_window": qps_window,
        "throughput_qps": qps_window,
        "qps_overall": round(len(successful) / args.secs, 4),
        "lat_mean_s": statistics.fmean(latencies) if latencies else 0.0,
        "lat_p50_s": _percentile(latencies, 0.50),
        "lat_p95_s": _percentile(latencies, 0.95),
        "lat_p99_s": _percentile(latencies, 0.99),
        "prompt_tokens_min": min(prompt_tokens) if prompt_tokens else 0,
        "prompt_tokens_mean": (
            statistics.fmean(prompt_tokens) if prompt_tokens else 0.0
        ),
        "completion_tokens_min": min(completion_tokens) if completion_tokens else 0,
        "completion_tokens_mean": (
            statistics.fmean(completion_tokens) if completion_tokens else 0.0
        ),
        "audio_duration_min_s": min(audio_durations) if audio_durations else 0.0,
        "audio_duration_mean_s": (
            statistics.fmean(audio_durations) if audio_durations else 0.0
        ),
        "all_measured_prompt_tokens_gt_3": bool(prompt_tokens)
        and min(prompt_tokens) > 3,
        "all_measured_audio_nonempty": bool(audio_durations)
        and min(audio_durations) > 0,
        "bucket_qps_cv": (
            round(
                statistics.pstdev(steady_bucket_values)
                / (statistics.fmean(steady_bucket_values) + 1e-9),
                3,
            )
            if steady_bucket_values
            else None
        ),
        "bucket_qps": bucket_qps,
        "wall_time_including_tail_s": wall_time_s,
    }
    summary["latency_mean_s"] = summary["lat_mean_s"]
    summary["latency_p95_s"] = summary["lat_p95_s"]

    if cpu_stat_delta is not None:
        cpu_seconds = cpu_stat_delta.get("usage_usec", 0) / 1e6
        summary.update(
            {
                "container_cpu_seconds": cpu_seconds,
                "container_cpu_average_cores": cpu_seconds / measurement_seconds,
                "container_cpu_seconds_per_success": (
                    cpu_seconds / len(in_window) if in_window else 0.0
                ),
                "cpu_stat_delta": cpu_stat_delta,
            }
        )
    return summary


async def _run(args: argparse.Namespace) -> None:
    (
        args.resolved_meta,
        dataset_parquet_path,
        args.resolved_dataset_parquet_sha256,
    ) = _resolve_dataset_source(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    client_affinity_start = _cpu_affinity_snapshot()
    _write_json(output_dir / "client-affinity-start.json", client_affinity_start)
    samples = load_seedtts_samples(args.resolved_meta, None, split=args.lang)
    if len(samples) != args.expected_samples:
        raise RuntimeError(
            "dataset sample count mismatch: "
            f"{len(samples)} != {args.expected_samples}"
        )

    sample_ids = [sample.sample_id for sample in samples]
    unique_reference_audio = len(
        {str(Path(sample.ref_audio).resolve()) for sample in samples}
    )
    dataset_contract = {
        "repo_id": args.dataset_repo_id,
        "requested_source": args.meta,
        "resolved_source": args.resolved_meta,
        "revision": args.dataset_revision or None,
        "parquet": str(dataset_parquet_path) if dataset_parquet_path else None,
        "parquet_sha256": args.resolved_dataset_parquet_sha256,
        "split": args.lang,
        "samples": len(samples),
        "sample_ids": sample_ids,
        "sample_ids_sha256": _combined_digest(sample_ids),
        "unique_reference_audio": unique_reference_audio,
        "nonempty_target_text": sum(
            bool(sample.target_text.strip()) for sample in samples
        ),
        "nonempty_reference_text": sum(
            bool(sample.ref_text.strip()) for sample in samples
        ),
        "existing_reference_audio": sum(
            Path(sample.ref_audio).is_file() for sample in samples
        ),
        "target_text_min_chars": min(len(sample.target_text) for sample in samples),
        "target_text_max_chars": max(len(sample.target_text) for sample in samples),
        "target_text_sha256": hashlib.sha256(
            "\n".join(sample.target_text for sample in samples).encode()
        ).hexdigest(),
        "reference_text_sha256": hashlib.sha256(
            "\n".join(sample.ref_text for sample in samples).encode()
        ).hexdigest(),
        "payload": {
            "endpoint": "/v1/audio/speech",
            "input": "sample.target_text",
            "references": [
                {
                    "audio_path": "sample.ref_audio",
                    "text": "sample.ref_text",
                }
            ],
            "voice": args.speaker,
            "response_format": "wav",
            "stream": False,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
        },
    }
    _write_json(output_dir / "request-contract-proof.json", dataset_contract)
    if (
        dataset_contract["nonempty_target_text"] != args.expected_samples
        or dataset_contract["nonempty_reference_text"] != args.expected_samples
        or dataset_contract["existing_reference_audio"] != args.expected_samples
        or (
            args.expected_unique_references > 0
            and unique_reference_audio != args.expected_unique_references
        )
    ):
        raise RuntimeError(f"invalid SeedTTS request contract: {dataset_contract}")

    save_audio_dir = tempfile.mkdtemp(
        dir="/dev/shm" if os.path.isdir("/dev/shm") else None
    )
    send_fn = make_tts_send_fn(
        args.model,
        f"{args.base_url.rstrip('/')}/v1/audio/speech",
        response_format="wav",
        stream=False,
        no_ref_audio=False,
        ref_format="references",
        voice=args.speaker,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        save_audio_dir=save_audio_dir,
    )

    try:
        timeout = aiohttp.ClientTimeout(total=args.request_timeout_s)
        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
        async with aiohttp.ClientSession(
            timeout=timeout, connector=connector
        ) as session:
            records, wall_time_s, cpu_stat_delta = await _run_closed_loop(
                session,
                samples,
                args,
                output_dir,
                send_fn,
            )

        _write_jsonl(output_dir / "requests.jsonl", records)
        summary = _summarize(
            records,
            args=args,
            wall_time_s=wall_time_s,
            cpu_stat_delta=cpu_stat_delta,
        )
        client_affinity_end = _cpu_affinity_snapshot()
        _write_json(output_dir / "client-affinity-end.json", client_affinity_end)
        summary["client_cpu_affinity_start"] = client_affinity_start
        summary["client_cpu_affinity_end"] = client_affinity_end
        # note (likaige): Preserve the primary result if diagnostics parsing fails.
        _write_json(output_dir / "summary.json", summary)
        if args.gpu_metrics_csv:
            measurement_start_epoch_s = float(
                (output_dir / "measurement-window-started.epoch").read_text(
                    encoding="utf-8"
                )
            )
            gpu_metrics = _summarize_nvidia_smi(
                Path(args.gpu_metrics_csv),
                start_epoch_s=measurement_start_epoch_s,
                duration_s=args.secs - args.warmup_secs,
            )
            summary["nvidia_smi_metrics"] = gpu_metrics
            _write_json(output_dir / "gpu-metrics-summary.json", gpu_metrics)
        _write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        if summary["errors"] != 0:
            raise RuntimeError(f"request failures observed: {summary['errors']}")
        if not math.isfinite(float(summary["qps_window"])) or not (
            float(summary["qps_window"]) > 0
        ):
            raise RuntimeError("measured QPS must be finite and positive")
        if args.require_full_dataset_coverage and not (
            summary["full_dataset_covered_total"]
            and summary["full_dataset_covered_window"]
        ):
            raise RuntimeError(
                "full SeedTTS coverage gate failed: "
                f"total={summary['unique_samples_total']}/{args.expected_samples}, "
                f"window={summary['unique_samples_window']}/{args.expected_samples}"
            )
        if not summary["all_measured_prompt_tokens_gt_3"]:
            raise RuntimeError(
                "prompt-token gate failed; /v1/audio/speech may not have "
                "received the intended non-empty SeedTTS text"
            )
        if not summary["all_measured_audio_nonempty"]:
            raise RuntimeError("one or more measured audio responses were empty")
    finally:
        shutil.rmtree(save_audio_dir, ignore_errors=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Closed-loop Higgs-TTS /v1/audio/speech benchmark."
    )
    parser.add_argument("--label", default="dp1-c96-r1")
    parser.add_argument("--base-url", default=_DEFAULT_BASE_URL)
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--meta", default=_DEFAULT_DATASET)
    parser.add_argument(
        "--dataset-repo-id",
        default=_DEFAULT_DATASET_REPO_ID,
    )
    parser.add_argument(
        "--dataset-revision",
        default=_DEFAULT_DATASET_REVISION,
    )
    parser.add_argument(
        "--dataset-parquet-sha256",
        default=_DEFAULT_DATASET_PARQUET_SHA256,
    )
    parser.add_argument("--lang", choices=["en", "zh"], default="en")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--secs", type=float, default=_DEFAULT_TOTAL_SECONDS)
    parser.add_argument(
        "--warmup-secs",
        type=float,
        default=_DEFAULT_WARMUP_SECONDS,
    )
    parser.add_argument(
        "--concurrency",
        "--conc",
        dest="concurrency",
        type=int,
        default=_DEFAULT_CONCURRENCY,
    )
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=_DEFAULT_MAX_NEW_TOKENS,
    )
    parser.add_argument("--voice-clone", action="store_true")
    parser.add_argument("--speaker", default=_DEFAULT_SPEAKER)
    parser.add_argument("--temperature", type=float, default=_DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=_DEFAULT_EXPECTED_SAMPLES,
    )
    parser.add_argument(
        "--expected-unique-references",
        type=int,
        default=_DEFAULT_EXPECTED_UNIQUE_REFERENCES,
    )
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--require-full-dataset-coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require every dataset sample to complete successfully both over the "
            "whole run and inside the measured window."
        ),
    )
    parser.add_argument("--collect-cpu-metrics", action="store_true")
    parser.add_argument(
        "--gpu-metrics-csv",
        help="Optional nvidia-smi query CSV sampled throughout the run.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.secs <= 0:
        parser.error("--secs must be positive")
    if not 0 <= args.warmup_secs < args.secs:
        parser.error("--warmup-secs must be non-negative and less than --secs")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.sample_offset < 0:
        parser.error("--sample-offset must be non-negative")
    if args.expected_samples <= 0:
        parser.error("--expected-samples must be positive")
    if args.expected_unique_references < 0:
        parser.error("--expected-unique-references must be non-negative")
    if args.dataset_revision and (
        len(args.dataset_revision) != 40
        or any(
            character not in "0123456789abcdef" for character in args.dataset_revision
        )
    ):
        parser.error("--dataset-revision must be a lowercase 40-character commit")
    if args.dataset_parquet_sha256 and (
        len(args.dataset_parquet_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in args.dataset_parquet_sha256
        )
    ):
        parser.error("--dataset-parquet-sha256 must be a lowercase SHA-256")
    if not args.voice_clone:
        parser.error("--voice-clone is required by the PR #1071 benchmark contract")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
