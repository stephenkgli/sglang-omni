#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

# Compare one scheduler at c96 with two schedulers at c192. Both arms keep
# frontend=2, give every scheduler replica capacity/CUDA graphs up to batch 96,
# and require every GPU-owning process to attach to a private CUDA MPS daemon.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-/opt/sglang-omni-kaige}"
SINGLE_RUNNER="${SINGLE_RUNNER:-${SCRIPT_DIR}/run_higgs_tts_single_gpu_throughput.sh}"
BENCHMARK_CLIENT="${BENCHMARK_CLIENT:-${SCRIPT_DIR}/benchmark_higgs_tts_throughput.py}"
MPS_VERIFIER="${MPS_VERIFIER:-${SCRIPT_DIR}/verify_higgs_tts_full_pipeline_mps.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"

readonly MODEL_PATH="bosonai/higgs-tts-3-4b"
readonly HF_HOME="/data/huggingface"
readonly DATASET_REPO_ID="zhaochenyang20/seed-tts-eval-arrow"
readonly DATASET_REVISION="81d1901582dee1293a537a6d945d084301712c41"
readonly DATASET_PARQUET_SHA256="5849b41b49cae996328c06d2c5791717c3bafc369bddfa1ec4f86761bb8bc0ca"
readonly FRONTEND_REPLICAS=2
readonly ENCODER_MEMORY_FRACTION=0.0245
readonly SECS=110
readonly WARMUP_SECS=20
readonly MAX_NEW_TOKENS=512
readonly DECODE_CUDA_GRAPH_MAX_FRAMES=512
readonly SERVER_THREADS=3
readonly CLIENT_THREADS=2
readonly PORT="${PORT:-8901}"
readonly BOOT_TIMEOUT_SECS="${BOOT_TIMEOUT_SECS:-3600}"
readonly AUTO_CPU_AFFINITY="${AUTO_CPU_AFFINITY:-1}"
readonly MPS_QUERY_TIMEOUT_SECONDS=10
readonly MPS_STARTUP_TIMEOUT_SECONDS=10
readonly MPS_DRAIN_TIMEOUT_SECONDS=40
readonly MPS_SHUTDOWN_TIMEOUT_SECONDS=10

GPU_SELECTOR="${CUDA_VISIBLE_DEVICES-}"
SERVER_CPUSET="${SERVER_CPUSET:-}"
CLIENT_CPUSET="${CLIENT_CPUSET:-}"
SERVER_NUMA_NODE="${SERVER_NUMA_NODE:-}"
CLIENT_NUMA_NODE="${CLIENT_NUMA_NODE:-}"
SERVER_AFFINITY_CPU_COUNT=""
CLIENT_AFFINITY_CPU_COUNT=""
SOURCE_COMMIT=""
RUN_ROOT=""
STATUS_FILE=""
MPS_STATE=""
MPS_CONTROL_PID=""
MPS_ARTIFACT_DIR=""
COMPLETED=0

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

validate_boolean() {
  local name="$1"
  local value="$2"
  [[ "${value}" == "0" || "${value}" == "1" ]] \
    || die "${name} must be 0 or 1"
}

pid_is_live() {
  local pid="$1"
  local status
  kill -0 "${pid}" 2>/dev/null || return 1
  status="$(ps -o stat= -p "${pid}" 2>/dev/null || true)"
  [[ -n "${status// /}" && "${status}" != Z* ]]
}

mps_query() {
  local command="$1"
  local timeout_seconds="${2:-${MPS_QUERY_TIMEOUT_SECONDS}}"
  CUDA_MPS_PIPE_DIRECTORY="${MPS_STATE}/pipe" \
    CUDA_MPS_LOG_DIRECTORY="${MPS_STATE}/log" \
    timeout "${timeout_seconds}" \
    nvidia-cuda-mps-control <<< "${command}" \
    2>> "${MPS_STATE}/control-query.err"
}

mps_control_pid() {
  local pid_file="${MPS_STATE}/pipe/nvidia-cuda-mps-control.pid"
  local pid
  [[ -r "${pid_file}" ]] || return 1
  read -r pid < "${pid_file}"
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 1
  printf '%s\n' "${pid}"
}

mps_server_clients() {
  local server_output
  local server_pid
  local client_output
  local client_pid
  server_output="$(mps_query get_server_list)" || return 1
  while read -r server_pid; do
    [[ "${server_pid}" =~ ^[0-9]+$ ]] || continue
    client_output="$(mps_query "get_client_list ${server_pid}")" || return 1
    while read -r client_pid; do
      [[ "${client_pid}" =~ ^[0-9]+$ ]] || continue
      printf '%s\t%s\n' "${server_pid}" "${client_pid}"
    done <<< "${client_output}"
  done <<< "${server_output}"
}

ensure_no_mps_daemon() {
  local matches
  matches="$(pgrep -af '[n]vidia-cuda-mps-(control|server)' || true)"
  [[ -z "${matches}" ]] \
    || die "an existing CUDA MPS daemon/server is present: ${matches//$'\n'/; }"
}

ensure_gpu_idle() {
  local active_pids
  active_pids="$(
    nvidia-smi -i "${GPU_SELECTOR}" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null \
      | sed '/^[[:space:]]*$/d'
  )"
  [[ -z "${active_pids}" ]] \
    || die "GPU ${GPU_SELECTOR} is already in use: ${active_pids//$'\n'/, }"
}

archive_mps_state() {
  [[ -n "${MPS_STATE}" && -d "${MPS_STATE}" ]] || return 0
  [[ -n "${MPS_ARTIFACT_DIR}" ]] || return 0
  [[ ! -e "${MPS_ARTIFACT_DIR}" ]] || return 0
  mkdir -p "${MPS_ARTIFACT_DIR}"
  cp -a "${MPS_STATE}/log" "${MPS_ARTIFACT_DIR}/log"
  local name
  for name in \
    manifest.txt control.pid control-start.err control-query.err \
    clients-after-drain.tsv; do
    [[ ! -f "${MPS_STATE}/${name}" ]] \
      || cp -p "${MPS_STATE}/${name}" "${MPS_ARTIFACT_DIR}/${name}"
  done
  ls -la "${MPS_STATE}/pipe" \
    > "${MPS_ARTIFACT_DIR}/pipe-directory-listing.txt"
  printf '%s\n' "${MPS_STATE}" \
    > "${MPS_ARTIFACT_DIR}/original-state-path.txt"
}

remove_private_mps_state() {
  case "${MPS_STATE}" in
    /tmp/higgs-topology-full-mps.*)
      rm -rf -- "${MPS_STATE}"
      ;;
    *)
      die "refusing to remove unexpected MPS state path: ${MPS_STATE}"
      ;;
  esac
}

start_private_mps() {
  local arm_dir="$1"
  local label="$2"
  ensure_no_mps_daemon
  ensure_gpu_idle

  local gpu_uuid
  gpu_uuid="$(
    nvidia-smi -i "${GPU_SELECTOR}" --query-gpu=uuid \
      --format=csv,noheader,nounits | tr -d '[:space:]'
  )"
  [[ -n "${gpu_uuid}" ]] || die "failed to resolve GPU UUID"

  MPS_STATE="$(mktemp -d /tmp/higgs-topology-full-mps.XXXXXX)"
  MPS_ARTIFACT_DIR="${arm_dir}/mps-daemon"
  mkdir -p "${MPS_STATE}/pipe" "${MPS_STATE}/log"
  chmod 700 "${MPS_STATE}" "${MPS_STATE}/pipe" "${MPS_STATE}/log"
  printf '%s\n' \
    "arm=${label}" \
    "gpu_selector=${GPU_SELECTOR}" \
    "gpu_uuid=${gpu_uuid}" \
    "scope=all_gpu_processes_in_server_tree" \
    > "${MPS_STATE}/manifest.txt"

  env \
    -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE \
    -u CUDA_MPS_PINNED_DEVICE_MEM_LIMIT \
    -u CUDA_MPS_CLIENT_PRIORITY \
    CUDA_VISIBLE_DEVICES="${gpu_uuid}" \
    CUDA_MPS_PIPE_DIRECTORY="${MPS_STATE}/pipe" \
    CUDA_MPS_LOG_DIRECTORY="${MPS_STATE}/log" \
    nvidia-cuda-mps-control -d \
    2>> "${MPS_STATE}/control-start.err"

  local deadline=$((SECONDS + MPS_STARTUP_TIMEOUT_SECONDS))
  while ((SECONDS < deadline)); do
    if mps_query get_default_active_thread_percentage 1 >/dev/null 2>&1; then
      break
    fi
    sleep 0.2
  done
  mps_query get_default_active_thread_percentage 1 >/dev/null 2>&1 \
    || die "private MPS daemon did not become ready"
  MPS_CONTROL_PID="$(mps_control_pid)" \
    || die "private MPS daemon PID file is missing or invalid"
  pid_is_live "${MPS_CONTROL_PID}" \
    || die "private MPS daemon exited during startup"
  printf '%s\n' "${MPS_CONTROL_PID}" > "${MPS_STATE}/control.pid"
}

stop_private_mps() {
  [[ -n "${MPS_STATE}" ]] || return 0
  local deadline=$((SECONDS + MPS_DRAIN_TIMEOUT_SECONDS))
  local mappings
  while true; do
    mappings="$(mps_server_clients)" || {
      archive_mps_state
      echo "ERROR: private MPS control query failed; state preserved at ${MPS_STATE}" >&2
      return 1
    }
    [[ -z "${mappings}" ]] && break
    if ((SECONDS >= deadline)); then
      printf '%s\n' "${mappings}" > "${MPS_STATE}/clients-after-drain.tsv"
      archive_mps_state
      echo "ERROR: MPS clients remain after server teardown; state preserved at ${MPS_STATE}" >&2
      return 1
    fi
    sleep 1
  done

  mps_query quit >/dev/null || {
    archive_mps_state
    echo "ERROR: failed to stop private MPS daemon; state preserved at ${MPS_STATE}" >&2
    return 1
  }
  deadline=$((SECONDS + MPS_SHUTDOWN_TIMEOUT_SECONDS))
  while pid_is_live "${MPS_CONTROL_PID}"; do
    if ((SECONDS >= deadline)); then
      archive_mps_state
      echo "ERROR: private MPS daemon did not exit; state preserved at ${MPS_STATE}" >&2
      return 1
    fi
    sleep 1
  done

  archive_mps_state
  remove_private_mps_state
  MPS_STATE=""
  MPS_CONTROL_PID=""
  MPS_ARTIFACT_DIR=""
}

on_exit() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${MPS_STATE}" ]]; then
    stop_private_mps || status=1
  fi
  if [[ -n "${STATUS_FILE}" ]]; then
    if [[ "${COMPLETED}" == "1" && "${status}" == "0" ]]; then
      printf 'complete\n' > "${STATUS_FILE}"
    else
      printf 'failed exit_status=%s\n' "${status}" > "${STATUS_FILE}"
    fi
  fi
  exit "${status}"
}

resolve_cpu_affinity() {
  local supplied=0
  [[ -n "${SERVER_CPUSET}" ]] && supplied=$((supplied + 1))
  [[ -n "${CLIENT_CPUSET}" ]] && supplied=$((supplied + 1))
  [[ -n "${SERVER_NUMA_NODE}" ]] && supplied=$((supplied + 1))
  [[ -n "${CLIENT_NUMA_NODE}" ]] && supplied=$((supplied + 1))

  if ((supplied == 0)); then
    [[ "${AUTO_CPU_AFFINITY}" == "1" ]] \
      || die "CPU affinity is unset and AUTO_CPU_AFFINITY=0"
    local resolved_text
    resolved_text="$(
      "${PYTHON_BIN}" - "${GPU_SELECTOR}" <<'PY'
import os
import pathlib
import subprocess
import sys


gpu_id = sys.argv[1]
pci_bus_id = subprocess.check_output(
    [
        "nvidia-smi",
        "-i",
        gpu_id,
        "--query-gpu=pci.bus_id",
        "--format=csv,noheader,nounits",
    ],
    text=True,
).strip().lower()
domain, separator, device = pci_bus_id.partition(":")
if not separator or len(domain) not in {4, 8}:
    raise SystemExit(f"unexpected GPU PCI bus id: {pci_bus_id}")
if len(domain) == 8:
    domain = domain[-4:]
pci_bus_id = f"{domain}:{device}"
numa_path = pathlib.Path(f"/sys/bus/pci/devices/{pci_bus_id}/numa_node")
if not numa_path.is_file():
    raise SystemExit(f"GPU NUMA path is unavailable: {numa_path}")
gpu_node = int(numa_path.read_text().strip())
if gpu_node < 0:
    raise SystemExit(f"GPU {gpu_id} has no usable NUMA node")

nodes: dict[int, list[int]] = {}
seen_cores: set[tuple[int, int]] = set()
for cpu in sorted(os.sched_getaffinity(0)):
    root = pathlib.Path(f"/sys/devices/system/cpu/cpu{cpu}")
    node_links = sorted(root.glob("node[0-9]*"))
    if len(node_links) != 1:
        continue
    node = int(node_links[0].name.removeprefix("node"))
    package = int((root / "topology/physical_package_id").read_text().strip())
    core = int((root / "topology/core_id").read_text().strip())
    core_key = (package, core)
    if core_key in seen_cores:
        continue
    seen_cores.add(core_key)
    nodes.setdefault(node, []).append(cpu)

server_cpus = nodes.get(gpu_node, [])
if not server_cpus:
    raise SystemExit(
        f"GPU-local NUMA node {gpu_node} exposes no available physical cores"
    )
client_nodes = [node for node, cpus in nodes.items() if node != gpu_node and cpus]
if not client_nodes:
    details = ", ".join(
        f"node{node}={len(cpus)}" for node, cpus in sorted(nodes.items())
    )
    raise SystemExit(
        f"no non-GPU NUMA node exposes available physical cores ({details})"
    )
client_node = min(client_nodes, key=lambda node: (-len(nodes[node]), node))
client_cpus = nodes[client_node]

print(",".join(map(str, server_cpus)))
print(",".join(map(str, client_cpus)))
print(gpu_node)
print(client_node)
print(len(server_cpus))
print(len(client_cpus))
PY
    )" || die "failed to auto-discover disjoint CPU/NUMA affinity"

    local resolved=()
    mapfile -t resolved <<< "${resolved_text}"
    [[ "${#resolved[@]}" == "6" ]] \
      || die "CPU affinity discovery returned ${#resolved[@]} fields, expected 6"
    SERVER_CPUSET="${resolved[0]}"
    CLIENT_CPUSET="${resolved[1]}"
    SERVER_NUMA_NODE="${resolved[2]}"
    CLIENT_NUMA_NODE="${resolved[3]}"
    SERVER_AFFINITY_CPU_COUNT="${resolved[4]}"
    CLIENT_AFFINITY_CPU_COUNT="${resolved[5]}"
    return
  fi

  ((supplied == 4)) \
    || die "set either none or all four CPU/NUMA affinity variables"
  local count_text
  count_text="$(
    "${PYTHON_BIN}" - "${SERVER_CPUSET}" "${CLIENT_CPUSET}" <<'PY'
import sys


def parse_cpu_list(value: str) -> set[int]:
    cpus: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise SystemExit(f"invalid CPU range: {item}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(item))
    if not cpus:
        raise SystemExit(f"empty CPU list: {value!r}")
    return cpus


print(len(parse_cpu_list(sys.argv[1])))
print(len(parse_cpu_list(sys.argv[2])))
PY
  )" || die "failed to count manually configured CPU sets"
  local counts=()
  mapfile -t counts <<< "${count_text}"
  [[ "${#counts[@]}" == "2" ]] \
    || die "CPU affinity counter returned ${#counts[@]} fields, expected 2"
  SERVER_AFFINITY_CPU_COUNT="${counts[0]}"
  CLIENT_AFFINITY_CPU_COUNT="${counts[1]}"
}

validate_attachment() {
  local arm_dir="$1"
  local key_path="${arm_dir}/manifest/attachment-key.txt"
  (
    cd "${arm_dir}/manifest"
    sha256sum \
      hostname.txt boot-id.txt gpu-identity.csv cpu-max.txt \
      cpuset-cpus-effective.txt cpuset-mems-effective.txt
  ) | sha256sum | cut -d' ' -f1 > "${key_path}"
  if [[ ! -f "${RUN_ROOT}/attachment-key.txt" ]]; then
    cp "${key_path}" "${RUN_ROOT}/attachment-key.txt"
  elif ! cmp -s "${key_path}" "${RUN_ROOT}/attachment-key.txt"; then
    die "active attachment changed between comparison arms"
  fi
}

run_arm() {
  local label="$1"
  local directory="$2"
  local scheduler_replicas="$3"
  local concurrency="$4"
  local scheduler_memory_fraction="$5"
  local scheduler_capacity_each=$((concurrency / scheduler_replicas))
  local arm_dir="${RUN_ROOT}/${directory}"

  echo "=== ${label}: scheduler=${scheduler_replicas} frontend=2 concurrency=${concurrency} full-pipeline MPS ==="
  start_private_mps "${arm_dir}" "${label}"

  env \
    -u CUDA_MPS_PIPE_DIRECTORY \
    -u CUDA_MPS_LOG_DIRECTORY \
    -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE \
    -u CUDA_MPS_PINNED_DEVICE_MEM_LIMIT \
    -u CUDA_MPS_CLIENT_PRIORITY \
    ARM="${label}" \
    FRONTEND_REPLICAS="${FRONTEND_REPLICAS}" \
    ENCODER_MEMORY_FRACTION="${ENCODER_MEMORY_FRACTION}" \
    SCHEDULER_REPLICAS="${scheduler_replicas}" \
    SCHEDULER_MEMORY_FRACTION="${scheduler_memory_fraction}" \
    SCHEDULER_MAX_RUNNING_REQUESTS="${scheduler_capacity_each}" \
    SCHEDULER_CUDA_GRAPH_MAX_BS="${scheduler_capacity_each}" \
    SERVER_MPS_MODE=required \
    SERVER_MPS_PIPE_DIRECTORY="${MPS_STATE}/pipe" \
    SERVER_MPS_LOG_DIRECTORY="${MPS_STATE}/log" \
    REPS=1 \
    RUN_DIR="${arm_dir}" \
    REPO_DIR="${REPO_DIR}" \
    BENCHMARK_CLIENT="${BENCHMARK_CLIENT}" \
    FULL_PIPELINE_MPS_VERIFIER="${MPS_VERIFIER}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    MODEL_PATH="${MODEL_PATH}" \
    HF_HOME="${HF_HOME}" \
    HF_HUB_OFFLINE=0 \
    HF_DATASETS_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0 \
    DATASET="${DATASET_REPO_ID}" \
    DATASET_REPO_ID="${DATASET_REPO_ID}" \
    DATASET_REVISION="${DATASET_REVISION}" \
    DATASET_PARQUET_SHA256="${DATASET_PARQUET_SHA256}" \
    DATASET_SOURCE_MODE=huggingface \
    CUDA_VISIBLE_DEVICES="${GPU_SELECTOR}" \
    PORT="${PORT}" \
    CAP="${concurrency}" \
    CONC="${concurrency}" \
    CUDA_GRAPH_MAX_BS="${concurrency}" \
    SECS="${SECS}" \
    WARMUP_SECS="${WARMUP_SECS}" \
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
    DECODE_CUDA_GRAPH_MAX_FRAMES="${DECODE_CUDA_GRAPH_MAX_FRAMES}" \
    SERVER_THREADS="${SERVER_THREADS}" \
    CLIENT_THREADS="${CLIENT_THREADS}" \
    SERVER_CPUSET="${SERVER_CPUSET}" \
    CLIENT_CPUSET="${CLIENT_CPUSET}" \
    SERVER_NUMA_NODE="${SERVER_NUMA_NODE}" \
    CLIENT_NUMA_NODE="${CLIENT_NUMA_NODE}" \
    SERVER_AFFINITY_CPU_COUNT="${SERVER_AFFINITY_CPU_COUNT}" \
    CLIENT_AFFINITY_CPU_COUNT="${CLIENT_AFFINITY_CPU_COUNT}" \
    CGROUP_CPU_QUOTA_POLICY=record_only \
    BOOT_TIMEOUT_SECS="${BOOT_TIMEOUT_SECS}" \
    PREFILL_COALESCING_MODE=required \
    PROFILE=1 \
    NSYS_SM_PROFILE=0 \
    REQUIRE_FULL_DATASET_COVERAGE=1 \
    bash "${SINGLE_RUNNER}"

  validate_attachment "${arm_dir}"
  stop_private_mps \
    || die "MPS cleanup failed for ${label}; no later arm will run"
  ensure_no_mps_daemon
  ensure_gpu_idle
}

summarize_results() {
  "${PYTHON_BIN}" - "${RUN_ROOT}" "${SOURCE_COMMIT}" <<'PY'
import json
import math
import pathlib
import sys


root = pathlib.Path(sys.argv[1])
source_commit = sys.argv[2]
specs = [
    {
        "label": "S1F2C96",
        "directory": "score-00-S1F2C96",
        "scheduler_replicas": 1,
        "frontend_replicas": 2,
        "concurrency": 96,
        "scheduler_capacity_each": 96,
    },
    {
        "label": "S2F2C192",
        "directory": "score-01-S2F2C192",
        "scheduler_replicas": 2,
        "frontend_replicas": 2,
        "concurrency": 192,
        "scheduler_capacity_each": 96,
    },
]
rows = []
for spec in specs:
    arm_dir = root / spec["directory"]
    rep_dir = arm_dir / "rep-1"
    summary = json.loads((rep_dir / "client/summary.json").read_text())
    binding = json.loads((rep_dir / "replica-binding-summary.json").read_text())
    attachment = json.loads(
        (rep_dir / "full-pipeline-mps-attachment.json").read_text()
    )
    config = json.loads((arm_dir / "pipeline-config.json").read_text())
    contract = dict(
        line.split("=", 1)
        for line in (arm_dir / "benchmark-contract.txt").read_text().splitlines()
        if "=" in line
    )
    observed_commit = (arm_dir / "manifest/git-commit.txt").read_text().strip()
    if observed_commit != source_commit:
        raise SystemExit(
            f"source commit drift in {spec['label']}: "
            f"{observed_commit} != {source_commit}"
        )

    qps = float(summary["qps_window"])
    if not math.isfinite(qps) or qps <= 0:
        raise SystemExit(f"invalid QPS in {spec['label']}: {qps}")
    if int(summary["errors"]) != 0:
        raise SystemExit(f"request errors in {spec['label']}")
    if not summary.get("all_measured_prompt_tokens_gt_3"):
        raise SystemExit(f"prompt-token gate failed in {spec['label']}")
    if not summary.get("all_measured_audio_nonempty"):
        raise SystemExit(f"audio gate failed in {spec['label']}")
    if not (
        summary.get("full_dataset_covered_total")
        and summary.get("full_dataset_covered_window")
    ):
        raise SystemExit(f"full-dataset coverage failed in {spec['label']}")

    expected_counts = {"audio_encoder": 2, "preprocessing": 2}
    if spec["scheduler_replicas"] == 2:
        expected_counts["tts_engine"] = 2
    if binding.get("replica_counts") != expected_counts:
        raise SystemExit(
            f"replica topology mismatch in {spec['label']}: {binding}"
        )
    if not binding.get("aligned") or not binding.get("frontend_aligned"):
        raise SystemExit(f"replica bindings are not aligned in {spec['label']}")
    for stage, replica_count in expected_counts.items():
        counts = binding["counts"][stage]
        if sorted(counts) != [str(index) for index in range(replica_count)]:
            raise SystemExit(
                f"binding IDs mismatch for {stage} in {spec['label']}: {counts}"
            )
        if max(counts.values()) - min(counts.values()) > 1:
            raise SystemExit(
                f"bindings are imbalanced for {stage} in {spec['label']}: {counts}"
            )

    if not attachment.get("exact_full_pipeline_attachment"):
        raise SystemExit(
            f"full-pipeline MPS attachment failed in {spec['label']}: "
            f"{attachment.get('violations')}"
        )
    expected_gpu_processes = 2 + spec["scheduler_replicas"] + 1
    attached_processes = attachment.get("expected_gpu_processes", [])
    if len(attached_processes) != expected_gpu_processes:
        raise SystemExit(
            f"MPS process count mismatch in {spec['label']}: "
            f"{len(attached_processes)} != {expected_gpu_processes}"
        )
    if len(attachment.get("mps_server_clients", {})) != 1:
        raise SystemExit(f"expected one MPS server in {spec['label']}")
    if len(attachment.get("mps_client_pids", [])) != expected_gpu_processes:
        raise SystemExit(f"MPS client count mismatch in {spec['label']}")

    stages = {stage["name"]: stage for stage in config["stages"]}
    expected_stage_replicas = {
        "preprocessing": 2,
        "audio_encoder": 2,
        "tts_engine": spec["scheduler_replicas"],
    }
    for stage_name, expected in expected_stage_replicas.items():
        if int(stages[stage_name]["num_replicas"]) != expected:
            raise SystemExit(
                f"pipeline config mismatch for {stage_name} in {spec['label']}"
            )

    expected_contract = {
        "cap": str(spec["concurrency"]),
        "concurrency": str(spec["concurrency"]),
        "logical_cuda_graph_max_bs": str(spec["concurrency"]),
        "frontend_replicas": "2",
        "scheduler_replicas": str(spec["scheduler_replicas"]),
        "scheduler_max_running_requests_each": "96",
        "scheduler_cuda_graph_max_bs_each": "96",
        "server_mps_mode": "required",
        "server_mps_enabled": "1",
        "vocoder_process": "isolated",
        "vocoder_compile_decode": "false",
        "vocoder_decode_cuda_graph_frame_counts": "1..512",
        "require_full_dataset_coverage": "1",
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise SystemExit(
                f"contract mismatch for {key} in {spec['label']}: "
                f"{contract.get(key)!r} != {expected!r}"
            )

    gpu_metrics = summary["nvidia_smi_metrics"]["metrics"]
    cpu_delta = summary.get("cpu_stat_delta") or {}
    rows.append(
        {
            **spec,
            "qps": qps,
            "latency_mean_s": float(summary["lat_mean_s"]),
            "latency_p50_s": float(summary["lat_p50_s"]),
            "latency_p95_s": float(summary["lat_p95_s"]),
            "latency_p99_s": float(summary["lat_p99_s"]),
            "completion_tokens_per_s": qps
            * float(summary["completion_tokens_mean"]),
            "generated_audio_seconds_per_s": qps
            * float(summary["audio_duration_mean_s"]),
            "bucket_qps_cv": float(summary["bucket_qps_cv"]),
            "successful_requests_window": int(summary["n_completions_window"]),
            "container_cpu_average_cores": float(
                summary["container_cpu_average_cores"]
            ),
            "cpu_nr_periods": int(cpu_delta.get("nr_periods", 0)),
            "cpu_nr_throttled": int(cpu_delta.get("nr_throttled", 0)),
            "cpu_throttled_usec": int(cpu_delta.get("throttled_usec", 0)),
            "gpu_device_busy_mean_percent": float(
                gpu_metrics["device_busy_percent"]["mean"]
            ),
            "gpu_memory_used_mean_mib": float(
                gpu_metrics["memory_used_mib"]["mean"]
            ),
            "mps_client_count": len(attachment["mps_client_pids"]),
            "mps_exact_full_pipeline_attachment": True,
        }
    )

baseline, candidate = rows
delta_qps = candidate["qps"] - baseline["qps"]
delta_percent = delta_qps / baseline["qps"] * 100.0
payload = {
    "decision": "accept" if delta_qps > 0 else "reject",
    "decision_scope": (
        "one S1F2C96 start followed by one S2F2C192 start; "
        "no drift, noise, or stability claim"
    ),
    "source_commit": source_commit,
    "primary_metric": "measurement-window successful QPS",
    "arm_order": [spec["label"] for spec in specs],
    "mps_scope": "all_gpu_processes_in_server_tree",
    "delta_qps": delta_qps,
    "delta_percent": delta_percent,
    "rows": rows,
}
(root / "summary.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)

lines = [
    "# Higgs TTS scheduler replicas under full-pipeline MPS",
    "",
    f"- Point result: **{payload['decision']}**",
    f"- QPS delta: {delta_qps:+.4f} ({delta_percent:+.2f}%)",
    "- Evidence scope: one A-to-B point comparison; no stability claim",
    "- Every configured GPU process passed exact private-MPS attachment validation",
    "",
    "| Arm | Scheduler | Frontend | Concurrency | QPS | P50 | P95 | MPS clients |",
    "|:---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['label']} | {row['scheduler_replicas']} | "
        f"{row['frontend_replicas']} | {row['concurrency']} | "
        f"{row['qps']:.4f} | {row['latency_p50_s']:.3f} s | "
        f"{row['latency_p95_s']:.3f} s | {row['mps_client_count']} |"
    )
(root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
}

write_artifact_manifest() {
  local manifest_tmp
  manifest_tmp="$(mktemp /tmp/higgs-artifact-manifest.XXXXXX)"
  (
    cd "${RUN_ROOT}"
    find . -type f ! -name artifact-manifest.sha256 -print0 \
      | sort -z \
      | xargs -0 sha256sum
  ) > "${manifest_tmp}"
  mv "${manifest_tmp}" "${RUN_ROOT}/artifact-manifest.sha256"
}

main() {
  [[ "${GPU_SELECTOR}" =~ ^[0-9]+$ ]] \
    || die "CUDA_VISIBLE_DEVICES must contain exactly one numeric physical GPU index"
  validate_boolean AUTO_CPU_AFFINITY "${AUTO_CPU_AFFINITY}"
  require_command cmp
  require_command git
  require_command mktemp
  require_command nvidia-cuda-mps-control
  require_command nvidia-smi
  require_command pgrep
  require_command ps
  require_command sha256sum
  require_command taskset
  require_command timeout
  require_command "${PYTHON_BIN}"
  [[ -d /data && -w /data ]] || die "/data must exist and be writable"
  [[ -d "${REPO_DIR}/sglang_omni" ]] \
    || die "sglang-omni checkout not found: ${REPO_DIR}"
  [[ -x "${SINGLE_RUNNER}" ]] || die "single runner is not executable: ${SINGLE_RUNNER}"
  [[ -f "${BENCHMARK_CLIENT}" ]] || die "benchmark client is missing: ${BENCHMARK_CLIENT}"
  [[ -f "${MPS_VERIFIER}" ]] || die "MPS verifier is missing: ${MPS_VERIFIER}"

  local inherited_mps_name
  for inherited_mps_name in \
    CUDA_MPS_PIPE_DIRECTORY \
    CUDA_MPS_LOG_DIRECTORY \
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE \
    CUDA_MPS_PINNED_DEVICE_MEM_LIMIT \
    CUDA_MPS_CLIENT_PRIORITY; do
    [[ -z "${!inherited_mps_name-}" ]] \
      || die "unset inherited ${inherited_mps_name} before running"
  done

  SOURCE_COMMIT="$(git -C "${REPO_DIR}" rev-parse HEAD)" \
    || die "failed to resolve source commit"
  [[ "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
    || die "invalid source commit: ${SOURCE_COMMIT}"
  [[ -z "$(git -C "${REPO_DIR}" status --porcelain --untracked-files=no)" ]] \
    || die "tracked source checkout is dirty"

  resolve_cpu_affinity
  RUN_ROOT="/data/higgs-s1f2-c96-vs-s2f2-c192-full-mps-$(date +%Y%m%d-%H%M%S)-$$"
  [[ ! -e "${RUN_ROOT}" ]] || die "result directory already exists: ${RUN_ROOT}"
  mkdir -p "${HF_HOME}" "${RUN_ROOT}"
  STATUS_FILE="${RUN_ROOT}/status.txt"
  printf 'running\n' > "${STATUS_FILE}"

  printf '%s\n' \
    "objective=compare scheduler=2/frontend=2/c192 against scheduler=1/frontend=2/c96 with matched per-scheduler capacity under full-pipeline MPS" \
    "evidence_scope=single A-to-B point comparison; no drift, noise, or stability claim" \
    "source_commit=${SOURCE_COMMIT}" \
    "model=${MODEL_PATH}" \
    "model_hf_url=https://huggingface.co/${MODEL_PATH}" \
    "dataset=${DATASET_REPO_ID}" \
    "dataset_hf_url=https://huggingface.co/datasets/${DATASET_REPO_ID}" \
    "dataset_revision=${DATASET_REVISION}" \
    "dataset_parquet_sha256=${DATASET_PARQUET_SHA256}" \
    "arm_order=S1F2C96,S2F2C192" \
    "S1F2C96=scheduler_replicas=1;frontend_replicas=2;concurrency=96;scheduler_capacity_each=96;scheduler_cuda_graph_max_bs_each=96" \
    "S2F2C192=scheduler_replicas=2;frontend_replicas=2;concurrency=192;scheduler_capacity_each=96;scheduler_cuda_graph_max_bs_each=96" \
    "mps=required_both_arms" \
    "mps_scope=all_gpu_processes_in_server_tree" \
    "mps_daemon=fresh_private_daemon_per_arm" \
    "mps_active_thread_percentage=default_100" \
    "seconds=${SECS}" \
    "warmup_seconds=${WARMUP_SECS}" \
    "measurement_seconds=$((SECS - WARMUP_SECS))" \
    "server_cpu_affinity=${SERVER_CPUSET};numa=${SERVER_NUMA_NODE};physical_cores=${SERVER_AFFINITY_CPU_COUNT};gpu_local=true" \
    "client_cpu_affinity=${CLIENT_CPUSET};numa=${CLIENT_NUMA_NODE};physical_cores=${CLIENT_AFFINITY_CPU_COUNT}" \
    > "${RUN_ROOT}/comparison-contract.txt"
  sha256sum \
    "${BASH_SOURCE[0]}" \
    "${SINGLE_RUNNER}" \
    "${BENCHMARK_CLIENT}" \
    "${MPS_VERIFIER}" \
    > "${RUN_ROOT}/harness-sha256.txt"

  echo "Result directory: ${RUN_ROOT}"
  echo "Server affinity: ${SERVER_AFFINITY_CPU_COUNT} physical cores on NUMA ${SERVER_NUMA_NODE} (GPU-local)"
  echo "Client affinity: ${CLIENT_AFFINITY_CPU_COUNT} physical cores on NUMA ${CLIENT_NUMA_NODE}"

  ensure_no_mps_daemon
  ensure_gpu_idle
  run_arm S1F2C96 score-00-S1F2C96 1 96 0.85
  run_arm S2F2C192 score-01-S2F2C192 2 192 0.425
  summarize_results
  COMPLETED=1
  printf 'complete\n' > "${STATUS_FILE}"
  write_artifact_manifest
  echo "Full-pipeline MPS comparison complete: ${RUN_ROOT}"
}

trap on_exit EXIT INT TERM
main "$@"
