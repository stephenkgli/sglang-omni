#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

# note (lkg): Compare the generic entry-stage replica path with a frozen full
# SeedTTS English closed-loop workload through /v1/audio/speech.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
BENCHMARK_CLIENT="${BENCHMARK_CLIENT:-${SCRIPT_DIR}/benchmark_higgs_tts_throughput.py}"
FULL_PIPELINE_MPS_VERIFIER="${FULL_PIPELINE_MPS_VERIFIER:-${SCRIPT_DIR}/verify_higgs_tts_full_pipeline_mps.py}"

GPU_SELECTOR="${CUDA_VISIBLE_DEVICES-}"
GPU_ID=""

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_BIN_DIR=""
MODEL_PATH="${MODEL_PATH:-bosonai/higgs-tts-3-4b}"
HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
readonly SEEDTTS_REPO_ID="zhaochenyang20/seed-tts-eval-arrow"
readonly SEEDTTS_PARQUET_SHA256="5849b41b49cae996328c06d2c5791717c3bafc369bddfa1ec4f86761bb8bc0ca"
DATASET="${DATASET:-/root/autodl-tmp/datasets/seed-tts-eval-arrow}"
DATASET_REPO_ID="${DATASET_REPO_ID:-${SEEDTTS_REPO_ID}}"
DATASET_REVISION="${DATASET_REVISION-}"
DATASET_PARQUET_SHA256="${DATASET_PARQUET_SHA256:-${SEEDTTS_PARQUET_SHA256}}"
readonly DATASET_SOURCE_MODE="${DATASET_SOURCE_MODE:-local_verified}"
SAMPLE_SELECTION="full_seedtts_order_worker_strided_cycle"
PORT="${PORT:-8901}"
BASE_URL="http://127.0.0.1:${PORT}"
RUN_DIR="${RUN_DIR:-}"

readonly CAP="${CAP:-96}"
readonly CONC="${CONC:-${CAP}}"
readonly CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-${CAP}}"
readonly REPS="${REPS:-1}"
readonly SECS="${SECS:-110}"
readonly WARMUP_SECS="${WARMUP_SECS:-20}"
readonly MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
readonly DECODE_CUDA_GRAPH_MAX_FRAMES="${DECODE_CUDA_GRAPH_MAX_FRAMES:-512}"
readonly COALESCE_REQUESTS="${COALESCE_REQUESTS:-32}"
readonly COALESCE_WAIT_MS="${COALESCE_WAIT_MS:-300}"
readonly EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-1088}"
readonly EXPECTED_UNIQUE_REFERENCES="${EXPECTED_UNIQUE_REFERENCES:-666}"
readonly SPEAKER=Ethan
readonly TEMPERATURE=0.7
readonly BOOT_TIMEOUT_SECS="${BOOT_TIMEOUT_SECS:-600}"
readonly ARM="${ARM:-baseline}"
readonly FRONTEND_REPLICAS="${FRONTEND_REPLICAS:-1}"
readonly ENCODER_MEMORY_FRACTION="${ENCODER_MEMORY_FRACTION:-0.0245}"
readonly SCHEDULER_REPLICAS="${SCHEDULER_REPLICAS:-1}"
readonly SCHEDULER_MEMORY_FRACTION="${SCHEDULER_MEMORY_FRACTION:-0.85}"
readonly SERVER_MPS_MODE="${SERVER_MPS_MODE:-disabled}"
readonly SERVER_MPS_PIPE_DIRECTORY="${SERVER_MPS_PIPE_DIRECTORY:-}"
readonly SERVER_MPS_LOG_DIRECTORY="${SERVER_MPS_LOG_DIRECTORY:-}"
readonly SERVER_MPS_GPU_UUID="${SERVER_MPS_GPU_UUID:-}"
# CAP and CUDA_GRAPH_MAX_BS describe the logical tts_engine stage.  Runtime
# overrides are copied to every physical replica, so divide them here instead
# of accidentally provisioning CAP independently on every scheduler replica.
readonly SCHEDULER_MAX_RUNNING_REQUESTS="${SCHEDULER_MAX_RUNNING_REQUESTS:-$(((CAP + SCHEDULER_REPLICAS - 1) / SCHEDULER_REPLICAS))}"
readonly SCHEDULER_CUDA_GRAPH_MAX_BS="${SCHEDULER_CUDA_GRAPH_MAX_BS:-$(((CUDA_GRAPH_MAX_BS + SCHEDULER_REPLICAS - 1) / SCHEDULER_REPLICAS))}"
readonly SERVER_THREADS="${SERVER_THREADS:-3}"
readonly CLIENT_THREADS="${CLIENT_THREADS:-2}"
readonly SERVER_CPUSET="${SERVER_CPUSET:-48-71}"
readonly CLIENT_CPUSET="${CLIENT_CPUSET:-0-23}"
readonly SERVER_NUMA_NODE="${SERVER_NUMA_NODE:-1}"
readonly CLIENT_NUMA_NODE="${CLIENT_NUMA_NODE:-0}"
readonly EXPECTED_AFFINITY_CPUS="${EXPECTED_AFFINITY_CPUS:-24}"
readonly SERVER_AFFINITY_CPU_COUNT="${SERVER_AFFINITY_CPU_COUNT:-${EXPECTED_AFFINITY_CPUS}}"
readonly CLIENT_AFFINITY_CPU_COUNT="${CLIENT_AFFINITY_CPU_COUNT:-${EXPECTED_AFFINITY_CPUS}}"
readonly CGROUP_CPU_QUOTA_POLICY="${CGROUP_CPU_QUOTA_POLICY:-exact_affinity}"
readonly REQUIRE_FULL_DATASET_COVERAGE="${REQUIRE_FULL_DATASET_COVERAGE:-1}"

# note (lkg): Lightweight cgroup CPU and nvidia-smi observers are enabled for
# every scored arm. CPU throttling is reported as an environment-interference
# metric; request correctness, coverage, graph, topology, and attachment drift
# remain hard validity gates. NSYS remains opt-in because it can perturb QPS.
PROFILE="${PROFILE:-1}"
NSYS_SM_PROFILE="${NSYS_SM_PROFILE:-0}"
NSYS_DURATION_S="${NSYS_DURATION_S:-90}"
PREFILL_COALESCING_MODE="${PREFILL_COALESCING_MODE:-required}"
PREFILL_COALESCING_SUPPORTED=""
PREFILL_COALESCING_ENABLED=""
PREFILL_COALESCE_REQUESTS_EFFECTIVE="disabled"
PREFILL_COALESCE_WAIT_MS_EFFECTIVE="disabled"
SERVER_MPS_ENABLED="0"
SERVER_CUDA_VISIBLE_DEVICES_EFFECTIVE=""

SERVER_PID=""
GPU_QUERY_PID=""
GPU_DMON_PID=""
NSYS_PID=""
CURRENT_REP_DIR=""
PIPELINE_CONFIG_PATH=""
GPU_PCI_BUS_ID=""
GPU_SYSFS_BUS_ID=""
GPU_NUMA_NODE=""

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

validate_positive_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] \
    || die "${name} must be a positive integer"
}

validate_nonnegative_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] \
    || die "${name} must be a non-negative integer"
}

validate_memory_budget() {
  "${PYTHON_BIN}" - \
    "${FRONTEND_REPLICAS}" \
    "${ENCODER_MEMORY_FRACTION}" \
    "${SCHEDULER_REPLICAS}" \
    "${SCHEDULER_MEMORY_FRACTION}" <<'PY'
import math
import sys

frontend_replicas = int(sys.argv[1])
encoder_fraction = float(sys.argv[2])
scheduler_replicas = int(sys.argv[3])
scheduler_fraction = float(sys.argv[4])
for name, value in (
    ("ENCODER_MEMORY_FRACTION", encoder_fraction),
    ("SCHEDULER_MEMORY_FRACTION", scheduler_fraction),
):
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise SystemExit(f"{name} must be finite and in (0, 1): {value}")
total = (
    frontend_replicas * encoder_fraction
    + scheduler_replicas * scheduler_fraction
    + 0.10
)
if total > 1.0 + 1e-9:
    raise SystemExit(f"GPU placement memory budget exceeds 1.0: {total:.9f}")
print(f"GPU placement memory budget: {total:.9f}")
PY
}

resolve_and_validate_dataset() {
  [[ "${DATASET_REPO_ID}" == "${SEEDTTS_REPO_ID}" ]] \
    || die "DATASET_REPO_ID must be ${SEEDTTS_REPO_ID}"
  if [[ "${DATASET_SOURCE_MODE}" == "huggingface" ]]; then
    [[ "${DATASET}" == "${SEEDTTS_REPO_ID}" ]] \
      || die "huggingface dataset mode requires DATASET=${SEEDTTS_REPO_ID}"
    [[ "${HF_HUB_OFFLINE:-1}" == "0" ]] \
      || die "huggingface dataset mode requires HF_HUB_OFFLINE=0"
    [[ "${HF_DATASETS_OFFLINE:-1}" == "0" ]] \
      || die "huggingface dataset mode requires HF_DATASETS_OFFLINE=0"
    if [[ -z "${DATASET_REVISION}" ]]; then
      DATASET_PARQUET_SHA256=""
    fi
    return
  fi
  [[ "${DATASET_SOURCE_MODE}" == "local_verified" ]] \
    || die "DATASET_SOURCE_MODE must be local_verified or huggingface"
  [[ -d "${DATASET}" ]] \
    || die "full SeedTTS dataset directory not found: ${DATASET}"
  DATASET="$(
    "${PYTHON_BIN}" -c \
      'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' \
      "${DATASET}"
  )" || die "failed to resolve dataset path: ${DATASET}"
  local parquet_path
  parquet_path="${DATASET}/data/en-00000-of-00001.parquet"
  [[ -f "${parquet_path}" ]] \
    || die "full SeedTTS parquet not found: ${parquet_path}"
  local parquet_sha256
  parquet_sha256="$(
    "${PYTHON_BIN}" - "${parquet_path}" <<'PY'
import hashlib
import pathlib
import sys

digest = hashlib.sha256()
with pathlib.Path(sys.argv[1]).open("rb") as input_file:
    for block in iter(lambda: input_file.read(1024 * 1024), b""):
        digest.update(block)
print(digest.hexdigest())
PY
  )" || die "failed to hash full SeedTTS parquet"
  [[ "${parquet_sha256}" == "${DATASET_PARQUET_SHA256}" ]] \
    || die \
      "full SeedTTS parquet SHA mismatch: ${parquet_sha256}"
}

validate_arm() {
  case "${ARM}" in
    baseline)
      [[ "${FRONTEND_REPLICAS}" == "1" ]] \
        || die "baseline requires FRONTEND_REPLICAS=1"
      [[ "${SCHEDULER_REPLICAS}" == "1" ]] \
        || die "baseline requires SCHEDULER_REPLICAS=1"
      ;;
    r2)
      [[ "${FRONTEND_REPLICAS}" == "2" ]] \
        || die "r2 requires FRONTEND_REPLICAS=2"
      [[ "${SCHEDULER_REPLICAS}" == "1" ]] \
        || die "r2 requires SCHEDULER_REPLICAS=1"
      ;;
    *)
      [[ "${ARM}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
        || die "ARM must be a filesystem-safe label"
      ;;
  esac
}

resolve_prefill_coalescing() {
  local stages_path="${REPO_DIR}/sglang_omni/models/higgs_tts/stages.py"
  local support
  support="$(
    "${PYTHON_BIN}" -c '
import ast
import pathlib
import sys

tree = ast.parse(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
factory = next(
    node
    for node in ast.walk(tree)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    and node.name == "create_sglang_tts_engine_executor"
)
parameters = {
    argument.arg
    for argument in (
        factory.args.posonlyargs + factory.args.args + factory.args.kwonlyargs
    )
}
required = {"prefill_coalesce_requests", "prefill_coalesce_wait_ms"}
print(int(factory.args.kwarg is not None or required <= parameters))
' "${stages_path}"
  )" || die "failed to inspect Higgs TTS prefill-coalescing support"
  [[ "${support}" == "0" || "${support}" == "1" ]] \
    || die "invalid prefill-coalescing support result: ${support}"
  PREFILL_COALESCING_SUPPORTED="${support}"

  case "${PREFILL_COALESCING_MODE}" in
    auto)
      PREFILL_COALESCING_ENABLED="${PREFILL_COALESCING_SUPPORTED}"
      ;;
    required)
      [[ "${PREFILL_COALESCING_SUPPORTED}" == "1" ]] \
        || die "current code does not support required prefill coalescing"
      PREFILL_COALESCING_ENABLED="1"
      ;;
    disabled)
      PREFILL_COALESCING_ENABLED="0"
      ;;
    *)
      die "PREFILL_COALESCING_MODE must be auto, required, or disabled"
      ;;
  esac

  if [[ "${PREFILL_COALESCING_ENABLED}" == "1" ]]; then
    PREFILL_COALESCE_REQUESTS_EFFECTIVE="${COALESCE_REQUESTS}"
    PREFILL_COALESCE_WAIT_MS_EFFECTIVE="${COALESCE_WAIT_MS}"
  fi
}

resolve_server_mps() {
  case "${SERVER_MPS_MODE}" in
    disabled)
      [[ -z "${SERVER_MPS_PIPE_DIRECTORY}" ]] \
        || die "disabled server MPS cannot receive a pipe directory"
      [[ -z "${SERVER_MPS_LOG_DIRECTORY}" ]] \
        || die "disabled server MPS cannot receive a log directory"
      [[ -z "${SERVER_MPS_GPU_UUID}" ]] \
        || die "disabled server MPS cannot receive a GPU UUID"
      SERVER_MPS_ENABLED="0"
      SERVER_CUDA_VISIBLE_DEVICES_EFFECTIVE="${GPU_SELECTOR}"
      ;;
    required)
      [[ "${SERVER_MPS_PIPE_DIRECTORY}" == /* ]] \
        || die "SERVER_MPS_PIPE_DIRECTORY must be an absolute path"
      [[ "${SERVER_MPS_LOG_DIRECTORY}" == /* ]] \
        || die "SERVER_MPS_LOG_DIRECTORY must be an absolute path"
      [[ "${SERVER_MPS_PIPE_DIRECTORY}" != "${SERVER_MPS_LOG_DIRECTORY}" ]] \
        || die "server MPS pipe and log directories must differ"
      [[ -d "${SERVER_MPS_PIPE_DIRECTORY}" ]] \
        || die "server MPS pipe directory not found"
      [[ -d "${SERVER_MPS_LOG_DIRECTORY}" ]] \
        || die "server MPS log directory not found"
      [[ -w "${SERVER_MPS_PIPE_DIRECTORY}" ]] \
        || die "server MPS pipe directory is not writable"
      [[ -w "${SERVER_MPS_LOG_DIRECTORY}" ]] \
        || die "server MPS log directory is not writable"
      [[ "${SERVER_MPS_GPU_UUID}" == GPU-* ]] \
        || die "SERVER_MPS_GPU_UUID must be the daemon-selected GPU UUID"
      local inherited_mps_name
      for inherited_mps_name in \
        CUDA_MPS_PIPE_DIRECTORY \
        CUDA_MPS_LOG_DIRECTORY \
        CUDA_MPS_ACTIVE_THREAD_PERCENTAGE \
        CUDA_MPS_PINNED_DEVICE_MEM_LIMIT \
        CUDA_MPS_CLIENT_PRIORITY; do
        if declare -p "${inherited_mps_name}" >/dev/null 2>&1; then
          die "unset inherited ${inherited_mps_name}; server MPS uses a private endpoint"
        fi
      done
      SERVER_MPS_ENABLED="1"
      local observed_gpu_uuid
      observed_gpu_uuid="$(
        nvidia-smi -i "${GPU_ID}" --query-gpu=uuid \
          --format=csv,noheader,nounits | tr -d '[:space:]'
      )" || die "failed to resolve the selected physical GPU UUID"
      [[ "${observed_gpu_uuid}" == GPU-* ]] \
        || die "invalid selected physical GPU UUID: ${observed_gpu_uuid}"
      [[ "${SERVER_MPS_GPU_UUID}" == "${observed_gpu_uuid}" ]] \
        || die \
          "MPS daemon/client GPU UUID mismatch: ${SERVER_MPS_GPU_UUID} != ${observed_gpu_uuid}"
      SERVER_CUDA_VISIBLE_DEVICES_EFFECTIVE="${SERVER_MPS_GPU_UUID}"
      ;;
    *)
      die "SERVER_MPS_MODE must be disabled or required"
      ;;
  esac
}

validate_gpu_selector() {
  [[ -n "${GPU_SELECTOR}" ]] \
    || die "CUDA_VISIBLE_DEVICES must be set to exactly one physical GPU index"
  [[ "${GPU_SELECTOR}" =~ ^[0-9]+$ ]] \
    || die "CUDA_VISIBLE_DEVICES must contain one numeric GPU index"
  GPU_ID="${GPU_SELECTOR}"
}

configure_gpu() {
  nvidia-smi -i "${GPU_ID}" >/dev/null \
    || die "CUDA_VISIBLE_DEVICES=${GPU_SELECTOR} is not a visible physical GPU"
  GPU_PCI_BUS_ID="$(
    nvidia-smi -i "${GPU_ID}" --query-gpu=pci.bus_id \
      --format=csv,noheader,nounits | tr -d '[:space:]'
  )"
  [[ -n "${GPU_PCI_BUS_ID}" ]] || die "failed to resolve GPU PCI bus id"
  GPU_SYSFS_BUS_ID="${GPU_PCI_BUS_ID,,}"
  if [[ "${GPU_SYSFS_BUS_ID}" =~ ^[[:xdigit:]]{8}: ]]; then
    GPU_SYSFS_BUS_ID="${GPU_SYSFS_BUS_ID:4}"
  fi
  [[ -r "/sys/bus/pci/devices/${GPU_SYSFS_BUS_ID}/numa_node" ]] \
    || die "GPU NUMA node is unavailable for ${GPU_SYSFS_BUS_ID}"
  GPU_NUMA_NODE="$(
    tr -d '[:space:]' \
      < "/sys/bus/pci/devices/${GPU_SYSFS_BUS_ID}/numa_node"
  )"
  [[ "${GPU_NUMA_NODE}" =~ ^[0-9]+$ ]] \
    || die "GPU ${GPU_ID} has no usable NUMA node: ${GPU_NUMA_NODE}"
  if [[ -z "${RUN_DIR}" ]]; then
    local run_root="/tmp"
    if [[ -d /root/autodl-tmp/sglang-eval-lab/runs ]]; then
      run_root="/root/autodl-tmp/sglang-eval-lab/runs"
    fi
    RUN_DIR="${run_root}/higgs-entry-replicas-${ARM}-gpu${GPU_ID}-$(date +%Y%m%d-%H%M%S)"
  fi
}

validate_cpu_bindings() {
  "${PYTHON_BIN}" - \
    "${SERVER_CPUSET}" \
    "${CLIENT_CPUSET}" \
    "${SERVER_NUMA_NODE}" \
    "${CLIENT_NUMA_NODE}" \
    "${SERVER_AFFINITY_CPU_COUNT}" \
    "${CLIENT_AFFINITY_CPU_COUNT}" \
    "${CGROUP_CPU_QUOTA_POLICY}" \
    "${EXPECTED_AFFINITY_CPUS}" \
    "${GPU_NUMA_NODE}" \
    "${RUN_DIR}/cpu-affinity-contract.json" <<'PY'
import json
import math
import os
import pathlib
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


def cpu_identity(cpu: int) -> tuple[int, tuple[int, int]]:
    root = pathlib.Path(f"/sys/devices/system/cpu/cpu{cpu}")
    node_paths = sorted(root.glob("node[0-9]*"))
    if len(node_paths) != 1:
        raise SystemExit(f"CPU {cpu} has unexpected NUMA links: {node_paths}")
    node = int(node_paths[0].name.removeprefix("node"))
    package = int((root / "topology/physical_package_id").read_text().strip())
    core = int((root / "topology/core_id").read_text().strip())
    return node, (package, core)


server_spec, client_spec = sys.argv[1], sys.argv[2]
server_node, client_node = int(sys.argv[3]), int(sys.argv[4])
server_width, client_width = int(sys.argv[5]), int(sys.argv[6])
quota_policy, quota_expected_cores = sys.argv[7], int(sys.argv[8])
gpu_node = int(sys.argv[9])
output_path = pathlib.Path(sys.argv[10])
server_cpus, client_cpus = parse_cpu_list(server_spec), parse_cpu_list(client_spec)
available = set(os.sched_getaffinity(0))

if len(server_cpus) != server_width or len(client_cpus) != client_width:
    raise SystemExit(
        "server/client affinity width mismatch: "
        f"server={len(server_cpus)}/{server_width}, "
        f"client={len(client_cpus)}/{client_width}"
    )
if server_cpus & client_cpus:
    raise SystemExit(f"server/client CPU sets overlap: {sorted(server_cpus & client_cpus)}")
if not (server_cpus | client_cpus) <= available:
    raise SystemExit("requested CPU sets exceed the container effective affinity")
if server_node == client_node:
    raise SystemExit("server and client must use different NUMA nodes")
if server_node != gpu_node:
    raise SystemExit(
        f"server NUMA node {server_node} is not GPU-local node {gpu_node}"
    )

identities = {
    "server": [cpu_identity(cpu) for cpu in sorted(server_cpus)],
    "client": [cpu_identity(cpu) for cpu in sorted(client_cpus)],
}
for name, expected_node, cpus in (
    ("server", server_node, server_cpus),
    ("client", client_node, client_cpus),
):
    nodes = {node for node, _core in identities[name]}
    cores = {core for _node, core in identities[name]}
    if nodes != {expected_node}:
        raise SystemExit(f"{name} CPUs span unexpected NUMA nodes: {sorted(nodes)}")
    if len(cores) != len(cpus):
        raise SystemExit(f"{name} CPU set includes SMT siblings: {sorted(cpus)}")

quota_text, period_text = pathlib.Path("/sys/fs/cgroup/cpu.max").read_text().split()
period = int(period_text)
quota = None if quota_text == "max" else int(quota_text)
quota_cores = None if quota is None else quota / period
if quota_policy == "exact_affinity":
    if quota_cores is None:
        raise SystemExit("benchmark requires a finite cgroup CPU quota")
    if not math.isclose(quota_cores, quota_expected_cores):
        raise SystemExit(
            f"cgroup quota is {quota_cores:g} cores, "
            f"expected {quota_expected_cores}"
        )
elif quota_policy != "record_only":
    raise SystemExit(f"unknown cgroup CPU quota policy: {quota_policy}")
burst = (
    pathlib.Path("/sys/fs/cgroup/cpu.max.burst").read_text().strip()
    if pathlib.Path("/sys/fs/cgroup/cpu.max.burst").is_file()
    else None
)
payload = {
    "server": {
        "cpulist": server_spec,
        "cpus": sorted(server_cpus),
        "cpu_count": len(server_cpus),
        "physical_core_count": len({core for _node, core in identities["server"]}),
        "numa_node": server_node,
        "gpu_local": True,
    },
    "client": {
        "cpulist": client_spec,
        "cpus": sorted(client_cpus),
        "cpu_count": len(client_cpus),
        "physical_core_count": len({core for _node, core in identities["client"]}),
        "numa_node": client_node,
        "gpu_local": False,
    },
    "disjoint": True,
    "container_effective_cpu_count": len(available),
    "cgroup_cpu_quota_policy": quota_policy,
    "cgroup_cpu_max": f"{quota_text} {period}",
    "cgroup_quota_cores": quota_cores,
    "cgroup_cpu_max_burst": burst,
}
output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

validate_gpu_idle() {
  local active_pids
  active_pids="$(
    nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null \
      | sed '/^[[:space:]]*$/d'
  )"
  [[ -z "${active_pids}" ]] \
    || die "GPU ${GPU_ID} already has compute clients: ${active_pids//$'\n'/, }"
}

record_attachment_identity() {
  mkdir -p "${RUN_DIR}/manifest"
  hostname > "${RUN_DIR}/manifest/hostname.txt"
  uname -a > "${RUN_DIR}/manifest/uname.txt"
  [[ -r /proc/sys/kernel/random/boot_id ]] \
    || die "kernel boot identity is unavailable"
  cp /proc/sys/kernel/random/boot_id "${RUN_DIR}/manifest/boot-id.txt"
  nvidia-smi -i "${GPU_ID}" \
    --query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total \
    --format=csv > "${RUN_DIR}/manifest/gpu-identity.csv"
  nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid,process_name \
    --format=csv > "${RUN_DIR}/manifest/gpu-processes-before.csv" 2>&1 || true
  nvidia-smi topo -m > "${RUN_DIR}/manifest/nvidia-topology.txt"
  lscpu > "${RUN_DIR}/manifest/lscpu.txt"
  lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE \
    > "${RUN_DIR}/manifest/lscpu-extended.txt"
  df -h / /root /root/autodl-tmp \
    > "${RUN_DIR}/manifest/disk-free.txt" 2>&1 || true
  local cgroup_file
  for cgroup_file in cpu.max cpuset.cpus.effective cpuset.mems.effective; do
    [[ -r "/sys/fs/cgroup/${cgroup_file}" ]] \
      || die "required cgroup-v2 identity is unavailable: ${cgroup_file}"
    cp "/sys/fs/cgroup/${cgroup_file}" \
      "${RUN_DIR}/manifest/${cgroup_file//./-}.txt"
  done
  for cgroup_file in \
    cpu.stat memory.current memory.max; do
    if [[ -r "/sys/fs/cgroup/${cgroup_file}" ]]; then
      cp "/sys/fs/cgroup/${cgroup_file}" \
        "${RUN_DIR}/manifest/${cgroup_file//./-}.txt"
    fi
  done
  "${PYTHON_BIN}" --version > "${RUN_DIR}/manifest/python-version.txt" 2>&1
  if "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
    "${PYTHON_BIN}" -m pip freeze > "${RUN_DIR}/manifest/pip-freeze.txt"
    printf 'python -m pip freeze\n' \
      > "${RUN_DIR}/manifest/package-inventory-tool.txt"
  elif command -v uv >/dev/null 2>&1; then
    uv pip freeze \
      --python "$(command -v "${PYTHON_BIN}")" \
      > "${RUN_DIR}/manifest/pip-freeze.txt"
    printf 'uv pip freeze\n' \
      > "${RUN_DIR}/manifest/package-inventory-tool.txt"
  else
    "${PYTHON_BIN}" - "${RUN_DIR}/manifest/pip-freeze.txt" <<'PY'
import importlib.metadata
import pathlib
import sys


rows = {
    f"{distribution.metadata['Name']}=={distribution.version}"
    for distribution in importlib.metadata.distributions()
    if distribution.metadata.get("Name")
}
pathlib.Path(sys.argv[1]).write_text(
    "\n".join(sorted(rows, key=str.lower)) + "\n",
    encoding="utf-8",
)
PY
    printf 'python importlib.metadata fallback\n' \
      > "${RUN_DIR}/manifest/package-inventory-tool.txt"
  fi
  sha256sum \
    "${BENCHMARK_CLIENT}" \
    "${FULL_PIPELINE_MPS_VERIFIER}" \
    "${BASH_SOURCE[0]}" \
    > "${RUN_DIR}/manifest/harness-sha256.txt"
  if git -C "${REPO_DIR}" rev-parse HEAD \
    > "${RUN_DIR}/manifest/git-commit.txt" 2>/dev/null; then
    git -C "${REPO_DIR}" status --short \
      > "${RUN_DIR}/manifest/git-status.txt"
  elif [[ -f "${REPO_DIR}/.source-commit" ]]; then
    cp "${REPO_DIR}/.source-commit" "${RUN_DIR}/manifest/git-commit.txt"
  else
    printf '%s\n' "${SOURCE_COMMIT:-unknown}" \
      > "${RUN_DIR}/manifest/git-commit.txt"
  fi
}

stop_pid() {
  local pid="$1"
  [[ -n "${pid}" ]] || return 0
  if kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
}

stop_observers() {
  stop_pid "${GPU_QUERY_PID}"
  stop_pid "${GPU_DMON_PID}"
  GPU_QUERY_PID=""
  GPU_DMON_PID=""
  if [[ -n "${NSYS_PID}" ]]; then
    wait "${NSYS_PID}" || true
    NSYS_PID=""
  fi
}

stop_server() {
  [[ -n "${SERVER_PID}" ]] || return 0
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null \
      || kill -TERM "${SERVER_PID}" 2>/dev/null \
      || true
    local attempt
    for ((attempt=0; attempt<30; attempt++)); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      kill -KILL -- "-${SERVER_PID}" 2>/dev/null \
        || kill -KILL "${SERVER_PID}" 2>/dev/null \
        || true
    fi
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
  sleep 3
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_observers
  stop_server
  exit "${status}"
}

record_command() {
  local output_path="$1"
  shift
  printf '%q ' "$@" > "${output_path}"
  printf '\n' >> "${output_path}"
}

record_and_validate_affinity_tree() {
  local root_pid="$1"
  local expected_cpulist="$2"
  local expected_node="$3"
  local output_path="$4"
  "${PYTHON_BIN}" - \
    "${root_pid}" \
    "${expected_cpulist}" \
    "${expected_node}" \
    "${output_path}" <<'PY'
import json
import os
import pathlib
import sys


def parse_cpu_list(value: str) -> set[int]:
    result: set[int] = set()
    for item in value.split(","):
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            result.update(range(int(start_text), int(end_text) + 1))
        else:
            result.add(int(item))
    return result


def read_ppid(pid: int) -> int | None:
    try:
        value = pathlib.Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    closing = value.rfind(")")
    fields = value[closing + 2 :].split()
    return int(fields[1])


def descendants(root_pid: int) -> set[int]:
    result = {root_pid}
    while True:
        added = False
        for proc_path in pathlib.Path("/proc").glob("[0-9]*"):
            pid = int(proc_path.name)
            if pid in result:
                continue
            if read_ppid(pid) in result:
                result.add(pid)
                added = True
        if not added:
            return result


root_pid = int(sys.argv[1])
expected = parse_cpu_list(sys.argv[2])
expected_node = int(sys.argv[3])
output_path = pathlib.Path(sys.argv[4])
rows = []
violations = []
for pid in sorted(descendants(root_pid)):
    proc_path = pathlib.Path(f"/proc/{pid}")
    try:
        process_affinity = set(os.sched_getaffinity(pid))
        command = (proc_path / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            errors="replace"
        ).strip()
        ppid = read_ppid(pid)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        continue

    thread_affinities: set[tuple[int, ...]] = set()
    for task_path in (proc_path / "task").glob("[0-9]*"):
        try:
            thread_affinities.add(
                tuple(sorted(os.sched_getaffinity(int(task_path.name))))
            )
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    variants = [list(value) for value in sorted(thread_affinities)]
    if process_affinity != expected:
        violations.append(
            f"pid {pid} process affinity {sorted(process_affinity)} != {sorted(expected)}"
        )
    if thread_affinities != {tuple(sorted(expected))}:
        violations.append(f"pid {pid} thread affinity variants {variants}")
    rows.append(
        {
            "pid": pid,
            "ppid": ppid,
            "command": command,
            "process_cpus": sorted(process_affinity),
            "thread_count": sum(1 for _ in (proc_path / "task").glob("[0-9]*")),
            "thread_affinity_variants": variants,
            "inductor_compile_worker": "torch/_inductor/compile_worker" in command,
        }
    )

if not any(row["pid"] == root_pid for row in rows):
    violations.append(f"root pid {root_pid} disappeared before affinity validation")
payload = {
    "root_pid": root_pid,
    "expected_cpulist": sys.argv[2],
    "expected_cpus": sorted(expected),
    "expected_cpu_count": len(expected),
    "expected_numa_node": expected_node,
    "process_count": len(rows),
    "inductor_compile_worker_processes": sum(
        row["inductor_compile_worker"] for row in rows
    ),
    "all_processes_and_threads_match": not violations,
    "violations": violations,
    "processes": rows,
}
output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
if violations:
    raise SystemExit("; ".join(violations[:10]))
PY
}

wait_for_server() {
  local deadline=$((SECONDS + BOOT_TIMEOUT_SECS))
  while ((SECONDS < deadline)); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      tail -n 200 "${CURRENT_REP_DIR}/server.log" >&2 || true
      die "server exited before becoming healthy"
    fi
    if curl --noproxy '*' --fail --silent --max-time 3 \
      "${BASE_URL}/health" >/dev/null; then
      return
    fi
    sleep 5
  done
  tail -n 200 "${CURRENT_REP_DIR}/server.log" >&2 || true
  die "server did not become healthy within ${BOOT_TIMEOUT_SECS}s"
}

validate_server_log() {
  local log_path="${CURRENT_REP_DIR}/server.log"
  local topology_path="${CURRENT_REP_DIR}/process-topology.txt"
  local fatal_pattern
  grep -F " ready with stages=" "${log_path}" > "${topology_path}" || true
  fatal_pattern="CUDA out of memory|BackendCompilerFailed|InductorError"
  fatal_pattern+="|torch\\._dynamo\\.exc\\.| 500 Internal Server Error"
  fatal_pattern+="|Higgs codec decode CUDA graph miss"
  if grep -Eiq "${fatal_pattern}" "${log_path}"; then
    grep -Ein "${fatal_pattern}" "${log_path}" >&2
    die "server log contains a fatal performance or correctness error"
  fi
  grep -Fq \
    "Captured ${DECODE_CUDA_GRAPH_MAX_FRAMES} Higgs codec decode CUDA graphs for frame counts 1..${DECODE_CUDA_GRAPH_MAX_FRAMES}" \
    "${log_path}" \
    || die "expected vocoder decode CUDA graph capture was not observed"
  if grep -Fq "torch.compile of the codec decode" "${log_path}"; then
    die "vocoder compile_decode must remain disabled"
  fi

  require_process_stages vocoder vocoder
  if ((SCHEDULER_REPLICAS == 1)); then
    require_process_stages pipeline tts_engine
    if grep -Fq "pipeline@r" "${topology_path}"; then
      die "unreplicated scheduler unexpectedly started replica processes"
    fi
  else
    local scheduler_replica
    for ((scheduler_replica=0; scheduler_replica<SCHEDULER_REPLICAS; scheduler_replica++)); do
      require_process_stages \
        "pipeline@r${scheduler_replica}" \
        "tts_engine@r${scheduler_replica}"
    done
    grep -Fq "Replicated stage 'tts_engine'" "${log_path}" \
      || die "tts_engine scheduler replica expansion was not observed"
  fi

  if ((FRONTEND_REPLICAS == 1)); then
    require_process_stages tts_frontend preprocessing audio_encoder
    if grep -Fq "tts_frontend@r" "${topology_path}"; then
      die "unreplicated frontend unexpectedly started replica processes"
    fi
  else
    local frontend_replica
    for ((frontend_replica=0; frontend_replica<FRONTEND_REPLICAS; frontend_replica++)); do
      require_process_stages \
        "tts_frontend@r${frontend_replica}" \
        "preprocessing@r${frontend_replica}" \
        "audio_encoder@r${frontend_replica}"
    done
    grep -Fq "Replicated stage 'preprocessing'" "${log_path}" \
      || die "preprocessing replica expansion was not observed"
    grep -Fq "Replicated stage 'audio_encoder'" "${log_path}" \
      || die "audio_encoder replica expansion was not observed"
  fi
}

validate_full_pipeline_mps_attachment() {
  local output_path="${CURRENT_REP_DIR}/full-pipeline-mps-attachment.json"
  if [[ "${SERVER_MPS_ENABLED}" == "1" ]]; then
    "${PYTHON_BIN}" "${FULL_PIPELINE_MPS_VERIFIER}" \
      --server-log "${CURRENT_REP_DIR}/server.log" \
      --pipeline-config "${PIPELINE_CONFIG_PATH}" \
      --pipe-directory "${SERVER_MPS_PIPE_DIRECTORY}" \
      --log-directory "${SERVER_MPS_LOG_DIRECTORY}" \
      --output "${output_path}" \
      || die "full-pipeline MPS attachment validation failed"
    return
  fi

  "${PYTHON_BIN}" - \
    "${PIPELINE_CONFIG_PATH}" \
    "${output_path}" <<'PY'
import json
import pathlib
import sys

config_path, output_path = map(pathlib.Path, sys.argv[1:])
config = json.loads(config_path.read_text(encoding="utf-8"))
mps_env = {
    stage["name"]: {
        name: value
        for name, value in stage.get("env", {}).items()
        if name.startswith("CUDA_MPS_")
    }
    for stage in config["stages"]
}
mps_env = {name: env for name, env in mps_env.items() if env}
payload = {
    "mode": "disabled",
    "mps_enabled": False,
    "mps_scope": "all_gpu_processes_in_server_tree",
    "config_stage_mps_env": mps_env,
    "disabled_contract_valid": not mps_env,
}
output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
if mps_env:
    raise SystemExit(f"disabled arm contains stage MPS env: {mps_env}")
PY
}

require_process_stages() {
  local process_name="$1"
  shift
  local line
  line="$(grep -F "Process ${process_name} ready with stages=" \
    "${CURRENT_REP_DIR}/server.log" | tail -n 1)"
  [[ -n "${line}" ]] || die "process ${process_name} did not become ready"
  local stage_name
  for stage_name in "$@"; do
    [[ "${line}" == *"'${stage_name}'"* ]] \
      || die "process ${process_name} is missing stage ${stage_name}: ${line}"
  done
}

validate_replica_bindings() {
  "${PYTHON_BIN}" - \
    "${CURRENT_REP_DIR}/server.log" \
    "${CURRENT_REP_DIR}/replica-binding-summary.json" \
    "${FRONTEND_REPLICAS}" \
    "${SCHEDULER_REPLICAS}" <<'PY'
import ast
import collections
import json
import pathlib
import re
import sys

log_path = pathlib.Path(sys.argv[1])
output_path = pathlib.Path(sys.argv[2])
frontend_replicas = int(sys.argv[3])
scheduler_replicas = int(sys.argv[4])
pattern = re.compile(r"bindings=(None|\{.*\})$")
parsed = []
for line in log_path.read_text(encoding="utf-8").splitlines():
    if "Coordinator submitted req=" not in line:
        continue
    match = pattern.search(line)
    if match:
        parsed.append(ast.literal_eval(match.group(1)))

if not parsed:
    raise SystemExit("no coordinator replica-binding evidence found")

replica_counts = {}
if frontend_replicas > 1:
    replica_counts.update(
        {
            "audio_encoder": frontend_replicas,
            "preprocessing": frontend_replicas,
        }
    )
if scheduler_replicas > 1:
    replica_counts["tts_engine"] = scheduler_replicas

if not replica_counts:
    if any(value is not None for value in parsed):
        raise SystemExit("unreplicated config unexpectedly assigned replica bindings")
    summary = {"requests": len(parsed), "bindings": None}
else:
    expected_keys = set(replica_counts)
    counts = {
        name: collections.Counter() for name in sorted(expected_keys)
    }
    for value in parsed:
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise SystemExit(f"unexpected replica binding: {value!r}")
        if (
            frontend_replicas > 1
            and value["audio_encoder"] != value["preprocessing"]
        ):
            raise SystemExit(f"frontend replica binding is not aligned: {value!r}")
        for name in expected_keys:
            replica_id = int(value[name])
            if not 0 <= replica_id < replica_counts[name]:
                raise SystemExit(f"invalid {name} replica id: {replica_id}")
            counts[name][replica_id] += 1
    for name, counter in counts.items():
        expected_ids = set(range(replica_counts[name]))
        if set(counter) != expected_ids or any(not counter[i] for i in expected_ids):
            raise SystemExit(f"all {name} replicas were not exercised: {counter}")
        if max(counter.values()) - min(counter.values()) > 1:
            raise SystemExit(f"round-robin {name} bindings are imbalanced: {counter}")
    summary = {
        "requests": len(parsed),
        "aligned": True,
        "frontend_aligned": None if frontend_replicas <= 1 else True,
        "replica_counts": replica_counts,
        "counts": {
            name: {str(key): count for key, count in sorted(counter.items())}
            for name, counter in counts.items()
        },
    }

output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
PY
}

validate_client_summary() {
  "${PYTHON_BIN}" - \
    "${CURRENT_REP_DIR}/client/summary.json" \
    "${REQUIRE_FULL_DATASET_COVERAGE}" \
    "${CLIENT_CPUSET}" \
    "${CLIENT_NUMA_NODE}" <<'PY'
import json
import math
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
require_coverage = sys.argv[2] == "1"
expected_cpu_spec = sys.argv[3]
expected_numa_node = int(sys.argv[4])
summary = json.loads(path.read_text(encoding="utf-8"))


def parse_cpu_list(value: str) -> list[int]:
    cpus: set[int] = set()
    for item in value.split(","):
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            cpus.update(range(int(start_text), int(end_text) + 1))
        else:
            cpus.add(int(item))
    return sorted(cpus)


qps = float(summary.get("qps_window", 0.0))
if not math.isfinite(qps) or qps <= 0:
    raise SystemExit(f"invalid measured QPS: {qps}")
if int(summary.get("errors", -1)) != 0:
    raise SystemExit(f"request failures observed: {summary.get('errors')}")
if not summary.get("all_measured_audio_nonempty"):
    raise SystemExit("measured audio correctness gate failed")
if not summary.get("all_measured_prompt_tokens_gt_3"):
    raise SystemExit("measured prompt-token correctness gate failed")
if require_coverage and not (
    summary.get("full_dataset_covered_total")
    and summary.get("full_dataset_covered_window")
):
    raise SystemExit("full SeedTTS coverage gate failed")
expected_cpus = parse_cpu_list(expected_cpu_spec)
for phase in ("start", "end"):
    affinity = summary.get(f"client_cpu_affinity_{phase}") or {}
    if affinity.get("process_cpus") != expected_cpus:
        raise SystemExit(
            f"client {phase} affinity mismatch: "
            f"{affinity.get('process_cpus')} != {expected_cpus}"
        )
    if affinity.get("numa_nodes") != [expected_numa_node]:
        raise SystemExit(
            f"client {phase} NUMA mismatch: {affinity.get('numa_nodes')}"
        )
    if int(affinity.get("physical_core_count", 0)) != len(expected_cpus):
        raise SystemExit(f"client {phase} affinity includes SMT siblings")
    if not affinity.get("all_threads_match_process"):
        raise SystemExit(f"client {phase} threads escaped the CPU affinity")
cpu_delta = summary.get("cpu_stat_delta")
if cpu_delta is not None:
    nr_throttled = int(cpu_delta.get("nr_throttled", 0))
    throttled_usec = int(cpu_delta.get("throttled_usec", 0))
    if nr_throttled or throttled_usec:
        nr_periods = int(cpu_delta.get("nr_periods", 0))
        period_fraction = nr_throttled / nr_periods if nr_periods else 0.0
        print(
            "warning: cgroup CPU throttling observed; retaining the window "
            "with an interference annotation: "
            f"nr_throttled={nr_throttled}, nr_periods={nr_periods}, "
            f"period_fraction={period_fraction:.6%}, "
            f"throttled_usec={throttled_usec}",
            file=sys.stderr,
        )
PY
}

write_pipeline_config() {
  PIPELINE_CONFIG_PATH="${RUN_DIR}/pipeline-config.json"
  taskset --cpu-list "${SERVER_CPUSET}" env \
    PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" - \
      "${PIPELINE_CONFIG_PATH}" \
      "${MODEL_PATH}" \
      "${ARM}" \
      "${FRONTEND_REPLICAS}" \
      "${ENCODER_MEMORY_FRACTION}" \
      "${SCHEDULER_REPLICAS}" \
      "${SCHEDULER_MEMORY_FRACTION}" \
      "${DECODE_CUDA_GRAPH_MAX_FRAMES}" \
      "${PREFILL_COALESCING_ENABLED}" \
      "${COALESCE_REQUESTS}" \
      "${COALESCE_WAIT_MS}" <<'PY'
import json
import pathlib
import sys

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.process_overrides import apply_stage_process_overrides
from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig

(
    output_path,
    model_path,
    arm,
    replicas_text,
    encoder_fraction_text,
    scheduler_replicas_text,
    scheduler_fraction_text,
    max_frames_text,
    prefill_enabled_text,
    coalesce_requests_text,
    coalesce_wait_ms_text,
) = sys.argv[1:]
replicas = int(replicas_text)
scheduler_replicas = int(scheduler_replicas_text)
max_frames = int(max_frames_text)
overrides = {
    "stages.preprocessing.num_replicas": replicas,
    "stages.audio_encoder.num_replicas": replicas,
    "stages.audio_encoder.runtime.resources.total_gpu_memory_fraction": float(
        encoder_fraction_text
    ),
    "stages.tts_engine.num_replicas": scheduler_replicas,
    "stages.tts_engine.runtime.resources.total_gpu_memory_fraction": float(
        scheduler_fraction_text
    ),
    "stages.vocoder.factory_args.compile_decode": False,
    "stages.vocoder.factory_args.decode_cuda_graph_frame_counts": list(
        range(1, max_frames + 1)
    ),
}
if replicas > 1:
    overrides["stages.audio_encoder.replica_devices"] = ",".join(
        ["0"] * replicas
    )
if scheduler_replicas > 1:
    overrides["stages.tts_engine.replica_devices"] = ",".join(
        ["0"] * scheduler_replicas
    )
if prefill_enabled_text == "1":
    overrides.update(
        {
            "stages.tts_engine.factory_args.prefill_coalesce_requests": int(
                coalesce_requests_text
            ),
            "stages.tts_engine.factory_args.prefill_coalesce_wait_ms": int(
                coalesce_wait_ms_text
            ),
        }
    )

config = ConfigManager(
    HiggsTtsPipelineConfig(model_path=model_path)
).merge_config(overrides)
config = apply_stage_process_overrides(
    config,
    isolate_stages=["vocoder"],
    stage_processes=None,
)
payload = config.model_dump(mode="json")
payload["config_cls"] = type(config).__name__
pathlib.Path(output_path).write_text(
    json.dumps(payload, indent=2) + "\n",
    encoding="utf-8",
)
PY
  sha256sum "${PIPELINE_CONFIG_PATH}" \
    > "${RUN_DIR}/pipeline-config.sha256"
}

start_server() {
  local server_env_options=(
    -u CUDA_MPS_PIPE_DIRECTORY
    -u CUDA_MPS_LOG_DIRECTORY
    -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
    -u CUDA_MPS_PINNED_DEVICE_MEM_LIMIT
    -u CUDA_MPS_CLIENT_PRIORITY
    -u CUDA_VISIBLE_DEVICES
  )
  local server_device_env=(
    CUDA_VISIBLE_DEVICES="${SERVER_CUDA_VISIBLE_DEVICES_EFFECTIVE}"
  )
  if [[ "${SERVER_MPS_ENABLED}" == "1" ]]; then
    # Use the same UUID as the private MPS daemon. Numeric selectors are
    # ambiguous after MPS remaps devices, while leaving this unset lets
    # PyTorch's NVML path count unrelated host GPUs before CUDA sees the
    # daemon's one-device namespace.
    server_device_env=(
      CUDA_VISIBLE_DEVICES="${SERVER_CUDA_VISIBLE_DEVICES_EFFECTIVE}"
      CUDA_MPS_PIPE_DIRECTORY="${SERVER_MPS_PIPE_DIRECTORY}"
      CUDA_MPS_LOG_DIRECTORY="${SERVER_MPS_LOG_DIRECTORY}"
    )
  fi
  local server_command=(
    setsid
    taskset
    --cpu-list "${SERVER_CPUSET}"
    env
    "${server_env_options[@]}"
    "${server_device_env[@]}"
    PATH="${PYTHON_BIN_DIR}:${PATH}"
    PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
    OMP_NUM_THREADS="${SERVER_THREADS}"
    MKL_NUM_THREADS="${SERVER_THREADS}"
    OPENBLAS_NUM_THREADS="${SERVER_THREADS}"
    NUMEXPR_NUM_THREADS="${SERVER_THREADS}"
    VECLIB_MAXIMUM_THREADS="${SERVER_THREADS}"
    BLIS_NUM_THREADS="${SERVER_THREADS}"
    SGL_OMNI_INTRAOP_THREADS="${SERVER_THREADS}"
    SGL_OMNI_INTEROP_THREADS=1
    SGLANG_OMNI_INTRAOP_THREADS="${SERVER_THREADS}"
    SGLANG_OMNI_INTEROP_THREADS=1
    SGLANG_OMNI_STARTUP_TIMEOUT="${BOOT_TIMEOUT_SECS}"
    TOKENIZERS_PARALLELISM=false
    HF_HOME="${HF_HOME}"
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
    NO_PROXY="127.0.0.1,localhost"
    no_proxy="127.0.0.1,localhost"
    "${PYTHON_BIN}" -m sglang_omni.cli serve
    --config "${PIPELINE_CONFIG_PATH}"
    --model-path "${MODEL_PATH}"
    --model-name "${MODEL_PATH}"
    --host 127.0.0.1
    --port "${PORT}"
    --allowed-local-media-path /
    --max-running-requests "${SCHEDULER_MAX_RUNNING_REQUESTS}"
    --cuda-graph-max-bs "${SCHEDULER_CUDA_GRAPH_MAX_BS}"
  )
  record_command "${CURRENT_REP_DIR}/server-command.txt" "${server_command[@]}"
  "${server_command[@]}" > "${CURRENT_REP_DIR}/server.log" 2>&1 &
  SERVER_PID=$!
  echo "${SERVER_PID}" > "${CURRENT_REP_DIR}/server.pid"
  wait_for_server
  validate_server_log
  validate_full_pipeline_mps_attachment
  record_and_validate_affinity_tree \
    "${SERVER_PID}" \
    "${SERVER_CPUSET}" \
    "${SERVER_NUMA_NODE}" \
    "${CURRENT_REP_DIR}/server-affinity-pre-client.json"
}

start_observers() {
  [[ "${PROFILE}" == "1" ]] || return 0
  local gpu_query_fields
  gpu_query_fields="timestamp,index,pci.bus_id,utilization.gpu,utilization.memory"
  gpu_query_fields+=",memory.used,power.draw,clocks.sm,clocks.mem"
  taskset --cpu-list "${CLIENT_CPUSET}" nvidia-smi -i "${GPU_ID}" \
    --query-gpu="${gpu_query_fields}" \
    --format=csv,nounits --loop-ms=500 \
    > "${CURRENT_REP_DIR}/nvidia-query.csv" 2>&1 &
  GPU_QUERY_PID=$!
  taskset --cpu-list "${CLIENT_CPUSET}" \
    nvidia-smi dmon -i "${GPU_ID}" -s pucvmet -d 1 -o DT \
    > "${CURRENT_REP_DIR}/nvidia-dmon.log" 2>&1 &
  GPU_DMON_PID=$!

  if [[ "${NSYS_SM_PROFILE}" == "1" ]]; then
    (
      sleep "${WARMUP_SECS}"
      exec taskset --cpu-list "${CLIENT_CPUSET}" nsys profile \
        --gpu-metrics-devices="${GPU_ID}" \
        --gpu-metrics-set=gh100 \
        -d "${NSYS_DURATION_S}" \
        -o "${CURRENT_REP_DIR}/nsys-sm" \
        -f true \
        sleep "$((NSYS_DURATION_S + 3))"
    ) > "${CURRENT_REP_DIR}/nsys.log" 2>&1 &
    NSYS_PID=$!
  fi
}

run_client() {
  local rep="$1"
  local client_command=(
    taskset
    --cpu-list "${CLIENT_CPUSET}"
    env
    PATH="${PYTHON_BIN_DIR}:${PATH}"
    PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
    OMP_NUM_THREADS="${CLIENT_THREADS}"
    MKL_NUM_THREADS="${CLIENT_THREADS}"
    OPENBLAS_NUM_THREADS="${CLIENT_THREADS}"
    NUMEXPR_NUM_THREADS="${CLIENT_THREADS}"
    TOKENIZERS_PARALLELISM=false
    HF_HOME="${HF_HOME}"
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
    NO_PROXY="127.0.0.1,localhost"
    no_proxy="127.0.0.1,localhost"
    "${PYTHON_BIN}" "${BENCHMARK_CLIENT}"
    --label "${ARM}-gpu${GPU_ID}-c${CONC}-r${rep}"
    --base-url "${BASE_URL}"
    --model "${MODEL_PATH}"
    --meta "${DATASET}"
    --dataset-repo-id "${DATASET_REPO_ID}"
    --dataset-revision "${DATASET_REVISION}"
    --dataset-parquet-sha256 "${DATASET_PARQUET_SHA256}"
    --lang en
    --output-dir "${CURRENT_REP_DIR}/client"
    --secs "${SECS}"
    --warmup-secs "${WARMUP_SECS}"
    --concurrency "${CONC}"
    --voice-clone
    --speaker "${SPEAKER}"
    --temperature "${TEMPERATURE}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --sample-offset 0
    --expected-samples "${EXPECTED_SAMPLES}"
    --expected-unique-references "${EXPECTED_UNIQUE_REFERENCES}"
    --request-timeout-s 600
  )
  if [[ "${REQUIRE_FULL_DATASET_COVERAGE}" == "1" ]]; then
    client_command+=(--require-full-dataset-coverage)
  else
    client_command+=(--no-require-full-dataset-coverage)
  fi
  if [[ "${PROFILE}" == "1" ]]; then
    client_command+=(--collect-cpu-metrics)
    client_command+=(
      --gpu-metrics-csv "${CURRENT_REP_DIR}/nvidia-query.csv"
    )
  fi
  record_command "${CURRENT_REP_DIR}/client-command.txt" "${client_command[@]}"
  "${client_command[@]}" | tee "${CURRENT_REP_DIR}/client.log"
}

run_repetition() {
  local rep="$1"
  CURRENT_REP_DIR="${RUN_DIR}/rep-${rep}"
  mkdir -p "${CURRENT_REP_DIR}"
  echo
  echo "=== ${ARM}: repetition ${rep}/${REPS} ==="
  start_server
  start_observers
  run_client "${rep}"
  stop_observers
  validate_server_log
  validate_replica_bindings
  validate_client_summary
  record_and_validate_affinity_tree \
    "${SERVER_PID}" \
    "${SERVER_CPUSET}" \
    "${SERVER_NUMA_NODE}" \
    "${CURRENT_REP_DIR}/server-affinity-post-client.json"
  nvidia-smi -i "${GPU_ID}" -q > "${CURRENT_REP_DIR}/nvidia-smi-after.txt"
  stop_server
}

print_summary() {
  "${PYTHON_BIN}" - "${RUN_DIR}" "${CONC}" <<'PY'
import glob
import json
import statistics
import sys

root, concurrency = sys.argv[1], int(sys.argv[2])
paths = sorted(glob.glob(f"{root}/rep-*/client/summary.json"))
rows = [json.load(open(path)) for path in paths]
if not rows:
    raise SystemExit("no benchmark summaries found")

qps = statistics.fmean(row["qps_window"] for row in rows)
lat = statistics.fmean(row["lat_mean_s"] for row in rows)
p95 = statistics.fmean(row["lat_p95_s"] for row in rows)
errors = sum(row["errors"] for row in rows)

print("\n=== Higgs entry-replica result summary ===")
print(f"repetitions={len(rows)} qps_mean={qps:.3f} "
      f"lat_mean={lat:.3f}s p95_mean={p95:.3f}s "
      f"qps*lat={qps * lat:.1f} errors={errors}")
if abs(qps * lat - concurrency) / concurrency > 0.15:
    print(
        f"WARNING: qps*lat={qps * lat:.1f} is not close to "
        f"concurrency={concurrency}"
    )
print(f"Raw results: {root}")
PY
}

main() {
  validate_gpu_selector
  validate_boolean PROFILE "${PROFILE}"
  validate_boolean NSYS_SM_PROFILE "${NSYS_SM_PROFILE}"
  validate_boolean \
    REQUIRE_FULL_DATASET_COVERAGE \
    "${REQUIRE_FULL_DATASET_COVERAGE}"
  validate_positive_integer CAP "${CAP}"
  validate_positive_integer CONC "${CONC}"
  validate_positive_integer CUDA_GRAPH_MAX_BS "${CUDA_GRAPH_MAX_BS}"
  validate_positive_integer REPS "${REPS}"
  validate_positive_integer FRONTEND_REPLICAS "${FRONTEND_REPLICAS}"
  validate_positive_integer SCHEDULER_REPLICAS "${SCHEDULER_REPLICAS}"
  validate_positive_integer \
    SCHEDULER_MAX_RUNNING_REQUESTS \
    "${SCHEDULER_MAX_RUNNING_REQUESTS}"
  validate_positive_integer \
    SCHEDULER_CUDA_GRAPH_MAX_BS \
    "${SCHEDULER_CUDA_GRAPH_MAX_BS}"
  validate_positive_integer SERVER_THREADS "${SERVER_THREADS}"
  validate_positive_integer CLIENT_THREADS "${CLIENT_THREADS}"
  validate_nonnegative_integer SERVER_NUMA_NODE "${SERVER_NUMA_NODE}"
  validate_nonnegative_integer CLIENT_NUMA_NODE "${CLIENT_NUMA_NODE}"
  validate_positive_integer EXPECTED_AFFINITY_CPUS "${EXPECTED_AFFINITY_CPUS}"
  validate_positive_integer \
    SERVER_AFFINITY_CPU_COUNT \
    "${SERVER_AFFINITY_CPU_COUNT}"
  validate_positive_integer \
    CLIENT_AFFINITY_CPU_COUNT \
    "${CLIENT_AFFINITY_CPU_COUNT}"
  case "${CGROUP_CPU_QUOTA_POLICY}" in
    exact_affinity|record_only) ;;
    *) die "CGROUP_CPU_QUOTA_POLICY must be exact_affinity or record_only" ;;
  esac
  validate_positive_integer SECS "${SECS}"
  validate_nonnegative_integer WARMUP_SECS "${WARMUP_SECS}"
  validate_positive_integer MAX_NEW_TOKENS "${MAX_NEW_TOKENS}"
  validate_positive_integer \
    DECODE_CUDA_GRAPH_MAX_FRAMES \
    "${DECODE_CUDA_GRAPH_MAX_FRAMES}"
  validate_positive_integer COALESCE_REQUESTS "${COALESCE_REQUESTS}"
  validate_nonnegative_integer COALESCE_WAIT_MS "${COALESCE_WAIT_MS}"
  validate_positive_integer EXPECTED_SAMPLES "${EXPECTED_SAMPLES}"
  validate_nonnegative_integer \
    EXPECTED_UNIQUE_REFERENCES \
    "${EXPECTED_UNIQUE_REFERENCES}"
  validate_positive_integer BOOT_TIMEOUT_SECS "${BOOT_TIMEOUT_SECS}"
  case "${DATASET_SOURCE_MODE}" in
    local_verified|huggingface) ;;
    *) die "DATASET_SOURCE_MODE must be local_verified or huggingface" ;;
  esac
  validate_arm
  ((WARMUP_SECS < SECS)) \
    || die "WARMUP_SECS must be smaller than SECS"
  ((DECODE_CUDA_GRAPH_MAX_FRAMES >= MAX_NEW_TOKENS)) \
    || die "decode CUDA graph frame domain must cover MAX_NEW_TOKENS"
  local scheduler_aggregate_capacity
  scheduler_aggregate_capacity=$((
    SCHEDULER_REPLICAS * SCHEDULER_MAX_RUNNING_REQUESTS
  ))
  ((scheduler_aggregate_capacity >= CAP)) \
    || die "aggregate scheduler capacity must cover logical CAP"
  ((scheduler_aggregate_capacity < CAP + SCHEDULER_REPLICAS)) \
    || die "aggregate scheduler capacity must be the ceiling split of CAP"
  ((SCHEDULER_CUDA_GRAPH_MAX_BS <= SCHEDULER_MAX_RUNNING_REQUESTS)) \
    || die "scheduler CUDA graph max batch cannot exceed per-replica capacity"
  [[ "${NSYS_SM_PROFILE}" == "0" || "${PROFILE}" == "1" ]] \
    || die "NSYS_SM_PROFILE=1 requires PROFILE=1"
  require_command "${PYTHON_BIN}"
  PYTHON_BIN_DIR="$(
    cd -- "$(dirname -- "$(command -v "${PYTHON_BIN}")")" && pwd
  )"
  require_command curl
  require_command lscpu
  require_command nvidia-smi
  require_command setsid
  require_command sha256sum
  require_command taskset
  validate_memory_budget
  if [[ "${NSYS_SM_PROFILE}" == "1" ]]; then
    require_command nsys
  fi
  [[ -d "${REPO_DIR}/sglang_omni" ]] \
    || die "sglang-omni repository not found: ${REPO_DIR}"
  [[ -f "${BENCHMARK_CLIENT}" ]] \
    || die "benchmark client not found: ${BENCHMARK_CLIENT}"
  [[ -f "${FULL_PIPELINE_MPS_VERIFIER}" ]] \
    || die "full-pipeline MPS verifier not found: ${FULL_PIPELINE_MPS_VERIFIER}"
  resolve_server_mps
  if [[ "${SERVER_MPS_ENABLED}" == "1" ]]; then
    require_command nvidia-cuda-mps-control
  fi
  resolve_and_validate_dataset
  resolve_prefill_coalescing
  printf \
    'Prefill coalescing: mode=%s supported=%s enabled=%s\n' \
    "${PREFILL_COALESCING_MODE}" \
    "${PREFILL_COALESCING_SUPPORTED}" \
    "${PREFILL_COALESCING_ENABLED}"

  configure_gpu
  validate_gpu_idle
  [[ -e "${RUN_DIR}" ]] && die "RUN_DIR already exists: ${RUN_DIR}"
  mkdir -p "${RUN_DIR}"
  validate_cpu_bindings
  write_pipeline_config
  record_attachment_identity
  cd "${REPO_DIR}"

  export NO_PROXY="127.0.0.1,localhost"
  export no_proxy="${NO_PROXY}"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

  printf '%s\n' "${CUDA_VISIBLE_DEVICES}" \
    > "${RUN_DIR}/cuda-visible-devices.txt"
  printf '%s\n' "${MODEL_PATH}" > "${RUN_DIR}/model-path.txt"
  printf '%s\n' "${HF_HOME}" > "${RUN_DIR}/hf-home.txt"
  printf '%s\n' "${DATASET}" > "${RUN_DIR}/dataset-path.txt"
  printf '%s\n' "${DATASET_REPO_ID}" > "${RUN_DIR}/dataset-repo-id.txt"
  printf '%s\n' "${DATASET_REVISION}" > "${RUN_DIR}/dataset-revision.txt"
  printf '%s\n' "${DATASET_PARQUET_SHA256}" \
    > "${RUN_DIR}/dataset-parquet-sha256.txt"
  printf '%s\n' \
    "cap=${CAP}" \
    "concurrency=${CONC}" \
    "logical_cuda_graph_max_bs=${CUDA_GRAPH_MAX_BS}" \
    "repetitions=${REPS}" \
    "total_seconds=${SECS}" \
    "warmup_seconds=${WARMUP_SECS}" \
    "measurement_seconds=$((SECS - WARMUP_SECS))" \
    "max_new_tokens=${MAX_NEW_TOKENS}" \
    "arm=${ARM}" \
    "frontend_replicas=${FRONTEND_REPLICAS}" \
    "frontend_replica_devices=$([[ "${FRONTEND_REPLICAS}" -gt 1 ]] && printf 'gpu0_repeated_%s' "${FRONTEND_REPLICAS}" || printf 'none')" \
    "encoder_memory_fraction=${ENCODER_MEMORY_FRACTION}" \
    "scheduler_replicas=${SCHEDULER_REPLICAS}" \
    "scheduler_replica_devices=$([[ "${SCHEDULER_REPLICAS}" -gt 1 ]] && printf 'gpu0_repeated_%s' "${SCHEDULER_REPLICAS}" || printf 'none')" \
    "scheduler_memory_fraction=${SCHEDULER_MEMORY_FRACTION}" \
    "scheduler_max_running_requests_each=${SCHEDULER_MAX_RUNNING_REQUESTS}" \
    "scheduler_cuda_graph_max_bs_each=${SCHEDULER_CUDA_GRAPH_MAX_BS}" \
    "scheduler_aggregate_capacity=$((SCHEDULER_REPLICAS * SCHEDULER_MAX_RUNNING_REQUESTS))" \
    "server_mps_mode=${SERVER_MPS_MODE}" \
    "server_mps_enabled=${SERVER_MPS_ENABLED}" \
    "server_mps_scope=all_gpu_processes_in_server_tree" \
    "server_mps_pipe_directory=${SERVER_MPS_PIPE_DIRECTORY:-disabled}" \
    "server_mps_log_directory=${SERVER_MPS_LOG_DIRECTORY:-disabled}" \
    "server_mps_gpu_uuid=${SERVER_MPS_GPU_UUID:-disabled}" \
    "server_mps_attachment_gate=exact_configured_gpu_process_pid_set" \
    "server_cuda_visible_devices=${SERVER_CUDA_VISIBLE_DEVICES_EFFECTIVE}" \
    "server_cuda_visible_devices_mode=$([[ "${SERVER_MPS_ENABLED}" == "1" ]] && printf 'gpu_uuid' || printf 'physical_numeric_index')" \
    "vocoder_process=isolated" \
    "vocoder_compile_decode=false" \
    "vocoder_decode_cuda_graph_frame_counts=1..${DECODE_CUDA_GRAPH_MAX_FRAMES}" \
    "server_threads=${SERVER_THREADS}" \
    "client_threads=${CLIENT_THREADS}" \
    "server_cpu_affinity=${SERVER_CPUSET}" \
    "server_numa_node=${SERVER_NUMA_NODE}" \
    "server_affinity_cpu_count=${SERVER_AFFINITY_CPU_COUNT}" \
    "server_gpu_local=true" \
    "client_cpu_affinity=${CLIENT_CPUSET}" \
    "client_numa_node=${CLIENT_NUMA_NODE}" \
    "client_affinity_cpu_count=${CLIENT_AFFINITY_CPU_COUNT}" \
    "server_client_affinity_disjoint=true" \
    "expected_samples=${EXPECTED_SAMPLES}" \
    "expected_unique_references=${EXPECTED_UNIQUE_REFERENCES}" \
    "dataset_repo_id=${DATASET_REPO_ID}" \
    "dataset_source_mode=${DATASET_SOURCE_MODE}" \
    "dataset_revision=${DATASET_REVISION:-unpinned}" \
    "dataset_snapshot_path=${DATASET}" \
    "dataset_parquet_sha256=${DATASET_PARQUET_SHA256:-unverified}" \
    "sample_selection=${SAMPLE_SELECTION}" \
    "prefill_coalescing_mode=${PREFILL_COALESCING_MODE}" \
    "prefill_coalescing_supported=${PREFILL_COALESCING_SUPPORTED}" \
    "prefill_coalescing_enabled=${PREFILL_COALESCING_ENABLED}" \
    "prefill_coalesce_requests_requested=${COALESCE_REQUESTS}" \
    "prefill_coalesce_wait_ms_requested=${COALESCE_WAIT_MS}" \
    "prefill_coalesce_requests_effective=${PREFILL_COALESCE_REQUESTS_EFFECTIVE}" \
    "prefill_coalesce_wait_ms_effective=${PREFILL_COALESCE_WAIT_MS_EFFECTIVE}" \
    "voice_clone=true" \
    "speaker=${SPEAKER}" \
    "temperature=${TEMPERATURE}" \
    "stream=false" \
    "api=/v1/audio/speech" \
    "request_input=SeedTTS.target_text" \
    "reference_format=references" \
    "response_format=wav" \
    "require_full_dataset_coverage=${REQUIRE_FULL_DATASET_COVERAGE}" \
    "cgroup_cpu_quota_policy=${CGROUP_CPU_QUOTA_POLICY}" \
    "cpu_binding=taskset; server/client use disjoint NUMA-local physical-core sets" \
    > "${RUN_DIR}/benchmark-contract.txt"
  nvidia-smi -i "${GPU_ID}" -q > "${RUN_DIR}/nvidia-smi-before.txt"
  cp /proc/self/status "${RUN_DIR}/runner-proc-status.txt"
  if [[ -r /sys/fs/cgroup/cpu.max ]]; then
    cp /sys/fs/cgroup/cpu.max "${RUN_DIR}/cpu.max.txt"
  fi

  local rep
  for ((rep=1; rep<=REPS; rep++)); do
    run_repetition "${rep}"
  done
  print_summary
  echo "Benchmark complete: ${RUN_DIR}"
}

trap cleanup EXIT INT TERM
main "$@"
