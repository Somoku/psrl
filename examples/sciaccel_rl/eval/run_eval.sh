#!/usr/bin/env bash
# run_eval.sh — end-to-end SciAccel-RL evaluation on one node.
#
# Serves a checkpoint as a fleet of vLLM replicas via psrl.eval.serve, runs Harbor
# episodes over the v2 dataset with eval_sciaccel.py, prints the summary, and tears
# the fleet down.
#
# Model-agnostic: pass --model / --served-model-name for any checkpoint.
#
# The anchor agents (oracle, nop) need no model, so --agent nop / --agent oracle
# skip the vLLM launch entirely. Run those FIRST: the sciaccel-rl README requires
# oracle = full score and nop = 0 on this machine before any agent number means
# anything. The nop pass also warms every Docker image (it walks the full
# env-build -> verifier-build -> grade path), which is what makes the subsequent
# model run fast.
#
# Usage:
#   # anchors + image warm-up (no GPU needed)
#   bash run_eval.sh --agent nop
#   bash run_eval.sh --agent oracle
#
#   # baseline a checkpoint over all tasks
#   bash run_eval.sh --model /path/to/ckpt --served-model-name my-ckpt
#
#   # quick smoke over one task per family
#   bash run_eval.sh --per-family 1 --families sign bounds accel --n-concurrent 3
#
# Options:
#   --model PATH            HF checkpoint to serve (default: Qwen3.5-9B).
#   --served-model-name N   Name the endpoint advertises (default: qwen35-9b).
#   --agent NAME            terminus-2 (default) | oracle | nop.
#   --dataset PATH          v2 Parquet (default: examples/sciaccel_rl/data/v2/all.parquet).
#   --output-dir PATH       Artefact directory (default: outputs/sciaccel_rl/eval/<agent>_<ts>).
#   --port N                Port of the first replica (default: 8000). A fleet of
#                           R replicas uses ports N .. N+R-1.
#   --replicas N            Independent vLLM servers to start (default: 4). Each
#                           takes --tp GPUs, so replicas * tp must fit the host.
#                           Endpoints are auto-discovered from the fleet's
#                           endpoints.json, so --api-base is rarely needed.
#   --tp N / --pp N         Per-replica parallelism (default: 2 / 1).
#   --max-model-len N       Context window (default: 32768). Bounded by KV cache, not
#                           by the model: Qwen3.5-9B is trained for 262144 but one
#                           2xH20 replica holds only ~140k KV tokens, so 32768 is what
#                           keeps ~4 sequences per replica running concurrently.
#                           Verify with: curl -s localhost:8000/metrics | grep num_gpu_blocks
#   --max-output-tokens N   Advisory output budget in terminus-2's model_info
#                           (default: 8192). METADATA ONLY on the litellm chat path:
#                           nothing sends it as a per-request max_tokens, so it does not
#                           bound generation -- --max-model-len does. Measured per-turn
#                           output was 219-549 tokens regardless of this value.
#   --max-turns N           Cap agent turns per trial (default: 25). 0 = unbounded.
#                           Every turn resends the whole transcript, so uncapped runs
#                           tend to exhaust the context window and die UNGRADED; a cap
#                           ends the loop cleanly so the verifier still scores it.
#   --max-per-instance N    Concurrent tasks per vLLM endpoint (default: 32). Total in
#                           flight is this times the endpoint count. Tasks are pulled
#                           from a shared queue, so a finished slot takes the next one.
#   --api-base URLS         Comma-separated endpoint list. Only needed with
#                           --reuse-server; otherwise endpoints are discovered from
#                           the fleet's endpoints.json.
#   -k N                    Attempts per task (default: 1).
#   -n N                    Concurrent Harbor trials (default: 8).
#   --temperature F         Sampling temperature (default: 1.0).
#   --categories A B        Category filter forwarded to eval_sciaccel.
#   --families A B          Family filter forwarded to eval_sciaccel.
#   --task-glob PATTERN     Task-name glob forwarded to eval_sciaccel.
#   --per-family N          Cap per (category, family, tree) group.
#   --limit N               Overall task cap.
#   --timeout-multiplier F  Scales task-declared agent timeouts (default: 1.0).
#   --build-timeout-multiplier F
#                           Scales task-declared image build timeouts (default: 2.0).
#   --skip-gpu-tasks        Drop tasks needing a GPU (default: on). Harbor's local
#                           Docker provider rejects them, so they can only be errors.
#   --with-gpu-tasks        Keep them, for a provider that does support GPUs.
#   --no-apt-mirror         Don't redirect apt to the internal Debian mirror during
#                           image builds. Only for hosts with a direct route to
#                           deb.debian.org; see config/apt-mirror-override.yaml.
#   --keep-server           Leave vLLM running after the eval (for a follow-up run).
#   --reuse-server          Do not launch vLLM; assume one is already on --port.
#   -h | --help             Print this help.
#
# NOTE(claude): No --tool-call-parser is passed to vLLM. Terminus-2 parses its own
# JSON/XML text protocol out of the assistant message, so enabling vLLM's
# tool-call extraction would strip the very text the agent needs.

set -euo pipefail

usage() { sed -n '2,74p' "$0"; }

PSRL_PATH=${PSRL_PATH:-$(python3 -c "import os, psrl; print(os.path.dirname(os.path.dirname(psrl.__file__)))")}
ENV_SCRIPT=${ENV_SCRIPT:-/apdcephfs_zwfy10/share_303541817/lhy/env/psrl.sh}

MODEL=${HF_MODEL_PATH:-/apdcephfs_zwfy10/share_303541817/lhy/models/Qwen3.5-9B}
SERVED_NAME="qwen35-9b"
AGENT="terminus-2"
DATASET="${PSRL_PATH}/examples/sciaccel_rl/data/v2/all.parquet"
OUTPUT_DIR=""
PORT=8000
TP=2
PP=1
REPLICAS=4
MAX_MODEL_LEN=32768
MAX_OUTPUT_TOKENS=8192
MAX_TURNS=25
MAX_PER_INSTANCE=32
API_BASE=""
N_ATTEMPTS=1
N_CONCURRENT=8
TEMPERATURE=1.0
TIMEOUT_MULTIPLIER=1.0
BUILD_TIMEOUT_MULTIPLIER=2.0
APT_MIRROR=1
SKIP_GPU_TASKS=1
KEEP_SERVER=0
REUSE_SERVER=0
CATEGORIES=()
FAMILIES=()
TASK_GLOB=""
PER_FAMILY=0
LIMIT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)               MODEL="$2"; shift 2 ;;
        --served-model-name)   SERVED_NAME="$2"; shift 2 ;;
        --agent)               AGENT="$2"; shift 2 ;;
        --dataset)             DATASET="$2"; shift 2 ;;
        --output-dir)          OUTPUT_DIR="$2"; shift 2 ;;
        --port)                PORT="$2"; shift 2 ;;
        --tp)                  TP="$2"; shift 2 ;;
        --pp)                  PP="$2"; shift 2 ;;
        --replicas)            REPLICAS="$2"; shift 2 ;;
        --max-model-len)       MAX_MODEL_LEN="$2"; shift 2 ;;
        --max-output-tokens)   MAX_OUTPUT_TOKENS="$2"; shift 2 ;;
        --max-turns)           MAX_TURNS="$2"; shift 2 ;;
        --max-per-instance)    MAX_PER_INSTANCE="$2"; shift 2 ;;
        --api-base)            API_BASE="$2"; shift 2 ;;
        -k|--n-attempts)       N_ATTEMPTS="$2"; shift 2 ;;
        -n|--n-concurrent)     N_CONCURRENT="$2"; shift 2 ;;
        --temperature)         TEMPERATURE="$2"; shift 2 ;;
        --timeout-multiplier)  TIMEOUT_MULTIPLIER="$2"; shift 2 ;;
        --build-timeout-multiplier) BUILD_TIMEOUT_MULTIPLIER="$2"; shift 2 ;;
        --no-apt-mirror)       APT_MIRROR=0; shift ;;
        --skip-gpu-tasks)      SKIP_GPU_TASKS=1; shift ;;
        --with-gpu-tasks)      SKIP_GPU_TASKS=0; shift ;;
        --keep-server)         KEEP_SERVER=1; shift ;;
        --reuse-server)        REUSE_SERVER=1; shift ;;
        --task-glob)           TASK_GLOB="$2"; shift 2 ;;
        --per-family)          PER_FAMILY="$2"; shift 2 ;;
        --limit)               LIMIT="$2"; shift 2 ;;
        --categories)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do CATEGORIES+=("$1"); shift; done ;;
        --families)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do FAMILIES+=("$1"); shift; done ;;
        -h|--help)             usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -f "${DATASET}" ]] || {
    echo "ERROR: dataset not found: ${DATASET}" >&2
    echo "Build it first: python -m examples.sciaccel_rl.prepare.build_dataset_v2 --repo <sciaccel-rl> --out-dir $(dirname "${DATASET}")" >&2
    exit 2
}

if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${PSRL_PATH}/outputs/sciaccel_rl/eval/${AGENT}_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${OUTPUT_DIR}"

# Anchor agents play the policy themselves; no endpoint is involved.
NEEDS_MODEL=1
if [[ "${AGENT}" == "oracle" || "${AGENT}" == "nop" ]]; then
    NEEDS_MODEL=0
fi

LAUNCHED_SERVER=0
SERVE_DIR="${OUTPUT_DIR}/serve"

# Kill the process GROUP, not the pid. Each replica is a process-group leader and
# its VLLM::Worker_TPn children join that group; signalling only the leader leaves
# the workers alive still holding every GPU.
cleanup() {
    if [[ "${LAUNCHED_SERVER}" -eq 1 && "${KEEP_SERVER}" -eq 0 && -f "${SERVE_DIR}/endpoints.json" ]]; then
        echo "[run_eval] Stopping the vLLM fleet..."
        python3 -c "
import json, os, signal, sys, time

pids = [e['pid'] for e in json.load(open(sys.argv[1]))['endpoints'] if e.get('pid')]
groups = []
for pid in pids:
    try:
        groups.append(os.getpgid(pid))
    except ProcessLookupError:
        pass
for gid in groups:
    try:
        os.killpg(gid, signal.SIGTERM)
        print(f'  SIGTERM -> process group {gid}')
    except (ProcessLookupError, PermissionError):
        pass
time.sleep(15)
for gid in groups:
    try:
        os.killpg(gid, signal.SIGKILL)
        print(f'  SIGKILL -> process group {gid} (ignored SIGTERM)')
    except (ProcessLookupError, PermissionError):
        pass
" "${SERVE_DIR}/endpoints.json" || true
    fi
}
trap cleanup EXIT

set +u
# shellcheck disable=SC1090
source "${ENV_SCRIPT}"
set -u

if [[ "${NEEDS_MODEL}" -eq 1 && "${REUSE_SERVER}" -eq 0 ]]; then
    [[ -d "${MODEL}" ]] || { echo "ERROR: model directory not found: ${MODEL}" >&2; exit 2; }
    echo "[run_eval] Serving ${MODEL} as ${SERVED_NAME}: ${REPLICAS} replica(s) x TP=${TP} from port ${PORT}..."
    # A fleet of independent servers, not one server with --data-parallel-size:
    # DP is broken in this repo's patched vLLM, and eval_sciaccel spreads its work
    # queue across every endpoint anyway.
    (cd "${PSRL_PATH}" && python3 -m psrl.eval.serve \
        topology=fleet \
        topology.replicas="${REPLICAS}" \
        topology.tp="${TP}" \
        topology.pp="${PP}" \
        topology.base_port="${PORT}" \
        server.checkpoint="${MODEL}" \
        server.served_model_name="${SERVED_NAME}" \
        server.max_model_len="${MAX_MODEL_LEN}" \
        env_script="${ENV_SCRIPT}" \
        output_dir="${SERVE_DIR}")
    LAUNCHED_SERVER=1
fi

# Discover endpoints from whatever the fleet actually brought up healthy, so the
# eval can never be pointed at a replica that failed to load.
if [[ "${NEEDS_MODEL}" -eq 1 && -z "${API_BASE}" && -f "${SERVE_DIR}/endpoints.json" ]]; then
    API_BASE="$(python3 -c "
import json, sys
payload = json.load(open(sys.argv[1]))
print(','.join(e['url'] for e in payload['endpoints']))
" "${SERVE_DIR}/endpoints.json")"
    echo "[run_eval] Discovered endpoints: ${API_BASE}"
fi

EVAL_ARGS=(
    --dataset "${DATASET}"
    --output-dir "${OUTPUT_DIR}"
    --agent "${AGENT}"
    --n-attempts "${N_ATTEMPTS}"
    --n-concurrent "${N_CONCURRENT}"
    --timeout-multiplier "${TIMEOUT_MULTIPLIER}"
    --build-timeout-multiplier "${BUILD_TIMEOUT_MULTIPLIER}"
    --max-per-instance "${MAX_PER_INSTANCE}"
)
if [[ "${NEEDS_MODEL}" -eq 1 ]]; then
    EVAL_ARGS+=(
        --served-model-name "${SERVED_NAME}"
        --api-base "${API_BASE:-http://127.0.0.1:${PORT}/v1}"
        --temperature "${TEMPERATURE}"
        --max-model-len "${MAX_MODEL_LEN}"
        --max-output-tokens "${MAX_OUTPUT_TOKENS}"
        --max-turns "${MAX_TURNS}"
    )
fi
[[ ${#CATEGORIES[@]} -gt 0 ]] && EVAL_ARGS+=(--categories "${CATEGORIES[@]}")
[[ "${APT_MIRROR}" -eq 0 ]]     && EVAL_ARGS+=(--no-apt-mirror)
[[ "${SKIP_GPU_TASKS}" -eq 1 ]] && EVAL_ARGS+=(--skip-gpu-tasks)
[[ ${#FAMILIES[@]} -gt 0 ]]   && EVAL_ARGS+=(--families "${FAMILIES[@]}")
[[ -n "${TASK_GLOB}" ]]       && EVAL_ARGS+=(--task-glob "${TASK_GLOB}")
[[ "${PER_FAMILY}" -gt 0 ]]   && EVAL_ARGS+=(--per-family "${PER_FAMILY}")
[[ "${LIMIT}" -gt 0 ]]        && EVAL_ARGS+=(--limit "${LIMIT}")

cd "${PSRL_PATH}"
PYTHONUNBUFFERED=1 python3 -m examples.sciaccel_rl.eval.eval_sciaccel "${EVAL_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/eval.log"
