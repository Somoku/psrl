#!/usr/bin/env bash
# serve_vllm_multinode.sh — start one vLLM replica on every host in a
# hosts file (data-parallel across hosts). Each host runs an independent
# OpenAI-compatible server with its own --tensor-parallel-size, so the
# *aggregate* throughput is `#hosts × host_throughput`.
#
# This helper does NOT do cross-node tensor parallelism (that requires a
# pre-started Ray cluster). If your model doesn't fit on a single host,
# start a Ray cluster yourself and use serve_vllm.sh with --tp bigger
# than the local GPU count + --distributed-executor-backend ray.
#
# What this script produces
# -------------------------
#
#   <outdir>/endpoints.txt     One `http://<host>:<port>` per successful host.
#   <outdir>/litellm_proxy.yaml A drop-in litellm proxy config that
#                              round-robins across every endpoint. Use with
#                              `litellm --config <path> --port 4000` to get
#                              a single URL for downstream clients.
#   <outdir>/host_logs/<host>.launch.log   Per-host stdout of the launch.
#
# Prerequisites
# -------------
#
#   * Shared FS: the repo (--repo-root) and the checkpoint are readable
#     from every host at the same path.
#   * Passwordless ssh to every host.
#   * Each host has CUDA + conda env + vLLM installed (same shared conda
#     env works well).
#
# Usage
# -----
#   bash serve_vllm_multinode.sh \
#       --hosts /jizhicfs/lhy/hosts/32GPUs \
#       --checkpoint /jizhicfs/lhy/checkpoints/my-step-1000 \
#       --tp 2 --port 8000 \
#       --served-model-name my-model \
#       --tool-call-parser hermes \
#       --outdir examples/mini_swe/output/serve/my_step1000
#
# Options
# -------
#   --hosts FILE                Hosts file (same format as load_all_nodes.sh).
#   --checkpoint PATH           Checkpoint path on shared FS. (required)
#   --served-model-name NAME    Model name exposed via /v1. Default: basename.
#                               Every host uses the same name so clients
#                               are agnostic to which replica they hit.
#   --port N                    Port bound on every host (default: 8000).
#   --tp N                      Per-host tensor-parallel-size (default: 1).
#   --pp N                      Per-host pipeline-parallel-size (default: 1).
#   --dp N                      Per-host data-parallel-size (default: 1).
#                               Combined: N_total_replicas = #hosts * DP.
#   --gpu-ids LIST              CUDA_VISIBLE_DEVICES on *every* host.
#   --max-model-len N
#   --gpu-memory-utilization F
#   --tool-call-parser NAME     Default: hermes.
#   --chat-template PATH        Shared-FS path to a jinja chat template.
#   --distributed-executor-backend BACKEND
#   --extra 'ARGS'              Extra vLLM args passed verbatim.
#   --wait-ready SECONDS        Per-host readiness timeout (default: 1800 =
#                               30min; first-time torch.compile on MoE
#                               models like Qwen3-30B can exceed 15min).
#   --outdir DIR                Where to write endpoints.txt + logs.
#                               Default: examples/mini_swe/output/serve/<ts>.
#   --repo-root PATH            Absolute path to psrl_agent repo on shared FS
#                               (default: /jizhicfs/lhy/psrl_agent).
#   --env-script PATH           Env script forwarded to every host's
#                               serve_vllm.sh. Must contain `conda activate`
#                               plus all the NCCL / UCX / vLLM / cudnn / torch
#                               LD_LIBRARY_PATH knobs. Default:
#                               /jizhicfs/lhy/env/psrl.sh (same as training).
#                               Pass '' to disable sourcing.
#   --ssh-user USER             Optional ssh username.
#   --dry-run                   Print the per-host ssh command and exit.
#
# Companion: stop by running
#   pssh -h <hosts> -i "pkill -f 'vllm.entrypoints.openai.api_server.*--port ${PORT}'"
# or per-host: bash serve_vllm_stop.sh <port>  (see docs).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HOSTS=""
CHECKPOINT=""
SERVED_NAME=""
PORT=8000
TP=1
PP=1
DP=1
GPU_IDS=""
MAX_MODEL_LEN=""
GPU_MEM_UTIL=""
TOOL_PARSER=""
CHAT_TEMPLATE=""
DIST_BACKEND=""
EXTRA=""
WAIT_READY=1800
OUTDIR=""
REPO_ROOT="/jizhicfs/lhy/psrl_agent"
ENV_SCRIPT="/jizhicfs/lhy/env/psrl.sh"
SSH_USER=""
DRY_RUN=0

usage() { sed -n '2,78p' "$0"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hosts)                         HOSTS="$2"; shift 2 ;;
        --checkpoint)                    CHECKPOINT="$2"; shift 2 ;;
        --served-model-name)             SERVED_NAME="$2"; shift 2 ;;
        --port)                          PORT="$2"; shift 2 ;;
        --tp)                            TP="$2"; shift 2 ;;
        --pp)                            PP="$2"; shift 2 ;;
        --dp)                            DP="$2"; shift 2 ;;
        --gpu-ids)                       GPU_IDS="$2"; shift 2 ;;
        --max-model-len)                 MAX_MODEL_LEN="$2"; shift 2 ;;
        --gpu-memory-utilization)        GPU_MEM_UTIL="$2"; shift 2 ;;
        --tool-call-parser)              TOOL_PARSER="$2"; shift 2 ;;
        --chat-template)                 CHAT_TEMPLATE="$2"; shift 2 ;;
        --distributed-executor-backend)  DIST_BACKEND="$2"; shift 2 ;;
        --extra)                         EXTRA="$2"; shift 2 ;;
        --wait-ready)                    WAIT_READY="$2"; shift 2 ;;
        --outdir)                        OUTDIR="$2"; shift 2 ;;
        --repo-root)                     REPO_ROOT="$2"; shift 2 ;;
        --env-script)                    ENV_SCRIPT="$2"; shift 2 ;;
        --ssh-user)                      SSH_USER="$2"; shift 2 ;;
        --dry-run)                       DRY_RUN=1; shift ;;
        -h|--help)                       usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$HOSTS" && -f "$HOSTS" ]]  || { echo "ERROR: --hosts FILE is required." >&2; exit 2; }
[[ -n "$CHECKPOINT" ]]            || { echo "ERROR: --checkpoint is required." >&2; exit 2; }
[[ -d "$CHECKPOINT" ]]            || { echo "ERROR: --checkpoint $CHECKPOINT is not a directory." >&2; exit 2; }
[[ -z "$SERVED_NAME" ]] && SERVED_NAME="$(basename "${CHECKPOINT%/}")"

if [[ -z "$OUTDIR" ]]; then
    OUTDIR="$REPO_ROOT/examples/mini_swe/output/serve/$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUTDIR/host_logs"
ENDPOINTS_FILE="$OUTDIR/endpoints.txt"
LITELLM_CFG="$OUTDIR/litellm_proxy.yaml"
: > "$ENDPOINTS_FILE"

# Clean host list (strip comments / blanks).
mapfile -t HOST_LIST < <(grep -Ev '^[[:space:]]*(#|$)' "$HOSTS")
NUM_HOSTS=${#HOST_LIST[@]}
[[ "$NUM_HOSTS" -gt 0 ]] || { echo "ERROR: no hosts in $HOSTS." >&2; exit 2; }

# Assemble the serve_vllm.sh invocation that every host will run.
SERVE_SCRIPT="$REPO_ROOT/examples/mini_swe/eval/serve_vllm.sh"
[[ -f "$SERVE_SCRIPT" ]] || { echo "ERROR: $SERVE_SCRIPT missing (expected on shared FS)." >&2; exit 2; }

REMOTE_ARGS=(
    --checkpoint "$CHECKPOINT"
    --served-model-name "$SERVED_NAME"
    --host 0.0.0.0
    --port "$PORT"
    --tp "$TP"
    --pp "$PP"
    --dp "$DP"
    --tool-call-parser "$TOOL_PARSER"
    --wait-ready "$WAIT_READY"
    --env-script "$ENV_SCRIPT"
)
[[ -n "$GPU_IDS" ]]        && REMOTE_ARGS+=(--gpu-ids "$GPU_IDS")
[[ -n "$MAX_MODEL_LEN" ]]  && REMOTE_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
[[ -n "$GPU_MEM_UTIL" ]]   && REMOTE_ARGS+=(--gpu-memory-utilization "$GPU_MEM_UTIL")
[[ -n "$CHAT_TEMPLATE" ]]  && REMOTE_ARGS+=(--chat-template "$CHAT_TEMPLATE")
[[ -n "$DIST_BACKEND" ]]   && REMOTE_ARGS+=(--distributed-executor-backend "$DIST_BACKEND")
[[ -n "$EXTRA" ]]          && REMOTE_ARGS+=(--extra "$EXTRA")

# shellcheck disable=SC2124
REMOTE_CMD="bash $(printf '%q ' "$SERVE_SCRIPT" "${REMOTE_ARGS[@]}")"

echo "=== serve_vllm_multinode ==="
echo "  hosts        : $HOSTS ($NUM_HOSTS hosts)"
echo "  checkpoint   : $CHECKPOINT"
echo "  served-name  : $SERVED_NAME"
echo "  port         : $PORT"
echo "  TP / PP / DP : $TP / $PP / $DP"
echo "  tool parser  : ${TOOL_PARSER:-<disabled>}"
echo "  env script   : ${ENV_SCRIPT:-<disabled>}"
echo "  outdir       : $OUTDIR"
echo "  per-host cmd :"
echo "    $REMOTE_CMD"
echo

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "(dry-run)"
    exit 0
fi

SSH_OPTS=(
    -o StrictHostKeyChecking=no
    -o UserKnownHostsFile=/dev/null
    -o LogLevel=ERROR
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=240
    -o BatchMode=yes
)
[[ -n "$SSH_USER" ]] && SSH_OPTS+=(-l "$SSH_USER")

# Launch every host in parallel: ssh runs directly in the background so its
# PID maps 1:1 to a host. We then `wait $pid || rc=$?` per PID to capture
# each host's exit code. The previous layout used a `( ssh ...; echo $? > .rc ) &`
# subshell pattern, which had a race: under `set -e` the subshell could exit
# before its `echo $?` line ran, leaving the summary loop reading a missing
# or stale .rc file.
PIDS=()
declare -A HOSTS_BY_PID
for host in "${HOST_LIST[@]}"; do
    log="$OUTDIR/host_logs/${host//[:\/]/_}.launch.log"
    ssh "${SSH_OPTS[@]}" "$host" bash -lc "$(printf '%q' "$REMOTE_CMD")" \
        >"$log" 2>&1 &
    pid=$!
    PIDS+=("$pid")
    HOSTS_BY_PID[$pid]=$host
done

# Wait per-PID so each `wait` can be `||`-tested under `set -e`. The
# trailing `|| rc=$?` keeps `set -e` happy and lets us record non-zero codes.
declare -A RC_BY_HOST
for pid in "${PIDS[@]}"; do
    rc=0
    wait "$pid" || rc=$?
    host="${HOSTS_BY_PID[$pid]}"
    RC_BY_HOST[$host]=$rc
done

# Summary.
echo "--- per-host launch summary ---"
n_ok=0
n_fail=0
for host in "${HOST_LIST[@]}"; do
    log="$OUTDIR/host_logs/${host//[:\/]/_}.launch.log"
    rc="${RC_BY_HOST[$host]:-?}"
    if [[ "$rc" == "0" ]]; then
        echo "http://$host:$PORT" >> "$ENDPOINTS_FILE"
        printf '  %-22s OK    -> http://%s:%d\n' "$host" "$host" "$PORT"
        n_ok=$((n_ok + 1))
    else
        printf '  %-22s FAIL  (rc=%s)  see %s\n' "$host" "$rc" "$log"
        n_fail=$((n_fail + 1))
    fi
done

# Write a litellm proxy config that load-balances across every endpoint.
# Usage: `litellm --config <LITELLM_CFG> --port 4000` then point clients at
# http://localhost:4000/v1 with model=<SERVED_NAME>.
{
    echo "# Generated by serve_vllm_multinode.sh @ $(date -Iseconds)"
    echo "model_list:"
    while IFS= read -r ep; do
        [[ -z "$ep" ]] && continue
        echo "  - model_name: ${SERVED_NAME}"
        echo "    litellm_params:"
        echo "      model: openai/${SERVED_NAME}"
        echo "      api_base: ${ep}/v1"
        echo "      api_key: dummy"
    done < "$ENDPOINTS_FILE"
    echo "router_settings:"
    echo "  routing_strategy: least-busy"
} > "$LITELLM_CFG"

echo
echo "=== done ==="
echo "  healthy      : $n_ok / $NUM_HOSTS"
echo "  failed       : $n_fail"
echo "  endpoints    : $ENDPOINTS_FILE"
echo "  litellm cfg  : $LITELLM_CFG"
echo
if [[ "$n_ok" -gt 0 ]]; then
    echo "To load-balance across every replica, start litellm proxy:"
    echo "  litellm --config $LITELLM_CFG --port 4000"
    echo "then in eval_swebench:"
    echo "  export OPENAI_API_BASE=http://localhost:4000/v1"
    echo "  export OPENAI_API_KEY=dummy"
    echo "  python -m examples.mini_swe.eval.eval_swebench --model $SERVED_NAME ..."
fi

[[ "$n_fail" -eq 0 ]] || exit 1
exit 0
