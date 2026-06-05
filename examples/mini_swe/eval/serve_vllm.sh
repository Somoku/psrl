#!/usr/bin/env bash
# serve_vllm.sh — start a single vLLM OpenAI-compatible server for a local
# checkpoint, configured the way mini-swe-agent / eval_swebench expects
# (chat completions + bash tool calls).
#
# Usage:
#   bash serve_vllm.sh --checkpoint /jizhicfs/lhy/checkpoints/my-step-1000 \
#                      --tp 2 --port 8000 --tool-call-parser hermes
#
# Options (all long-form):
#   --checkpoint PATH           Absolute path to an HF-compatible checkpoint
#                               directory. (required)
#   --served-model-name NAME    Name exposed via /v1 endpoint. Clients use
#                               this as the `model` field. Default: basename
#                               of --checkpoint.
#   --host HOST                 Bind host (default: 0.0.0.0).
#   --port N                    Bind port (default: 8000).
#   --tp N                      --tensor-parallel-size (default: 1).
#   --pp N                      --pipeline-parallel-size (default: 1).
#   --dp N                      --data-parallel-size (default: 1). vLLM 0.6+.
#                               Runs N replicas behind one server. Needs
#                               TP*PP*DP GPUs visible.
#   --tool-call-parser NAME     Chat-template tool parser. Default: '' (disabled).
#                               PSRL trains with mswea_bash_command text blocks,
#                               not OpenAI tool-call JSON, so plain text-completion
#                               mode is correct for PSRL checkpoints.
#                               Only set this for external models that natively
#                               output tool-call JSON:
#                               Qwen2/Qwen3/Hermes:  hermes
#                               Llama3/Llama4:       llama3_json
#                               Mistral:             mistral
#                               DeepSeek v3:         deepseek_v3
#   --chat-template PATH        Path to a custom jinja chat template.
#                               Only needed if the checkpoint's tokenizer
#                               config doesn't already have one.
#   --gpu-ids LIST              Comma-separated GPU IDs set as
#                               CUDA_VISIBLE_DEVICES before launching vLLM.
#                               Default: inherit existing value.
#   --max-model-len N           --max-model-len passed to vLLM.
#   --gpu-memory-utilization F  --gpu-memory-utilization (default: 0.9).
#   --distributed-executor-backend BACKEND
#                               vLLM --distributed-executor-backend. Leave
#                               empty to let vLLM auto-select. Set to 'ray'
#                               for cross-node TP (requires a pre-started Ray
#                               cluster; this script does NOT start one).
#   --extra 'ARGS'              Catchall: additional CLI args forwarded
#                               verbatim to vLLM. Example:
#                                 --extra '--trust-remote-code --quantization awq'
#   --log-file PATH             Where to write stdout/stderr
#                               (default: /tmp/vllm_<port>.log).
#   --pid-file PATH             Where to write the server PID
#                               (default: /tmp/vllm_<port>.pid).
#   --env-script PATH           Env script to source before launching vLLM.
#                               Must contain `conda activate` plus all the
#                               NCCL / UCX / vLLM / cudnn / torch
#                               LD_LIBRARY_PATH knobs. The default
#                               (/jizhicfs/lhy/env/psrl.sh) is the same file
#                               training uses; pass '' to disable sourcing.
#   --foreground                Don't background; stream logs to this shell.
#                               Useful for debugging; incompatible with the
#                               multinode fan-out launcher.
#   --wait-ready SECONDS        After launching, poll /v1/models until the
#                               server responds or SECONDS elapse, then exit.
#                               0 disables (default: 1800 = 30min, enough for
#                               first-time torch.compile on MoE models like
#                               Qwen3-30B; dense models usually ready in <5min).
#   --dry-run                   Print the command that would run and exit.
#   -h | --help                 Print this help.
#
# Exit code:
#   0  server launched (and, when --wait-ready > 0, passed the health check).
#   2  argument error.
#   3  health check timed out — the server process may still be alive; inspect
#      --log-file for the reason.

set -euo pipefail

CHECKPOINT=""
SERVED_NAME=""
HOST="0.0.0.0"
PORT=8000
TP=1
PP=1
DP=1
TOOL_PARSER=""
CHAT_TEMPLATE=""
GPU_IDS=""
MAX_MODEL_LEN=""
GPU_MEM_UTIL="0.9"
DIST_BACKEND=""
EXTRA=""
LOG_FILE=""
PID_FILE=""
ENV_SCRIPT="/jizhicfs/lhy/env/psrl.sh"
FOREGROUND=0
WAIT_READY=1800
DRY_RUN=0

usage() { sed -n '2,68p' "$0"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)                    CHECKPOINT="$2"; shift 2 ;;
        --served-model-name)             SERVED_NAME="$2"; shift 2 ;;
        --host)                          HOST="$2"; shift 2 ;;
        --port)                          PORT="$2"; shift 2 ;;
        --tp)                            TP="$2"; shift 2 ;;
        --pp)                            PP="$2"; shift 2 ;;
        --dp)                            DP="$2"; shift 2 ;;
        --tool-call-parser)              TOOL_PARSER="$2"; shift 2 ;;
        --chat-template)                 CHAT_TEMPLATE="$2"; shift 2 ;;
        --gpu-ids)                       GPU_IDS="$2"; shift 2 ;;
        --max-model-len)                 MAX_MODEL_LEN="$2"; shift 2 ;;
        --gpu-memory-utilization)        GPU_MEM_UTIL="$2"; shift 2 ;;
        --distributed-executor-backend)  DIST_BACKEND="$2"; shift 2 ;;
        --extra)                         EXTRA="$2"; shift 2 ;;
        --log-file)                      LOG_FILE="$2"; shift 2 ;;
        --pid-file)                      PID_FILE="$2"; shift 2 ;;
        --env-script)                    ENV_SCRIPT="$2"; shift 2 ;;
        --foreground)                    FOREGROUND=1; shift ;;
        --wait-ready)                    WAIT_READY="$2"; shift 2 ;;
        --dry-run)                       DRY_RUN=1; shift ;;
        -h|--help)                       usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$CHECKPOINT" ]]            || { echo "ERROR: --checkpoint is required." >&2; exit 2; }
[[ -d "$CHECKPOINT" || -f "$CHECKPOINT/config.json" ]] \
    || { echo "ERROR: --checkpoint $CHECKPOINT not a readable HF directory." >&2; exit 2; }

[[ -z "$SERVED_NAME" ]] && SERVED_NAME="$(basename "${CHECKPOINT%/}")"
[[ -z "$LOG_FILE" ]] && LOG_FILE="/tmp/vllm_${PORT}.log"
[[ -z "$PID_FILE" ]] && PID_FILE="/tmp/vllm_${PORT}.pid"

# Source the env script (handles conda activation + NCCL / UCX / vLLM /
# library-path setup). Skipped only when explicitly disabled with --env-script ''.
#
# psrl.sh chains into /jizhicfs/lhy/activate which references $no_proxy /
# $LD_LIBRARY_PATH unconditionally, so we temporarily disable `set -u` around
# the source. The rest of this script keeps nounset checking.
if [[ -n "$ENV_SCRIPT" ]]; then
    [[ -f "$ENV_SCRIPT" ]] || { echo "ERROR: --env-script $ENV_SCRIPT not found." >&2; exit 2; }
    set +u
    # shellcheck disable=SC1090
    source "$ENV_SCRIPT"
    set -u
fi

# Apply GPU pinning (overrides whatever the caller set).
if [[ -n "$GPU_IDS" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi


# Build vLLM command.
CMD=(
    python -m vllm.entrypoints.openai.api_server
    --model "$CHECKPOINT"
    --served-model-name "$SERVED_NAME"
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size "$TP"
    --pipeline-parallel-size "$PP"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
)
if [[ "$DP" -gt 1 ]]; then
    CMD+=(--data-parallel-size "$DP")
    # vLLM v1 forces DP synchronization to use gloo TCP instead of NCCL
    # when async_scheduling is on AND the model is MoE (vllm/config/vllm.py).
    # gloo TCP is fragile; disable async_scheduling for MoE+DP to keep NCCL.
    _IS_MOE=0
    if [[ -f "$CHECKPOINT/config.json" ]]; then
        python3 -c "
import json, sys
cfg = json.load(open('$CHECKPOINT/config.json'))
arch = cfg.get('architectures', [''])[0].lower()
moe = 'moe' in arch or cfg.get('num_experts') or cfg.get('num_local_experts')
sys.exit(0 if moe else 1)
" 2>/dev/null && _IS_MOE=1
    fi
    if [[ "$_IS_MOE" -eq 1 ]]; then
        CMD+=(--async-scheduling false)
    fi
fi
if [[ -n "$TOOL_PARSER" ]]; then
    CMD+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER")
fi
if [[ -n "$CHAT_TEMPLATE" ]]; then
    CMD+=(--chat-template "$CHAT_TEMPLATE")
fi
if [[ -n "$MAX_MODEL_LEN" ]]; then
    CMD+=(--max-model-len "$MAX_MODEL_LEN")
fi
if [[ -n "$DIST_BACKEND" ]]; then
    CMD+=(--distributed-executor-backend "$DIST_BACKEND")
fi
if [[ -n "$EXTRA" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=($EXTRA)
    CMD+=("${EXTRA_ARR[@]}")
fi

HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname)"

echo "=== serve_vllm ==="
echo "  host         : $HOSTNAME_SHORT"
echo "  checkpoint   : $CHECKPOINT"
echo "  served-name  : $SERVED_NAME"
echo "  bind         : http://$HOST:$PORT/v1"
echo "  TP / PP / DP : $TP / $PP / $DP"
echo "  tool parser  : ${TOOL_PARSER:-<disabled>}"
echo "  GPU IDs      : ${CUDA_VISIBLE_DEVICES:-<inherit>}"
echo "  env script   : ${ENV_SCRIPT:-<disabled>}"
echo "  log file     : $LOG_FILE"
echo "  pid file     : $PID_FILE"
echo "  cmd:"
printf '    %q ' "${CMD[@]}"; echo

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "(dry-run: not launching.)"
    exit 0
fi

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$(dirname "$PID_FILE")"

if [[ "$FOREGROUND" -eq 1 ]]; then
    echo "(launching in foreground — press Ctrl-C to stop.)"
    exec "${CMD[@]}"
fi

# Background launch; nohup + setsid so a parent ssh exit doesn't kill it.
nohup setsid "${CMD[@]}" >"$LOG_FILE" 2>&1 </dev/null &
PID=$!
echo "$PID" > "$PID_FILE"
echo "[$HOSTNAME_SHORT] vLLM started (pid=$PID), log=$LOG_FILE"

if [[ "$WAIT_READY" -le 0 ]]; then
    exit 0
fi

# Health check against a localhost-relative URL (0.0.0.0 binds on all ifs but
# curling 0.0.0.0 fails on some kernels).
PROBE_HOST="$HOST"
if [[ "$HOST" == "0.0.0.0" ]]; then PROBE_HOST="127.0.0.1"; fi
URL="http://${PROBE_HOST}:${PORT}/v1/models"

echo "[$HOSTNAME_SHORT] waiting up to ${WAIT_READY}s for $URL ..."
start_ts=$(date +%s)
while :; do
    # Kill path: if the vllm process died, bail early.
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "[$HOSTNAME_SHORT] ERROR: vLLM process $PID exited before becoming ready." >&2
        echo "--- last 40 lines of $LOG_FILE ---" >&2
        tail -n 40 "$LOG_FILE" >&2 || true
        exit 3
    fi
    if curl -fsS --max-time 5 "$URL" >/dev/null 2>&1; then
        echo "[$HOSTNAME_SHORT] READY: $URL (pid=$PID)"
        exit 0
    fi
    now=$(date +%s)
    if (( now - start_ts >= WAIT_READY )); then
        echo "[$HOSTNAME_SHORT] ERROR: health check timed out after ${WAIT_READY}s." >&2
        echo "--- last 40 lines of $LOG_FILE ---" >&2
        tail -n 40 "$LOG_FILE" >&2 || true
        exit 3
    fi
    sleep 3
done
