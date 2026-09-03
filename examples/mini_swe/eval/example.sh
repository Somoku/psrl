#!/usr/bin/env bash
# example.sh — end-to-end eval of a PSRL checkpoint on the 80-problem
# Verified subset across 4 hosts.
#
# Usage (run from the repo root):
#   bash examples/mini_swe/eval/example.sh
#
# What this script does:
#   1. Start one vLLM replica per host (TP=4; text-completion mode,
#      no tool-call parser).
#   2. Set OPENAI_API_BASE so each eval worker calls its local vLLM directly
#      (no central proxy; avoids litellm CLI startup hang in corp proxy envs).
#   3. Run multi-node eval (one shard per host, 8 concurrent workers each).
#   4. Print the merged summary.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

MODEL="${PSRL_WORKSPACE}/models/SWE-agent-LM-7B"
SERVED_MODEL_NAME="SWE-agent-LM-7B"
HOSTS_FILE="${PSRL_WORKSPACE}/hosts/32GPUs_another"
DATASET="examples/mini_swe/data/verified_subset_80/train.parquet"
OUTPUT_DIR="$SCRIPT_DIR/output"

SERVE_PORT=8000
TP=2
DP=4
WORKERS_PER_NODE=8
GRADER_TIMEOUT=1800
SSH_TIMEOUT=7200

# Shared-FS env script (conda + NCCL / UCX / vLLM / library paths).
ENV_SCRIPT="${PSRL_WORKSPACE}/env/psrl.sh"

# Source locally; temporarily disable -u because the env script references
# some vars (no_proxy, LD_LIBRARY_PATH) that may be unset in a fresh shell.
set +u
# shellcheck disable=SC1090
source "$ENV_SCRIPT"
set -u

SERVE_OUTDIR="$OUTPUT_DIR/serve"
EVAL_OUTDIR="$OUTPUT_DIR/eval"

mkdir -p "$SERVE_OUTDIR" "$EVAL_OUTDIR"

# ---------------------------------------------------------------------------
# Step 1 — fan vLLM to every host (text-completion, no tool-call parser)
# ---------------------------------------------------------------------------
echo "=== Step 1: starting vLLM on all hosts ==="
# One endpoint per host, with DP replicas behind it: Step 2 gives every eval shard
# a single OPENAI_API_BASE, so topology.replicas must stay 1 and the parallelism
# goes into topology.dp. (A fleet of N ports would need N URLs, which
# eval_swebench_multinode cannot consume.)
#
# The launcher exits non-zero only when healthy endpoints drop below
# topology.min_healthy_frac; a partial set is still usable, so the gate below reads
# endpoints.json rather than trusting the exit code.
set +e
python -m psrl.eval.serve \
    topology=multinode \
    topology.hosts_file="$HOSTS_FILE" \
    topology.replicas=1 \
    topology.tp="$TP" \
    topology.dp="$DP" \
    topology.base_port="$SERVE_PORT" \
    topology.wait_ready_sec=1800 \
    server.checkpoint="$MODEL" \
    server.served_model_name="$SERVED_MODEL_NAME" \
    server.max_model_len=32768 \
    env_script="$ENV_SCRIPT" \
    output_dir="$SERVE_OUTDIR"
SERVE_RC=$?
set -e
# server.tool_call_parser is empty in every preset — mini-swe-agent parses its own
# bash blocks, so no override is needed. First-time torch.compile on Qwen3-30B can
# take ~20min per host; wait_ready_sec=1800 gives a 30min ceiling.

ENDPOINTS_JSON="$SERVE_OUTDIR/endpoints.json"
if [[ ! -s "$ENDPOINTS_JSON" ]]; then
    echo "ERROR: no healthy endpoints after serve step (rc=$SERVE_RC). Aborting." >&2
    echo "Check per-host logs under $SERVE_OUTDIR/hosts/ and remote /tmp/vllm_${SERVE_PORT}.log." >&2
    exit 1
fi

N_HEALTHY=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['n_endpoints'])" "$ENDPOINTS_JSON")
N_HOSTS=$(grep -cEv '^[[:space:]]*(#|$)' "$HOSTS_FILE")
echo
echo "Healthy endpoints: $N_HEALTHY / $N_HOSTS"
python -c "
import json, sys
for e in json.load(open(sys.argv[1]))['endpoints']:
    print(f\"  {e['url']}\")
" "$ENDPOINTS_JSON"
if [[ "$N_HEALTHY" -lt "$N_HOSTS" ]]; then
    echo
    echo "NOTE: continuing with $N_HEALTHY healthy host(s); the launcher logged which"
    echo "hosts contributed nothing. (Tail the vLLM log on a failed host; if it is still"
    echo " in 'compilation.py' it only needs more time — bump topology.wait_ready_sec.)"
fi
echo

# ---------------------------------------------------------------------------
# Step 2 — set up each eval shard to call its local vLLM directly
# ---------------------------------------------------------------------------
# We bypass the litellm proxy entirely: each eval host runs its own vLLM on
# SERVE_PORT, so we set OPENAI_API_BASE=http://localhost:SERVE_PORT/v1 and
# pass it to every remote host via --set-env.  eval_swebench_multinode.py
# will forward this as an env var to each shard's eval process, and because
# every host has a local vLLM, requests stay on the same machine — no proxy,
# no cross-host routing.
#
# We unset http_proxy/https_proxy on each remote host for the LLM call scope
# via NO_PROXY containing localhost/127.0.0.1, which covers the direct
# http://localhost:PORT request.
echo "=== Step 2: configuring per-host vLLM endpoints ==="
export OPENAI_API_BASE="http://localhost:${SERVE_PORT}/v1"
export OPENAI_API_KEY="dummy"
export NO_PROXY="localhost,127.0.0.1,${NO_PROXY:-}"
export no_proxy="localhost,127.0.0.1,${no_proxy:-}"
echo "Each eval shard will call its own local vLLM at $OPENAI_API_BASE"
echo "NO_PROXY includes localhost so http_proxy is bypassed for LLM calls"

# ---------------------------------------------------------------------------
# Step 3 — multi-node eval
# ---------------------------------------------------------------------------
echo "=== Step 3: running multi-node eval ==="
cd "$REPO_ROOT"
python -m examples.mini_swe.eval.eval_swebench_multinode \
    --hosts "$HOSTS_FILE" \
    --dataset "$DATASET" \
    --output-dir "$EVAL_OUTDIR" \
    --model "$SERVED_MODEL_NAME" \
    --model-class "examples.mini_swe.eval.xml_fc_model.XmlFcModel" \
    --config "examples/mini_swe/config/swebench_agent_config_full_sweagent.yaml" \
    --workers-per-node "$WORKERS_PER_NODE" \
    --grader-timeout "$GRADER_TIMEOUT" \
    --ssh-timeout "$SSH_TIMEOUT" \
    --repo-root "$REPO_ROOT" \
    --env-script "$ENV_SCRIPT"

# ---------------------------------------------------------------------------
# Step 4 — summary + cleanup
# ---------------------------------------------------------------------------
echo
echo "=== Results ==="
EVAL_OUTDIR="$EVAL_OUTDIR" python - <<'PY'
import json, pathlib, os
summary_path = pathlib.Path(os.environ["EVAL_OUTDIR"]) / "summary.json"
if summary_path.exists():
    s = json.loads(summary_path.read_text())
    print(f"Resolved  : {s['resolved']}/{s['total']} ({s['resolve_rate']:.1%})")
    print(f"Avg turns : {s['avg_turns']:.1f}")
    print(f"Wall clock: {s.get('elapsed_s', '?')}s")
    print(f"Output    : {summary_path.parent}")
else:
    print(f"WARNING: {summary_path} not found.")
PY

echo
echo "To stop vLLM replicas:"
echo "  pssh -h $HOSTS_FILE -i \"pkill -f 'vllm.entrypoints.openai.api_server.*--port $SERVE_PORT'\""
