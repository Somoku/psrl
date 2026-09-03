#!/usr/bin/env bash
# warm_status.sh — report how warm each node's task-image cache is, and skip the ones
# that are already done.
#
# Re-running a warm pass on an already-warm node is safe but not free: buildkit still
# resolves every layer, so it costs ~16 s per task (~5 min for 144) versus ~170 s per
# task cold. This tells you which nodes actually need the work.
#
# There is no single "is it warm" flag to read. The signal is behavioural: build ONE
# task and time it. Under ~40 s means the expensive layers (apt, git clone, make) were
# reused; minutes means they were rebuilt. That probe is itself a partial warm, so it is
# never wasted.
#
# Usage:
#   bash warm_status.sh --hosts /tmp/newhosts
#   bash warm_status.sh --hosts-list 28.49.55.40,28.49.36.157
#   bash warm_status.sh --hosts /tmp/newhosts --warm      # probe, then warm the cold ones
#
# Options:
#   --hosts FILE       Hosts file, one address per line; '#' and blanks ignored.
#   --hosts-list LIST  Comma-separated addresses instead of a file.
#   --warm             After probing, launch a full warm pass on every cold node.
#   --threshold S      Seconds below which a node counts as warm (default: 40).
#   --dataset PATH     v2 parquet (default: examples/sciaccel_rl/data/v2/all.parquet).
#   --concurrency N    Concurrency for the full warm pass (default: 8). Lowered from 12
#                      to reduce transient registry failures on cold nodes. The causal
#                      link is NOT proven: two nodes running the identical 12-way pass
#                      lost 20 tasks and 0 tasks. 8 trades ~50% more wall time for fewer
#                      retries. Raise it if warm time matters more.
#   -h | --help        Print this help.

set -euo pipefail

PSRL_PATH=${PSRL_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}

HOSTS_FILE=""
HOSTS_LIST=""
DO_WARM=0
THRESHOLD=40
DATASET="${PSRL_PATH}/examples/sciaccel_rl/data/v2/all.parquet"
CONCURRENCY=8

usage() { sed -n '2,30p' "$0"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hosts)        HOSTS_FILE="$2"; shift 2 ;;
        --hosts-list)   HOSTS_LIST="$2"; shift 2 ;;
        --warm)         DO_WARM=1; shift ;;
        --threshold)    THRESHOLD="$2"; shift 2 ;;
        --dataset)      DATASET="$2"; shift 2 ;;
        --concurrency)  CONCURRENCY="$2"; shift 2 ;;
        -h|--help)      usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -n "${HOSTS_FILE}" ]]; then
    [[ -f "${HOSTS_FILE}" ]] || { echo "ERROR: hosts file not found: ${HOSTS_FILE}" >&2; exit 2; }
    mapfile -t HOSTS < <(grep -Ev '^[[:space:]]*(#|$)' "${HOSTS_FILE}")
elif [[ -n "${HOSTS_LIST}" ]]; then
    IFS=',' read -r -a HOSTS <<< "${HOSTS_LIST}"
else
    echo "ERROR: pass --hosts FILE or --hosts-list LIST." >&2
    usage >&2
    exit 2
fi
[[ ${#HOSTS[@]} -gt 0 ]] || { echo "ERROR: no hosts." >&2; exit 2; }

# One representative task. Its Dockerfile shares every expensive layer with the other
# 144, so its build time is a faithful proxy for the whole bank. A cheap 2D repair task
# is used so the probe itself is quick. Verified present in data/v2/all.parquet -- a name
# that matches nothing makes run_eval exit with "No tasks matched the filters".
PROBE_TASK='sciaccel/laps-repair-bounds-2d-mhdrhs-l264'

echo "=== warm_status ==="
echo "  hosts     : ${#HOSTS[@]}"
echo "  probe task: ${PROBE_TASK}"
echo "  threshold : ${THRESHOLD}s (under this = warm)"
echo

PROBE_ROOT="${PSRL_PATH}/outputs/sciaccel_rl/eval/warm_probe"
mkdir -p "${PROBE_ROOT}"
declare -a COLD=()
declare -a WARM=()

for host in "${HOSTS[@]}"; do
    out="${PROBE_ROOT}/${host//./_}_$(date +%H%M%S)"
    cmd="cd ${PSRL_PATH} && bash examples/sciaccel_rl/eval/run_eval.sh --agent nop \
--dataset ${DATASET} --task-glob $(printf '%q' "${PROBE_TASK}") --output-dir ${out} \
--skip-gpu-tasks --max-per-instance 1 -n 1"

    printf '  %-18s probing... ' "${host}"
    if ! ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
            "${host}" bash -lc "$(printf '%q' "${cmd}")" >"${out}.log" 2>&1; then
        echo "PROBE FAILED (see ${out}.log)"
        COLD+=("${host}")
        continue
    fi

    secs="$(python3 -c "
import json, sys
try:
    rows = [json.loads(l) for l in open(sys.argv[1])]
except OSError:
    print(-1); raise SystemExit
vals = [r.get('env_setup_seconds') for r in rows if r.get('env_setup_seconds')]
print(int(vals[0]) if vals else -1)
" "${out}/results.jsonl" 2>/dev/null || echo -1)"

    if [[ "${secs}" -lt 0 ]]; then
        echo "no timing recorded -> treating as cold"
        COLD+=("${host}")
    elif [[ "${secs}" -le "${THRESHOLD}" ]]; then
        echo "env_setup ${secs}s -> WARM"
        WARM+=("${host}")
    else
        echo "env_setup ${secs}s -> COLD"
        COLD+=("${host}")
    fi
done

echo
echo "  warm: ${#WARM[@]} ${WARM[*]:-}"
echo "  cold: ${#COLD[@]} ${COLD[*]:-}"

if [[ "${DO_WARM}" -eq 0 ]]; then
    echo
    echo "Re-run with --warm to warm the cold nodes, or warm them by hand:"
    for host in "${COLD[@]:-}"; do
        [[ -n "${host}" ]] && echo "  ssh ${host} 'cd ${PSRL_PATH} && bash examples/sciaccel_rl/eval/run_eval.sh --agent nop --dataset ${DATASET} --output-dir <out> --skip-gpu-tasks --max-per-instance ${CONCURRENCY} -n ${CONCURRENCY}'"
    done
    exit 0
fi

if [[ ${#COLD[@]} -eq 0 ]]; then
    echo
    echo "Every node is already warm; nothing to do."
    exit 0
fi

echo
echo "=== warming ${#COLD[@]} cold node(s) ==="
for host in "${COLD[@]}"; do
    # Refuse to start a second pass on a node that is already running one. Two passes
    # writing the same output directory is not merely wasteful: the second truncates the
    # first's results.jsonl, so completed work is reported as never having happened.
    # Observed here -- 131 finished trials were erased this way.
    if ssh -o BatchMode=yes -o ConnectTimeout=15 "${host}" \
           'pgrep -f "run_eval.sh --agent nop" >/dev/null 2>&1'; then
        echo "  SKIP ${host}: a nop pass is already running there"
        continue
    fi
    # Timestamped, so re-running never overwrites an earlier pass's results.
    out="${PSRL_PATH}/outputs/sciaccel_rl/eval/nop_warm_${host//./_}_$(date +%m%d_%H%M%S)"
    cmd="cd ${PSRL_PATH} && nohup setsid bash examples/sciaccel_rl/eval/run_eval.sh \
--agent nop --dataset ${DATASET} --output-dir ${out} --skip-gpu-tasks \
--max-per-instance ${CONCURRENCY} -n ${CONCURRENCY} > /tmp/nop_warm.log 2>&1 < /dev/null &"
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${host}" \
        bash -lc "$(printf '%q' "${cmd}")" >/dev/null 2>&1 || true
    echo "  launched on ${host} -> ${out}"
    # Stagger: launching several at once has raced and silently dropped a host.
    sleep 3
done

echo
echo "Watch progress with (output dirs are timestamped, so glob them):"
echo "  for D in ${PSRL_PATH}/outputs/sciaccel_rl/eval/nop_warm_*/results.jsonl; do"
echo "    echo \"\$(dirname \$D | xargs basename): \$(wc -l < \$D)/144\""
echo "  done"
echo
echo "Then repair the transient registry failures and check the anchor:"
echo "  bash examples/sciaccel_rl/prepare/warm_repair.sh --results <out>/results.jsonl --host <host>"
