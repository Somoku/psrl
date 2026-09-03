#!/usr/bin/env bash
# start_distributed_training.sh — preflight the cluster, then launch the 3-node run.
#
# Everything expensive is already done as of 2026-08-31: all three nodes are Docker
# provisioned, image-warmed, and nop-anchor verified. This script only checks that the
# cluster is actually free, starts Ray, and launches training. It refuses rather than
# proceeds when a check fails, because every failure mode below produces a hang or a
# silently wrong run rather than a clean error.
#
# Usage:
#   bash examples/sciaccel_rl/prepare/start_distributed_training.sh
#   bash examples/sciaccel_rl/prepare/start_distributed_training.sh --check-only
#
# Options:
#   --hosts FILE     Hostfile; FIRST line becomes the Ray head. Default: hosts/24GPUs.
#   --check-only     Run the preflight checks and stop.
#   --force          Skip the free-GPU check. Only if you know the residents are yours.
#   -h | --help      Print this help.

set -euo pipefail

PSRL_PATH=${PSRL_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
HOSTS_FILE="${PSRL_WORKSPACE:-/apdcephfs_zwfy10/share_303541817/lhy}/hosts/24GPUs"
CHECK_ONLY=0
FORCE=0

usage() { sed -n '2,20p' "$0"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hosts)      HOSTS_FILE="$2"; shift 2 ;;
        --check-only) CHECK_ONLY=1; shift ;;
        --force)      FORCE=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -f "${HOSTS_FILE}" ]] || { echo "ERROR: hostfile not found: ${HOSTS_FILE}" >&2; exit 2; }
mapfile -t HOSTS < <(grep -Ev '^[[:space:]]*(#|$)' "${HOSTS_FILE}")
[[ ${#HOSTS[@]} -eq 3 ]] || {
    echo "ERROR: expected 3 hosts for the 8-gen + 16-train layout, got ${#HOSTS[@]}." >&2
    echo "For a 2-node run edit fsdp_qwen35_9b.sh: NNODES=2, TRAIN_NNODES=1, TRAIN_FSDP=8." >&2
    exit 2
}

echo "=== preflight: ${#HOSTS[@]} hosts from ${HOSTS_FILE} ==="
FAIL=0

# 1. GPUs must be free. The script declares train_pool = [8] * 2 and four TP=2 rollout
#    pools, committing all 24. If a node's GPUs are taken, Ray cannot place the bundles
#    and waits forever instead of erroring -- so this is checked first and hard.
echo "-- free GPUs"
for host in "${HOSTS[@]}"; do
    used="$(ssh -o BatchMode=yes -o ConnectTimeout=15 "${host}" \
        'nvidia-smi --query-gpu=memory.used --format=csv,noheader | awk "{s+=\$1} END {print s+0}"' 2>/dev/null || echo -1)"
    if [[ "${used}" -lt 0 ]]; then
        printf '   %-18s UNREACHABLE\n' "${host}"; FAIL=1
    elif [[ "${used}" -gt 2048 ]]; then
        printf '   %-18s BUSY (%s MiB in use)\n' "${host}" "${used}"
        [[ "${FORCE}" -eq 1 ]] || FAIL=1
    else
        printf '   %-18s free\n' "${host}"
    fi
done

# 2. No foreign Ray. ray_start.sh force-stops Ray on every host in the file, which would
#    kill another user's cluster without asking. This happened here: a 2d12h old
#    lgm-slime cluster was found on one node.
echo "-- no foreign Ray processes"
for host in "${HOSTS[@]}"; do
    n="$(ssh -o BatchMode=yes -o ConnectTimeout=15 "${host}" \
        'pgrep -af "gcs_server|raylet" 2>/dev/null | grep -v "bash -c" | wc -l' 2>/dev/null || echo -1)"
    if [[ "${n}" -gt 0 ]]; then
        printf '   %-18s %s Ray process(es) present -- CONFIRM THEY ARE YOURS\n' "${host}" "${n}"
        ssh -o BatchMode=yes "${host}" \
            'pgrep -af "gcs_server|raylet" 2>/dev/null | grep -v "bash -c" | sed "s|/lib/python3.*||" | head -3' 2>/dev/null | sed 's/^/       /'
        FAIL=1
    else
        printf '   %-18s clean\n' "${host}"
    fi
done

# 3. Image cache warm. Probed directly on the node rather than inferred from local
#    result files: a pass launched on the head node writes a directory with no IP in its
#    name, so filename matching reported an already-warm node as cold. Judged by build
#    time, never by cache directory size -- buildkit's GC shrank that dir from 18 GB to
#    3.1 GB here while builds stayed at 16 s.
echo "-- image cache (base images present = the expensive layers are local)"
for host in "${HOSTS[@]}"; do
    n="$(ssh -o BatchMode=yes -o ConnectTimeout=15 "${host}" \
        'c=0; for i in python:3.13-slim debian:bookworm-slim alpine:3.19; do docker image inspect $i >/dev/null 2>&1 && c=$((c+1)); done; echo $c' 2>/dev/null || echo -1)"
    layers="$(ssh -o BatchMode=yes -o ConnectTimeout=15 "${host}" \
        'docker images -q 2>/dev/null | wc -l' 2>/dev/null || echo 0)"
    if [[ "${n}" -lt 0 ]]; then
        printf '   %-18s UNREACHABLE\n' "${host}"; FAIL=1
    elif [[ "${n}" -lt 3 ]]; then
        printf '   %-18s only %s/3 base images -- run provision_docker_nodes.sh\n' "${host}" "${n}"; FAIL=1
    elif [[ "${layers}" -lt 50 ]]; then
        printf '   %-18s base images ok but only %s images cached -- run a nop warm pass\n' "${host}" "${layers}"; FAIL=1
    else
        printf '   %-18s warm (3/3 base images, %s images cached)\n' "${host}" "${layers}"
    fi
done

echo
if [[ "${FAIL}" -ne 0 ]]; then
    echo "PREFLIGHT FAILED -- not starting. Fix the items above, or pass --force for the GPU check only."
    exit 1
fi
echo "PREFLIGHT OK"

[[ "${CHECK_ONLY}" -eq 0 ]] || exit 0

echo
echo "=== starting Ray (head = ${HOSTS[0]}) ==="
cd "${PSRL_PATH}"
bash examples/ray/ray_start.sh "${HOSTS_FILE}"

# Ray reports resources asynchronously; a node that has joined but not yet registered
# its GPUs makes the trainer see fewer than 24 and mis-place bundles.
echo
echo "=== waiting for 24 GPUs to register ==="
for _ in $(seq 1 30); do
    sleep 10
    n="$(python3 -c "
import ray
ray.init(address='auto', log_to_driver=False)
print(int(ray.cluster_resources().get('GPU', 0)))
" 2>/dev/null || echo 0)"
    echo "   GPUs visible to Ray: ${n}/24"
    [[ "${n}" -ge 24 ]] && break
done
[[ "${n}" -ge 24 ]] || { echo "ERROR: only ${n}/24 GPUs registered. Not launching." >&2; exit 1; }

LOG_DIR="${PSRL_PATH}/examples/sciaccel_rl/psrl_logs/GRPO-sciaccel-v2-Qwen35-9B"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/train_$(date +%m%d_%H%M%S).log"
echo "${LOG}" > /tmp/train_log

echo
echo "=== launching training ==="
echo "  log: ${LOG}"
nohup setsid bash examples/sciaccel_rl/fsdp_qwen35_9b.sh > "${LOG}" 2>&1 < /dev/null &
echo "  pid: $!"
echo
echo "Watch with:  tail -f ${LOG}"
echo "Baseline to beat (Qwen3.5-9B, same settings): 7/144 solved, mean reward_repair 0.0497"
