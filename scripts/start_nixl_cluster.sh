#!/bin/bash
# Bring up a Ray cluster for the multi-node NIXL tests.
#
#   scripts/start_nixl_cluster.sh <head-ip> <worker-ip>...
#
# Every node is started through scripts/nixl_env.sh, which sources the cluster's shared
# environment. That is the point of this script rather than running `ray start` by hand:
#
#   * Ray refuses to join a cluster whose head runs a different interpreter
#     ("Version mismatch: ... Python 3.12.13 / 3.12.2"). An interactive shell and a
#     non-interactive `ssh host cmd` resolve different condas, so starting the head
#     locally and the workers over ssh is enough to produce that skew.
#   * Ray and gRPC must not go through the HTTP proxy the cluster exports, or the GCS
#     connection times out. The proxy is unset here for the same reason the cluster's
#     own env file flags it.
#   * A fresh `ray stop --force` everywhere first: node IDs are regenerated on restart,
#     and actors pinned with NodeAffinitySchedulingStrategy to a previous node ID sit in
#     PENDING_CREATION forever rather than failing. A partial restart is worse than none.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_SH="${REPO}/scripts/nixl_env.sh"
RAY_BIN="${PSRL_RAY_BIN:-/apdcephfs_zwfy10/share_303541817/lhy/anaconda3/bin/ray}"
PORT="${PSRL_RAY_PORT:-6399}"
GPUS="${PSRL_RAY_GPUS:-8}"

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <head-ip> [worker-ip ...]" >&2
    exit 2
fi
HEAD_IP="$1"; shift
WORKERS=("$@")

echo "=== stopping any previous cluster ==="
for ip in "$HEAD_IP" "${WORKERS[@]}"; do
    timeout 90 ssh -o BatchMode=yes -o StrictHostKeyChecking=no -n "$ip" \
        "${RAY_BIN} stop --force >/dev/null 2>&1; true" 2>/dev/null \
        || "${RAY_BIN}" stop --force >/dev/null 2>&1
done
"${RAY_BIN}" stop --force > /dev/null 2>&1
sleep 5

echo "=== starting head on ${HEAD_IP} ==="
# shellcheck disable=SC1090
source "${ENV_SH}"
unset http_proxy https_proxy
if ! timeout 240 "${RAY_BIN}" start --head \
        --node-ip-address="${HEAD_IP}" --port="${PORT}" --num-gpus="${GPUS}" \
        2>&1 | grep -q "Ray runtime started"; then
    echo "ERROR: head failed to start" >&2
    exit 1
fi

for ip in "${WORKERS[@]}"; do
    echo "=== joining ${ip} ==="
    joined=$(timeout 260 ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$ip" \
        "source ${ENV_SH} >/dev/null 2>&1; unset http_proxy https_proxy; \
         ${VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR:+export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=${VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR};} \
         timeout 200 ${RAY_BIN} start --address='${HEAD_IP}:${PORT}' --num-gpus=${GPUS} 2>&1 \
         | grep -c 'Ray runtime started'" 2>/dev/null | tail -1)
    [ "${joined:-0}" = "1" ] || { echo "ERROR: ${ip} failed to join" >&2; exit 1; }
done

echo "=== verifying ==="
python - "$HEAD_IP" "$PORT" <<'PY'
import sys
import ray

ray.init(address=f"{sys.argv[1]}:{sys.argv[2]}", ignore_reinit_error=True, logging_level="ERROR")
alive = [n for n in ray.nodes() if n["Alive"]]
gpus = int(sum(n["Resources"].get("GPU", 0) for n in alive))
print(f"CLUSTER_READY nodes={len(alive)} gpus={gpus}")
PY
