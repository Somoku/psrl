#!/bin/bash
# N GPU pairs reading concurrently across two nodes (or one), for aggregate NIC ingress.
#
#   tests/nixl/scripts/run_pair_bw.sh <target-ip> <initiator-ip> <pairs> [bench args...]
#
# Target i serves from GPU i on port ${BASE_PORT:-41011}+i; initiator i reads into GPU i. Initiators
# meet at a shared-filesystem barrier after their handshake, so the timed loops overlap
# and the aggregate is total bytes over the union window (see bench_peer_whole_tensor.py).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TARGET=$1; INIT=$2; PAIRS=$3; shift 3
RUN="${REPO}/tests/nixl/log/pair_bw_$(date +%s)"
mkdir -p "$RUN"
ENV="source ${REPO}/scripts/nixl_env.sh >/dev/null 2>&1; unset http_proxy https_proxy; cd ${REPO}; export PYTHONPATH=${REPO}"
PY=python  # nixl_env.sh puts the cluster interpreter on PATH
for i in $(seq 0 $((PAIRS-1))); do
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no -n "$TARGET" \
    "$ENV; nohup timeout 2400 $PY tests/nixl/bench_peer_whole_tensor.py --role target --ip $TARGET --port $((${BASE_PORT:-41011}+i*200)) --port-file $RUN/port_$i --gpu $i --hold-s 1500 $* > $RUN/target_$i.log 2>&1 &"
  sleep 1  # concurrent starts were seen to fail with "Failed to bind metadata listener socket"
done
# An initiator that connects before its target listens gets "Connection refused" and
# then waits forever for a descriptor notification, so start them only once every
# target's agent is up.
for _ in $(seq 1 600); do
  up=$(grep -l "agent up" $RUN/target_*.log 2>/dev/null | wc -l)
  [ "$up" -ge "$PAIRS" ] && break
  if grep -q "Error" $RUN/target_*.log 2>/dev/null; then grep -h "Error" $RUN/target_*.log | head -3; exit 1; fi
  sleep 1
done
for i in $(seq 0 $((PAIRS-1))); do
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no -n "$INIT" \
    "$ENV; timeout 2400 $PY tests/nixl/bench_peer_whole_tensor.py --role initiator --ip $TARGET --port-file $RUN/port_$i --gpu $((i+${INIT_GPU_OFFSET:-0})) --barrier-dir $RUN/barrier --barrier-n $PAIRS $* > $RUN/init_$i.log 2>&1" &
done
wait
$PY - "$RUN" <<'PYEOF'
import glob, json, sys
rs = [json.loads(l.split("RESULT ", 1)[1]) for f in sorted(glob.glob(sys.argv[1] + "/init_*.log"))
      for l in open(f) if l.startswith("RESULT ")]
print(f"{len(rs)} initiators reported")
for mode in ("cells", "whole", "flat"):
    if not rs or mode not in rs[0]:
        continue
    ws = [r[mode]["window"] for r in rs]
    moved = sum(r[mode]["bytes_moved"] for r in rs)
    span = max(w[1] for w in ws) - min(w[0] for w in ws)
    per = [r[mode]["GB_per_s"] for r in rs]
    print(f"{mode:5s} desc/client={rs[0][mode]['descriptors']} per-pair best GB/s min={min(per)} max={max(per)} "
          f"aggregate={moved / span / 1e9:.1f} GB/s over {span:.2f}s")
PYEOF
echo "logs: $RUN"
