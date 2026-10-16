#!/bin/bash
# Run N bench_cross_node_bw initiators concurrently, aligned with --start-at.
# Run on the initiator node:
#   ssh <init-node> bash /path/to/run_bw_initiators.sh <target-ip> <pairs> <start-at> [port-base]
#
# Outputs one RESULT line per pair, then an AGGREGATE line.
set -uo pipefail

TARGET_IP=$1
PAIRS=${2:-8}
START_AT=${3}
PORT_BASE=${4:-32600}

REPO=/apdcephfs_zwfy10/share_303541817/ls/psrl-wip/psrl
CONDA=/apdcephfs_zwfy10/share_303541817/lhy/anaconda3
ENV_NAME=base
PY=${CONDA}/bin/python

source ${CONDA}/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ${ENV_NAME} 2>/dev/null || true
source $REPO/scripts/nixl_env.sh 2>/dev/null || true
unset http_proxy https_proxy

export PYTHONPATH=$REPO

mkdir -p /tmp/w12_bwinit

for i in $(seq 0 $((PAIRS-1))); do
  PORT=$((PORT_BASE + i))
  $PY $REPO/tests/nixl/bench_cross_node_bw.py \
    --mode initiator --ip $TARGET_IP --port $PORT --cuda $i \
    --name $i --total-gib 4 --tensors 64 --reps 3 \
    --start-at $START_AT \
    > /tmp/w12_bwinit/ini_${PAIRS}_$i.log 2>&1 &
done

wait

# Collect results
python3 - $PAIRS <<'PYEOF'
import sys, os, json
pairs = int(sys.argv[1])
best_list = []
for i in range(pairs):
    path = f"/tmp/w12_bwinit/ini_{pairs}_{i}.log"
    found = False
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line.startswith("RESULT"):
                toks = {}
                for tok in line.split()[1:]:
                    k, _, v = tok.partition("=")
                    try: toks[k] = float(v)
                    except: toks[k] = v
                gbps = toks.get("best", 0)
                best_list.append(gbps)
                print(f"  pair {i}: {gbps:.2f} GB/s")
                found = True
                break
    if not found:
        lines = open(path).readlines() if os.path.exists(path) else []
        last = lines[-1].strip() if lines else "missing"
        print(f"  pair {i}: no RESULT — {last}")
if best_list:
    agg = sum(best_list)
    print(f"AGGREGATE pairs={pairs} min={min(best_list):.2f} max={max(best_list):.2f} sum={agg:.1f} GB/s")
PYEOF
