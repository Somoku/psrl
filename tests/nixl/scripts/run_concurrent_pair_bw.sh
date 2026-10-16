#!/bin/bash
# Measure cross-node NIXL READ aggregate bandwidth as a function of concurrent pair count.
#
# Usage:
#   tests/nixl/scripts/run_concurrent_pair_bw.sh <target-ip> <initiator-ip> [max-pairs] [bench args...]
#
# Sweeps pairs = 1, 2, 4, <max-pairs> (default 8).
# Each pair gets its own unique --name so NIXL agent names never collide.
# Uses --start-at to align all initiators for a given pair count.
#
# Example:
#   tests/nixl/scripts/run_concurrent_pair_bw.sh 28.59.83.232 29.162.224.39 8 \
#     --total-gib 4 --tensors 64 --reps 3
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TARGET=$1; INIT=$2; MAX_PAIRS=${3:-8}; shift 3

BASE_PORT=${BASE_PORT:-32100}
SERVE_S=600
RUN="/tmp/w12_cpbw_$(date +%s)"
mkdir -p "$RUN"

# Use absolute paths so SSH non-interactive shell works without conda activate.
# psrl.sh sets ENV_NAME=base; Python lives at ${CONDA}/envs/${ENV_NAME}/bin/python.
CONDA=/apdcephfs_zwfy10/share_303541817/lhy/anaconda3
PY=${CONDA}/envs/base/bin/python
CONDA_LIB=${CONDA}/envs/base/lib

# nixl_env.sh exports UCX_NET_DEVICES and LD_LIBRARY_PATH; replicate the minimal set.
UCX_DEVS="bond1,bond2,bond3,bond4,bond5,bond6,bond7,bond8,mlx5_bond_1:1,mlx5_bond_4:1,mlx5_bond_3:1,mlx5_bond_2:1,mlx5_bond_7:1,mlx5_bond_6:1,mlx5_bond_8:1,mlx5_bond_5:1"
NIXL_ENV="export UCX_NET_DEVICES='${UCX_DEVS}'; export LD_LIBRARY_PATH='${CONDA_LIB}:\${LD_LIBRARY_PATH:-}'; export UCX_IB_ROCE_REACHABILITY_MODE=all; unset http_proxy https_proxy"
ENV="$NIXL_ENV; cd ${REPO}; export PYTHONPATH=${REPO}"

# Remote log dirs — fixed paths on each node, no cross-node path passing
TGT_LOGDIR=/tmp/w12_cpbw_tgt
INIT_LOGDIR=/tmp/w12_cpbw_init

echo "=== Concurrent pair bandwidth sweep: TARGET=$TARGET INIT=$INIT MAX_PAIRS=$MAX_PAIRS ==="
echo "Target logs: ${TARGET}:${TGT_LOGDIR}  Init logs: ${INIT}:${INIT_LOGDIR}"

# Kill any stale bench processes on both nodes
ssh -o BatchMode=yes -o StrictHostKeyChecking=no -n "$TARGET" \
  "pkill -f bench_cross_node_bw 2>/dev/null; mkdir -p $TGT_LOGDIR; true" 2>/dev/null || true
ssh -o BatchMode=yes -o StrictHostKeyChecking=no -n "$INIT" \
  "pkill -f bench_cross_node_bw 2>/dev/null; mkdir -p $INIT_LOGDIR; true" 2>/dev/null || true
sleep 1

# Launch all targets up front (they serve for SERVE_S seconds, covering all sweeps)
echo "Starting $MAX_PAIRS target agents..."
for i in $(seq 0 $((MAX_PAIRS-1))); do
  PORT=$((BASE_PORT + i))
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no -n "$TARGET" \
    "$ENV; nohup $PY ${REPO}/tests/nixl/bench_cross_node_bw.py \
      --mode target --ip $TARGET --port $PORT --cuda $i \
      --name $i $* --serve-s $SERVE_S \
      > ${TGT_LOGDIR}/tgt_$i.log 2>&1 &" 2>/dev/null
  sleep 0.3
done

# Wait until all targets are up
echo -n "Waiting for $MAX_PAIRS targets..."
for attempt in $(seq 1 120); do
  up=$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$TARGET" \
    "grep -rl 'waiting for peer' ${TGT_LOGDIR}/ 2>/dev/null | wc -l" 2>/dev/null | grep -v no_proxy | tail -1)
  up=${up:-0}
  if [ "$up" -ge "$MAX_PAIRS" ]; then
    echo " $up/$MAX_PAIRS up"
    break
  fi
  echo -n "."
  sleep 1
done
if [ "${up:-0}" -lt "$MAX_PAIRS" ]; then
  echo " only $up/$MAX_PAIRS up after 120s; checking logs:"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$TARGET" \
    "tail -5 ${TGT_LOGDIR}/tgt_0.log 2>/dev/null" 2>/dev/null | grep -v no_proxy
  exit 1
fi

# Sweep pair counts
RESULTS=""
for PAIRS in 1 2 4 $MAX_PAIRS; do
  # Skip duplicates if MAX_PAIRS is already in the list
  case "$PAIRS" in
    $MAX_PAIRS) ;;
    *) [ "$PAIRS" -gt "$MAX_PAIRS" ] && continue ;;
  esac

  echo ""
  echo "--- pairs=$PAIRS ---"
  START_AT=$(python3 -c "import time; print(time.time() + 60)")

  for i in $(seq 0 $((PAIRS-1))); do
    PORT=$((BASE_PORT + i))
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no -n "$INIT" \
      "$ENV; $PY tests/nixl/bench_cross_node_bw.py \
        --mode initiator --ip $TARGET --port $PORT --cuda $i \
        --name $i $* --start-at $START_AT \
        > $RUN/ini_p${PAIRS}_$i.log 2>&1" &
  done
  wait

  # Collect per-pair results and compute aggregate
  python3 - "$RUN" "$PAIRS" <<'PYEOF'
import glob, json, sys, os
run, pairs = sys.argv[1], int(sys.argv[2])
best_list = []
for i in range(pairs):
    path = f"{run}/ini_p{pairs}_{i}.log"
    if not os.path.exists(path):
        print(f"  pair {i}: log missing")
        continue
    for line in open(path):
        if line.startswith("RESULT"):
            d = {}
            for tok in line.split()[1:]:
                k, _, v = tok.partition("=")
                try: d[k] = float(v)
                except: d[k] = v
            best_list.append(d.get("best", 0))
            print(f"  pair {i}: {d.get('best', '?')} GB/s")
            break
    else:
        # show last few lines for diagnosis
        lines = open(path).readlines()
        print(f"  pair {i}: no RESULT — last line: {lines[-1].strip() if lines else 'empty'}")

if best_list:
    agg = sum(best_list)
    print(f"AGGREGATE pairs={pairs} min={min(best_list):.2f} max={max(best_list):.2f} sum={agg:.1f} GB/s")
PYEOF
done

echo ""
echo "=== Done. Logs in $RUN ==="
