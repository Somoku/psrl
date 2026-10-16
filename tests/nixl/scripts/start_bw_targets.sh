#!/bin/bash
# Start N bench_cross_node_bw target agents, one per GPU.
# Run this script directly on the target node via SSH:
#   ssh <target-node> bash /path/to/start_bw_targets.sh <target-ip> <pairs> [port-base]
#
# Each agent writes its log to /tmp/w12_bwtgt/tgt_<i>.log
set -uo pipefail

TARGET_IP=$1
PAIRS=${2:-8}
PORT_BASE=${3:-32500}

REPO=/apdcephfs_zwfy10/share_303541817/ls/psrl-wip/psrl
CONDA=/apdcephfs_zwfy10/share_303541817/lhy/anaconda3
ENV_NAME=base
# base env lives directly in $CONDA/bin, not $CONDA/envs/base/bin
PY=${CONDA}/bin/python

# Source the cluster env to get full UCX_NET_DEVICES, LD_LIBRARY_PATH, etc.
source ${CONDA}/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ${ENV_NAME} 2>/dev/null || true
source $REPO/scripts/nixl_env.sh 2>/dev/null || true
unset http_proxy https_proxy

export PYTHONPATH=$REPO

mkdir -p /tmp/w12_bwtgt
echo "Starting $PAIRS target agents on $TARGET_IP..."

for i in $(seq 0 $((PAIRS-1))); do
  PORT=$((PORT_BASE + i))
  $PY $REPO/tests/nixl/bench_cross_node_bw.py \
    --mode target --ip $TARGET_IP --port $PORT --cuda $i \
    --name $i --total-gib 4 --tensors 64 --reps 3 --serve-s 600 \
    > /tmp/w12_bwtgt/tgt_$i.log 2>&1 &
  echo "  GPU $i: port $PORT pid $!"
done

# Wait for all agents to be up
echo -n "Waiting for agents..."
for attempt in $(seq 1 60); do
  up=$(grep -rl 'waiting for peer' /tmp/w12_bwtgt/ 2>/dev/null | wc -l)
  [ "$up" -ge "$PAIRS" ] && echo " $up/$PAIRS up" && exit 0
  echo -n "."
  sleep 1
done
echo " timed out"
exit 1
