#!/bin/bash
# Run one configuration-matrix cell of the NIXL e2e harness on one node.
#
#   tests/nixl/scripts/run_e2e_cell.sh <node-ip> <cell-name> [hydra overrides...]
#
# The node must run its own single-node Ray head (`ray start --head`), so the harness's
# `ray.init()` attaches to it and train/gen land on that node only. Output goes to
# tests/nixl/log/matrix/<cell-name>/ (driver log + per-client logs); the last lines of the
# driver log carry the [PASS]/[MEASURE] verdicts.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
NODE=$1; CELL=$2; shift 2
OUT="${REPO}/tests/nixl/log/matrix/${CELL}"
rm -rf "$OUT"; mkdir -p "$OUT"
PY=python  # nixl_env.sh puts the cluster interpreter on PATH
ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$NODE" \
  "source ${REPO}/scripts/nixl_env.sh >/dev/null 2>&1; unset http_proxy https_proxy; \
   export PSRL_REPO=${REPO}; cd ${REPO}/tests/nixl; \
   PYTHONUNBUFFERED=1 timeout ${CELL_TIMEOUT_S:-3600} $PY test_nixl_e2e.py --config-name nixl_e2e_1node \
     logging.path=${OUT} $* > ${OUT}/driver.log 2>&1; echo \"exit=\$?\" >> ${OUT}/driver.log"
tail -1 "$OUT/driver.log"
