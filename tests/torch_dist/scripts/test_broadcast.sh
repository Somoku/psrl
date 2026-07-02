#!/bin/bash
# Run torch.distributed broadcast test with torchrun (5 processes).
# Requires: GPU node with psrl environment activated (≥5 GPUs).
#
# Usage:
#   bash tests/torch_dist/scripts/test_broadcast.sh

set -xeuo pipefail

PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")

export MASTER_ADDR=localhost
export MASTER_PORT=12345
torchrun --nproc_per_node=5 ${PSRL_PATH}/tests/torch_dist/test_broadcast.py 2>&1 | tee test_broadcast.log
