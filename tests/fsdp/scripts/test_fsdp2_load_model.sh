#!/bin/bash
# Run FSDP2 model loading test with torchrun (2 GPUs).
# Requires: GPU node with psrl environment activated.
#
# Usage:
#   bash tests/fsdp/scripts/test_fsdp2_load_model.sh

set -xeuo pipefail

PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")

export MASTER_ADDR=localhost
export MASTER_PORT=12345
torchrun --nproc_per_node=2 ${PSRL_PATH}/tests/fsdp/test_fsdp2_load_model.py 2>&1 | tee test_fsdp2_load_model.log
