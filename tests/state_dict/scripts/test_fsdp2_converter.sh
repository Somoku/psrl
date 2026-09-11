#!/bin/bash
# Run FSDP2 converter integration test with torchrun (2 GPUs).
# Requires a GPU node, the PSRL environment, and a checkpoint at `PSRL_WORKSPACE`.
#
# Usage:
#   PSRL_WORKSPACE=/path/to/workspace bash tests/state_dict/scripts/test_fsdp2_converter.sh

set -xeuo pipefail

PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")

export MASTER_ADDR=localhost
export MASTER_PORT=12345
torchrun --nproc_per_node=2 ${PSRL_PATH}/tests/state_dict/test_fsdp2_converter.py 2>&1 | tee test_fsdp2_converter.log
