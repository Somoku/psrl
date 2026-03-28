#!/bin/bash
# Run Megatron model initialization test.
# Requires: GPU node with psrl + megatron environment activated.
#
# Usage:
#   bash tests/megatron/scripts/test_megatron_model_init.sh

set -xeuo pipefail

PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")

export PSRL_LOGGING_PATH=${PSRL_PATH}/tests/megatron/log
export PSRL_LOGGING_LEVEL=INFO
mkdir -p ${PSRL_LOGGING_PATH}

python ${PSRL_PATH}/tests/megatron/test_megatron_model_init.py 2>&1 | tee test_megatron_model_init.log
