#!/bin/bash
# Run vLLM converter integration test with torchrun (2 GPUs, old ParameterMapping API).
# Tests: vLLM → HuggingFace state dict conversion via ParameterMapping.
# Requires: GPU node with psrl environment, model checkpoint at PSRL_WORKSPACE.
#
# Usage:
#   PSRL_WORKSPACE=/path/to/workspace bash tests/state_dict/scripts/test_vllm_converter.sh

set -xeuo pipefail

PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")

export MASTER_ADDR=localhost
export MASTER_PORT=12345
torchrun --nproc_per_node=2 ${PSRL_PATH}/tests/state_dict/test_vllm_converter.py 2>&1 | tee test_vllm_converter.log
