#!/bin/bash
# Integration test: vLLM weight conversion via SupportsWeightLayoutSpec (new API).
# Tests TP=1, TP=2, and spec consistency check.
# Requires: GPU node with psrl environment, model checkpoint at PSRL_WORKSPACE.
#
# Usage:
#   PSRL_WORKSPACE=/path/to/workspace bash tests/state_dict/scripts/test_vllm_converter_new_api.sh

set -xeuo pipefail

PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")
TEST_DIR="${PSRL_PATH}/tests/state_dict"

export MASTER_ADDR=localhost
export MASTER_PORT=12346
export PSRL_WORKSPACE=${PSRL_WORKSPACE:-./psrl_workspace}

echo "=== Test 1: TP=1 new API ==="
torchrun --nproc_per_node=1 "${TEST_DIR}/test_vllm_converter.py" 2>&1 | tee test_vllm_converter_new_api_tp1.log
grep -E "PASS|FAIL|Error" test_vllm_converter_new_api_tp1.log

echo "=== Test 2: TP=2 new API ==="
torchrun --nproc_per_node=2 "${TEST_DIR}/test_vllm_converter.py" 2>&1 | tee test_vllm_converter_new_api_tp2.log
grep -E "PASS|FAIL|Error" test_vllm_converter_new_api_tp2.log

echo "=== Test 3: Spec consistency ==="
torchrun --nproc_per_node=1 "${TEST_DIR}/test_vllm_converter.py" consistency 2>&1 | tee test_vllm_converter_consistency.log
grep -E "PASS|FAIL|Error" test_vllm_converter_consistency.log
