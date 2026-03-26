#!/bin/bash
# Integration test: vLLM weight conversion via SupportsWeightLayoutSpec (no ParameterMapping)
export MASTER_ADDR=localhost
export MASTER_PORT=12346
export PSRL_WORKSPACE=${PSRL_WORKSPACE:-./psrl_workspace}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Test 1: TP=1 new API ==="
torchrun --nproc_per_node=1 "${TEST_DIR}/test_vllm_converter.py" 2>&1 | tee test_vllm_converter_new_api_tp1.log
grep -E "PASS|FAIL|Error" test_vllm_converter_new_api_tp1.log

echo "=== Test 2: TP=2 new API ==="
torchrun --nproc_per_node=2 "${TEST_DIR}/test_vllm_converter.py" 2>&1 | tee test_vllm_converter_new_api_tp2.log
grep -E "PASS|FAIL|Error" test_vllm_converter_new_api_tp2.log

echo "=== Test 3: Spec consistency ==="
torchrun --nproc_per_node=1 "${TEST_DIR}/test_vllm_converter.py" consistency 2>&1 | tee test_vllm_converter_consistency.log
grep -E "PASS|FAIL|Error" test_vllm_converter_consistency.log
