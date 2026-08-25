#!/usr/bin/env bash
# E3b memory sweep: one engine process per max_num_batched_tokens value N.
#
# Usage:
#   bash run_chunked_prefill_memory_sweep.sh [TP] [MODEL_PATH] [GPU_UTIL]
#
# Positional arguments (all optional):
#   TP           Tensor parallel size (default: 1)
#   MODEL_PATH   Path to the HuggingFace model (default: ${PSRL_WORKSPACE}/models/SWE-agent-LM-7B)
#   GPU_UTIL     gpu_memory_utilization (default: 0.90)
#
# For each N in N_VALUES, this script starts a fresh engine, reads the memory
# breakdown (activation reservation + KV capacity), and exits.  Results are
# written to individual JSON files under exp/chunked_prefill/e3b/.
#
# Because peak_activation is profiled once at engine startup under
# max_num_batched_tokens and then frozen, a fresh process per N is the ONLY
# way to observe the true reservation cost for each N value.
set -euo pipefail

source ${PSRL_WORKSPACE}/env/psrl.sh

GEN_TP=${1:-1}
HF_MODEL_PATH=${2:-${PSRL_WORKSPACE}/models/SWE-agent-LM-7B}
GPU_UTIL=${3:-0.90}

PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")
E3B_DIR="${PSRL_PATH}/examples/bench/rollout/exp/chunked_prefill/e3b"
mkdir -p "${E3B_DIR}"

# N values to sweep.
N_VALUES=(512 1024 2048 4096 8192 16384 32768 65536)

echo "E3b memory sweep: TP=${GEN_TP} util=${GPU_UTIL}"
echo "Output directory: ${E3B_DIR}"

for N in "${N_VALUES[@]}"; do
    echo "--- Probing N=${N} ---"
    set -x
    PYTHONUNBUFFERED=1 python -m psrl.bench.chunked_prefill.memory_probe \
        --model "${HF_MODEL_PATH}" \
        --max-batched-tokens "${N}" \
        --tp "${GEN_TP}" \
        --gpu-util "${GPU_UTIL}" \
        --max-model-len 32768 \
        --output "${E3B_DIR}/e3b_TP${GEN_TP}_N${N}_util$(echo ${GPU_UTIL} | tr -d '.').json" \
        2>&1 | tee "${E3B_DIR}/e3b_TP${GEN_TP}_N${N}_util$(echo ${GPU_UTIL} | tr -d '.').log"
    set +x
    echo "--- Done N=${N} ---"
done

echo "E3b sweep complete. Results in ${E3B_DIR}"
