#!/bin/bash
set -e
set -o pipefail
trap 'echo "[ERROR] Failed at line $LINENO: $BASH_COMMAND" >&2; exit 1' ERR

VLLM_PATH=${VLLM_PATH:-}
VERL_PATH=${VERL_PATH:-}
MAX_JOBS=${MAX_JOBS:-32}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSRL_PATH="$(dirname "$SCRIPT_DIR")"
THIRD_PARTY_PATH="$PSRL_PATH/third_party"
mkdir -p $THIRD_PARTY_PATH

echo "0. Install uv to boost installation speed"
python -m pip install uv

echo "1. Install pytorch and tensordict"
python -m uv pip install "torch==2.9.1" "torchvision==0.24.1" "torchaudio==2.9.1" --index-url https://download.pytorch.org/whl/cu128
python -m uv pip install "triton==3.5.1" "tensordict==0.10.0" torchdata

echo "2. Install basic packages"
python -m uv pip install "transformers[hf_xet]==5.5.0" accelerate datasets peft hf-transfer matplotlib flask click==8.2.1 \
    "numpy<2.0.0" "pyarrow>=19.0.1" pandas paramiko sortedcontainers \
    ray[default]==2.49.1 codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler blobfile xgrammar \
    pytest py-spy pre-commit ruff meson ninja pynvml requests einops trl==0.26.2

python -m uv pip uninstall pynvml nvidia-ml-py
python -m uv pip install --no-cache-dir "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1" "nvidia-cudnn-frontend>=1.13.0"

echo "4. Install FlashAttention and FlashInfer"
# Install flash-attn-2.8.1
FLASH_ATTN_CUDA_ARCHS=90 \
FLASH_ATTENTION_FORCE_BUILD="TRUE" \
FLASH_ATTENTION_FORCE_CXX11_ABI="FALSE" \
FLASH_ATTENTION_SKIP_CUDA_BUILD="FALSE" \
python -m uv pip install --no-cache-dir --no-build-isolation "flash-attn==2.8.1" 

# Install flashinfer-python
python -m uv pip install --no-cache-dir --no-build-isolation "flashinfer-python==0.5.3"

python -m uv pip install flash-linear-attention==0.4.2

echo "5. Install apex"
mkdir -p apex_src
pushd apex_src
git clone https://github.com/NVIDIA/apex.git && \
cd apex && \
MAX_JOB=$MAX_JOBS python -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
popd
rm -rf apex_src

echo "6. Install vllm and verl"
if [ -z "$VLLM_PATH" ]; then
    pushd $THIRD_PARTY_PATH
    git clone -b v0.18.1 https://github.com/vllm-project/vllm.git
    popd
fi
pushd $VLLM_PATH
python use_existing_torch.py
python -m uv pip install -r requirements/build.txt
python -m uv pip install --no-build-isolation -e .
popd

# Reinstall transformers because vllm v0.18.1 uses transformers==4.57.3
python -m uv pip install "transformers[hf_xet]==5.5.0"

if [ -z "$VERL_PATH" ]; then
    pushd $THIRD_PARTY_PATH
    git clone https://github.com/volcengine/verl.git
    VERL_PATH=$THIRD_PARTY_PATH/verl
    cd $VERL_PATH
    git checkout 3824689
    popd
fi
pushd $VERL_PATH
python -m uv pip install -e .
popd

echo "7. Apply patch for vllm"
pushd $PSRL_PATH/patch/vllm
bash apply_patch.sh
popd

echo "8. Apply patch for verl"
pushd $PSRL_PATH/patch/verl
bash apply_patch.sh
popd

echo "Successfully installed all basic packages"
