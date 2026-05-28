#!/bin/bash

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
python -m uv pip install --no-cache-dir "torch==2.9.1" "torchvision==0.24.1" "torchaudio==2.9.1" --index-url https://download.pytorch.org/whl/cu128
python -m uv pip install --no-cache-dir "triton==3.5.1" "tensordict==0.10.0" torchdata

echo "2. Install basic packages"
python -m uv pip install "transformers[hf_xet]>=4.55.4" accelerate datasets peft hf-transfer matplotlib flask click==8.2.1 \
    "numpy<2.0.0" "pyarrow>=19.0.1" pandas paramiko sortedcontainers \
    ray[default]==2.49.1 codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler blobfile xgrammar \
    pytest py-spy pre-commit ruff meson ninja pynvml requests einops trl

python -m uv pip uninstall -y pynvml nvidia-ml-py
python -m uv pip install --no-cache-dir "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1" "nvidia-cudnn-frontend>=1.13.0"

echo "4. Install FlashAttention and FlashInfer"
# Install flash-attn-2.8.1
FLASH_ATTN_CUDA_ARCHS=128 \
FLASH_ATTENTION_FORCE_BUILD="TRUE" \
FLASH_ATTENTION_FORCE_CXX11_ABI="FALSE" \
FLASH_ATTENTION_SKIP_CUDA_BUILD="FALSE" \
python -m uv pip install -U "flash-attn==2.8.1" --no-build-isolation

# Install flashinfer-python
python -m uv pip install --no-cache-dir --no-build-isolation "flashinfer-python==0.5.3"

echo "5. Install apex"
mkdir -p apex_src
pushd apex_src
git clone https://github.com/NVIDIA/apex.git && \
cd apex && \
MAX_JOB=$MAX_JOBS python -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
popd
rm -rf apex_src

echo "6. Install SMG (with PSRL policies)"
# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust/Cargo not found. Please install Rust first:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo "  source \$HOME/.cargo/env"
    exit 1
fi

echo "Building SMG from source..."
pushd $THIRD_PARTY_PATH
git clone https://github.com/Somoku/smg.git -b psrl-dev
cd smg
# Build release binary with PSRL policies
cargo build --release
python -m uv pip install -e crates/grpc_client/python/
python -m uv pip install -e crates/psrl_state/python/
python -m uv pip install -e grpc_servicer/

echo "Build python binding of smg..."
python -m uv pip install maturin
cd bindings/python
maturin develop --features vendored-openssl
popd

# Verify the binding is importable
python -c "from smg.router import Router; print('  ✓ smg.router binding installed successfully')" || {
    echo "Error: smg.router binding failed to install"
    exit 1
}

echo "7. Install vllm and verl"
if [ -z "$VLLM_PATH" ]; then
    pushd $THIRD_PARTY_PATH
    git clone -b releases/v0.18.1 https://github.com/vllm-project/vllm.git
    VLLM_PATH=$THIRD_PARTY_PATH/vllm
    popd
fi
pushd $VLLM_PATH
python use_existing_torch.py
python -m uv pip install -r requirements/build.txt
python -m uv pip install --no-build-isolation -e .
popd

if [ -z "$VERL_PATH" ]; then
    pushd $THIRD_PARTY_PATH
    git clone https://github.com/volcengine/verl.git
    VERL_PATH=$THIRD_PARTY_PATH/verl
    cd $VERL_PATH
    git checkout 03bc8411
    popd
fi
pushd $VERL_PATH
python -m uv pip install -e .
popd

pushd $PSRL_PATH/patch/vllm
bash apply_patch.sh
popd

echo "8. Apply patch for verl"
pushd $PSRL_PATH/patch/verl
bash apply_patch.sh
popd

echo "9. Install torch_memory_saver"
python -m uv pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@d64a6394d1e09c613fab90260054cecc2684586d --no-cache-dir --force-reinstall

echo "Successfully installed all basic packages"
