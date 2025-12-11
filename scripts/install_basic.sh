#!/bin/bash

VLLM_PATH=${VLLM_PATH:-}
VERL_PATH=${VERL_PATH:-}
MAX_JOBS=${MAX_JOBS:-32}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSRL_PATH="$(dirname "$SCRIPT_DIR")"
THIRD_PARTY_PATH="$PSRL_PATH/third_party"
mkdir -p $THIRD_PARTY_PATH

echo "1. Install pytorch and tensordict"
python -m pip install --no-cache-dir "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" --index-url https://download.pytorch.org/whl/cu128
# python -m pip install --no-cache-dir "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-cache-dir "tensordict==0.10.0" torchdata

echo "2. Install xformers"
# python -m pip install -v --no-build-isolation -U "git+https://github.com/facebookresearch/xformers.git@v0.0.29.post3#egg=xformers"
python -m pip install -v --no-build-isolation -U "git+https://github.com/facebookresearch/xformers.git@v0.0.31#egg=xformers"

echo "3. Install basic packages"
python -m pip install "transformers[hf_xet]>=4.55.4" accelerate datasets peft hf-transfer matplotlib flask click==8.2.1 \
    "numpy<2.0.0" "pyarrow>=19.0.1" pandas paramiko sortedcontainers \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler blobfile xgrammar \
    pytest py-spy pyext pre-commit ruff meson ninja pynvml requests einops

python -m pip uninstall -y pynvml nvidia-ml-py
python -m pip install --no-cache-dir "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1" "nvidia-cudnn-frontend>=1.13.0"

echo "4. Install FlashAttention and FlashInfer"
# Install flash-attn-2.7.2.post1
python -m pip install --no-cache-dir --no-build-isolation "flash-attn==2.7.2.post1" 
# Install flashinfer-0.2.7.post1
python -m pip install --no-cache-dir --no-build-isolation "flashinfer-python==0.2.7.post1"

echo "5. Install apex"
mkdir -p apex_src
pushd apex_src
git clone https://github.com/NVIDIA/apex.git && \
cd apex && \
MAX_JOB=$MAX_JOBS pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
popd
rm -rf apex_src

echo "6. May need to fix opencv"
python -m pip install opencv-python
python -m pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"

echo "7. Install vllm and verl"
if [ -z "$VLLM_PATH" ]; then
    pushd $THIRD_PARTY_PATH
    git clone -b v0.10.2 https://github.com/vllm-project/vllm.git
    VLLM_PATH=$THIRD_PARTY_PATH/vllm
    popd
fi
pushd $VLLM_PATH
cp $PSRL_PATH/patch/vllm/use_existing_torch.py .
python use_existing_torch.py
python -m pip install -r requirements/build.txt
python -m pip install --no-build-isolation -e .
popd

if [ -z "$VERL_PATH" ]; then
    pushd $THIRD_PARTY_PATH
    git clone https://github.com/volcengine/verl.git
    VERL_PATH=$THIRD_PARTY_PATH/verl
    cd VERL_PATH
    git checkout 6ff2b43
    popd
fi
pushd $VERL_PATH
python -m pip install -e .
popd

echo "8. Apply patch for vllm and verl"
pushd $PSRL_PATH/patch/vllm
bash apply_patch.sh
popd

pushd $PSRL_PATH/patch/verl
bash apply_patch.sh
popd

echo "9. Downgrade uvloop to 0.21.0"
python -m pip install uvloop==0.21.0

echo "Successfully installed all basic packages"
