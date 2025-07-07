#!/bin/bash

USE_MEGATRON=${USE_MEGATRON:-1}

VLLM_PATH=${VLLM_PATH:-2}
VERL_PATH=${VERL_PATH:-3}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSRL_PATH="$(dirname "$SCRIPT_DIR")"

export MAX_JOBS=32

echo "1. install pytorch and tensordict"
pip install --no-cache-dir "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" --index-url https://download.pytorch.org/whl/cu128
pip install --no-cache-dir "tensordict==0.6.2" torchdata

echo "2. install basic packages"
pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pyext pre-commit ruff

pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

echo "3. install apex"
mkdir -p apex_src
pushd apex_src
git clone https://github.com/NVIDIA/apex.git && \
cd apex && \
MAX_JOB=32 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
popd
rm -rf apex_src

echo "4. install FlashAttention and FlashInfer"
# Install flash-attn-2.7.4.post1
pip install --no-cache-dir "flash-attn==2.7.4.post1" --no-build-isolation

# Install flashinfer-0.2.7.post1
pip install --no-cache-dir "flashinfer-python==0.2.7.post1"


if [ $USE_MEGATRON -eq 1 ]; then
    echo "5. install TransformerEngine and Megatron"
    echo "Notice that TransformerEngine installation can take very long time, please be patient"
    NVTE_FRAMEWORK=pytorch pip3 install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.2
    pip3 install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.0rc3
fi


echo "6. May need to fix opencv"
pip install opencv-python
pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"

echo "7. Install vllm and verl"
if [ -z "$VLLM_PATH" ]; then
    mkdir -p vllm_src
    pushd vllm_src
    git clone --b v0.9.0.1 https://github.com/vllm-project/vllm.git
    VLLM_PATH=$(pwd)/vllm
    popd
fi
pushd $VLLM_PATH
python use_existing_torch.py
pip install -r requirements/build.txt
pip install --no-build-isolation -e .
popd

if [ -z "$VERL_PATH" ]; then
    mkdir -p verl_src
    pushd verl_src
    git clone --b v0.4.1.x https://github.com/volcengine/verl.git
    VERL_PATH=$(pwd)/verl
    popd
fi
pushd $VERL_PATH
pip install -e .
popd

echo "8. Apply patch for vllm and verl"
pushd $VLLM_PATH/patch/vllm
bash apply_patch.sh
popd

pushd $PSRL_PATH/patch/verl
bash apply_patch.sh
popd

if [ $USE_MEGATRON -eq 1 ]; then
    echo "7. Install cudnn python package (avoid being overridden)"
    pip install nvidia-cudnn-cu12==9.8.0.87
fi

echo "Successfully installed all packages"
