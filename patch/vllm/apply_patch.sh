#!/bin/bash
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/v0.9.0.1.patch"

if [ ! -f "$PATCH_FILE" ]; then
    echo "Error: Patch file $PATCH_FILE does not exist."
    exit 1
fi

echo "Searching vllm install path..."

echo "Try to find vllm by python import..."

VLLM_PATH=$(python3 -c "
import sys
import os
sys.stdout = open(os.devnull, 'w')
import vllm
sys.stdout = sys.__stdout__
print(os.path.dirname(vllm.__file__))
" 2>/dev/null || echo "")

if [ -n "$VLLM_PATH" ] && [ -d "$VLLM_PATH" ]; then
    VLLM_PATH=$(dirname "$VLLM_PATH")
    echo "Found vllm path: $VLLM_PATH"
else
    VLLM_PATH=""
fi

if [ -z "$VLLM_PATH" ] || [ ! -d "$VLLM_PATH" ]; then
    echo "Error: Could not find vllm installation path."
    echo "Please ensure that vllm is installed and try again."
    exit 1
fi

echo "Found vllm path: $VLLM_PATH"

echo "Check if vllm is installed editably..."
IS_EDITABLE=false

if pip list -e 2>/dev/null | grep -q "vllm"; then
    IS_EDITABLE=true
    echo "vllm is installed in editable mode."
fi

if [ "$IS_EDITABLE" = false ]; then
    echo "Error: vllm is not installed in editable mode."
    echo "Please install vllm in editable mode using:"
    echo "pip install -e /path/to/vllm"
    echo ""
    echo "If you want to apply the patch anyway, please use --force option."
    if [ "$1" != "--force" ]; then
        exit 1
    else
        echo "Applying patch forcefully..."
    fi
fi

TARGET_FILES=(
    "vllm/model_executor/layers/fused_moe/fused_moe.py"
    "vllm/model_executor/layers/quantization/utils/fp8_utils.py"
    "vllm/v1/sample/metadata.py"
    "vllm/v1/sample/sampler.py"
    "vllm/v1/worker/gpu_input_batch.py"
    "vllm/v1/worker/gpu_model_runner.py"
)
for file in "${TARGET_FILES[@]}"; do
    if [ ! -f "$VLLM_PATH/$file" ]; then
        echo "Error: Target file $VLLM_PATH/$file does not exist."
        exit 1
    fi
done

echo "Applying patch to vllm..."
cd "$VLLM_PATH"

if git apply --check "$PATCH_FILE" > /dev/null 2>&1; then
    echo "Patch is valid, applying..."
    if git apply "$PATCH_FILE"; then
        echo "Patch applied successfully."
    else
        echo "Error: Failed to apply patch."
        exit 1
    fi
else
    echo "Error: Patch is not valid or cannot be applied cleanly."
    exit 1
fi

echo ""
echo "Patch applied successfully to the following files:"
for file in "${TARGET_FILES[@]}"; do
    echo "  - $VLLM_PATH/$file"
done