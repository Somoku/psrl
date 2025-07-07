#!/bin/bash
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/v0.4.1.x.patch"

if [ ! -f "$PATCH_FILE" ]; then
    echo "Error: Patch file $PATCH_FILE does not exist."
    exit 1
fi

echo "Searching verl install path..."

echo "Try to find verl by python import..."
VERL_PATH=$(python3 -c "
import sys
import os
sys.stdout = open(os.devnull, 'w')
import verl
sys.stdout = sys.__stdout__
print(os.path.dirname(verl.__file__))
" 2>/dev/null || echo "")

if [ -n "$VERL_PATH" ] && [ -d "$VERL_PATH" ]; then
    VERL_PATH=$(dirname "$VERL_PATH")
    echo "Found verl path: $VERL_PATH"
else
    VERL_PATH=""
fi

if [ -z "$VERL_PATH" ] || [ ! -d "$VERL_PATH" ]; then
    echo "Error: Could not find verl installation path."
    echo "Please ensure that verl is installed and try again."
    exit 1
fi

echo "Found verl path: $VERL_PATH"

echo "Check if verl is installed editably..."
IS_EDITABLE=false

if pip list -e 2>/dev/null | grep -q "verl"; then
    IS_EDITABLE=true
    echo "verl is installed in editable mode."
fi

if [ "$IS_EDITABLE" = false ]; then
    echo "Error: verl is not installed in editable mode."
    echo "Please install verl in editable mode using:"
    echo "pip install -e /path/to/verl"
    echo ""
    echo "If you want to apply the patch anyway, please use --force option."
    if [ "$1" != "--force" ]; then
        exit 1
    else
        echo "Applying patch forcefully..."
    fi
fi

TARGET_FILES=(
    "verl/single_controller/ray/base.py"
    "verl/workers/actor/dp_actor.py"
    "verl/workers/megatron_workers.py"
)
for file in "${TARGET_FILES[@]}"; do
    if [ ! -f "$VERL_PATH/$file" ]; then
        echo "Error: Target file $VERL_PATH/$file does not exist."
        exit 1
    fi
done

echo "Applying patch to verl..."
cd "$VERL_PATH"

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
    echo "  - $VERL_PATH/$file"
done