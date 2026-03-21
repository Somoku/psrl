#!/bin/bash
set -e
set -o pipefail
trap 'echo "[ERROR] Failed at line $LINENO: $BASH_COMMAND" >&2; exit 1' ERR

TMS_PATH=${TMS_PATH:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSRL_PATH="$(dirname "$SCRIPT_DIR")"
THIRD_PARTY_PATH="$PSRL_PATH/third_party"
mkdir -p $THIRD_PARTY_PATH

echo "1. Install torch_memory_saver"
if [ -z "$TMS_PATH" ]; then
    pushd $THIRD_PARTY_PATH
    git clone https://github.com/fzyzcjy/torch_memory_saver.git
    TMS_PATH=$THIRD_PARTY_PATH/torch_memory_saver
    popd
fi
pushd $TMS_PATH
git checkout d64a6394d1e09c613fab90260054cecc2684586d
rm -rf ./*.so ./build
python -m pip uninstall torch_memory_saver -y
python -m pip install --no-cache-dir -e .
popd

echo "Successfully installed torch_memory_saver"
