#!/bin/bash
set -e
set -o pipefail
trap 'echo "[ERROR] Failed at line $LINENO: $BASH_COMMAND" >&2; exit 1' ERR

CUDA_PATH=${CUDA_PATH:-"/usr/local/cuda"}
MAX_JOBS=${MAX_JOBS:-32}
# NIXL v1.4.1 is tested against UCX v1.22.x. UCX_REF is the git ref to build
# when the detected UCX is older than REQUIRED_UCX_VERSION.
UCX_REF="v1.22.x"
REQUIRED_UCX_VERSION="1.22.0"
NIXL_REF="v1.4.1"
UCX_PREFIX="/usr"
INSTALL_UCX=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSRL_PATH="$(dirname "$SCRIPT_DIR")"
THIRD_PARTY_PATH="$PSRL_PATH/third_party"
mkdir -p $THIRD_PARTY_PATH

if command -v ucx_info >/dev/null 2>&1; then
    echo "Detected existing UCX installation via ucx_info"
    UCX_INFO_OUTPUT=$(ucx_info -v 2>/dev/null || true)
    DETECTED_VERSION=$(echo "$UCX_INFO_OUTPUT" | grep -Eo '([0-9]+\.){2}[0-9]+' | head -n1)
    DETECTED_PREFIX=$(echo "$UCX_INFO_OUTPUT" | grep -i -- '--prefix=' | sed -E 's/.*--prefix=([^ ]+).*/\1/' | head -n1)

    if [ -n "$DETECTED_PREFIX" ]; then
        UCX_PREFIX="$DETECTED_PREFIX"
    fi

    if [ -n "$DETECTED_VERSION" ] && [ "$(printf '%s\n%s\n' "$DETECTED_VERSION" "$REQUIRED_UCX_VERSION" | sort -V | head -n1)" = "$REQUIRED_UCX_VERSION" ]; then
        echo "UCX version $DETECTED_VERSION found at $UCX_PREFIX (>= $REQUIRED_UCX_VERSION), skipping UCX build."
        INSTALL_UCX=false
    else
        echo "UCX version $DETECTED_VERSION found at $UCX_PREFIX (< $REQUIRED_UCX_VERSION), will build UCX $REQUIRED_UCX_VERSION."
    fi
else
    echo "UCX info was not found. Building UCX $REQUIRED_UCX_VERSION."
fi

if $INSTALL_UCX; then
    echo "1. Install ucx"
    UCX_PREFIX="$THIRD_PARTY_PATH/ucx"
    mkdir -p $THIRD_PARTY_PATH/ucx_src
    pushd $THIRD_PARTY_PATH/ucx_src
    git clone -b $UCX_REF https://github.com/openucx/ucx.git
    cd ucx

    # Checking Mellanox NICs
    MLX_OPTS=""
    if lspci | grep -i mellanox > /dev/null || command -v ibstat > /dev/null; then
        echo "Mellanox NIC detected, adding Mellanox-specific options"
        MLX_OPTS="--with-rdmacm \
                  --with-mlx5   \
                  --with-ib-hw-tm"
    fi

    ./autogen.sh && ./configure     \
        --prefix=$UCX_PREFIX        \
        --enable-shared             \
        --disable-static            \
        --disable-doxygen-doc       \
        --enable-optimizations      \
        --enable-cma                \
        --enable-devel-headers      \
        --without-go                \
        --with-cuda=$CUDA_PATH      \
        --with-verbs                \
        --with-dm                   \
        --enable-mt                 \
        $MLX_OPTS &&                \
    make -j $MAX_JOBS &&            \
    make -j $MAX_JOBS install-strip &&  \
    ldconfig
    popd
    rm -rf $THIRD_PARTY_PATH/ucx_src
else
    echo "1. Skip UCX installation"
fi

# NIXL v1.4.1 requires a C++20 compiler (GCC >= 11 or Clang >= 14).
if ! echo 'int main(){return 0;}' | ${CXX:-g++} -std=c++20 -x c++ - -o /dev/null 2>/dev/null; then
    echo "[ERROR] NIXL v1.4.1 requires a C++20 compiler (GCC >= 11 or Clang >= 14);" \
         "${CXX:-g++} does not support -std=c++20." >&2
    exit 1
fi

echo "2. Install nixl"
mkdir -p $THIRD_PARTY_PATH/nixl_src
pushd $THIRD_PARTY_PATH/nixl_src
git clone -b $NIXL_REF https://github.com/ai-dynamo/nixl.git
cd nixl
mkdir -p build
# PSRL only uses the UCX backend. Build no other plugins.
meson setup build \
    --prefix=$THIRD_PARTY_PATH/nixl \
    -Dbuild_docs=false \
    -Ducx_path=$UCX_PREFIX \
    -Dinstall_headers=true \
    -Ddisable_gds_backend=false
cd build
ninja -j $MAX_JOBS
ninja install -j $MAX_JOBS
cd ..
python -m pip install .
python -m pip install build/src/bindings/python/nixl-meta/nixl-*-py3-none-any.whl
popd
rm -rf $THIRD_PARTY_PATH/nixl_src

echo "Successfully installed all packages for nixl"
