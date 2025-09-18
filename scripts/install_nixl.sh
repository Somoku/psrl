#!/bin/bash

CUDA_PATH=${CUDA_PATH:-"/usr/local/cuda"}
MAX_JOBS=${MAX_JOBS:-32}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSRL_PATH="$(dirname "$SCRIPT_DIR")"
THIRD_PARTY_PATH="$PSRL_PATH/third_party"
mkdir -p $THIRD_PARTY_PATH

echo "1. Install ucx"
mkdir -p $THIRD_PARTY_PATH/ucx_src
pushd $THIRD_PARTY_PATH/ucx_src
git clone -b v1.19.x https://github.com/openucx/ucx.git
cd ucx

# Checking Mellanox NICs
MLX_OPTS=""
if lspci | grep -i mellanox > /dev/null || command -v ibstat > /dev/null; then
    echo "Mellanox NIC detected, adding Mellanox-specific options"
    MLX_OPTS="--with-rdmacm \
              --with-mlx5   \
              --with-ib-hw-tm"
fi

./autogen.sh && ./configure         \
    --prefix=$THIRD_PARTY_PATH/ucx  \
    --enable-shared                 \
    --disable-static                \
    --disable-doxygen-doc           \
    --enable-optimizations          \
    --enable-cma                    \
    --enable-devel-headers          \
    --without-go                    \
    --with-cuda=$CUDA_PATH          \
    --with-verbs                    \
    --with-dm                       \
    --enable-mt.                    \
    $MLX_OPTS &&                    \
make -j $MAX_JOBS &&                \
make -j $MAX_JOBS install-strip &&  \
ldconfig
popd
rm -rf $THIRD_PARTY_PATH/ucx_src

# Recommend to use gcc 11.x.x, gcc-toolset-13 may have error with nixl
echo "2. Install nixl"
mkdir -p $THIRD_PARTY_PATH/nixl_src
pushd $THIRD_PARTY_PATH/nixl_src
git clone -b release/0.5.1 https://github.com/ai-dynamo/nixl.git
cd nixl
mkdir -p build
# Disable obj backend
sed -i "s/subdir('obj')/# subdir('obj')/" "$THIRD_PARTY_PATH/nixl_src/nixl/src/plugins/meson.build"
meson setup build \
    --prefix=$THIRD_PARTY_PATH/nixl \
    -Dbuild_docs=false \
    -Ducx_path=$THIRD_PARTY_PATH/ucx \
    -Dinstall_headers=true \
    -Ddisable_gds_backend=false
cd build
ninja -j $MAX_JOBS
ninja install -j $MAX_JOBS
cd ..
python -m pip install .
popd
rm -rf $THIRD_PARTY_PATH/nixl_src

echo "Successfully installed all packages for nixl"