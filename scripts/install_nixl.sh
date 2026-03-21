#!/bin/bash
set -e
set -o pipefail
trap 'echo "[ERROR] Failed at line $LINENO: $BASH_COMMAND" >&2; exit 1' ERR

CUDA_PATH=${CUDA_PATH:-"/usr/local/cuda"}
MAX_JOBS=${MAX_JOBS:-32}
REQUIRED_UCX_VERSION="1.20.0"
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
    echo "ucx_info not found; will build UCX $REQUIRED_UCX_VERSION."
fi

if $INSTALL_UCX; then
    echo "1. Install ucx"
    UCX_PREFIX="$THIRD_PARTY_PATH/ucx"
    mkdir -p $THIRD_PARTY_PATH/ucx_src
    pushd $THIRD_PARTY_PATH/ucx_src
    git clone -b $REQUIRED_UCX_VERSION https://github.com/openucx/ucx.git
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

# Recommend to use gcc 11.x.x, gcc-toolset-13 may have error with nixl
echo "2. Install nixl"
mkdir -p $THIRD_PARTY_PATH/nixl_src
pushd $THIRD_PARTY_PATH/nixl_src
git clone -b 0.10.1 https://github.com/ai-dynamo/nixl.git
cd nixl
mkdir -p build
# Disable obj backend
sed -i "s/subdir('obj')/# subdir('obj')/" "$THIRD_PARTY_PATH/nixl_src/nixl/src/plugins/meson.build"

# Fix metadata_stream: background acceptClientsAsync() thread never exits when the
# listener socket is closed at agent teardown.  Three bugs, fixed together:
#
#   1. closeStream() forgets to reset socketFd to -1 after close(), so the
#      destructor's "if (socketFd != -1)" guard is useless and any later access
#      to the stale fd is unguarded.
#
#   2. ~nixlMDStreamListener() joins the thread *before* closing the socket,
#      so the thread is stuck in accept() and join() deadlocks forever.
#      The fix: close the socket first (interrupts accept()), then join.
#
#   3. acceptClientsAsync() treats every accept() failure as a loggable error.
#      After our fix the socket-closed errno values (EBADF, ENOTSOCK, EINVAL)
#      become the normal shutdown signal and must cause a clean exit, not a log
#      storm.  EAGAIN/EWOULDBLOCK (non-blocking socket, no client yet) should
#      also be silent.
#
# These three changes together eliminate the repeated
#   "Cannot accept client connection: Socket operation on non-socket [88]"
# messages that appear when Ray reuses the fd that nixl released.
STREAM_H="$THIRD_PARTY_PATH/nixl_src/nixl/src/utils/stream/metadata_stream.h"
STREAM_CPP="$THIRD_PARTY_PATH/nixl_src/nixl/src/utils/stream/metadata_stream.cpp"

# Fix 1: add #include <atomic> and a stopping_ flag to nixlMDStreamListener.
# The atomic flag lets acceptClientsAsync() know it should exit cleanly.
sed -i 's/#include <thread>/#include <atomic>\n#include <thread>/' "$STREAM_H"
sed -i 's/        std::thread listenerThread;/        std::atomic<bool> stopping_{false};\n        std::thread listenerThread;/' "$STREAM_H"

# Fixes 2-4: multi-line edits handled by Python (sed can't match across lines).
python3 - "$STREAM_CPP" <<'PYEOF'
import sys, re

path = sys.argv[1]
src  = open(path).read()

# Fix 2: reset socketFd to -1 in closeStream() so the stale fd is never reused.
src = src.replace(
    'void nixlMetadataStream::closeStream() {\n'
    '   if (socketFd != -1) {\n'
    '        close(socketFd);\n'
    '   }\n'
    '}',
    'void nixlMetadataStream::closeStream() {\n'
    '   if (socketFd != -1) {\n'
    '        close(socketFd);\n'
    '        socketFd = -1;  // prevent stale-fd reuse after close\n'
    '   }\n'
    '}'
)

# Fix 3: ~nixlMDStreamListener(): set stopping_ and close the socket *before*
# joining the thread.  This unblocks accept() so join() can complete promptly.
src = src.replace(
    'nixlMDStreamListener::~nixlMDStreamListener() {\n'
    '    if (listenerThread.joinable()) {\n'
    '        listenerThread.join();\n'
    '    }\n'
    '    if (csock >= 0) {\n'
    '            close(csock);\n'
    '    }\n'
    '}',
    'nixlMDStreamListener::~nixlMDStreamListener() {\n'
    '    // Signal the background thread to stop, then close the listening socket\n'
    '    // so that the blocking accept() call is interrupted before we join().\n'
    '    stopping_.store(true);\n'
    '    closeStream();  // closes socketFd and resets it to -1\n'
    '    if (listenerThread.joinable()) {\n'
    '        listenerThread.join();\n'
    '    }\n'
    '    if (csock >= 0) {\n'
    '            close(csock);\n'
    '    }\n'
    '}'
)

# Fix 4: acceptClientsAsync(): exit cleanly when the socket is closed (EBADF /
# ENOTSOCK / EINVAL), and stay silent for EAGAIN / EWOULDBLOCK (non-blocking
# socket with no pending client).  Only log a genuine unexpected error.
src = src.replace(
    'void nixlMDStreamListener::acceptClientsAsync() {\n'
    '    while(true) {\n'
    '        int clientSocket = accept(socketFd, NULL, NULL);\n'
    '        if (clientSocket < 0) {\n'
    '            NIXL_PERROR << "Cannot accept client connection";\n'
    '            continue;\n'
    '        }',
    'void nixlMDStreamListener::acceptClientsAsync() {\n'
    '    while (!stopping_.load()) {\n'
    '        int clientSocket = accept(socketFd, NULL, NULL);\n'
    '        if (clientSocket < 0) {\n'
    '            // Socket was closed (shutdown signal) — exit the loop cleanly.\n'
    '            if (errno == EBADF || errno == ENOTSOCK || errno == EINVAL) {\n'
    '                break;\n'
    '            }\n'
    '            // Non-blocking socket has no pending connection yet — not an error.\n'
    '            if (errno == EAGAIN || errno == EWOULDBLOCK) {\n'
    '                continue;\n'
    '            }\n'
    '            NIXL_PERROR << "Cannot accept client connection";\n'
    '            continue;\n'
    '        }'
)

open(path, 'w').write(src)
print("metadata_stream.cpp patched successfully.")
PYEOF

# Disable err handling for ucp (will make NIXL READ slower 10x!)
echo "Applying nixl patch..."
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