#!/bin/bash
# Provision one node with the UCX 1.22 + DOCA GPUNetIO stack that nixl_ep needs.
#
# Why these specific versions (see memory/nixl-ep-requires-ucx122-gpunetio.md):
#   * nixl_ep's kernels use the UCP GPU device API (ucp_device_local_mem_list_h),
#     which first exists in UCX 1.22.0. UCX 1.20.x fails with "identifier undefined".
#   * They also reach uct/ib/mlx5/gdaki/gdaki.cuh, which UCX installs only under
#     `if HAVE_GPUNETIO`. A UCX built without it still installs uct_device_impl.h,
#     which includes that header unconditionally -- so the install is self-inconsistent.
#   * 1.22 vendors DOCA GPUNetIO as a submodule, so no system-DOCA version matching.
#
# Idempotent: exits early if gdaki.cuh is already present under a 1.22+ UCX.
set -euo pipefail

LOG=/tmp/provision_nixl_ep.log
exec > >(tee -a "$LOG") 2>&1
echo "=== provision start $(date -Is) on $(hostname) ==="

UCX_VER=v1.22.0
SRC=/tmp/ucx_122_provision
JOBS=${JOBS:-32}

have_ucx=$(ucx_info -v 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)
if [ -f /usr/include/uct/ib/mlx5/gdaki/gdaki.cuh ] && [ "${have_ucx}" = "1.22.0" ]; then
    echo "already provisioned: ucx=${have_ucx}, gdaki.cuh present. nothing to do."
    exit 0
fi
echo "current ucx=${have_ucx:-none}; proceeding"

# --- 1. DOCA GPUNetIO SDK -------------------------------------------------
# Only the SDK headers/libs are needed; doca-all would replace the whole RDMA stack.
if [ ! -f /opt/mellanox/doca/include/doca_gpunetio.h ]; then
    echo "--- installing doca-sdk-gpunetio ---"
    cat > /etc/yum.repos.d/doca.repo <<'REPO'
[doca]
name=NVIDIA DOCA
baseurl=https://linux.mellanox.com/public/repo/doca/latest/rhel8/x86_64/
enabled=1
gpgcheck=1
gpgkey=https://linux.mellanox.com/public/repo/doca/latest/rhel8/x86_64/RPM-GPG-KEY-doca
REPO
    yum --disablerepo="*" --enablerepo="doca" install -y \
        doca-sdk-gpunetio doca-sdk-gpunetio-devel
else
    echo "--- doca gpunetio already present ---"
fi

# --- 2. UCX 1.22 with the vendored GPUNetIO submodule ---------------------
echo "--- building ucx ${UCX_VER} ---"
rm -rf "$SRC"
git clone --depth 1 -b "$UCX_VER" https://github.com/openucx/ucx.git "$SRC"
cd "$SRC"
# GDAKI lives behind HAVE_GPUNETIO and its headers come from this submodule.
git submodule update --init --recursive

./autogen.sh
# Flags mirror the pre-existing build on this cluster so nothing else regresses.
./configure \
    --prefix=/usr \
    --disable-logging --disable-debug --disable-assertions --disable-params-check \
    --enable-mt --enable-shared --enable-optimizations --enable-devel-headers \
    --enable-cma --disable-static --disable-doxygen-doc --without-go --without-java \
    --with-verbs --with-rdmacm --with-dm --without-rocm \
    --with-cuda=/usr/local/cuda

# Fail loudly if GPUNetIO was not picked up: a silent "no" here produces a UCX that
# installs uct_device_impl.h without the gdaki.cuh it includes.
if ! grep -q 'MLX5 modules:.*gda' config.log; then
    echo "ERROR: configure did not enable the gda module; GPUNetIO was not detected." >&2
    grep -iE 'gpunetio|gdaki|MLX5 modules' config.log | tail -20 >&2
    exit 1
fi
echo "--- gda module enabled, building ---"

make -j"$JOBS"
make -j"$JOBS" install
ldconfig

# --- 3. Verify ------------------------------------------------------------
new_ucx=$(ucx_info -v 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
[ -f /usr/include/uct/ib/mlx5/gdaki/gdaki.cuh ] || { echo "ERROR: gdaki.cuh missing after install" >&2; exit 1; }
grep -ql ucp_device_local_mem_list_h /usr/include/ucp/api/device/ucp_device_impl.h \
    || { echo "ERROR: UCP GPU device API missing after install" >&2; exit 1; }
echo "=== provision OK: ucx=${new_ucx}, gdaki.cuh present, ucp device API present ==="
