#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SMG_DIR="${SMG_DIR:-${PROJECT_ROOT}/third_party/smg}"

if [[ ! -f "${SMG_DIR}/Cargo.toml" ]]; then
    echo "Error: SMG source directory not found: ${SMG_DIR}" >&2
    exit 1
fi

python -m uv pip uninstall -y smg smg-grpc-proto smg-grpc-servicer psrl-state-grpc-proto

# Ensure the protobuf-stub generator is pinned to a version whose bundled protoc
# stamps gencode 6.x (compatible with the protobuf 6.33 runtime). grpcio-tools >=
# 1.81 stamps gencode 7.35, which the 6.33 runtime refuses to load (VersionError).
python -m uv pip install --no-cache-dir "grpcio-tools==1.78.0"

(
    cd -- "${SMG_DIR}"
    cargo build --release
    # --no-build-isolation: generate stubs with the ambient grpcio-tools==1.78.0
    # above rather than an isolated build env that would pull grpcio-tools >= 1.81.
    python -m uv pip install --no-build-isolation -e crates/grpc_client/python/
    python -m uv pip install --no-build-isolation -e crates/psrl_state/python/
    python -m uv pip install -e grpc_servicer/

    cd -- bindings/python
    maturin develop --features vendored-openssl
)
