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

(
    cd -- "${SMG_DIR}"
    cargo build --release
    python -m uv pip install -e crates/grpc_client/python/
    python -m uv pip install -e crates/psrl_state/python/
    python -m uv pip install -e grpc_servicer/

    cd -- bindings/python
    maturin develop --features vendored-openssl
)
