#!/usr/bin/env bash
#
# End-to-end PSRL training flow for the session-router/TITO mini-SWE-agent v1 loop.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "${SCRIPT_DIR}/fsdp_qwen_7b_swe_smith.sh" "$@"
