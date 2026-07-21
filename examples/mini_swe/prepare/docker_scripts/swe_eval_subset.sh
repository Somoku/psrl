#!/usr/bin/env bash
# Prefetch Docker images for the SWE-Gym Subset (100 instances, 2 repos).
# Much faster than the full 2438 — only ~100 unique images to pull.
#
# Run from this directory (examples/mini_swe/prepare/docker_scripts/):
#   bash swe_gym_subset.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPARE_DIR="$(dirname "$SCRIPT_DIR")"

WORKERS=${SWE_PREFETCH_WORKERS:-1}



bash "$SCRIPT_DIR/load_all_nodes.sh" \
    --hosts /jizhicfs/lhy/hosts/32GPUs \
    --image-dir /jizhicfs/lhy/docker_images/swe_eval
