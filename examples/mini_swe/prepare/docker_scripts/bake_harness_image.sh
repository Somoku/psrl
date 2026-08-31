#!/usr/bin/env bash
# bake_harness_image.sh — bake per-task harness derivative images.
#
# Each SWE task uses its OWN per-problem base image (the parquet's
# `sandbox_overrides.environment.image`), so a single global baked image is
# meaningless. This script derives, from one base image or from every unique
# image in a parquet, a derivative image
#     psrl/swebench-harness:<sha12(base-image)>
# that adds Node >= 18 + npm, the Claude Code CLI, and a seeded npm cache.
# Runner (`examples/mini_swe/runner.py`) selects the derivative when it exists
# locally and otherwise falls back to the original image + per-sandbox tarball
# install, so a missing bake never blocks training.
#
# Usage (once per worker host that creates sandboxes):
#   bash bake_harness_image.sh swebench/swesmith.x86_64.foo:latest          # one image
#   bash bake_harness_image.sh --parquet ../../data/swe_smith_py_1k/train.parquet  # all unique images
#
# Env overrides:
#   AGENT_NODE_TARBALL / AGENT_CC_TARBALL  (defaults as in the run script)
#   NPM_CACHE_DIR        host dir mounted as the bake container's npm cache
#                        (default /tmp/psrl-npm-cache) so the ~90MB platform
#                        package is downloaded once, not per baked image
#   PSRL_HARNESS_IMAGE_TAG  explicit output tag (single-image mode only)
set -euo pipefail

NODE_TARBALL="${AGENT_NODE_TARBALL:-/shared/artifacts/node-v22-linux-x64.tar.xz}"
CC_TARBALL="${AGENT_CC_TARBALL:-/shared/artifacts/anthropic-ai-claude-code-2.1.233.tgz}"
NPM_CACHE_DIR="${NPM_CACHE_DIR:-/tmp/psrl-npm-cache}"

for f in "$NODE_TARBALL" "$CC_TARBALL"; do
    [ -f "$f" ] || { echo "missing host tarball: $f (set AGENT_NODE_TARBALL / AGENT_CC_TARBALL)" >&2; exit 1; }
done
mkdir -p "$NPM_CACHE_DIR"

# Decompress the Node tarball on the host once (cached) so the bake container
# only ever runs a plain `tar -xf` — base images may lack xz-utils or a tar
# built with xz support.
NODE_TAR_MOUNT="$NODE_TARBALL"
if [[ "$NODE_TARBALL" == *.xz ]]; then
    NODE_TAR_PLAIN="${NODE_TARBALL%.xz}.plain.tar"
    if [ ! -f "$NODE_TAR_PLAIN" ]; then
        python3 -c '
import lzma, shutil, sys
src = lzma.open(sys.argv[1], "rb")
dst = open(sys.argv[2], "wb")
shutil.copyfileobj(src, dst)
src.close()
dst.close()
' "$NODE_TARBALL" "$NODE_TAR_PLAIN"
    fi
    NODE_TAR_MOUNT="$NODE_TAR_PLAIN"
fi

digest() { printf '%s' "$1" | sha256sum | cut -c1-12; }

version="$(
    python3 -c '
import json, sys, tarfile
t = tarfile.open(sys.argv[1])
pkg = json.load(t.extractfile("package/package.json"))
print(pkg["version"])
' "$CC_TARBALL"
)"

# npm registry: default to the Tencent mirror (direct, in no_proxy) so the
# platform-package fetch never depends on the container reaching npmjs.org.
NPM_REGISTRY="${NPM_REGISTRY:-https://mirrors.tencent.com/npm/}"

# Forward the host's proxy env into the bake container (same as the training
# sandboxes get via runner.py forward_env); without it npm cannot reach the
# registry and the install hangs.
_proxy_env_args=()
for k in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY NPM_REGISTRY; do
    if [ -n "${!k:-}" ]; then
        _proxy_env_args+=("-e" "$k=${!k}")
    fi
done

install_script='set -euo pipefail
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1 \
   || ! node -e "process.exit(+process.versions.node.split(\".\")[0] >= 18 ? 0 : 1)" >/dev/null 2>&1; then
  mkdir -p /opt/node22
  if tar -tf /tmp/node22.tarball >/dev/null 2>&1; then
    tar --no-same-owner -xf /tmp/node22.tarball -C /opt/node22 --strip-components=1
  elif command -v xz >/dev/null 2>&1; then
    xz -dc /tmp/node22.tarball | tar --no-same-owner -xf - -C /opt/node22 --strip-components=1
  else
    echo "Node tarball is compressed but xz is unavailable." >&2; exit 127
  fi
  ln -sf /opt/node22/bin/node /usr/local/bin/node
  ln -sf /opt/node22/bin/npm  /usr/local/bin/npm
  ln -sf /opt/node22/bin/npx  /usr/local/bin/npx
  hash -r
fi
npm install -g --prefix=/usr/local --no-audit --no-fund --prefer-offline --registry="${NPM_REGISTRY:-https://mirrors.tencent.com/npm/}" /tmp/claude-code.tgz
/usr/local/bin/claude --version'

bake_one() {
    local base_image="$1"
    local tag="${PSRL_HARNESS_IMAGE_TAG:-psrl/swebench-harness:$(digest "$base_image")}"
    if docker image inspect "$tag" >/dev/null 2>&1; then
        echo "skip (already baked): $tag"
        return 0
    fi
    if ! docker image inspect "$base_image" >/dev/null 2>&1; then
        echo "base image not present locally; attempting docker pull: $base_image" >&2
        docker pull "$base_image" || { echo "cannot obtain base image: $base_image" >&2; return 1; }
    fi
    echo "baking $base_image + node + claude-code@$version -> $tag"
    local container_name="psrl-harness-bake-$$"
    local cid
    cid="$(docker create --name "$container_name" \
        "${_proxy_env_args[@]}" \
        -v "$NODE_TAR_MOUNT:/tmp/node22.tarball:ro" \
        -v "$CC_TARBALL:/tmp/claude-code.tgz:ro" \
        -v "$NPM_CACHE_DIR:/root/.npm" \
        "$base_image" tail -f /dev/null)"
    docker start "$cid"
    local rc=0
    docker exec "$cid" bash -lc "$install_script" || rc=$?
    if [ "$rc" -ne 0 ]; then
        docker rm -f "$container_name" >/dev/null 2>&1 || true
        echo "install failed for $base_image (rc=$rc)" >&2
        return 1
    fi
    docker commit "$container_name" "$tag" >/dev/null
    docker rm -f "$container_name" >/dev/null 2>&1 || true
    echo "baked: $tag"
}

if [ "${1:-}" = "--parquet" ]; then
    parquet="${2:?--parquet requires a parquet path}"
    images="$(
        python3 -c '
import pandas as pd, sys
df = pd.read_parquet(sys.argv[1], columns=["extra_info"])
seen = set()
for ei in df["extra_info"]:
    img = (ei.get("sandbox_overrides", {}) or {}).get("environment", {}).get("image")
    if img and img not in seen:
        seen.add(img)
        print(img)
' "$parquet"
    )"
    echo "found $(wc -l <<<"$images") unique images in $parquet"
    while IFS= read -r img; do
        [ -z "$img" ] && continue
        bake_one "$img"
    done <<<"$images"
else
    base_image="${1:-${BASE_IMAGE:-}}"
    [ -n "$base_image" ] || { echo "usage: bake_harness_image.sh <image> | --parquet <path>" >&2; exit 2; }
    bake_one "$base_image"
fi
