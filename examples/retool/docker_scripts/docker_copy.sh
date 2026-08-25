#!/bin/bash
# set -v

# Copy a shared-fs docker image tar to all cluster nodes and docker load.
#
# Required env vars:
#   DOCKER_NODE_IPS   — comma-separated list of ip:gpu_count pairs (e.g. 192.168.1.1:8,192.168.1.2:8)
#   DOCKER_IMAGE_DIR  — source directory containing the tar file
#   DOCKER_IMAGE_FILE — tar filename (basename)
#
# Optional env vars:
#   DOCKER_NODE_NUM      — limit to first N nodes (default: all)
#   DOCKER_IMAGE_TAG — if set, retag the loaded image to this repo:tag on every node
#
# Example:
#   DOCKER_NODE_IPS=192.168.1.1:8,192.168.1.2:8 DOCKER_NODE_NUM=8 \
#     DOCKER_IMAGE_DIR=/path/to/dir DOCKER_IMAGE_FILE=my.tar \
#     DOCKER_IMAGE_TAG=code_sandbox:server \
#     ./docker_copy.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=docker_common.sh
. "$SCRIPT_DIR/docker_common.sh"

if [ -z "$DOCKER_NODE_IPS" ] || [ -z "$DOCKER_IMAGE_DIR" ] || [ -z "$DOCKER_IMAGE_FILE" ]; then
    echo "[docker_copy.sh] Required env vars not set."
    echo "Usage:"
    echo "  DOCKER_NODE_IPS=ip1:8,ip2:8 DOCKER_NODE_NUM=8 \\"
    echo "    DOCKER_IMAGE_DIR=/path/to/dir DOCKER_IMAGE_FILE=my.tar \\"
    echo "    ./docker_copy.sh"
    exit 1
fi

docker_cluster_init || exit 1
# pssh -H expects space-separated hosts, not comma-separated
hosts_str="${DOCKER_CLUSTER_HOSTS[*]}"

echo "=== Copying tar to all nodes in parallel ==="
pssh -t 3600 -H "$hosts_str" -i "cp $DOCKER_IMAGE_DIR/$DOCKER_IMAGE_FILE /tmp/"

echo "=== Loading docker image on all nodes in parallel ==="
# Remote script: load tar, extract the image ref from docker load output, retag, cleanup.
# docker load prints either:
#   "Loaded image: repo:tag"         — tagged image
#   "Loaded image ID: sha256:abc..."  — untagged / <none>:<none>
# We use that to retag without needing the caller to know the original tag.
REMOTE_CMD='
set -e
TAR=/tmp/'"$DOCKER_IMAGE_FILE"'
NEW_TAG='"$DOCKER_IMAGE_TAG"'

load_out=$(docker load -i "$TAR")
echo "$load_out"
rm "$TAR"

if [ -n "$NEW_TAG" ]; then
    src=$(printf "%s\n" "$load_out" | sed -n "s/^Loaded image: //p; s/^Loaded image ID: //p" | tail -n 1)
    [ -n "$src" ] || { echo "[docker_copy.sh] Could not detect loaded image ref"; exit 1; }
    docker tag "$src" "$NEW_TAG"
    echo "Tagged $src -> $NEW_TAG"
fi
'
pssh -t 3600 -H "$hosts_str" -i "$REMOTE_CMD"

echo "=== Copy completed ==="
