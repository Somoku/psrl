#!/bin/bash
# set -v

# Copy a shared-fs docker image tar to all cluster nodes and docker load.
#
# Positional (override env when non-empty; use "" only to skip a slot intentionally):
#   $1=DOCKER_NODE_IPS   $2=DOCKER_NODE_NUM   $3=DOCKER_IMAGE_DIR   $4=DOCKER_IMAGE_FILE
# Env (same names): DOCKER_NODE_IPS, DOCKER_NODE_NUM, DOCKER_IMAGE_DIR, DOCKER_IMAGE_FILE
#
# Example:
#   ./docker_copy.sh '28.49.196.175:8,28.49.196.77:8' 8 /jizhicfs/.../SandboxFusion code_sandbox_server.tar

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=docker_common.sh
. "$SCRIPT_DIR/docker_common.sh"

[ -n "$1" ] && DOCKER_NODE_IPS="$1"
[ -n "$2" ] && DOCKER_NODE_NUM="$2"
[ -n "$3" ] && DOCKER_IMAGE_DIR="$3"
[ -n "$4" ] && DOCKER_IMAGE_FILE="$4"

DOCKER_IMAGE_DIR=${DOCKER_IMAGE_DIR:-"/jizhicfs/johnnyslin/sandbox-docker/SandboxFusion"}
DOCKER_IMAGE_FILE=${DOCKER_IMAGE_FILE:-"code_sandbox_server.tar"}

docker_cluster_init || exit 1
hosts_str=$(docker_cluster_hosts_csv)

echo "=== Copying tar to all nodes in parallel ==="
pssh -t 3600 -H "$hosts_str" -i "cp $DOCKER_IMAGE_DIR/$DOCKER_IMAGE_FILE /tmp/"
echo "=== Loading docker image on all nodes in parallel ==="
pssh -t 3600 -H "$hosts_str" -i "docker load -i /tmp/$DOCKER_IMAGE_FILE && rm /tmp/$DOCKER_IMAGE_FILE"

echo "=== Copy completed ==="
