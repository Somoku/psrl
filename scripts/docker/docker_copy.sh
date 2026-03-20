#!/bin/bash
# set -v

# This script load the sandbox_fusion_server docker image from tar file to all worker nodes.
# Requires SANDBOX_NODE_NUM and SANDBOX_NODE_IPS (env or args: $1=SANDBOX_NODE_IPS, $2=SANDBOX_NODE_NUM).

[ -n "$1" ] && SANDBOX_NODE_IPS="$1"
[ -n "$2" ] && SANDBOX_NODE_NUM="$2"

SANDBOX_SERVER_PATH=${SANDBOX_SERVER_PATH:-"/jizhicfs/johnnyslin/sandbox-docker/SandboxFusion"}

if [ -z "$SANDBOX_NODE_IPS" ]; then
    echo "Error: SANDBOX_NODE_IPS is not set"
    exit 1
fi

if [ -z "$SANDBOX_NODE_NUM" ]; then
    SANDBOX_NODE_NUM=$(echo "$SANDBOX_NODE_IPS" | sed "s/:.//g; s/,/\\n/g" | wc -l)
fi

mapfile -t hosts < <(echo "$SANDBOX_NODE_IPS" | sed "s/:.//g; s/,/\\n/g" | head -n $SANDBOX_NODE_NUM)
if [ ${#hosts[@]} -eq 0 ]; then
    echo "Error: SANDBOX_NODE_IPS is empty or invalid"
    exit 1
fi

MANAGER=${hosts[0]}
WORKERS=("${hosts[@]:0}")

# Build comma-separated host list for parallel pssh
hosts_str=$(IFS=,; echo "${hosts[*]}")

# 3. Copy tar and load image on all nodes in parallel (pssh fans out to all hosts concurrently)
echo "=== Copying tar to all nodes in parallel ==="
pssh -t 3600 -H "$hosts_str" -i "cp $SANDBOX_SERVER_PATH/code_sandbox_server.tar /tmp/"
echo "=== Loading docker image on all nodes in parallel ==="
pssh -t 3600 -H "$hosts_str" -i "docker load -i /tmp/code_sandbox_server.tar && rm /tmp/code_sandbox_server.tar"

echo "=== Copy completed ==="