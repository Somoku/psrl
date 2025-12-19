#!/bin/bash
# set -v

# This script load the sandbox_fusion_server docker image from tar file to all worker nodes.

NODE_NUM=
SANDBOX_SERVER_PATH=${SANDBOX_SERVER_PATH:-"$HOME/sandbox_fusion_server"}

if [ -z "$NODE_IP_LIST" ]; then
    echo "Error: NODE_IP_LIST is not set"
    exit 1
fi

if [ -z "$NODE_NUM" ]; then
    NODE_NUM=$(echo "$NODE_IP_LIST" | sed "s/:.//g; s/,/\\n/g" | wc -l)
fi

mapfile -t hosts < <(echo "$NODE_IP_LIST" | sed "s/:.//g; s/,/\\n/g" | head -n $NODE_NUM)
if [ ${#hosts[@]} -eq 0 ]; then
    echo "Error: NODE_IP_LIST is empty"
    exit 1
fi

hosts=("${hosts[@]:0}")
MANAGER=${hosts[0]}
WORKERS=("${hosts[@]:1}")

# 3. Leave Swarm
for node in "${WORKERS[@]}"; do
    pssh -t 3600 -H $node -i "cp $SANDBOX_SERVER_PATH/code_sandbox_server.tar /tmp/"
    pssh -t 3600 -H $node -i "docker load -i /tmp/code_sandbox_server.tar && rm /tmp/code_sandbox_server.tar"
done

echo "=== Copy completed ==="