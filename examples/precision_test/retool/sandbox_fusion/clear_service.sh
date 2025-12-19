#!/bin/bash

NODE_NUM=

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

echo "=== Clean docker swarm resources ==="

# 1. Remove service
echo "1. Remove service..."
pssh -H $MANAGER -i "docker service rm sandbox-service" 2>/dev/null || echo "Service does not exist or has been removed"

# 2. Remove network
echo "2. Remove network..."
pssh -H $MANAGER -i "docker network rm sandbox-overlay" 2>/dev/null || echo "Network does not exist or has been removed"

# 3. Leave Swarm
for node in $MANAGER "${WORKERS[@]}"; do
    echo "Node $node leaving Swarm..."
    pssh -H $node -i "docker swarm leave --force" 2>/dev/null || true
done

echo "=== Clean up completed ==="