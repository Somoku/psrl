#!/bin/bash
# Requires SANDBOX_NODE_NUM and SANDBOX_NODE_IPS (env or args: $1=SANDBOX_NODE_IPS, $2=SANDBOX_NODE_NUM).

[ -n "$1" ] && SANDBOX_NODE_IPS="$1"
[ -n "$2" ] && SANDBOX_NODE_NUM="$2"

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