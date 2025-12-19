#!/bin/bash
# set -v

NODE_NUM=
IMAGE_NAME=code_sandbox:server

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

start_docker() {
    local host=$1
    echo "Launching Docker on $host..."
    
    pssh -H $host -i "
        pkill dockerd || true
        export http_proxy=
        export https_proxy=
        export no_proxy=
        nohup dockerd > /var/log/docker.log 2>&1 &
        sleep 3
        
        for i in {1..30}; do
            if docker version >/dev/null 2>&1; then
                echo 'Docker launched successfully'
                break
            fi
            sleep 3
            if [ \$i -eq 30 ]; then
                echo 'Error: Docker failed to start on $host'
                exit 1
            fi
        done
    "
}

check_docker() {
    local host=$1
    if pssh -H $host -i "docker version" >/dev/null 2>&1; then
        echo "$host: Docker is running"
        return 0
    else
        echo "$host: Docker is not running"
        return 1
    fi
}

echo "0. Checking Docker on all nodes..."
for node in $MANAGER "${WORKERS[@]}"; do
    if check_docker $node; then
        echo "$node: Docker is already running, skipping restart"
    else
        echo "$node: Starting Docker..."
        start_docker $node
    fi
done

echo "1. Initializing Swarm on manager..."
pssh -H $MANAGER -i "docker swarm init --advertise-addr $MANAGER"

# Get join token
TOKEN=$(ssh $MANAGER "docker swarm join-token worker -q")
JOIN_CMD="docker swarm join --token $TOKEN $MANAGER:2377"

echo "2. Adding workers to swarm..."
echo $JOIN_CMD
for worker in "${WORKERS[@]}"; do
    echo "Adding $worker..."
    pssh -H $worker -i "$JOIN_CMD"
done

for worker in $MANAGER "${WORKERS[@]}"; do
    echo "Removing service on $worker..."
    pssh -H $worker -i "docker service rm sandbox-service 2>/dev/null"
done

echo "3. Creating overlay network..."
pssh -H $MANAGER -i "docker network rm sandbox-overlay" 2>/dev/null || echo "Network does not exist or has been removed"
sleep 5
pssh -H $MANAGER -i "docker network create --driver=overlay --attachable sandbox-overlay"

echo "4. Deploying sandbox service..."
pssh -H $MANAGER -i "docker service create \
    --name sandbox-service \
    --network sandbox-overlay \
    --replicas $NODE_NUM \
    --publish published=8080,target=8080,mode=host \
    --cap-add ALL \
    --env ALLOWED_HOSTS=* \
    --env CSRF_TRUSTED_ORIGINS=* \
    --env FORWARDED_ALLOW_IPS=* \
    --env DISABLE_HOST_CHECK=true \
    $IMAGE_NAME \
    make run-online HOST=0.0.0.0"

echo "5. Validating deployment..."
pssh -H $MANAGER -i "
    echo '=== Service status ==='
    docker service ls
    echo '=== Node status ==='
    docker node ls
    echo '=== Service detail ==='
    docker service ps sandbox-service
"
