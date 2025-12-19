#!/bin/bash
# set -v

# Launch Docker on all nodes
# ./docker-manager.sh start all

# Check Docker status of node 192.168.1.102
# ./docker-manager.sh status 192.168.1.102

# Check Docker logs of node 192.168.1.101
# ./docker-manager.sh logs 192.168.1.101

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

ACTION=${1:-status}
HOST=${2:-all}

manage_docker() {
    local host=$1
    local action=$2
    
    case $action in
        start)
            echo "Launching Docker on $host..."
            pssh -H $host -i "nohup dockerd > /var/log/docker.log 2>&1 &"
            ;;
        stop)
            echo "Stopping Docker on $host..."
            pssh -H $host -i "pkill dockerd"
            ;;
        restart)
            echo "Restarting Docker on $host..."
            pssh -H $host -i "(pkill dockerd; nohup dockerd > /var/log/docker.log 2>&1 &)"
            ;;
        status)
            echo "Checking Docker status on $host..."
            if pssh -H $host -i "docker version >/dev/null 2>&1"; then
                echo "✓ $host: Docker is running"
            else
                echo "✗ $host: Docker is not running"
            fi
            ;;
        logs)
            echo "Viewing Docker logs on $host..."
            pssh -H $host -i "tail -f /var/log/docker.log"
            ;;
        *)
            echo "Usage: $0 {start|stop|restart|status|logs} [host]"
            ;;
    esac
}

if [ "$HOST" = "all" ]; then
    for node in $MANAGER "${WORKERS[@]}"; do
        manage_docker $node $ACTION
    done
else
    manage_docker $HOST $ACTION
fi