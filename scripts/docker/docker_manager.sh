#!/bin/bash
# set -v

# Launch Docker on all nodes
# ./docker_manager.sh start all

# Check Docker status of node 28.49.198.139
# ./docker_manager.sh status 28.49.198.139

# Check Docker logs of node 28.49.198.139
# ./docker_manager.sh logs 28.49.198.139
#
# Requires SANDBOX_NODE_NUM and SANDBOX_NODE_IPS (env; $1/$2 are used for action/host):
#   SANDBOX_NODE_NUM=8 SANDBOX_NODE_IPS="ip1:8,ip2:8,..." ./docker_manager.sh start all

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