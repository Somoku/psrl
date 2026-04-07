#!/bin/bash
# set -v

# Docker daemon on cluster nodes (pssh).
#
# Cluster: DOCKER_NODE_IPS [, DOCKER_NODE_NUM] — env or positional (see below).
#
# Invocation (two styles; both use the same cluster env names):
#   A) Env cluster + action:
#        DOCKER_NODE_IPS='ip1:8,ip2:8' [DOCKER_NODE_NUM=8] ./docker_manager.sh start all
#   B) Positional cluster + action (same order as docker_copy.sh for the first two args):
#        ./docker_manager.sh 'ip1:8,ip2:8' [num_or_action] [action] [host]
#      If $2 is start|stop|restart|status|logs, node count is inferred (all IPs); then $2=action, $3=host.
#      Otherwise: $2=DOCKER_NODE_NUM (optional), $3=action, $4=host.
#
# Examples:
#   ./docker_manager.sh status 28.49.198.139
#   ./docker_manager.sh '28.49.196.175:8,28.49.196.77:8' start all
#   ./docker_manager.sh '28.49.196.175:8,28.49.196.77:8' 8 start all

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=docker_common.sh
. "$SCRIPT_DIR/docker_common.sh"

case "${1:-}" in
  start|stop|restart|status|logs)
    ACTION=$1
    HOST=${2:-all}
    ;;
  *)
    if [ -n "$1" ]; then
      DOCKER_NODE_IPS="$1"
      case "${2:-}" in
        start|stop|restart|status|logs)
          ACTION=$2
          HOST=${3:-all}
          ;;
        *)
          [ -n "$2" ] && DOCKER_NODE_NUM="$2"
          ACTION=${3:-status}
          HOST=${4:-all}
          ;;
      esac
    else
      ACTION=status
      HOST=all
    fi
    ;;
esac

docker_cluster_init || exit 1
MANAGER=$DOCKER_CLUSTER_MANAGER
WORKERS=("${DOCKER_CLUSTER_WORKERS[@]}")

manage_docker() {
  local host=$1
  local action=$2

  case $action in
    start)
      echo "Launching Docker on $host..."
      pssh -H "$host" -i "nohup dockerd > /var/log/docker.log 2>&1 &"
      ;;
    stop)
      echo "Stopping Docker on $host..."
      pssh -H "$host" -i "pkill dockerd"
      ;;
    restart)
      echo "Restarting Docker on $host..."
      pssh -H "$host" -i "(pkill dockerd; nohup dockerd > /var/log/docker.log 2>&1 &)"
      ;;
    status)
      echo "Checking Docker status on $host..."
      if pssh -H "$host" -i "docker version >/dev/null 2>&1"; then
        echo "✓ $host: Docker is running"
      else
        echo "✗ $host: Docker is not running"
      fi
      ;;
    logs)
      echo "Viewing Docker logs on $host..."
      pssh -H "$host" -i "tail -f /var/log/docker.log"
      ;;
    *)
      echo "Usage: $0 {start|stop|restart|status|logs} [host]"
      echo "   or: $0 <DOCKER_NODE_IPS> [DOCKER_NODE_NUM] {start|stop|restart|status|logs} [host]"
      echo "   or: $0 <DOCKER_NODE_IPS> {start|stop|restart|status|logs} [host]  (num = all IPs)"
      ;;
  esac
}

if [ "$HOST" = "all" ]; then
  for node in "$MANAGER" "${WORKERS[@]}"; do
    manage_docker "$node" "$ACTION"
  done
else
  manage_docker "$HOST" "$ACTION"
fi
