# Shared helpers for docker_*.sh in this directory — source from those scripts; do not execute.
# Cluster env (set directly or via script-specific positional args before use):
#   DOCKER_NODE_IPS  — comma-separated ip:gpu_count, e.g. 192.168.1.1:8,192.168.1.2:8
#   DOCKER_NODE_NUM  — optional; use only the first N hosts from the list (default: all entries)
# After docker_cluster_init succeeds:
#   DOCKER_CLUSTER_HOSTS   — bash array of plain IPs
#   DOCKER_CLUSTER_MANAGER — first host
#   DOCKER_CLUSTER_WORKERS — remaining hosts (array)

docker_cluster_ips_to_lines() {
  printf '%s' "$DOCKER_NODE_IPS" | sed "s/:.//g; s/,/\\n/g"
}

# Fills DOCKER_CLUSTER_* globals; returns 0 on success.
docker_cluster_init() {
  if [ -z "$DOCKER_NODE_IPS" ]; then
    echo "Error: DOCKER_NODE_IPS is not set"
    return 1
  fi
  if [ -z "$DOCKER_NODE_NUM" ]; then
    DOCKER_NODE_NUM=$(docker_cluster_ips_to_lines | wc -l)
  fi
  mapfile -t DOCKER_CLUSTER_HOSTS < <(docker_cluster_ips_to_lines | head -n "$DOCKER_NODE_NUM")
  if [ ${#DOCKER_CLUSTER_HOSTS[@]} -eq 0 ]; then
    echo "Error: DOCKER_NODE_IPS is empty or invalid"
    return 1
  fi
  DOCKER_CLUSTER_MANAGER="${DOCKER_CLUSTER_HOSTS[0]}"
  DOCKER_CLUSTER_WORKERS=("${DOCKER_CLUSTER_HOSTS[@]:1}")
  return 0
}

docker_cluster_hosts_csv() {
  (IFS=,; echo "${DOCKER_CLUSTER_HOSTS[*]}")
}
