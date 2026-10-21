#!/usr/bin/env bash
# Starts node-250 as a p3b node-plane (role=node) serving the docker backend.
# The control plane is a separate process on this host (deploy_control.json);
# SDK clients connect to its listen address, not to this node plane.
set -e
source /root/env/env.sh          # sets no_proxy for cluster-internal direct routing
export PATH=/usr/local/go/bin:$PATH
pkill -f 'deploy_node_250.json' 2>/dev/null; sleep 1
rm -f /run/sandboxd.sock 2>/dev/null
nohup /usr/local/bin/sandboxd -config /home/psrl/p3b/deploy_node_250.json     >> /tmp/sandboxd_node.log 2>&1 &
sleep 4
pgrep -f deploy_node_250.json > /dev/null && echo "node-250 node-plane started OK (port 9090)" || { echo "FAILED"; tail -5 /tmp/sandboxd_node.log; exit 1; }
