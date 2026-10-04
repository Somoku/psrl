#!/usr/bin/env bash
# Starts node-250 as a p3b node-plane (role=node) serving docker backend.
# Cluster control lives on node-97; SDK clients connect there.
set -e
source /root/env/env.sh          # sets no_proxy for cluster-internal direct routing
export PATH=/usr/local/go/bin:$PATH
pkill -f 'deploy_node_250.json' 2>/dev/null; sleep 1
rm -f /run/sandboxd.sock 2>/dev/null
nohup /usr/local/bin/sandboxd -config /home/psrl/p3b/deploy_node_250.json     >> /tmp/sandboxd_node.log 2>&1 &
sleep 4
pgrep -f deploy_node_250.json > /dev/null && echo "node-250 node-plane started OK (port 36001)" || { echo "FAILED"; tail -5 /tmp/sandboxd_node.log; exit 1; }
