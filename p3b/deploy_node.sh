#!/bin/bash
# Restart one sandboxd plane from a config.
#
# The pattern is "sandboxd -config <path>", with the binary's last character
# bracketed. Both halves are load-bearing:
#
#   - the config path, because node-250 runs the control plane and its node
#     plane as two processes on one machine, and a pattern matching only the
#     binary would take the other one down;
#   - "sandboxd -config", because the invoking SSH command line contains the
#     config path too, and a pattern of the path alone kills the session
#     running this script before the restart happens.
#
# The bracket stops the pattern from matching pkill's own command line.
set -u

CONFIG=${1:?usage: deploy_node.sh <config.json> [logfile]}
LOG=${2:-/var/log/sandboxd.log}

source /root/env/env.sh >/dev/null 2>&1

pattern="sandbox[d] -config $CONFIG"

pkill -f "$pattern" || true
sleep 1

setsid nohup /usr/local/bin/sandboxd -config "$CONFIG" >"$LOG" 2>&1 </dev/null &
sleep 4

echo "---PROCESS---"
pgrep -af "$pattern" || echo "NOT RUNNING"
echo "---LOG---"
tail -8 "$LOG"
