#!/bin/bash
# Install a staged sandboxd binary and restart the planes named on the command
# line. Each plane is a config path; node-250 passes two, node-97 one.
set -u

STAGED=${STAGED:-/tmp/sandboxd.new}

source /root/env/env.sh >/dev/null 2>&1

install -m 0755 "$STAGED" /usr/local/bin/sandboxd
echo "installed: $(sha256sum /usr/local/bin/sandboxd | cut -c1-16)"

for config in "$@"; do
    pkill -f "sandbox[d] -config $config" || true
done
sleep 2

for config in "$@"; do
    log="/var/log/sandboxd-$(basename "$config" .json).log"
    setsid nohup /usr/local/bin/sandboxd -config "$config" >"$log" 2>&1 </dev/null &
done
sleep 5

for config in "$@"; do
    echo "---$(basename "$config")---"
    pgrep -af "sandbox[d] -config $config" || echo "NOT RUNNING"
    tail -3 "/var/log/sandboxd-$(basename "$config" .json).log"
done
