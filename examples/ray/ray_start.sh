#!/bin/bash
env_file="${PSRL_WORKSPACE}/env/psrl.sh"
source ${env_file}

PORT=8887                # Ray节点通信端口
DASHBOARD_PORT=8265      # Ray Dashboard端口

# 读取hostfile
mapfile -t hosts < <(echo "$NODE_IP_LIST" | sed "s/:.//g; s/,/\\n/g" | head -n $NODE_NUM)
if [ ${#hosts[@]} -eq 0 ]; then
    echo "Error: NODE_IP_LIST is empty"
    exit 1
fi

hosts=("${hosts[@]:0}")
HEAD_IP=${hosts[0]}
workers=( "${hosts[@]:1}" )

# unset http_proxy && \
# unset https_proxy && \

# 启动Head节点
echo "Starting Head node at ${HEAD_IP}"
pssh -H "${HEAD_IP}" -i \
    "source ${env_file} && \
    ray start --head \
    --port=${PORT} \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=${DASHBOARD_PORT} \
    --num-cpus=32"

# 启动Worker节点
if [ ${#workers[@]} -gt 0 ]; then
    echo "Starting ${#workers[@]} Worker nodes"
    pssh -H "${workers[*]}" -i \
        "source ${env_file} && \
        ray start --address=${HEAD_IP}:${PORT} \
        --num-cpus=32"
fi
