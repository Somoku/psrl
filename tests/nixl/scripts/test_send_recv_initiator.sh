#!/bin/bash
# Run the initiator side of the NIXL send/recv test.
# First start the target (test_send_recv_target.sh) on the remote node,
# then run this initiator script.
#
# Required env vars:
#   SEND_IP   — IP of the local (initiator) node
#   RECV_IP   — IP of the remote (target) node
#
# Usage:
#   SEND_IP=192.168.1.1 RECV_IP=192.168.1.2 bash tests/nixl/scripts/test_send_recv_initiator.sh

set -xeuo pipefail

PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")

GPU_ID=-1
IP=${SEND_IP}

# Uncomment to enable debug logging:
# export UCX_LOG_LEVEL=debug
# export NIXL_LOG_LEVEL=debug
export UCX_NET_DEVICES="bond1,bond2,bond3,bond4,bond5,bond6,bond7,bond8,mlx5_bond_1:1,mlx5_bond_4:1,mlx5_bond_3:1,mlx5_bond_2:1,mlx5_bond_7:1,mlx5_bond_6:1,mlx5_bond_8:1,mlx5_bond_5:1"
echo "UCX_NET_DEVICES: ${UCX_NET_DEVICES}"

# Run raw tensor send/recv test (initiator side)
PYTHONUNBUFFERED=1 python ${PSRL_PATH}/tests/nixl/test_send_recv.py --ip ${IP} --mode initiator --cuda ${GPU_ID}

# To test with model weights instead, comment the line above and uncomment:
# PYTHONUNBUFFERED=1 python ${PSRL_PATH}/tests/nixl/test_send_recv_model.py \
#     --ip ${IP} --mode initiator --cuda ${GPU_ID} \
#     --model_path ${PSRL_WORKSPACE}/models/Qwen2.5-Math-7B
