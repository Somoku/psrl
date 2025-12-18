# First run the target script
# Then run the initiator script
# export UCX_LOG_LEVEL=debug

PSRL_WORKSPACE=${PSRL_WORKSPACE}/psrl
CONDA_ENV_FILE=${PSRL_WORKSPACE}/../activate
CONDA_ENV_NAME=psrl-lhy-new
GPU_ID=-1
IP=${SEND_IP}

source ${CONDA_ENV_FILE} 
conda activate ${CONDA_ENV_NAME} 
# export UCX_LOG_LEVEL=debug
# export NIXL_LOG_LEVEL=debug
export UCX_NET_DEVICES="bond1,bond2,bond3,bond4,bond5,bond6,bond7,bond8,mlx5_bond_1:1,mlx5_bond_4:1,mlx5_bond_3:1,mlx5_bond_2:1,mlx5_bond_7:1,mlx5_bond_6:1,mlx5_bond_8:1,mlx5_bond_5:1" 
echo "UCX_NET_DEVICES: ${UCX_NET_DEVICES}"

PYTHONUNBUFFERED=1 python ${PSRL_WORKSPACE}/psrl/unit_tests/nixl/test_send_recv.py --ip ${IP} --mode initiator --cuda ${GPU_ID}
# PYTHONUNBUFFERED=1 python ${PSRL_WORKSPACE}/psrl/unit_tests/nixl/test_send_recv_model.py --ip ${IP} --mode initiator --cuda ${GPU_ID} --model_path ${PSRL_WORKSPACE}/models/Qwen2.5-Math-7B
