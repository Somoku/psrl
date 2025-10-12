# First run the target script
# Then run the initiator script
# export UCX_LOG_LEVEL=debug
export UCX_NET_DEVICES="bond1,bond2,bond3,bond4,bond5,bond6,bond7,bond8,mlx5_bond_1:1,mlx5_bond_4:1,mlx5_bond_3:1,mlx5_bond_2:1,mlx5_bond_7:1,mlx5_bond_6:1,mlx5_bond_8:1,mlx5_bond_5:1"
python test_send_recv.py --ip 28.49.57.252 --mode target --cuda 0