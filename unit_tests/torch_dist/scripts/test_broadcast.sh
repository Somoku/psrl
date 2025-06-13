export MASTER_ADDR=localhost
export MASTER_PORT=12345
torchrun --nproc_per_node=5 test_broadcast.py