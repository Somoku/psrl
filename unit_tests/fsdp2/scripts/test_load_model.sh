export MASTER_ADDR=localhost
export MASTER_PORT=12345
torchrun --nproc_per_node=2 test_load_model.py