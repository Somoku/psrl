export MASTER_ADDR=localhost
export MASTER_PORT=12345
torchrun --nproc_per_node=2 test_fsdp_converter.py 2>&1 | tee test_fsdp_converter_2.log