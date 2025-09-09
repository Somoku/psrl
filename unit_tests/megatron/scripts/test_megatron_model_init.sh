export PSRL_LOGGING_PATH=${PSRL_WORKSPACE}/psrl/unit_tests/nixl/log
export PSRL_LOGGING_LEVEL=INFO
python test_megatron_model_init.py 2>&1 | tee test_megatron_model_init.log