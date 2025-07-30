export PSRL_LOGGING_PATH=/jizhicfs/lhy/psrl/unit_tests/nixl/log
export PSRL_LOGGING_LEVEL=INFO
python3 test_nixl_e2e.py 2>&1 | tee test_nixl_e2e.log