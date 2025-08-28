export PSRL_LOGGING_PATH=${PSRL_WORKSPACE}/psrl/unit_tests/nixl/log
export PSRL_LOGGING_LEVEL=INFO
python test_nixl_e2e.py 2>&1 | tee test_nixl_e2e.log