#!/bin/bash
set -xeuo pipefail

export PSRL_LOGGING_PATH=${PSRL_WORKSPACE}/psrl/unit_tests/nixl/log
export PSRL_LOGGING_LEVEL=INFO
cd ${PSRL_WORKSPACE}/psrl/unit_tests/nixl

CASE=3

# HSDP 16 GPUs Case
if [ $CASE -eq 0 ]; then
    PYTHONUNBUFFERED=1 python test_nixl_e2e.py \
        test.num_train=8 \
        test.num_gen=8 \
        test.train_engine_type=fsdp_hybrid \
        test.fsdp_hybrid.ddp_size=2 \
        test.fsdp_hybrid.fsdp_size=4 \
        test.gen.tensor_parallel_size=2 \
        test.gen.pipeline_parallel_size=2 \
        model.path=${PSRL_WORKSPACE}/models/Qwen2.5-3B-Instruct \
        2>&1 | tee test_nixl_e2e.log
fi

# Megatron 16 GPUs Case
if [ $CASE -eq 1 ]; then
    PYTHONUNBUFFERED=1 python test_nixl_e2e.py \
        test.num_train=8 \
        test.num_gen=8 \
        test.train_engine_type=megatron \
        test.megatron.tensor_model_parallel_size=4 \
        test.megatron.pipeline_model_parallel_size=1 \
        test.megatron.virtual_pipeline_model_parallel_size=1 \
        test.megatron.context_parallel_size=1 \
        test.gen.tensor_parallel_size=4 \
        test.gen.pipeline_parallel_size=1 \
        model.path=${PSRL_WORKSPACE}/models/Qwen2.5-32B \
        2>&1 | tee test_nixl_e2e.log
fi

# HSDP 64 GPUs Case
if [ $CASE -eq 2 ]; then
        PYTHONUNBUFFERED=1 python test_nixl_e2e.py \
        test.num_train=32 \
        test.num_gen=32 \
        test.train_engine_type=fsdp_hybrid \
        test.fsdp_hybrid.ddp_size=4 \
        test.fsdp_hybrid.fsdp_size=8 \
        test.gen.tensor_parallel_size=2 \
        test.gen.pipeline_parallel_size=1 \
        model.path=${PSRL_WORKSPACE}/models/Qwen2.5-3B-Instruct \
        2>&1 | tee test_nixl_e2e.log
fi

# Megatron 64 GPUs Case
if [ $CASE -eq 3 ]; then
    PYTHONUNBUFFERED=1 python test_nixl_e2e.py \
        nixl.max_pinned_temp_memory_slots=4 \
        test.num_train=32 \
        test.num_gen=32 \
        test.train_engine_type=megatron \
        test.megatron.tensor_model_parallel_size=8 \
        test.megatron.pipeline_model_parallel_size=2 \
        test.megatron.virtual_pipeline_model_parallel_size=1 \
        test.megatron.context_parallel_size=1 \
        test.gen.tensor_parallel_size=4 \
        test.gen.pipeline_parallel_size=1 \
        model.path=${PSRL_WORKSPACE}/models/Qwen2.5-32B \
        2>&1 | tee test_nixl_e2e.log
fi