#!/bin/bash
set -xeuo pipefail

PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")
export PSRL_LOGGING_PATH=${PSRL_PATH}/tests/nixl/log
export PSRL_LOGGING_LEVEL=INFO
cd ${PSRL_PATH}/tests/nixl

# Pick a CASE via env override, e.g. `CASE=4 bash scripts/test_nixl_e2e.sh`.
# Default is the original Qwen2.5-3B HSDP smoke case.
CASE=${CASE:-0}

# NOTE(lhy): HSDP/FSDP precision is not aligned with megatron, because we
# use FSDP1 in the unit test. The all-ones init / pull-equality verification
# does not depend on training-side precision, so this is fine.

# CASE 0 — Qwen2.5-3B-Instruct, HSDP train (1×8) + vLLM gen (TP=2 PP=2), 16 GPU
if [ $CASE -eq 0 ]; then
    PYTHONUNBUFFERED=1 python test_nixl_e2e.py \
        test.num_train=8 \
        test.num_gen=8 \
        test.train_engine_type=fsdp_hybrid \
        test.fsdp_hybrid.ddp_size=1 \
        test.fsdp_hybrid.fsdp_size=8 \
        test.gen.tensor_parallel_size=2 \
        test.gen.pipeline_parallel_size=2 \
        model.path=${PSRL_WORKSPACE}/models/Qwen2.5-3B-Instruct \
        2>&1 | tee test_nixl_e2e.log
fi

# CASE 1 — Qwen2.5-32B, Megatron train (TP=4 PP=2) + vLLM gen (TP=4), 16 GPU
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

# CASE 2 — Qwen2.5-3B-Instruct, HSDP train (4×8) + vLLM gen (TP=2), 64 GPU
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

# CASE 3 — Qwen2.5-32B, Megatron train (TP=8 PP=2) + vLLM gen (TP=4), 64 GPU
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

# CASE 4 — Qwen3-1.7B (dense), HSDP train (2×4) + vLLM gen (TP=2 DP=4), 8 GPU.
# Targets the Qwen3 build_weight_layout(): qkv_proj split + gate_up_proj split,
# no MoE / no GDN.
if [ $CASE -eq 4 ]; then
    PYTHONUNBUFFERED=1 python test_nixl_e2e.py \
        test.num_train=8 \
        test.num_gen=8 \
        test.train_engine_type=fsdp_hybrid \
        test.fsdp_hybrid.ddp_size=2 \
        test.fsdp_hybrid.fsdp_size=4 \
        test.gen.tensor_parallel_size=2 \
        test.gen.pipeline_parallel_size=1 \
        test.gen.expert_parallel_size=1 \
        test.gen.data_parallel_size=4 \
        model.path=${PSRL_WORKSPACE}/models/Qwen3-1.7B \
        model.train_dtype=float32 \
        2>&1 | tee test_nixl_e2e.log
fi

# CASE 5 — Qwen3-30B-A3B-Instruct-2507 (Qwen3-MoE, 128 routed experts).
# Megatron train: TP=2 PP=1 DP=4 EP=4 (8 GPU); vLLM gen: TP=4 EP=4. Targets
# the fused_moe transform inside qwen3_moe.build_weight_layout() alongside
# qkv_proj / gate_up_proj splits, with EP > 1 on both sides.
if [ $CASE -eq 5 ]; then
    PYTHONUNBUFFERED=1 python test_nixl_e2e.py \
        test.num_train=8 \
        test.num_gen=8 \
        test.train_engine_type=megatron \
        test.megatron.tensor_model_parallel_size=2 \
        test.megatron.pipeline_model_parallel_size=1 \
        test.megatron.virtual_pipeline_model_parallel_size=1 \
        test.megatron.context_parallel_size=1 \
        test.megatron.expert_model_parallel_size=4 \
        test.megatron.expert_tensor_parallel_size=1 \
        test.gen.tensor_parallel_size=4 \
        test.gen.pipeline_parallel_size=1 \
        test.gen.expert_parallel_size=4 \
        model.path=${PSRL_WORKSPACE}/models/Qwen3-30B-A3B-Instruct-2507 \
        model.train_dtype=bfloat16 \
        2>&1 | tee test_nixl_e2e.log
fi

# CASE 6 — Qwen3.5-4B (multimodal Qwen3.5 dense, hybrid full + linear attn).
# Megatron train: TP=2 PP=2 DP=2 (8 GPU); vLLM gen: TP=2 DP=4. Targets the
# qwen3_5 build_weight_layout(): qkv_proj + gate_up_proj + in_proj_qkvz /
# in_proj_ba on the GDN layers.
if [ $CASE -eq 6 ]; then
    PYTHONUNBUFFERED=1 python test_nixl_e2e.py \
        test.num_train=8 \
        test.num_gen=8 \
        test.train_engine_type=megatron \
        test.megatron.tensor_model_parallel_size=2 \
        test.megatron.pipeline_model_parallel_size=2 \
        test.megatron.virtual_pipeline_model_parallel_size=1 \
        test.megatron.context_parallel_size=1 \
        test.megatron.expert_model_parallel_size=1 \
        test.megatron.expert_tensor_parallel_size=1 \
        test.gen.tensor_parallel_size=2 \
        test.gen.pipeline_parallel_size=1 \
        test.gen.data_parallel_size=4 \
        model.path=${PSRL_WORKSPACE}/models/Qwen3.5-4B \
        model.trust_remote_code=true \
        model.train_dtype=bfloat16 \
        2>&1 | tee test_nixl_e2e.log
fi

# CASE 7 — Qwen3.5-35B-A3B (multimodal Qwen3.5-MoE, 256 experts).
# Megatron train: TP=4 PP=1 DP=2 EP=2 (8 GPU); vLLM gen: TP=8 EP=8. Targets
# fused_moe + qkv_proj + GDN with EP active.
if [ $CASE -eq 7 ]; then
    PYTHONUNBUFFERED=1 python test_nixl_e2e.py \
        test.num_train=8 \
        test.num_gen=8 \
        test.train_engine_type=megatron \
        test.megatron.tensor_model_parallel_size=1 \
        test.megatron.pipeline_model_parallel_size=1 \
        test.megatron.virtual_pipeline_model_parallel_size=1 \
        test.megatron.context_parallel_size=1 \
        test.megatron.expert_model_parallel_size=8 \
        test.megatron.expert_tensor_parallel_size=1 \
        test.gen.tensor_parallel_size=4 \
        test.gen.pipeline_parallel_size=1 \
        test.gen.expert_parallel_size=4 \
        test.gen.data_parallel_size=2 \
        model.path=${PSRL_WORKSPACE}/models/Qwen3.5-35B-A3B \
        model.trust_remote_code=true \
        model.train_dtype=bfloat16 \
        2>&1 | tee test_nixl_e2e.log
fi

# CASE 8 — Moonlight-16B-A3B (DeepseekV3 architecture via auto_map).
# Megatron train: TP=2 PP=1 DP=4 EP=4 (8 GPU); vLLM gen: TP=4 DP=2 EP=4. Targets
# deepseek_v2 build_weight_layout: MLA (q + kv_a + kv_b) + fused_moe with
# shared experts, EP > 1 on both sides. trust_remote_code routes the auto_map
# back to AutoModelForCausalLM.
if [ $CASE -eq 8 ]; then
    PYTHONUNBUFFERED=1 python test_nixl_e2e.py \
        test.num_train=8 \
        test.num_gen=8 \
        test.train_engine_type=megatron \
        test.megatron.tensor_model_parallel_size=2 \
        test.megatron.pipeline_model_parallel_size=1 \
        test.megatron.virtual_pipeline_model_parallel_size=1 \
        test.megatron.context_parallel_size=1 \
        test.megatron.expert_model_parallel_size=4 \
        test.megatron.expert_tensor_parallel_size=1 \
        test.gen.tensor_parallel_size=4 \
        test.gen.pipeline_parallel_size=1 \
        test.gen.data_parallel_size=2 \
        test.gen.expert_parallel_size=4 \
        model.path=${PSRL_WORKSPACE}/models/Moonlight-16B-A3B \
        model.trust_remote_code=true \
        model.train_dtype=bfloat16 \
        2>&1 | tee test_nixl_e2e.log
fi
