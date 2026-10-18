#!/bin/bash
# GRPO on Qwen3-8B with Megatron, weights moved through the parameter server.
#
# This is the baseline arm of the direct-supply ablation. `qwen3_8b_megatron_direct.sh`
# is the same run with `psrl.ps_mode=nixl_direct`, so the two must stay aligned on data,
# batch sizes, step count, and validation. Compare their per-step training and
# validation metrics to see whether direct supply carries the same weights the PS path
# does.
#
# Training and rollout each take one full 8-GPU node. Start the Ray cluster with eight
# GPUs per node (`PSRL_RAY_GPUS=8 scripts/start_nixl_cluster.sh <head> <worker>`) and
# set `LOCAL_IP` to a cluster node, because `PSManager` is pinned there.
set -xeuo pipefail

PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")

HF_MODEL_PATH=${HF_MODEL_PATH:-${PSRL_PATH}/models/Qwen3-8B}
train_files=${TRAIN_FILES:-/apdcephfs_zwfy10_303541817/share_303541817/ls/data/dapo/dapo-math-17k.parquet}
test_files=${TEST_FILES:-/apdcephfs_zwfy10_303541817/share_303541817/ls/data/dapo/aime-2024.parquet}

OUTPUT_DIR=${PSRL_PATH}/output
mkdir -p "$OUTPUT_DIR"

staleness=${STALENESS:-3}
project_name=psrl_example
experiment_name=GRPO-Qwen3-8B-megatron-staleness_${staleness}
CKPTS_DIR=${OUTPUT_DIR}/ckpts/"${project_name}"/"${experiment_name}"
LOG_DIR=${OUTPUT_DIR}/logs/$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOG_DIR"
mkdir -p "$CKPTS_DIR"


train_batch_size=${TRAIN_BATCH_SIZE:-16}
rollout_N=${ROLLOUT_N:-8}

# Ten steps and validation every five. The first validation runs before any update, so
# the two arms share a common origin, and the last runs regardless of `test_freq`.
total_training_steps=${TOTAL_TRAINING_STEPS:-10}
test_freq=${TEST_FREQ:-5}

TP=${TP:-4}
PP=${PP:-1}
CP=${CP:-1}
EP=${EP:-1}
ETP=${ETP:-1}

GEN_TP=1
GEN_PP=1

VAL_TP=1
VAL_PP=1

GEN_NNODES=1
GEN_NGPUS_PER_NODE=8
GEN_INSTANCES=$(( (${GEN_NNODES} * ${GEN_NGPUS_PER_NODE}) / ( ${GEN_TP} * ${GEN_PP} ) )) # Number of generation instances
GEN_NGPUS_PER_NODE_PER_INSTANCE=$(( ${GEN_TP} * ${GEN_PP} )) # Number of GPUs per node for generation per instance

TRAIN_NNODES=1
TRAIN_NGPUS_PER_NODE=8

VAL_INSTANCES=$(( (${TRAIN_NNODES} * ${TRAIN_NGPUS_PER_NODE}) / ( ${VAL_TP} * ${VAL_PP} ) )) # Number of validation instances
VAL_NGPUS_PER_NODE_PER_INSTANCE=$(( ${VAL_TP} * ${VAL_PP} )) # Number of GPUs per node for validation per instance


# data length config
max_prompt_length=2048
max_response_length=8192

k=${k:-1}
max_token_len_per_gpu=$(( k * (max_prompt_length + max_response_length) ))

# train rollout params
train_temperature=1.2
train_top_p=1.0
train_top_k=-1

# val rollout params
val_temperature=0
val_rollot_n=1
val_top_p=1.0
val_top_k=-1
val_do_sample=False


# Refuse to start on a cluster where the roles can share a node. The same check guards
# the direct script, so both ablation arms are placed identically.
: "${LOCAL_IP:?LOCAL_IP must name a node in the Ray cluster.}"
bash "$(dirname "$0")/check_two_node_split.sh" \
    "$TRAIN_NGPUS_PER_NODE" "$GEN_INSTANCES" "$VAL_INSTANCES"


python3 -m psrl.trainer.main_ppo --config-path=./config \
    --config-name='ppo_megatron_trainer' \
    psrl.ps_manager_ip=${LOCAL_IP} \
    psrl.rollout_n=${rollout_N} \
    psrl.staleness=${staleness} \
    psrl.staleness_buffer_entries=${train_batch_size} \
    psrl.ps_mode=nixl_cpu \
    psrl.logging_path=${LOG_DIR} \
    psrl.log_prob.enable_rollout_engine_log_prob=True \
    psrl.deployment.n_rollout_instances=${GEN_INSTANCES} \
    psrl.deployment.rollout_nnodes_per_instance=1 \
    psrl.deployment.rollout_ngpus_per_node_per_instance=${GEN_NGPUS_PER_NODE_PER_INSTANCE} \
    psrl.deployment.n_validate_instances=${VAL_INSTANCES} \
    psrl.deployment.validate_nnodes_per_instance=1 \
    psrl.deployment.validate_ngpus_per_node_per_instance=${VAL_NGPUS_PER_NODE_PER_INSTANCE} \
    psrl.deployment.train_nnodes=${TRAIN_NNODES} \
    psrl.deployment.train_ngpus_per_node=${TRAIN_NGPUS_PER_NODE} \
    psrl.nixl.server_port=27837 \
    psrl.group_post_process.enable=False \
    psrl.group_post_process.name=dynamic_sampling_filter \
    \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$train_batch_size \
    data.train_max_samples=${TRAIN_MAX_SAMPLES:-512} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=8 \
    data.truncation='error' \
    data.return_multi_modal_inputs=False \
    data.shuffle=True \
    \
    train_actor_rollout_ref.nccl_timeout=6000 \
    train_actor_rollout_ref.model.path=$HF_MODEL_PATH \
    train_actor_rollout_ref.model.use_remove_padding=True \
    train_actor_rollout_ref.actor.optim.lr=1e-6 \
    train_actor_rollout_ref.actor.ppo_mini_batch_size=$train_batch_size \
    train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.actor.use_kl_loss=False \
    train_actor_rollout_ref.actor.kl_loss_coef=0.01 \
    train_actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    train_actor_rollout_ref.actor.entropy_coeff=0 \
    train_actor_rollout_ref.actor.rollout_n=$rollout_N \
    train_actor_rollout_ref.actor.use_dynamic_bsz=False \
    train_actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_token_len_per_gpu} \
    train_actor_rollout_ref.actor.megatron.param_offload=False \
    train_actor_rollout_ref.actor.megatron.optimizer_offload=True \
    train_actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    train_actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    train_actor_rollout_ref.actor.megatron.context_parallel_size=$CP \
    train_actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP \
    train_actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP \
    train_actor_rollout_ref.actor.megatron.use_mbridge=True \
    train_actor_rollout_ref.actor.megatron.use_remove_padding=True \
    \
    train_actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    train_actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    train_actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    train_actor_rollout_ref.rollout.val_kwargs.do_sample=${val_do_sample} \
    train_actor_rollout_ref.rollout.val_kwargs.n=${val_rollot_n} \
    train_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.rollout.tensor_model_parallel_size=${VAL_TP} \
    train_actor_rollout_ref.rollout.pipeline_model_parallel_size=${VAL_PP} \
    train_actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    train_actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_token_len_per_gpu} \
    train_actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    train_actor_rollout_ref.rollout.max_num_batched_tokens=20480 \
    \
    gen_actor_rollout_ref.rollout.name=vllm \
    gen_actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    gen_actor_rollout_ref.rollout.pipeline_model_parallel_size=${GEN_PP} \
    gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    gen_actor_rollout_ref.rollout.max_num_batched_tokens=20480 \
    gen_actor_rollout_ref.rollout.agent.default_agent_loop=generate_only_agent \
    gen_actor_rollout_ref.rollout.temperature=${train_temperature} \
    gen_actor_rollout_ref.rollout.top_p=${train_top_p} \
    gen_actor_rollout_ref.rollout.top_k=${train_top_k} \
    \
    reward.launch_reward_fn_async=True \
    \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.val_before_train=True \
    trainer.save_freq=-1 \
    trainer.test_freq=${test_freq} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.total_training_steps=${total_training_steps} \
    trainer.total_epochs=1 $@ 2>&1 | tee "${experiment_name}.log"
