#!/bin/bash
# Trainer-direct supply with rollout and training colocated on the same GPUs.
#
# Same run as `qwen3_8b_megatron_direct.sh` -- data, batch sizes, parallel plan, step
# count and validation schedule -- with `psrl.colocate_rollout_and_train=True`. That is
# the variable under test here: everything else must stay aligned so a divergence is
# attributable to colocation rather than to the workload.
#
# Deployment: one 8-GPU node. Train (TP4 -> 2 DP x 4 TP) and 8 vLLM TP1 instances stay
# resident on the same cards, one worker of each role per GPU; the 8 validate instances
# sleep until the validation switch and time-share with the trainer. The bundle
# fractions in `psrl.yaml` are what make Ray pair them instead of letting two workers
# of one role claim a card.
#
# Because the vLLM servers share HBM with the trainer, they use a lower
# `gpu_memory_utilization` than the disaggregate script: the checkpoint needs room for
# the trainer's parameters and activations. The fractions are scheduling shares, not
# memory splits; the memory bound is the utilization setting.
set -xeuo pipefail

PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")

HF_MODEL_PATH=${HF_MODEL_PATH:-${PSRL_PATH}/models/Qwen3-8B}
# GSM8K, the dataset the direct-supply precision baseline was measured on. Point both
# arms at the same files when comparing metrics.
train_files=${TRAIN_FILES:-/apdcephfs_zwfy10_303541817/share_303541817/ls/data/gsm8k/train.parquet}
test_files=${TEST_FILES:-/apdcephfs_zwfy10_303541817/share_303541817/ls/data/gsm8k/test.parquet}

OUTPUT_DIR=${PSRL_PATH}/output
mkdir -p "$OUTPUT_DIR"

staleness=${STALENESS:-3}
project_name=psrl_example
experiment_name=GRPO-Qwen3-8B-megatron-direct-colocate-staleness_${staleness}
CKPTS_DIR=${OUTPUT_DIR}/ckpts/"${project_name}"/"${experiment_name}"
LOG_DIR=${OUTPUT_DIR}/logs/$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOG_DIR"
mkdir -p "$CKPTS_DIR"

train_batch_size=${TRAIN_BATCH_SIZE:-16}
rollout_N=${ROLLOUT_N:-8}

total_training_steps=${TOTAL_TRAINING_STEPS:-10}
test_freq=${TEST_FREQ:-5}

TP=${TP:-4}
PP=${PP:-1}
CP=${CP:-1}
EP=${EP:-1}
ETP=${ETP:-1}

GEN_TP=${GEN_TP:-1}
GEN_PP=1

VAL_TP=1
VAL_PP=1

# One node; train, rollout and validate share its GPUs. `total_nnodes=1` also makes
# the launcher reserve every other node's GPU slots, so a larger cluster cannot spread
# the workers and silently turn the run into a disaggregate one.
TRAIN_NNODES=1
TRAIN_NGPUS_PER_NODE=8

GEN_NNODES=1
GEN_NGPUS_PER_NODE=8
GEN_INSTANCES=$(( (${GEN_NNODES} * ${GEN_NGPUS_PER_NODE}) / ( ${GEN_TP} * ${GEN_PP} ) ))
GEN_NGPUS_PER_NODE_PER_INSTANCE=$(( ${GEN_TP} * ${GEN_PP} ))

VAL_INSTANCES=$(( (${TRAIN_NNODES} * ${TRAIN_NGPUS_PER_NODE}) / ( ${VAL_TP} * ${VAL_PP} ) ))
VAL_NGPUS_PER_NODE_PER_INSTANCE=$(( ${VAL_TP} * ${VAL_PP} ))

max_prompt_length=2048
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
# The model's native context is 40960, but a rollout never exceeds prompt + response.
# On a shared card the KV budget is small, and reserving for 40960 tokens makes vLLM
# demand 5.6 GiB before it will start; 10240 is what the workload can actually use.
max_model_len=$(( max_prompt_length + max_response_length ))

k=${k:-1}
max_token_len_per_gpu=$(( k * (max_prompt_length + max_response_length) ))

train_temperature=1.2
train_top_p=1.0
train_top_k=-1

val_temperature=0
val_rollot_n=1
val_top_p=1.0
val_top_k=-1
val_do_sample=False

: "${LOCAL_IP:?LOCAL_IP must name a node in the Ray cluster.}"

python3 -m psrl.trainer.main_ppo --config-path=./config \
    --config-name='ppo_megatron_trainer' \
    psrl.ps_manager_ip=${LOCAL_IP} \
    psrl.rollout_n=${rollout_N} \
    psrl.staleness=${staleness} \
    psrl.staleness_buffer_entries=${train_batch_size} \
    psrl.ps_mode=nixl_direct \
    psrl.colocate_rollout_and_train=True \
    psrl.checkpoint.use_dcp_save=False \
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
    psrl.deployment.total_nnodes=1 \
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
    train_actor_rollout_ref.rollout.gpu_memory_utilization=${VAL_GPU_MEM_UTIL:-0.3} \
    train_actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    train_actor_rollout_ref.rollout.max_num_batched_tokens=20480 \
    \
    gen_actor_rollout_ref.rollout.name=vllm \
    gen_actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    gen_actor_rollout_ref.rollout.pipeline_model_parallel_size=${GEN_PP} \
    gen_actor_rollout_ref.rollout.gpu_memory_utilization=${GEN_GPU_MEM_UTIL:-0.3} \
    gen_actor_rollout_ref.rollout.max_model_len=${max_model_len} \
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
    trainer.save_freq=${SAVE_FREQ:--1} \
    trainer.test_freq=${test_freq} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.total_training_steps=${total_training_steps} \
    trainer.total_epochs=1 $@ 2>&1 | tee "${experiment_name}.log"
