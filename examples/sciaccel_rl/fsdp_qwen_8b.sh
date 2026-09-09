#!/usr/bin/env bash
set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export RAY_prestart_worker_first_driver=false
export RAY_num_workers_soft_limit=0
export RAY_memory_monitor_refresh_ms=0

PSRL_PATH=${PSRL_PATH:-$(python3 -c "import os, psrl; print(os.path.dirname(os.path.dirname(psrl.__file__)))")}

# --- Model and data ---
HF_MODEL_PATH=${HF_MODEL_PATH:-/apdcephfs_zwfy10/share_303541817/lhy/models/Qwen3-8B}
train_files=${PSRL_PATH}/examples/sciaccel_rl/data/train.parquet

if [[ ! -d "${HF_MODEL_PATH}" ]]; then
    echo "ERROR: model directory not found: ${HF_MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -f "${train_files}" ]]; then
    echo "ERROR: training parquet not found: ${train_files}" >&2
    exit 1
fi

# --- Experiment ---
project_name=sciaccel_rl
experiment_name=GRPO-sciaccel-laps-cpu-8B
OUTPUT_DIR=${OUTPUT_DIR:-${PSRL_PATH}/examples/sciaccel_rl}
CKPTS_DIR=${OUTPUT_DIR}/ckpts/${project_name}/${experiment_name}
PSRL_LOG_DIR=${OUTPUT_DIR}/psrl_logs/${experiment_name}
mkdir -p "${CKPTS_DIR}" "${PSRL_LOG_DIR}"

# --- Agent loop config ---
agent_loop_config_path=${PSRL_PATH}/examples/sciaccel_rl/config/sciaccel_agent_config.yaml
reward_path=${PSRL_PATH}/examples/sciaccel_rl/reward.py

# --- Batch and sequence lengths (aligned with SkyRL Harbor defaults) ---
train_batch_size=1
rollout_N=8
max_prompt_length=2048
max_response_length=30720
max_model_len=32768
# The packing budget must cover the full prompt and response sequence.
max_tokens_per_gpu=$(( max_prompt_length + max_response_length ))
max_num_batched_tokens=32768
max_turns=32

# --- Chain-of-thought handling across turns ---

# `multi_thinking` needs Qwen3's accumulated-thinking template to keep its matched
# tags. `disable_thinking` and the trajectory modes must run on the model's own
# template, so they leave the override empty. Deriving the path here keeps the two
# settings from drifting apart, which renders a broken prompt.
thinking_template=${thinking_template:-multi_thinking}
if [ "${thinking_template}" = "multi_thinking" ]; then
    chat_template_path=${PSRL_PATH}/examples/sciaccel_rl/config/qwen3_acc_thinking.jinja2
    chat_template_arg="+gen_actor_rollout_ref.rollout.chat_template=${chat_template_path}"
else
    chat_template_arg=""
fi

# --- Deployment: single node, 8 GPU (4 gen + 4 train) ---
NNODES=1
NGPUS_PER_NODE=8

GEN_TP=1
GEN_PP=1
GEN_NNODES=1
GEN_NGPUS_PER_NODE=4
GEN_INSTANCES=$(((GEN_NNODES * GEN_NGPUS_PER_NODE) / (GEN_TP * GEN_PP)))
GEN_NGPUS_PER_NODE_PER_INSTANCE=$((GEN_TP * GEN_PP))

TRAIN_SP=1
TRAIN_FSDP=4
TRAIN_NNODES=1
TRAIN_NGPUS_PER_NODE=4

VAL_TP=1
VAL_PP=1
VAL_INSTANCES=$(((TRAIN_NNODES * TRAIN_NGPUS_PER_NODE) / (VAL_TP * VAL_PP)))
VAL_NGPUS_PER_NODE_PER_INSTANCE=$((VAL_TP * VAL_PP))

# --- GRPO and optimizer ---
actor_lr=1e-6
use_kl_loss=True
kl_loss_coef=0.001
clip_ratio_low=0.2
clip_ratio_high=0.3
total_training_steps=100
save_freq=50
test_freq=10

PYTHONUNBUFFERED=1 python3 -m psrl.trainer.main_ppo \
    psrl.ps_manager_ip=${LOCAL_IP:-127.0.0.1} \
    psrl.ps_mode=nixl_cpu \
    psrl.rollout_n=${rollout_N} \
    psrl.staleness=0 \
    psrl.staleness_buffer_entries=${train_batch_size} \
    psrl.rollout_gateway.trajectory_id_strategy=auto \
    psrl.agentic_rl.batch_agg_mode=request \
    psrl.agentic_rl.thinking_template=${thinking_template} \
    psrl.agentic_rl.trajectory_output.enable=True \
    psrl.agentic_rl.turn_output.enable=True \
    psrl.logging_path=${PSRL_LOG_DIR} \
    psrl.log_prob.enable_rollout_engine_log_prob=True \
    psrl.deployment.n_rollout_instances=${GEN_INSTANCES} \
    psrl.deployment.rollout_nnodes_per_instance=1 \
    psrl.deployment.rollout_ngpus_per_node_per_instance=${GEN_NGPUS_PER_NODE_PER_INSTANCE} \
    psrl.deployment.n_validate_instances=${VAL_INSTANCES} \
    psrl.deployment.validate_nnodes_per_instance=1 \
    psrl.deployment.validate_ngpus_per_node_per_instance=${VAL_NGPUS_PER_NODE_PER_INSTANCE} \
    psrl.deployment.train_nnodes=${TRAIN_NNODES} \
    psrl.deployment.train_ngpus_per_node=${TRAIN_NGPUS_PER_NODE} \
    psrl.deployment.total_nnodes=${NNODES} \
    psrl.group_post_process.enable=False \
    psrl.colocate_validate_and_train=False \
    \
    gen_actor_rollout_ref.rollout.name=vllm \
    gen_actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    gen_actor_rollout_ref.rollout.pipeline_model_parallel_size=${GEN_PP} \
    gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    gen_actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    gen_actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    ${chat_template_arg} \
    gen_actor_rollout_ref.rollout.n=${rollout_N} \
    gen_actor_rollout_ref.rollout.temperature=1.0 \
    gen_actor_rollout_ref.rollout.top_p=1.0 \
    gen_actor_rollout_ref.rollout.top_k=-1 \
    gen_actor_rollout_ref.rollout.multi_turn.enable=True \
    gen_actor_rollout_ref.rollout.multi_turn.max_turns=${max_turns} \
    gen_actor_rollout_ref.rollout.agent.agent_loop_config_path=${agent_loop_config_path} \
    gen_actor_rollout_ref.rollout.agent.default_agent_loop=sciaccel \
    gen_actor_rollout_ref.rollout.agent.traj_reward_mode=traj \
    \
    train_actor_rollout_ref.model.path=${HF_MODEL_PATH} \
    train_actor_rollout_ref.actor.optim.lr=${actor_lr} \
    train_actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    train_actor_rollout_ref.actor.optim.weight_decay=0.1 \
    train_actor_rollout_ref.actor.ppo_mini_batch_size=${train_batch_size} \
    train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.actor.use_dynamic_bsz=True \
    train_actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_tokens_per_gpu} \
    train_actor_rollout_ref.actor.rollout_n=${rollout_N} \
    train_actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    train_actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    train_actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    train_actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    train_actor_rollout_ref.actor.entropy_coeff=0 \
    train_actor_rollout_ref.actor.loss_agg_mode=token-mean \
    train_actor_rollout_ref.actor.grad_clip=1.0 \
    train_actor_rollout_ref.actor.strategy=fsdp2 \
    train_actor_rollout_ref.actor.fsdp_config.fsdp_size=${TRAIN_FSDP} \
    train_actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +train_actor_rollout_ref.actor.use_rollout_log_probs=True \
    \
    train_actor_rollout_ref.rollout.tensor_model_parallel_size=${VAL_TP} \
    train_actor_rollout_ref.rollout.pipeline_model_parallel_size=${VAL_PP} \
    train_actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    train_actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    train_actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    train_actor_rollout_ref.rollout.val_kwargs.n=1 \
    train_actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    train_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_tokens_per_gpu} \
    \
    reward.launch_reward_fn_async=True \
    reward.active_managers='[dapo]' \
    reward.managers.dapo.reward_fn.0.path=${reward_path} \
    reward.managers.dapo.reward_fn.0.name=compute_score \
    reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.enable=False \
    reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.len=${max_response_length} \
    reward.managers.dapo.reward_kwargs.max_resp_len=${max_response_length} \
    \
    data.train_files=${train_files} \
    data.val_files=${train_files} \
    data.train_batch_size=${train_batch_size} \
    data.prompt_key=prompt \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    data.reward_model_dicts.0.reward_loop_type=dapo \
    data.reward_model_dicts.0.reward_fn=compute_score \
    \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.val_before_train=False \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=15 \
    trainer.total_training_steps=${total_training_steps} \
    "$@" 2>&1 | tee "${OUTPUT_DIR}/${experiment_name}.log"
