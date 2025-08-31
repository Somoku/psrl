#!/bin/bash
set -x

source ${PSRL_WORKSPACE}/env/psrl.sh

HOME=${PSRL_WORKSPACE}
MODEL_PATH=${PSRL_WORKSPACE}/models/Qwen2.5-0.5B-Instruct
GLOBAL_BATCH_SIZE=16
GEN_TP=2 # TP in the generation side
TRAIN_TP=2 # TP in the training side for validation

NNODES=3
NGPUS_PER_NODE=8

PS_NNODES=1
PS_NGPUS_PER_NODE=${NGPUS_PER_NODE} 

GEN_NNODES=$(( ${NNODES} - ${PS_NNODES} )) # Number of nodes for generation
GEN_NGPUS_PER_NODE=4 # Number of GPUs per node for generation
GEN_INSTANCES=$(( (${GEN_NNODES} * ${GEN_NGPUS_PER_NODE}) / ${GEN_TP} )) # Number of generation instances

TRAIN_NNODES=${GEN_NNODES} # Number of nodes for training
TRAIN_NGPUS_PER_NODE=$(( ${NGPUS_PER_NODE} - ${GEN_NGPUS_PER_NODE} )) # Number of GPUs per node for training

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

PYTHONUNBUFFERED=1 python3 -m psrl.trainer.main_ppo \
    psrl.staleness=2 \
    psrl.staleness_buffer_entries=${GLOBAL_BATCH_SIZE} \
    psrl.gen_mode=batch \
    psrl.ps_mode=cpu_ref \
    psrl.logging_path=${PSRL_WORKSPACE}/psrl/examples/grpo_trainer/fsdp/psrl_log \
    psrl.log_prob.enable_inference_engine_log_prob=True \
    psrl.log_prob.enable_proxy_log_prob=False \
    psrl.deployment.n_rollout_instances=${GEN_INSTANCES} \
    psrl.deployment.rollout_nnodes_per_instance=1 \
    psrl.deployment.rollout_ngpus_per_node_per_instance=${GEN_TP} \
    psrl.deployment.train_nnodes=${TRAIN_NNODES} \
    psrl.deployment.train_ngpus_per_node=${TRAIN_NGPUS_PER_NODE} \
    psrl.deployment.ps_nnodes=${PS_NNODES} \
    psrl.deployment.ps_ngpus_per_node=${PS_NGPUS_PER_NODE} \
    \
    gen_actor_rollout_ref.model.path="$MODEL_PATH" \
    gen_actor_rollout_ref.rollout.mode=sync \
    gen_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    gen_actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    gen_actor_rollout_ref.rollout.n=4 \
    gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    gen_actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    \
    train_actor_rollout_ref.model.path="$MODEL_PATH" \
    train_actor_rollout_ref.model.use_remove_padding=True \
    train_actor_rollout_ref.model.enable_gradient_checkpointing=True \
    train_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    train_actor_rollout_ref.rollout.tensor_model_parallel_size=${TRAIN_TP} \
    train_actor_rollout_ref.rollout.n=4 \
    train_actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    train_actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    train_actor_rollout_ref.actor.optim.lr=1e-6 \
    train_actor_rollout_ref.actor.ppo_mini_batch_size=${GLOBAL_BATCH_SIZE} \
    train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.actor.fsdp_config.param_offload=False \
    train_actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    train_actor_rollout_ref.actor.use_kl_loss=True \
    train_actor_rollout_ref.actor.kl_loss_coef=0.001 \
    train_actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    train_actor_rollout_ref.actor.entropy_coeff=0 \
    train_actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    train_actor_rollout_ref.ref.fsdp_config.param_offload=False \
    \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path="$MODEL_PATH" \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    \
    algorithm.use_kl_in_reward=False \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=${GLOBAL_BATCH_SIZE} \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger=['console'] \
    trainer.project_name='psrl_fsdp_grpo_test' \
    trainer.experiment_name='batch' \
    trainer.total_training_steps=20 \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.total_epochs=30 2>&1 | tee psrl_fsdp_grpo_test-batch.log
