#!/bin/bash
set -x

PSRL_WORKSPACE=/jizhicfs/johnnyslin
source ${PSRL_WORKSPACE}/env/verl_H20.sh

export WANDB_API_KEY=8c63c5f4a504550818e34fadd4000eb1de2b3f30

HOME=${PSRL_WORKSPACE}
PSRL_PATH=${PSRL_WORKSPACE}/psrl

HF_MODEL_PATH=/jizhicfs/lhy/models/Qwen2.5-0.5B-Instruct
DIST_CKPT_PATH=/jizhicfs/lhy/models/mcore_ckpt/Qwen2.5-0.5B-Instruct
python3 ${PSRL_PATH}/scripts/converter_hf_to_mcore.py --hf_model_path $HF_MODEL_PATH --output_path $DIST_CKPT_PATH

GLOBAL_BATCH_SIZE=16
GEN_TP=2 # TP in the generation side
GEN_PP=2 # PP in the generation side

TRAIN_TP=2 # TP in the training side for validation
PP=2
CP=1

NNODES=3
NGPUS_PER_NODE=8

PS_NNODES=1
PS_NGPUS_PER_NODE=${NGPUS_PER_NODE} 

GEN_NNODES=$(( ${NNODES} - ${PS_NNODES} )) # Number of nodes for generation
GEN_NGPUS_PER_NODE=4 # Number of GPUs per node for generation
GEN_INSTANCES=$(( (${GEN_NNODES} * ${GEN_NGPUS_PER_NODE}) / ( ${GEN_TP} * ${GEN_PP} ) )) # Number of generation instances
GEN_NGPUS_PER_NODE_PER_INSTANCE=$(( ${GEN_TP} * ${GEN_PP} )) # Number of GPUs per node for generation per instance]

TRAIN_NNODES=${GEN_NNODES} # Number of nodes for training
TRAIN_NGPUS_PER_NODE=$(( ${NGPUS_PER_NODE} - ${GEN_NGPUS_PER_NODE} )) # Number of GPUs per node for training

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

bash $HOME/kill.sh 3

PYTHONUNBUFFERED=1 python3 -m psrl.trainer.main_ppo --config-path=./config --config-name='ppo_megatron_trainer' \
    psrl.staleness=2 \
    psrl.staleness_buffer_entries=${GLOBAL_BATCH_SIZE} \
    psrl.gen_mode=stream \
    psrl.ps_mode=cpu_ref \
    psrl.logging_path=${PSRL_WORKSPACE}/psrl/examples/ppo_trainer/megatron/psrl_log \
    psrl.log_prob.enable_inference_engine_log_prob=True \
    psrl.log_prob.enable_proxy_log_prob=False \
    psrl.deployment.n_rollout_instances=${GEN_INSTANCES} \
    psrl.deployment.rollout_nnodes_per_instance=1 \
    psrl.deployment.rollout_ngpus_per_node_per_instance=${GEN_NGPUS_PER_NODE_PER_INSTANCE} \
    psrl.deployment.train_nnodes=${TRAIN_NNODES} \
    psrl.deployment.train_ngpus_per_node=${TRAIN_NGPUS_PER_NODE} \
    psrl.deployment.ps_nnodes=${PS_NNODES} \
    psrl.deployment.ps_ngpus_per_node=${PS_NGPUS_PER_NODE} \
    \
    gen_actor_rollout_ref.model.path="$HF_MODEL_PATH" \
    gen_actor_rollout_ref.rollout.mode=psrl_async \
    gen_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    gen_actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    gen_actor_rollout_ref.rollout.pipeline_model_parallel_size=${GEN_PP} \
    gen_actor_rollout_ref.rollout.n=1 \
    gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    gen_actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    \
    train_actor_rollout_ref.model.path="$HF_MODEL_PATH" \
    train_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    train_actor_rollout_ref.rollout.tensor_model_parallel_size=${TRAIN_TP} \
    train_actor_rollout_ref.rollout.n=1 \
    train_actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    train_actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    train_actor_rollout_ref.actor.optim.lr=1e-6 \
    train_actor_rollout_ref.actor.ppo_mini_batch_size=${GLOBAL_BATCH_SIZE} \
    train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.actor.use_kl_loss=False \
    train_actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TRAIN_TP} \
    train_actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    train_actor_rollout_ref.actor.megatron.context_parallel_size=$CP \
    train_actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    train_actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    train_actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TRAIN_TP} \
    train_actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP \
    train_actor_rollout_ref.ref.megatron.context_parallel_size=$CP \
    train_actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    train_actor_rollout_ref.ref.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    \
    critic.optim.lr=1e-5 \
    critic.model.path="$HF_MODEL_PATH" \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.megatron.tensor_model_parallel_size=${TRAIN_TP} \
    critic.megatron.pipeline_model_parallel_size=$PP \
    critic.megatron.context_parallel_size=$CP \
    critic.megatron.use_dist_checkpointing=True \
    critic.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    \
    reward_model.launch_reward_fn_async=True \
    \
    algorithm.use_kl_in_reward=False \
    algorithm.adv_estimator=gae \
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
    trainer.project_name='psrl_megatron_ppo_test' \
    trainer.experiment_name='stream' \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=$NNODES \
    trainer.total_training_steps=20 \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.total_epochs=30 2>&1 | tee psrl_megatron_ppo_test-stream.log

bash $HOME/occupy.sh 3