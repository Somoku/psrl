#!/usr/bin/env bash
# Train Qwen3.5-9B on the SciAccel v2 task bank with GRPO.
# Environment variables override model, sequence, topology, and checkpoint settings.

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
# Expand allocator segments to reduce fragmentation from uneven activation sizes.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_prestart_worker_first_driver=false
export RAY_num_workers_soft_limit=0
export RAY_memory_monitor_refresh_ms=0

# Keep Ray sockets within the Unix path limit and off the shared filesystem.
export TMPDIR=${SCIACCEL_TMPDIR:-/tmp}
mkdir -p "${TMPDIR}"

PSRL_PATH=${PSRL_PATH:-$(python3 -c "import os, psrl; print(os.path.dirname(os.path.dirname(psrl.__file__)))")}

# --- Model and data ---

# The 9B model uses the shorter context configured below.
HF_MODEL_PATH=${HF_MODEL_PATH:-/apdcephfs_zwfy10/share_303541817/lhy/models/Qwen3.5-9B}
train_files=${PSRL_PATH}/examples/sciaccel_rl/data/v2/train.parquet
val_files=${PSRL_PATH}/examples/sciaccel_rl/data/v2/val.parquet

if [[ ! -d "${HF_MODEL_PATH}" ]]; then
    echo "ERROR: model directory not found: ${HF_MODEL_PATH}" >&2
    exit 1
fi
for f in "${train_files}" "${val_files}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: parquet not found: ${f}" >&2
        echo "Build it: python -m examples.sciaccel_rl.prepare.build_dataset_v2 --repo <sciaccel-rl> --out-dir $(dirname "${f}")" >&2
        exit 1
    fi
done

# --- Experiment ---
project_name=sciaccel_rl
experiment_name=GRPO-sciaccel-v2-Qwen35-9B
OUTPUT_DIR=${OUTPUT_DIR:-${PSRL_PATH}/examples/sciaccel_rl}
CKPTS_DIR=${OUTPUT_DIR}/ckpts/${project_name}/${experiment_name}
PSRL_LOG_DIR=${OUTPUT_DIR}/psrl_logs/${experiment_name}
mkdir -p "${CKPTS_DIR}" "${PSRL_LOG_DIR}"

# --- Agent loop config ---
agent_loop_config_path=${PSRL_PATH}/examples/sciaccel_rl/config/sciaccel_agent_config_v2.yaml
reward_path=${PSRL_PATH}/examples/sciaccel_rl/reward.py

# --- Batch and sequence lengths ---

# Batch size controls requests and packed sequences per step.
train_batch_size=${TRAIN_BATCH_SIZE:-8}
rollout_N=8
# Keep prompts large enough for the longest task instruction.
max_prompt_length=2048
# Fit the response window within 9B activation memory.
max_response_length=${MAX_RESPONSE_LENGTH:-64512}
max_model_len=$(( max_prompt_length + max_response_length ))
# The packing budget must cover the full prompt and response sequence.
max_tokens_per_gpu=$(( max_prompt_length + max_response_length ))
max_num_batched_tokens=${max_model_len}
# Cap turns so Harbor can grade before the shorter context window fills.
max_turns=${MAX_TURNS:-25}

# Spread Harbor containers across nodes with one agent loop worker per node.
AGENT_LOOP_WORKERS=${AGENT_LOOP_WORKERS:-3}

# Bound admitted sequences to the rollout engine's KV capacity.
# Revisit this value when batch size, context length, or engine count changes.
MAX_CONCURRENT_SEQS_PER_INSTANCE=${MAX_CONCURRENT_SEQS_PER_INSTANCE:-32}

# Keep HTTP concurrency above the KV-aware admission gate.
SERVER_MAX_CONCURRENCY=${SERVER_MAX_CONCURRENCY:-64}

# --- Chain-of-thought handling across turns ---

# Preserve accumulated thinking bytes across turns with the Qwen3.5 template.
thinking_template=multi_thinking
chat_template_path=${PSRL_PATH}/examples/sciaccel_rl/config/qwen35_acc_thinking.jinja2

# --- Deployment: 3 nodes x 8 GPU = 24 (8 generation + 16 training) ---

# Validation overlays training GPUs because generation and training consume all devices.
NNODES=3
NGPUS_PER_NODE=8

# Split generation GPUs across tensor-parallel rollout instances.
GEN_TP=2
GEN_PP=1
GEN_NNODES=1
GEN_NGPUS_PER_NODE=8
GEN_INSTANCES=$(((GEN_NNODES * GEN_NGPUS_PER_NODE) / (GEN_TP * GEN_PP)))
GEN_NGPUS_PER_NODE_PER_INSTANCE=$((GEN_TP * GEN_PP))

# Sequence parallelism remains disabled for incompatible VLM input shapes.
TRAIN_SP=${TRAIN_SP:-1}
# FSDP shards the model across all 16 training GPUs.
TRAIN_FSDP=16
TRAIN_NNODES=2
TRAIN_NGPUS_PER_NODE=8

# Keep validation engines modest because they share training GPUs.
VAL_TP=2
VAL_PP=1
VAL_INSTANCES=2
VAL_NGPUS_PER_NODE_PER_INSTANCE=$((VAL_TP * VAL_PP))

# --- GRPO and optimizer ---
actor_lr=1e-6
use_kl_loss=True
kl_loss_coef=0.001
clip_ratio_low=0.2
clip_ratio_high=0.3
total_training_steps=${TOTAL_TRAINING_STEPS:-200}
save_freq=25
# Full agentic validation is expensive, so run it infrequently.
test_freq=25

PYTHONUNBUFFERED=1 python3 -m psrl.trainer.main_ppo \
    psrl.ps_manager_ip=${LOCAL_IP:-127.0.0.1} \
    psrl.ps_mode=nixl_cpu \
    psrl.rollout_n=${rollout_N} \
    `# 1, so rollout for step N+1 overlaps training for step N instead of the GPUs idling` \
    `# through each phase. It also doubles requests in flight, since max_concurrency is` \
    `# rollout_n * staleness_buffer_entries * (staleness + 1), which is why the admission` \
    `# gate above is load-bearing.` \
    psrl.staleness=1 \
    psrl.staleness_buffer_entries=${train_batch_size} \
    psrl.rollout_gateway.trajectory_id_strategy=auto \
    psrl.rollout_gateway.server_max_concurrency=${SERVER_MAX_CONCURRENCY} \
    psrl.rollout_coordination.routing_strategy.max_concurrent_seqs_per_instance=${MAX_CONCURRENT_SEQS_PER_INSTANCE} \
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
    `# DAPO-style dynamic sampling is OFF by default. It drops groups whose rewards are all` \
    `# identical, which under GRPO contribute exactly nothing (advantage is reward minus group` \
    `# mean over std, so zero variance means zero gradient). The step-1 timing argues FOR it --` \
    `# update_actor was 2069 s of a 3210 s step, 64%, while rollout was only 751 s, 23% -- so` \
    `# trading rollout for less training is the right direction on paper.` \
    `#` \
    `# The risk is that it never fills a batch: the measured per-episode solve rate is 5.7%` \
    `# (40 of 704), so P(all 8 rollouts identical) is 0.63 and only ~37% of groups survive,` \
    `# needing ~2.7x more rollout per step. Enable with GROUP_FILTER=True once a baseline is` \
    `# established, and watch that the buffer still reaches its group count.` \
    psrl.group_post_process.enable=${GROUP_FILTER:-False} \
    psrl.group_post_process.name=dynamic_sampling_filter \
    algorithm.filter_groups.metric=seq_final_reward \
    psrl.colocate_validate_and_train=True \
    \
    gen_actor_rollout_ref.rollout.name=vllm \
    gen_actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    gen_actor_rollout_ref.rollout.pipeline_model_parallel_size=${GEN_PP} \
    gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    gen_actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    gen_actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    +gen_actor_rollout_ref.rollout.chat_template=${chat_template_path} \
    gen_actor_rollout_ref.rollout.n=${rollout_N} \
    gen_actor_rollout_ref.rollout.temperature=1.0 \
    gen_actor_rollout_ref.rollout.top_p=1.0 \
    gen_actor_rollout_ref.rollout.top_k=-1 \
    gen_actor_rollout_ref.rollout.multi_turn.enable=True \
    gen_actor_rollout_ref.rollout.multi_turn.max_turns=${max_turns} \
    gen_actor_rollout_ref.rollout.agent.agent_loop_config_path=${agent_loop_config_path} \
    gen_actor_rollout_ref.rollout.agent.default_agent_loop=sciaccel \
    gen_actor_rollout_ref.rollout.agent.num_workers=${AGENT_LOOP_WORKERS} \
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
    `# Optimizer offload is back ON, which is also what validate_config wants unless TMS` \
    `# covers the training workers. It used to be incompatible with the fused log-prob kernel:` \
    `# with offload the lm_head is a DTensor whose local shard lives on CPU, full_tensor()` \
    `# returns a CPU tensor, and qwen3_5.py only converted the activations dtype, so the` \
    `# matmul died with "mat2 is on cpu, different from other tensors on cuda:0". That is now` \
    `# fixed at the source: qwen3_5.py moves the weights to the activations device first, so` \
    `# this path no longer depends on the offload setting either way.` \
    train_actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    `# Chunked log-prob computation, needed IN ADDITION to SP. SP shards activations through` \
    `# the layer stack, but the engine gathers before the head, so lm_head still sees the full` \
    `# sequence: with SP=8 and this off, the OOM was 43.53 GiB, which reverses to 94,112 tokens` \
    `# rather than the sharded 12,288. Qwen3.5's vocab is 248,320, so the unfused logits are` \
    `# 45.5 GiB bf16 / 90.9 GiB after the fp32 upcast for log_softmax, while this path chunks` \
    `# and peaks at 10.8 GiB (measured at the full 98,304 budget).` \
    `#` \
    `# It needs monkey_patch.py's embeds slice to use padding=True so the shard's token count` \
    `# matches the engine's padded labels, which is fixed there.` \
    train_actor_rollout_ref.model.use_fused_kernels=True \
    train_actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
    `# Sequence parallelism. See the TRAIN_SP definition above: this is the lever that shards` \
    `# activations for a ~98k-token packed sequence, and it needed the text-only fix in` \
    `# transformer_impl.py to work on a multimodal checkpoint.` \
    train_actor_rollout_ref.actor.ulysses_sequence_parallel_size=${TRAIN_SP} \
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
    data.val_files=${val_files} \
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
    `# Truncated importance sampling, matching the dapo_trainer convention (rollout_is=token,` \
    `# threshold 2.0). This is load-bearing rather than optional here because staleness=1` \
    `# means a step trains on trajectories generated by the PREVIOUS weights, so the rollout` \
    `# and training policies genuinely differ and the uncorrected gradient is biased. TIS` \
    `# reweights per token by the behaviour-vs-current ratio, clipped at 2.0 so a few` \
    `# high-ratio tokens cannot dominate an update. It consumes the rollout log-probs that` \
    `# use_rollout_log_probs and enable_rollout_engine_log_prob already produce.` \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.val_before_train=False \
    trainer.total_training_steps=${total_training_steps} \
    trainer.resume_mode=auto \
    "$@" 2>&1 | tee "${OUTPUT_DIR}/${experiment_name}.log"
