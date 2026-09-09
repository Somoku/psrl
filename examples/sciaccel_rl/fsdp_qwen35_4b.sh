#!/usr/bin/env bash
# Train Qwen3.5-4B on the SciAccel v2 task bank with GRPO.
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

# The 4B model leaves more activation memory for long contexts.
HF_MODEL_PATH=${HF_MODEL_PATH:-/apdcephfs_zwfy10_303541817/share_303541817/lhy/models/Qwen3.5-4B}
# Localization hint strength for the repair tasks. `L1` adds file, line, and defect
# note, `L2` drops the line, and `L3` is the unhinted control. Validation is always
# unhinted, so scores stay comparable across levels.
HINT_LEVEL=${HINT_LEVEL:-L1}
# `v2_repair` holds the 99 single edit repair tasks, every one of them hinted.
# `v2_hint` adds the 44 excised routine tasks, which take a whole subroutine body
# (median 32 lines, max 1020) and cannot be helped by a location hint.
DATA_DIR=${DATA_DIR:-${PSRL_PATH}/examples/sciaccel_rl/data/mitgcm-biogeo_repair_easy}
train_files=${DATA_DIR}/${HINT_LEVEL}_train.parquet
# Hinted validation, so the split matches the training distribution. Every dataset
# also ships an unhinted `val.parquet` for measuring unaided localization, but a
# hint-trained model scores near zero on it for the wrong reason. Override with
# VAL_FILES to use it deliberately.
val_files=${VAL_FILES:-${DATA_DIR}/${HINT_LEVEL}_val.parquet}

if [[ ! -d "${HF_MODEL_PATH}" ]]; then
    echo "ERROR: model directory not found: ${HF_MODEL_PATH}" >&2
    exit 1
fi
for f in "${train_files}" "${val_files}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: parquet not found: ${f}" >&2
        echo "Build it: python -m examples.sciaccel_rl.prepare.build_dataset_v2 --repo <sciaccel-rl> --out-dir $(dirname "${f}") --categories repair --hint-level all" >&2
        exit 1
    fi
done

# --- Experiment ---
project_name=sciaccel_rl_mit
# The dataset directory is part of the identity, because a repair only run and a
# mixed run at the same hint level are different experiments.
experiment_name=GRPO-sciaccel-Qwen35-4B-$(basename "${DATA_DIR}")-${HINT_LEVEL}
OUTPUT_DIR=${OUTPUT_DIR:-${PSRL_PATH}/examples/sciaccel_rl}
CKPTS_DIR=${OUTPUT_DIR}/ckpts/${project_name}/${experiment_name}
PSRL_LOG_DIR=${OUTPUT_DIR}/psrl_logs/${experiment_name}
mkdir -p "${CKPTS_DIR}" "${PSRL_LOG_DIR}"

# --- Agent loop config ---
agent_loop_config_path=${PSRL_PATH}/examples/sciaccel_rl/config/sciaccel_agent_config_v2.yaml
reward_path=${PSRL_PATH}/examples/sciaccel_rl/reward.py

# --- Batch and sequence lengths ---

# Batch size controls requests and packed sequences per step.
train_batch_size=${TRAIN_BATCH_SIZE:-16}
rollout_N=8
# Keep prompts large enough for the longest task instruction.
max_prompt_length=2048
# Long terminal output requires most of the context budget.
max_response_length=${MAX_RESPONSE_LENGTH:-65536}
# The serving window must equal the training budget, not exceed it. `max_model_len` is
# forwarded to terminus-2 as `max_input_tokens`, so any headroom here is headroom the
# agent will actually use, and TITO then hands the trainer a response longer than
# `max_response_length`. An earlier attempt to add 4096 slack to dodge a prompt overflow
# instead moved the wall: overflow errors went from a handful to 3028, and 14% of
# episodes exceeded the training budget.
max_model_len=$(( max_prompt_length + max_response_length ))
# The packing budget must cover the longest sequence without exceeding the window.
max_tokens_per_gpu=${MAX_TOKENS_PER_GPU:-${max_model_len}}
if (( max_tokens_per_gpu < max_model_len )); then
    echo "ERROR: max_tokens_per_gpu (${max_tokens_per_gpu}) must be >= max_model_len (${max_model_len})." >&2
    echo "rearrange_micro_batches requires max_token_len >= max_seq_len." >&2
    exit 1
fi
max_num_batched_tokens=${max_model_len}
# The turn cap lets Harbor grade delivered work before unbounded context growth.
# Left at 50 deliberately. Measured cost is about 1104 response tokens per turn, so 50
# turns already spends 55k of the 65536 response budget and 59 turns would exhaust it.
# Raising the cap without also raising `max_response_length` just converts
# `max_turns_exceeded` into `max_response_length_exceeded`, which grades no better. The
# response budget cannot grow either: reserved memory peaked at 85 GB of 95 GB.
max_turns=${MAX_TURNS:-50}

# Nodes allowed to host agent loop workers, and therefore Docker containers. A node whose
# daemon has degraded still accepts actors and then hangs every episode it is handed, so
# excluding it is the only way to keep training moving without waiting on a reboot.
# Measured: a healthy node starts 16 containers in 2 s, one carrying 89 orphaned
# fuse-overlayfs mounts could not start 16 within 280 s. Empty means every alive node.
AGENT_NODE_IPS=${AGENT_NODE_IPS:-28.49.55.85,28.49.196.175}

# One agent loop worker per allowed node. Workers are placed round-robin, so more workers
# than nodes stacks several on one node and multiplies its container count by exactly the
# factor `harbor.max_concurrent_episodes` is there to bound.
if [ -n "${AGENT_NODE_IPS}" ]; then
    AGENT_LOOP_WORKERS=${AGENT_LOOP_WORKERS:-$(awk -F, '{print NF}' <<< "${AGENT_NODE_IPS}")}
else
    AGENT_LOOP_WORKERS=${AGENT_LOOP_WORKERS:-3}
fi

# Bound admitted sequences to the rollout engine's KV capacity.
# Revisit this value when batch size, context length, or engine count changes.
MAX_CONCURRENT_SEQS_PER_INSTANCE=${MAX_CONCURRENT_SEQS_PER_INSTANCE:-32}

# Keep HTTP concurrency above the KV-aware admission gate.
SERVER_MAX_CONCURRENCY=${SERVER_MAX_CONCURRENCY:-64}

# --- Chain-of-thought handling across turns ---

# `multi_thinking` preserves accumulated thinking bytes across turns and needs the
# accumulating Qwen3.5 template. `disable_thinking` and the trajectory modes must run
# on the model's own template, so they leave the override empty. Deriving the path
# here keeps the two settings from drifting apart, which renders a broken prompt.
thinking_template=${thinking_template:-multi_thinking}
if [ "${thinking_template}" = "multi_thinking" ]; then
    chat_template_path=${PSRL_PATH}/examples/sciaccel_rl/config/qwen35_acc_thinking.jinja2
    chat_template_arg="+gen_actor_rollout_ref.rollout.chat_template=${chat_template_path}"
else
    chat_template_arg=""
fi

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

# Sequence parallelism keeps long-context activations within device memory.
TRAIN_SP=${TRAIN_SP:-4}
# Hybrid sharding keeps all-gathers within each training node.
TRAIN_FSDP=${TRAIN_FSDP:-8}
TRAIN_NNODES=2
TRAIN_NGPUS_PER_NODE=8

# Keep validation engines modest because they share training GPUs.
VAL_TP=2
VAL_PP=1
VAL_INSTANCES=2
VAL_NGPUS_PER_NODE_PER_INSTANCE=$((VAL_TP * VAL_PP))

# --- GRPO and optimizer ---
actor_lr=1e-6
# KL to the reference is off. Measured at 6e-4 it contributed nothing to the loss while
# still paying for a reference forward pass every step. SkyRL's agentic recipes drop it
# too. Turning it off also frees the memory the ref model held, which matters because
# reserved memory peaked at 85 GB of the H20's 95 GB.
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.3
total_training_steps=${TOTAL_TRAINING_STEPS:-200}
# Save often enough to bound work lost after a failure.
save_freq=${SAVE_FREQ:-10}
# Full agentic validation is expensive, so run it infrequently.
test_freq=${TEST_FREQ:-200}

PYTHONUNBUFFERED=1 python3 -m psrl.trainer.main_ppo \
    psrl.ps_manager_ip=${LOCAL_IP:-127.0.0.1} \
    psrl.ps_mode=nixl_cpu \
    psrl.rollout_n=${rollout_N} \
    `# 1, so rollout for step N+1 overlaps training for step N instead of the GPUs idling` \
    `# through each phase. It also doubles requests in flight, since max_concurrency is` \
    `# rollout_n * staleness_buffer_entries * (staleness + 1), which is why the admission gate` \
    `# above is load-bearing.` \
    `#` \
    `# This needs the losses.py width-matching fix: no_padding_2_padding pads the model output` \
    `# to this micro-batch's own max response length (max_response_len is only set on the` \
    `# left-right padding path, never on NO_PADDING), while old_log_probs was padded under the` \
    `# grouping it was stored with. Those groupings only differ once staleness > 0, which is` \
    `# why step 1 passed and step 2 died on "size of tensor a (273) vs b (337)".` \
    psrl.staleness=${STALENESS:-1} \
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
    ${chat_template_arg} \
    gen_actor_rollout_ref.rollout.n=${rollout_N} \
    gen_actor_rollout_ref.rollout.temperature=1.0 \
    gen_actor_rollout_ref.rollout.top_p=1.0 \
    gen_actor_rollout_ref.rollout.top_k=-1 \
    gen_actor_rollout_ref.rollout.multi_turn.enable=True \
    gen_actor_rollout_ref.rollout.multi_turn.max_turns=${max_turns} \
    gen_actor_rollout_ref.rollout.agent.agent_loop_config_path=${agent_loop_config_path} \
    gen_actor_rollout_ref.rollout.agent.default_agent_loop=sciaccel \
    gen_actor_rollout_ref.rollout.agent.num_workers=${AGENT_LOOP_WORKERS} \
    `# Restrict which nodes host agent loop workers, and therefore Docker containers.` \
    `# Set AGENT_NODE_IPS='' to fall back to every alive node.` \
    ${AGENT_NODE_IPS:+gen_actor_rollout_ref.rollout.agent.node_ips=[${AGENT_NODE_IPS}]} \
    gen_actor_rollout_ref.rollout.agent.traj_reward_mode=traj \
    `# DAPO Overlong Filtering. 46% of episodes in the previous run ended on a harness` \
    `# budget (838 max_turns_exceeded plus 81 max_response_length_exceeded of 1983), and` \
    `# 724 of those scored exactly 0. Training them as failures penalises every token in` \
    `# the longest trajectories, and under token-mean the cheapest way to shed that` \
    `# penalty is to shorten each turn: measured 1125 to 327 tokens per turn over 16` \
    `# steps, which spent the turn cap faster, pushed max_turns_exceeded from 29% to 39%` \
    `# and collapsed the score from 0.573 at step 11 to 0.078 at step 16. Masking keeps` \
    `# the reward in the GRPO baseline while removing the gradient. Set False to A/B.` \
    gen_actor_rollout_ref.rollout.agent.overlong_filtering=${OVERLONG_FILTERING:-True} \
    \
    train_actor_rollout_ref.model.path=${HF_MODEL_PATH} \
    train_actor_rollout_ref.actor.optim.lr=${actor_lr} \
    `# Short warmup. At 10 steps the first 10 updates ran at 10% to 90% of the target lr,` \
    `# so a run that only reached step 14 had barely trained and its reward curve was` \
    `# almost pure sampling noise. 3 steps still eases in the first updates.` \
    train_actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS:-3} \
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
    `# The overlong penalty is off. It fired on 28% of samples, but 119 of those 142 were` \
    `# already-failing wall cases scoring ~0.07, so it mostly re-punished known failures.` \
    `# Meanwhile it hit ~23 episodes that FINISHED, and those score 1.0 some 64% of the` \
    `# time, so it was penalising successful repairs for taking a while. It also widened` \
    `# reward to a 2.0 range, inflating advantage variance for a non-task reason. Length` \
    `# is a symptom of failing to localize the defect here, not a cause worth shaping.` \
    reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.enable=False \
    reward.managers.dapo.reward_kwargs.max_resp_len=${max_response_length} \
    \
    data.train_files=${train_files} \
    data.val_files=${val_files} \
    data.train_batch_size=${train_batch_size} \
    `# The bank is grouped by category on disk, and verl's vendored legacy_data.yaml` \
    `# defaults shuffle to False, so an unshuffled run spends its first two steps` \
    `# entirely on restore tasks and never sees a hinted repair task.` \
    data.shuffle=True \
    data.seed=${DATA_SEED:-1} \
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
    `# Dr. GRPO: center advantages within the group but do NOT divide by the group std.` \
    `# Dividing amplifies noise in near-degenerate groups, where 7 of 8 rollouts score 0` \
    `# and one scores 1, because the tiny std blows that single sample up. This task is` \
    `# close to bimodal, so that case is common. SkyRL's Harbor recipes also set it off.` \
    algorithm.norm_adv_by_std_in_grpo=False \
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
