#!/usr/bin/env bash
# fsdp_qwen35_9b.sh — GRPO on the SciAccel-RL v2 task bank with Qwen3.5-9B.
#
# PREFER fsdp_qwen35_4b.sh. The 9B does train -- one step completed in 3210 s (update_actor
# 2069 s of it) with grad_norm 0.00115 and a 53.5 GiB peak -- but its activations only fit a
# 49152 window, and at that window 12 of 13 episodes ended on `max_response_length_exceeded`
# instead of the turn cap, so most trajectories were being cut off. The 4B affords 66560 and
# ends on turns instead. Keep this script for the 9B comparison, and note that the length and
# turn values below were last tuned on the 4B: for a real 9B run set
# MAX_RESPONSE_LENGTH=47104.
#
# Differences from fsdp_qwen_8b.sh, which trained Qwen3-8B on the v1 bank of a single
# task. Every one is grounded in a measurement from examples/sciaccel_rl/eval/FINDINGS.md:
#
#   * data/v2/train.parquet (128 tasks) with val.parquet (17) HELD OUT. The old script
#     set data.val_files=train_files, so its val curve measured memorization. The v2
#     split is stratified by (category, family, tree), so 17 tasks still cover every
#     task shape.
#   * max_turns 50, not 32. The 9B pinned a 25-turn cap on 85% of eval trials and
#     averaged 48.7 of 50 -- it is turn-starved, and episodes that never finish deliver
#     nothing, which scores the ladder's "nothing usable" 0.0.
#   * max_model_len 98304, not 32768. A 50-turn transcript reaches ~71k tokens and a
#     25-turn one already reaches ~35.5k, so 32768 guarantees a context overflow before
#     the turn cap. At 98304 the eval saw ZERO overflows in 144 trials.
#   * GEN_TP=2, not 1. Qwen3.5-9B is multimodal (it loads a vision tower even for text)
#     and a 98304 window needs the KV room: measured 8,796 KV blocks at TP=2 versus
#     18,743 at TP=4 on H20.
#
# Prerequisites, in order:
#
#   1. Build the dataset:
#        python -m examples.sciaccel_rl.prepare.build_dataset_v2 \
#            --repo <sciaccel-rl> --out-dir examples/sciaccel_rl/data/v2
#   2. Provision Docker on EVERY node, then warm the image cache on EVERY node. Harbor
#      containers run wherever the agent-loop process runs and rollout is spread across
#      the cluster, so any node can be asked to build a task environment:
#        bash examples/sciaccel_rl/prepare/provision_docker_nodes.sh --hosts <hostfile>
#        # then, on each node:
#        bash examples/sciaccel_rl/eval/run_eval.sh --agent nop \
#            --dataset examples/sciaccel_rl/data/v2/all.parquet
#      The nop pass doubles as the anchor: expect every by_category mean score ~= 0 and
#      an empty floor_mismatch. A non-zero nop score means the reward ladder credits a
#      non-delivery, and training on it would be meaningless. Unwarmed, the first steps
#      pay ~1 min per task; warmed, env_setup is ~16 s (measured).
#   3. Start the Ray cluster across the nodes, head node first:
#        bash examples/ray/ray_start.sh ${PSRL_WORKSPACE}/hosts/24GPUs
#      NOTE that script passes `--num-cpus=32`. Each Harbor container runs 4 MPI ranks,
#      so 32 advertised CPUs caps concurrent episodes at ~8 per node regardless of the
#      384 physical cores. Raise it if rollout throughput is CPU-bound.
#   4. Run THIS script on the Ray head node.
#   5. Optionally record the pre-RL baseline to compare the training curve against:
#        bash examples/sciaccel_rl/eval/run_eval.sh --model <Qwen3.5-9B> \
#            --max-turns 50 --max-model-len 98304
#      Measured for Qwen3.5-9B: 7/144 solved, mean reward_repair 0.0497.
#
# Usage:
#   bash examples/sciaccel_rl/fsdp_qwen35_9b.sh
#   HF_MODEL_PATH=/path/to/ckpt bash examples/sciaccel_rl/fsdp_qwen35_9b.sh

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
# Let the caching allocator grow segments instead of fragmenting into fixed blocks. The
# backward pass here OOMed asking for 2.06 GiB while 4.29 GiB was already reserved-but-unused,
# which is the fragmentation signature this setting addresses (PyTorch's own OOM message
# recommends it). Activations for a 98,304-token packed sequence arrive in very uneven sizes,
# so fixed-block reuse wastes a lot.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_prestart_worker_first_driver=false
export RAY_num_workers_soft_limit=0
export RAY_memory_monitor_refresh_ms=0

# Ray puts its plasma-store socket under the temp root, and AF_UNIX paths cannot exceed
# 107 bytes. A long inherited TMPDIR (e.g. a 62-char shared-FS scratch dir) projects to
# ~129 bytes and Ray dies at startup with `validate_socket_filename failed`. Ray resolves
# its root from `tempfile.gettempdir()`, i.e. TMPDIR, unless `ray.init(_temp_dir=...)`
# overrides it -- so pinning TMPDIR is what actually fixes this. Local /tmp also keeps
# the socket off the shared FUSE mount, which does not report free inodes and has
# produced ENOSPC failures under load.
export TMPDIR=${SCIACCEL_TMPDIR:-/tmp}
mkdir -p "${TMPDIR}"

PSRL_PATH=${PSRL_PATH:-$(python3 -c "import os, psrl; print(os.path.dirname(os.path.dirname(psrl.__file__)))")}

# --- Model and data ---
# See fsdp_qwen35_4b.sh for the smaller sibling, which affords a larger context window
# because its activations run ~0.62x on hidden_size.
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
# Episodes per step. `train_batch_size * rollout_n * (staleness + 1)` requests are in flight,
# and this is the main lever on step WALL CLOCK because TITO forks every 50-turn episode into
# ~50 training sequences: at 16 the buffer held 6,509 sequences totalling ~107 M tokens over
# ~500 micro-steps, measured at ~68 min of training per step. 8 halves that.
#
# Override to 1 (TRAIN_BATCH_SIZE=1) to smoke-test the pipeline: one group of 8 rollouts per
# step turns a ~1 h cycle into minutes, which is the right way to prove several steps run
# end to end before paying for a full-size run. `staleness_buffer_entries` and
# `ppo_mini_batch_size` are derived from this below, so they stay consistent.
train_batch_size=${TRAIN_BATCH_SIZE:-8}
rollout_N=8
# Instructions measured with the Qwen3.5-9B tokenizer over all 145 v2 tasks:
# 776 min, 1010 median, 1450 max. 2048 clears the longest with room to spare, and
# `data.truncation=error` means an under-sized value would crash rather than silently
# clip. Everything left over goes to the response, which is where the transcript grows.
max_prompt_length=2048
# 64512, giving a 66,560-token window. The 9B could only afford 47104 because its activations
# hit 87.6 GiB at the full budget; the 4B runs ~0.62x on hidden_size, so ~37 GiB at this
# length. Measured over 802 real episodes the response length is median 25,039 / p90 43,927 /
# p99 69,185, and at 49152 twelve of thirteen episodes ended on max_response_length_exceeded
# rather than the turn cap, with overflow prompts clustered just past the limit. This window
# plus max_turns=25 should let most episodes end on turns instead.
max_response_length=${MAX_RESPONSE_LENGTH:-64512}
max_model_len=$(( max_prompt_length + max_response_length ))
# Token-packing budget for the training and log-prob forward passes.
# `rearrange_micro_batches` asserts `max_token_len >= max_seq_len`, where max_seq_len is
# the FULL packed sequence (prompt + response), so this must cover both.
max_tokens_per_gpu=$(( max_prompt_length + max_response_length ))
max_num_batched_tokens=${max_model_len}
# 25, down from the eval's 50, because the prompt accumulates every prior turn and the
# window is now 49152 (see max_response_length). At 50 turns against that window 12 of 13
# episodes ended on `max_response_length_exceeded` rather than `max_turns_exceeded`, with
# overflow prompts clustered just past the cap (median 49,207 against a 49,152 limit), so the
# turn budget was not the binding constraint at all -- the context was, and the tail of each
# episode was being discarded.
#
# This costs real capability and is a deliberate trade: the eval measured delivery rising from
# 9% to 36% when turns went 25 -> 50. Raise it back together with max_response_length once
# activation memory allows the full window again. The cap itself is load-bearing either way:
# terminus-2 defaults to 1000000 episodes, so without it an agent walks into the context limit
# and dies UNGRADED instead of scoring.
max_turns=${MAX_TURNS:-25}

# One agent loop worker per node. `ray_trainer.init_workers` round-robins the workers over
# the alive nodes, so this count is also what spreads Harbor containers across the cluster.
# The repo default is 1, which pins EVERY episode's containers to a single node: measured
# 347 containers and load 332 on one 384-core node while the other two sat at 3 and 0. Each
# LAPS task runs 4 MPI ranks, so that node was oversubscribed roughly 4x and
# `TmuxSession.start()` -- normally seconds -- blew past its 360 s budget, failing 79 of the
# first 94 episodes with `Agent setup timed out`. Raising the timeout would only have hidden
# the imbalance.
AGENT_LOOP_WORKERS=${AGENT_LOOP_WORKERS:-3}

# Max concurrent sequences admitted per rollout instance. This is the real admission gate:
# it is both PSRL's in-flight cap and, via oc.select in psrl_rollout.yaml, vLLM's own
# `max_num_seqs`. The repo default is 1024, so each engine accepted far more sequences than
# its KV could hold and the excess sat in the scheduler queue instead of decoding.
#
# Measured on H20: 8,796 KV blocks x 16 = 140,736 tokens per TP=2 engine, i.e. 562,944 across
# the 4 rollout engines. A 50-turn transcript reaches ~71k tokens, so the fleet holds only
# about 8 of them at once. With a 1024-deep window, generation ran at 0.31 turns/min per
# episode: median 28 turns and p90 46 after 90 minutes, so episodes died on wall clock before
# the 50-turn cap and were graded 0.0 for delivering nothing.
#
# 32 x 4 engines admits 128, which is exactly the request count at train_batch_size=8, so
# nothing queues on admission. Re-derive if train_batch_size changes. At 256 in flight, 40 per
# engine was measured to be too many: a turn-50 prefix is ~33k tokens and only 4 of those fit
# in an engine's 140,736-token KV pool, so requests waited past terminus-2's 900 s litellm
# timeout and that turn was discarded (389 such timeouts in one rollout, runner.py sets
# timeout 900 with max_retries 0). Dropping to 20 took that count to 0. Judge this by the
# litellm.Timeout count and the turn distribution, never by GPU utilization.
MAX_CONCURRENT_SEQS_PER_INSTANCE=${MAX_CONCURRENT_SEQS_PER_INSTANCE:-32}

# HTTP generation concurrency PER rollout engine. Kept well above
# MAX_CONCURRENT_SEQS_PER_INSTANCE so the admission gate above is what binds, not the HTTP
# client: this one only caps outstanding POSTs and cannot see the KV pool.
SERVER_MAX_CONCURRENCY=${SERVER_MAX_CONCURRENCY:-64}

# --- Chain-of-thought handling across turns ---
# One of multi_traj / longest_traj / multi_thinking / disable_thinking. See
# psrl/trainer/config/psrl/agentic_rl.yaml for what each does and
# examples/sciaccel_rl/knowledge.md for the measurements.
#
# multi_thinking: the reasoning parser is turned off so the whole generation, the model's
# own '</think>' included, stays inline in `content`. The accumulating template below then
# replays every prior turn's <think> block, so the history TITO stored and the history
# terminus-2 replays are byte-identical and the episode stays ONE trajectory whose prompt
# carries several CoT segments. That is what token-level on-policy training needs.
#
# The template MUST be the Qwen3.5 variant. Qwen3.5 prefills '<think>\n' into the
# generation prompt, so the model emits only a closing '</think>' and never a matched
# pair. qwen3_acc_thinking.jinja2 replays `content` verbatim, which on 3.5 would leave an
# unmatched '</think>' and drop the prefill. Verified on Qwen3.5-9B: with the 3.5 template
# prompt(k) + response(k) is a strict token prefix of prompt(k+1) for every turn, while
# the stock 3.5 template breaks that at token 10 by stripping prior-turn CoT.
thinking_template=multi_thinking
chat_template_path=${PSRL_PATH}/examples/sciaccel_rl/config/qwen35_acc_thinking.jinja2

# --- Deployment: 3 nodes x 8 GPU = 24 (8 generation + 16 training) ---
#
# Ray places bundles by available resources; nothing here pins a pool to a named host.
# `main_ppo.init_resource_pool_mgr` builds:
#   train_pool        = [TRAIN_NGPUS_PER_NODE] * TRAIN_NNODES,  bundle fraction 0.9
#   rollout_pool_<i>  = [GEN_NGPUS_PER_NODE_PER_INSTANCE] * GEN_NNODES_PER_INSTANCE
#   validate_pool_<i> = same shape, bundle fraction 1.0 - 0.9 = 0.1
#
# Validation therefore OVERLAYS the training GPUs rather than claiming its own, which is
# required here: 16 train + 8 gen already commits all 24 GPUs, so a dedicated validation
# pool would have nothing to sit on and Ray would hang waiting for placement.
#
# All three nodes need the Docker image cache warmed. Harbor containers run wherever the
# agent-loop process runs, and rollout is spread across the cluster, so any node can be
# asked to build a task environment. See prepare/provision_docker_nodes.sh.
NNODES=3
NGPUS_PER_NODE=8

# 8 GPUs of generation. TP=2 per instance: Qwen3.5-9B is multimodal, so it loads a vision
# tower even for text-only serving, and the 98304 window needs KV room -- measured 8,796
# KV blocks at TP=2 (140,736 tokens) versus 18,743 at TP=4. Four TP=2 instances keeps
# rollout parallelism while leaving each instance enough cache for a ~71k transcript.
GEN_TP=2
GEN_PP=1
GEN_NNODES=1
GEN_NGPUS_PER_NODE=8
GEN_INSTANCES=$(((GEN_NNODES * GEN_NGPUS_PER_NODE) / (GEN_TP * GEN_PP)))
GEN_NGPUS_PER_NODE_PER_INSTANCE=$((GEN_TP * GEN_PP))

# Ulysses sequence parallel degree. 1, i.e. OFF. This is empirical, not a preference.
#
# SP would be the right tool for a ~98k-token sequence, and verl does register the VLM slicing
# hook for this model (monkey_patch.py calls patch_vlm_for_ulysses_input_slicing on
# Qwen3_5TextModel). But its two halves disagree on token count on this path: the engine runs
# `ulysses_pad` on the rmpad input_ids and rolled labels, while the wrapper slices
# `inputs_embeds` built from a differently-shaped batch. Measured mismatches were 512 label
# rows against 447 embed rows, then 512 against 315 after forcing padding=True -- gaps of
# 65 and 1,576 tokens, so this is a structural disagreement, not a padding rounding error.
#
# SP also does NOT shard the lm_head even when it works: the engine gathers before the head,
# so with SP=8 and fused kernels off the OOM was 43.53 GiB, which reverses to 94,112 tokens
# rather than the sharded 12,288. Activations and logits therefore need SEPARATE fixes, and
# only the logits one (fused kernels, below) is reliable here.
TRAIN_SP=${TRAIN_SP:-1}
# FSDP shards the model across all 16 training GPUs.
TRAIN_FSDP=16
TRAIN_NNODES=2
TRAIN_NGPUS_PER_NODE=8

# Validation reuses the training GPUs at bundle fraction 0.1 (see colocate below), so
# these instances do NOT add to the 24-GPU budget. Keep the count modest: each instance
# is a full vLLM engine that has to fit alongside the FSDP shards already resident, and
# validation runs the 17 held-out tasks as full agentic episodes (~750 s each in the eval).
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
# Validation runs the 16 held-out tasks through full agentic episodes, which is expensive
# (the eval measured ~750 s median per trial), so keep it infrequent. `val_before_train` is
# off below, so the first validation lands at this step rather than at step 0.
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
