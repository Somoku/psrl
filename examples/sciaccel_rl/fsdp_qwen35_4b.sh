#!/usr/bin/env bash
# fsdp_qwen35_4b.sh — GRPO on the SciAccel-RL v2 task bank with Qwen3.5-4B.
#
# This is the configuration that actually trains. It was reached by fixing four real bugs
# and measuring every remaining knob, so the notes below record WHY each value is what it
# is rather than restating what it is.
#
# Why 4B rather than 9B: the 9B trains, but the packed sequence drives activation memory and
# the 9B could only afford a 49152 window, at which 12 of 13 episodes ended on
# `max_response_length_exceeded` instead of the turn cap -- the tail of nearly every episode
# was being discarded. The 4B has hidden_size 2560 against 4096, so activations run ~0.62x
# and a 66560 window fits. Measured after the switch: 12 of 12 episodes ended on
# `max_turns_exceeded` with ZERO context overflows, and rollout went from 3.4 to 5.2
# turns/min per episode.
#
# Prerequisites, in order:
#
#   1. Build the dataset:
#        python -m examples.sciaccel_rl.prepare.build_dataset_v2 \
#            --repo <sciaccel-rl> --out-dir examples/sciaccel_rl/data/v2 \
#            --categories repair implementation
#      The `--categories` filter drops laps-accel-cuda, which needs a GPU inside its Harbor
#      container and fails with "EnvironmentType.DOCKER environment does not support GPU
#      allocation" because all 24 GPUs are already committed to vLLM and FSDP.
#   2. Provision Docker on EVERY node, then warm the image cache on EVERY node. Harbor
#      containers run wherever the agent-loop process runs and rollout is spread across the
#      cluster, so any node can be asked to build a task environment:
#        bash examples/sciaccel_rl/prepare/provision_docker_nodes.sh --hosts <hostfile>
#        # then, on each node:
#        bash examples/sciaccel_rl/eval/run_eval.sh --agent nop \
#            --dataset examples/sciaccel_rl/data/v2/all.parquet --skip-gpu-tasks
#      The nop pass doubles as the anchor: expect every by_category mean score ~= 0 and an
#      empty floor_mismatch. Unwarmed, the first steps pay ~1 min per task; warmed, env_setup
#      is ~16 s.
#   3. Start the Ray cluster, head node first:
#        bash examples/ray/ray_start.sh ${PSRL_WORKSPACE}/hosts/24GPUs
#      That script passes `--num-cpus=32`, and each Harbor container runs 4 MPI ranks, so 32
#      advertised CPUs caps concurrent episodes at ~8 per node regardless of the 384 physical
#      cores. Raise it if rollout throughput is CPU-bound.
#   4. Run THIS script on the Ray head node. It needs the psrl conda env, so either source
#      ${PSRL_WORKSPACE}/env/psrl.sh first or launch it inside a shell that does -- running
#      under conda base picks python 3.12.7 and Ray rejects it against the cluster's 3.12.13.
#
# This script REQUIRES three verl patches, committed inside third_party/verl (that directory
# is not tracked by the outer repo):
#   * 460e61be -- get_rope_index attached to the processor class rather than the instance
#     (instance binding makes the processor unpicklable, which kills vLLM EngineCore's
#     process_input_sockets thread and hangs the cluster with no traceback); flash-attn's
#     2D-only Triton cross-entropy fed flattened logits; lm_head weights moved to the
#     activations device.
#   * 5d735263 -- rank-0 tqdm over the micro-batch loop, which is ~64% of a step and was
#     otherwise silent.
#
# Usage:
#   bash examples/sciaccel_rl/fsdp_qwen35_4b.sh
#   TRAIN_BATCH_SIZE=1 TOTAL_TRAINING_STEPS=5 bash examples/sciaccel_rl/fsdp_qwen35_4b.sh
#   HF_MODEL_PATH=/path/to/ckpt bash examples/sciaccel_rl/fsdp_qwen35_4b.sh
#
# Env overrides: HF_MODEL_PATH, TRAIN_BATCH_SIZE, TOTAL_TRAINING_STEPS, MAX_RESPONSE_LENGTH,
# MAX_TURNS, TRAIN_SP, TRAIN_FSDP, MAX_TOKENS_PER_GPU, SAVE_FREQ, TEST_FREQ, GROUP_FILTER,
# MAX_CONCURRENT_SEQS_PER_INSTANCE, SERVER_MAX_CONCURRENCY, AGENT_LOOP_WORKERS.

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
# Qwen3.5-4B, not 9B. Same architecture and the same 248,320-entry vocab, but hidden_size
# 2560 against 4096, so activations run ~0.62x: the 9B needed 87.6 GiB at 98,304 tokens, and
# the 4B should need ~37 GiB at 66,560. That is what buys back the context the 9B could not
# afford. Point HF_MODEL_PATH at the 9B to go back.
HF_MODEL_PATH=${HF_MODEL_PATH:-/apdcephfs_zwfy10_303541817/share_303541817/lhy/models/Qwen3.5-4B}
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
experiment_name=GRPO-sciaccel-v2-Qwen35-4B
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
train_batch_size=${TRAIN_BATCH_SIZE:-16}
rollout_N=8
# Instructions measured with the Qwen3.5-9B tokenizer over all 145 v2 tasks:
# 776 min, 1010 median, 1450 max. 2048 clears the longest with room to spare, and
# `data.truncation=error` means an under-sized value would crash rather than silently
# clip. Everything left over goes to the response, which is where the transcript grows.
max_prompt_length=2048
# 96256, giving a 98,304-token window. Raised from 64512 because 477 of 1024 episodes (47%)
# ended on `max_response_length_exceeded` rather than finishing, with response_length/mean pinned
# at 58-60k of the 64512 cap on every step -- a truncated distribution, not a natural one.
#
# The reason the window fills is that terminal output, not the model, dominates it. Measured over
# 1024 trajectories: prompt median 1,869 / assistant 24,182 / env 34,784 tokens, so 59% of the
# budget is env output that is masked out of the loss. That output is almost entirely gfortran
# source listings and compile errors (425 caret listings and 379 error lines across a 25-file
# sample), because the agent iterates compile-fix-compile and reads the whole listing back each
# time. Trimming that at the source would be worth more than any window increase, but this is the
# lever available without touching the agent.
max_response_length=${MAX_RESPONSE_LENGTH:-96256}
max_model_len=$(( max_prompt_length + max_response_length ))
# Token-packing budget for the training and log-prob forward passes, and the single biggest
# memory knob: `rearrange_micro_batches` asserts `max_token_len >= max_seq_len` against the
# LONGEST ACTUAL sequence, so this has to cover a maximal episode and cannot be set below the
# window.
#
# Do NOT raise it above max_model_len. Measured at 66560: allocated 66.18 GiB and reserved
# 78.65 GiB of 95, so only 17% headroom was left at the SMALLER window. Scaling that by the
# 1.48x token increase projects ~98 GiB allocated, which OOMs outright. Packing two maximal
# sequences per micro-batch is unaffordable here.
#
# Where the memory goes, since gradient checkpointing is already on: checkpointed layer inputs
# are only 32 x T x H x 2B = 10.2 GiB, and the T x V logits (30.8 GiB) never materialize because
# use_fused_kernels chunks them. The remaining bulk is in-layer recompute peaks, which scale with
# the tokens in a micro-batch -- which is exactly this number. Hence: keep it minimal.
max_tokens_per_gpu=${MAX_TOKENS_PER_GPU:-${max_model_len}}
if (( max_tokens_per_gpu < max_model_len )); then
    echo "ERROR: max_tokens_per_gpu (${max_tokens_per_gpu}) must be >= max_model_len (${max_model_len});" >&2
    echo "       rearrange_micro_batches asserts max_token_len >= max_seq_len and will abort." >&2
    exit 1
fi
max_num_batched_tokens=${max_model_len}
# 50. This was cut to 25 for the 9B at a 49152 window, where 12 of 13 episodes ended on
# `max_response_length_exceeded` with overflow prompts clustered just past the cap (median
# 49,207 against 49,152): the context was binding, not the turn budget, so spending turns was
# pointless. The 4B at 66560 inverts that, and the numbers are unambiguous.
#
# Measured over the first three training steps at 25 turns: 270 of 271 episodes ended on
# `max_turns_exceeded` and exactly one on `max_response_length_exceeded`, with
# prompt_length/mean at 17,933 of the 66,560 available. Not one episode finished on its own, so
# every trajectory was truncated mid-task and `reward_repair` was 0.0 across all 144 graded
# trials -- which makes every GRPO group zero-variance and the whole step a no-op. Meanwhile the
# eval, which runs 50 turns, solved 20 tasks outright on this same task bank.
#
# So the turn cap was the binding constraint and 27% context utilization was the evidence. The
# eval also measured delivery going 9% -> 36% across 25 -> 50 turns. Rollout wall clock roughly
# doubles, which is worth paying for a reward signal that is currently identically zero.
#
# Expect SOME overflow at 50, unlike at 25. Per-step prompt_length/max over those three steps
# was 45,602 / 66,515 / 54,333, so the longest episodes already reach the 66,560 ceiling at half
# this budget while the mean sits near 18k. Doubling turns therefore trades a minority of
# episodes ending on `max_response_length_exceeded` for the majority getting far enough to be
# graded at all. That is the right trade while reward is identically zero, but if overflows come
# to dominate the fix is a larger window (or a summarizing agent), not another turn cut.
#
# The cap itself is load-bearing at any value: terminus-2 defaults to 1000000 episodes, so
# without it an agent walks into the context limit and dies UNGRADED instead of scoring.
max_turns=${MAX_TURNS:-50}

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
#
# Watch this after the window went to 98,304. At GEN_TP=4 the KV pool is ~299,888 tokens, so 32
# concurrent maximal transcripts oversubscribe it ~10x, and only 3 maximal sequences fit at once.
# Left at 32 because the observed median is ~59k assistant+env rather than the 98k ceiling, so
# typical occupancy is ~5 sequences, and vLLM preempts rather than failing when it runs short.
# The failure mode is slow rollout, not a crash. If litellm.Timeout reappears, drop this first.
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

# Ulysses sequence parallel degree. 2, sharding each sequence across a GPU pair.
#
# This was 1 (OFF) because the fused kernel asserted on mismatched row counts, measured as 512
# logits rows against 447 and then 315 labels. That was a real double-slice, now fixed: the
# engine slices `input_ids_rmpad_rolled` at transformer_impl.py:1032 and passes it as
# `shift_labels` (:1108), and qwen3_5.py sliced it a SECOND time, leaving labels at 1/sp^2 while
# the hidden states, sliced once by patch_vlm_for_ulysses_input_slicing, stayed at 1/sp.
# dense_common.py never slices for exactly this reason -- upstream 575d5a8a defined
# `shift_labels` to mean "already prepared, do not touch" -- and qwen3_5.py was the only model
# file in the tree still calling `ulysses_pad_and_slice_inputs` on them.
#
# SP is what makes the 98,304-token window affordable at all. Measured at the old 66,560 window:
# 66.18 GiB allocated and 78.65 GiB reserved of 95, i.e. 17% headroom at the SMALLER size.
# Scaling by 1.48x projects ~98 GiB, which OOMs. SP=2 halves the activation term to ~51 GiB.
#
# 2 rather than 8: the group stays within a node so its all-gather is NVLink, and SP does NOT
# shard the lm_head (the engine gathers before the head), so a larger degree buys progressively
# less. With SP=8 and fused kernels off the OOM was 43.53 GiB, which reverses to 94,112
# unsharded tokens rather than the 12,288 a sharded head would imply. `use_fused_kernels=True`
# below is what actually keeps the logits off the card, and it stays on.
TRAIN_SP=${TRAIN_SP:-4}
# HSDP: shard within a node, replicate across the two training nodes. `create_device_mesh`
# builds a (ddp=world_size/fsdp_size, fsdp=fsdp_size) mesh whenever fsdp_size < world_size, and
# FSDP2 passes that mesh straight to `fully_shard`, so 8 gives a 2x8 mesh over the 16 training
# GPUs. Every all-gather then stays on NVLink and only the gradient reduce crosses IB.
#
# This is a bandwidth trade, not a memory saving: params+grads resident goes from 0.93 to
# 1.86 GiB per rank because each rank now holds 1/8 of the model instead of 1/16. That is
# affordable only because the 4B is small -- measured `After FSDP` was 1.07 GB allocated and
# 3.88 GB reserved, i.e. under 8% of the 48.1 GB peak, with activations holding the other 92%.
# Set this to 16 to go back to pure FSDP across both nodes.
TRAIN_FSDP=${TRAIN_FSDP:-8}
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
# A step costs about two hours here, so checkpoint often enough that a crash never throws away
# more than one. Keep this at or below TOTAL_TRAINING_STEPS or the run finishes having saved
# nothing: at the previous default of 25 a 12-step verification run produced no checkpoint at all.
save_freq=${SAVE_FREQ:-10}
# Validation runs the 16 held-out tasks through full agentic episodes, which is expensive
# (the eval measured ~750 s median per trial), so keep it infrequent. `val_before_train` is
# off below, so the first validation lands at this step rather than at step 0.
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
