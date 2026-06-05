# Configuration

PSRL uses [Hydra](https://hydra.cc/) with [OmegaConf](https://omegaconf.readthedocs.io/)
for hierarchical, composable configuration management.

## Configuration System

All configuration lives under `psrl/trainer/config/`. Hydra composes a single merged
config from multiple YAML files at runtime, allowing you to:

- Override individual parameters from the command line
- Swap entire config groups (e.g., switch from FSDP to Megatron backend)
- Use variable interpolation (e.g., `${psrl.staleness}`) across config files

## Top-Level Config

The entry point is `psrl/trainer/config/ppo_trainer.yaml`, which composes the following
groups via Hydra defaults:

| Group | Config Key | Config Path | Description |
|---|---|---|---|
| `psrl` | `psrl` | `psrl/psrl.yaml` | PSRL-specific settings (staleness, deployment, routing) |
| `actor` | `train_actor_rollout_ref.actor` | `actor/dp_actor.yaml` | Actor model training config |
| `rollout` (train) | `train_actor_rollout_ref.rollout` | `rollout/rollout.yaml` | Rollout config used by training workers |
| `rollout` (gen) | `gen_actor_rollout_ref.rollout` | `rollout/rollout.yaml` | Rollout config used by generation workers |
| `data` | `data` | `data/data.yaml` | Dataset and dataloader config |
| `ref` | `train_actor_rollout_ref.ref` | `ref/dp_ref.yaml` | Reference model config |
| `model` (train) | `train_actor_rollout_ref.model` | `model/hf_model.yaml` | HuggingFace model loading config (trainer) |
| `model` (gen) | `gen_actor_rollout_ref.model` | `model/hf_model.yaml` | HuggingFace model loading config (gen worker) |
| `critic` | `critic` | `critic/dp_critic.yaml` | Critic model config |
| `reward_model` | `reward_model` | `reward_model/dp_reward_model.yaml` | Reward model config |
| `reward_manager` | `reward_manager` | `reward_manager.yaml` | Rule-based reward function config |
| `algorithm` | `algorithm` | `algorithm/rollout_correction.yaml` | Algorithm hyperparameters (PPO/GRPO/DAPO) |
| `trainer` | `trainer` | *(inline in ppo_trainer.yaml)* | Training loop settings (epochs, logging, checkpoints) |

For Megatron-LM training, use `ppo_megatron_trainer.yaml` instead, which
selects Megatron-specific actor, critic, and model configs with TP/PP/CP/EP support.

## veRL-Managed Configuration

Most of the config groups above (`train_actor_rollout_ref`, `data`, `algorithm`,
`trainer`, `rollout`, `critic`, `reward_model`, etc.) are inherited directly from
[veRL](https://github.com/volcengine/verl) with minimal changes.

:::{tip}
For the full reference of veRL-managed config groups, including actor training
hyperparameters, optimizer settings, rollout sampling parameters, data loading,
algorithm coefficients, and trainer loop settings, refer to the official veRL
configuration documentation:

**→ [veRL Configuration Reference](https://verl.readthedocs.io/en/latest/examples/config.html)**
:::

PSRL extends veRL with the `psrl` config group (described below) and adds a
`gen_actor_rollout_ref` sub-tree for configuring the decoupled generation cluster
separately from the training cluster.

---

## PSRL Config Reference

The primary PSRL configuration file is `psrl/trainer/config/psrl/psrl.yaml`. Below is a
categorized reference of all parameter groups.

### Top-Level Parameters

`colocate`
: Whether to colocate rollout and training workers on the same nodes/GPUs. When `True`,
  training and generation share the same GPU set, TMS is used to swap memory between
  phases.
  **Default:** `False`

`ps_manager_ip`
: IP address of the Parameter Server Manager process. All inter-component communication
  (NIXL, LMCache Controller, reward service) defaults to this address.
  **Default:** `127.0.0.1`

`reward_service_ip`
: IP address of the reward scoring service.
  **Default:** `${psrl.ps_manager_ip}`

`logging_path`
: Base directory for all PSRL log outputs (trajectory dumps, profiling files, etc.).
  **Default:** `~/psrl_logs`

### Core Settings

`staleness`
: Maximum version gap between generation and training. `0` = fully synchronous
  (generation blocks until training consumes). Values `>0` allow that many rollout
  buffers to be generated ahead of training consumption.
  **Default:** `0`

`staleness_buffer_entries`
: Number of prompts in each staleness buffer (effective training batch size).
  Each buffer must be fully filled before it can be consumed by training.
  **Default:** `512`

`rollout_n`
: Number of responses generated per prompt. Set `>1` for GRPO/DAPO group sampling
  (e.g., 8).
  **Default:** `1`

`ps_mode`
: Parameter server weight synchronization mode.
  - `cpu_ref`: CPU-based reference model (simpler setup, no NIXL required)
  - `nixl_cpu`: GPU-direct RDMA via NIXL (recommended for production)

  **Default:** `cpu_ref`

### Deployment

Resource allocation for training and rollout clusters.

`deployment.n_rollout_instances`
: Number of independent rollout (generation) vLLM instances.
  **Default:** `1`

`deployment.n_validate_instances`
: Number of validation rollout instances (typically colocated with training).
  **Default:** `1`

`deployment.rollout_nnodes_per_instance`
: Nodes allocated to each rollout instance.
  **Default:** `1`

`deployment.rollout_ngpus_per_node_per_instance`
: GPUs per node for each rollout instance.
  **Default:** `1`

`deployment.validate_nnodes_per_instance`
: Nodes allocated to each validation instance.
  **Default:** `1`

`deployment.validate_ngpus_per_node_per_instance`
: GPUs per node for each validation instance.
  **Default:** `1`

`deployment.train_nnodes`
: Nodes allocated to the training cluster.
  **Default:** `1`

`deployment.train_ngpus_per_node`
: GPUs per node in the training cluster.
  **Default:** `1`

`deployment.total_nnodes`
: Total nodes in the job. When set, excess nodes are blocked from scheduling to
  prevent colocated validate workers from spilling onto idle nodes. Set to match
  `NNODES` in your launch script.
  **Default:** `null`

`deployment.heterogeneous_rollout`
: Enable per-instance configuration of rollout resources. When `enable: True`, each
  rollout instance can be individually configured:

  | Sub-field | Description |
  |---|---|
  | `enable` | Master switch. **Default:** `False` |
  | `n_rollout_instances` | Mirrors `psrl.deployment.n_rollout_instances` |
  | `rollout_nnodes_per_instance` | List of per-instance node counts (length = `n_rollout_instances`) |
  | `rollout_ngpus_per_node_per_instance` | List of per-instance GPU counts |
  | `tensor_model_parallel_size_per_instance` | List of per-instance TP sizes |
  | `pipeline_model_parallel_size_per_instance` | List of per-instance PP sizes |

### Colocate & Fuse Settings

`colocate_validate_and_train`
: Whether to colocate validation and training workers on the same nodes.
  **Default:** `True`

`fuse_rollout_with_validate`
: Whether to dispatch generation requests to both rollout and validate instances,
  effectively using validate instances as extra rollout capacity.
  **Default:** `True`

### Status Collection

The rollout coordinator collects real-time engine statistics to enable smart routing
and sync decisions.

`status_collection.enable`
: Whether to enable engine status collection.
  **Default:** `True`

`status_collection.engine_sync_interval_in_ms`
: How often each vLLM engine pushes its status to the rollout coordinator (ms).
  **Default:** `100`

`status_collection.coordinator_sync_interval_in_ms`
: How often the coordinator aggregates engine statuses and pushes to the router (ms).
  **Default:** `100`

`status_collection.dump_logging_to_file_level`
: Granularity of status logs written to disk. Options: `none`, `partial_rollout`,
  `prompt`, `generation`, `all`.
  **Default:** `all`

`status_collection.dump_logging_to_file_interval_in_ms`
: File logging flush interval (ms).
  **Default:** `500`

### Partial Rollout

Allows generation to be interrupted and resumed, preventing long sequences from
blocking the training pipeline.

`partial_rollout.enable`
: Whether to enable partial rollout interruption.
  **Default:** `True`

`partial_rollout.interrupt_as_prompt`
: If `True`, interrupted trajectories are treated as new prompts (the partial
  generation becomes part of the next prompt). If `False`, they are retried from
  scratch.
  **Default:** `False`

```{seealso}
{doc}`../design/flexible_rollout`, Detailed design of the partial rollout mechanism
and how it interacts with the staleness system.
```

### Redundant Rollout

Generates more trajectories than needed for training, allowing the system to select the
best subset and discard redundant or slow samples.

`redundant_rollout.enable`
: Whether to enable redundant rollout generation.
  **Default:** `False`

`redundant_rollout.alg_global_batch_size`
: Required batch size for the algorithm (buffers are considered ready at this size).
  **Default:** `${psrl.staleness_buffer_entries}`

`redundant_rollout.alg_rollout_n`
: Number of responses required by the algorithm per prompt.
  **Default:** `${psrl.rollout_n}`

`redundant_rollout.redundant_global_batch_size`
: Actual number of trajectory *prompts* generated (must be ≥ `alg_global_batch_size`).
  **Default:** `${psrl.staleness_buffer_entries}`

`redundant_rollout.redundant_rollout_n`
: Actual number of responses generated per prompt (must be ≥ `alg_rollout_n`).
  **Default:** `${psrl.rollout_n}`

### Routing Strategy

Controls how generation requests are dispatched across rollout instances.

`routing_strategy.method`
: Routing algorithm. Options:
  - `random`: uniform random assignment
  - `round_robin`: cyclic assignment
  - `request_num_balance`: route to the instance with fewest active requests
  - `throughput_optimal`: maximize global throughput using a cost model
  - `throughput_optimal_with_budget`: throughput-optimal with per-request token budget
  - `kv_cache_aware`: route based on KV cache utilization and prefix overlap

  **Default:** `request_num_balance`

`routing_strategy.kv_transfer`
: When re-routing a request to a different instance, optionally transfer its
  accumulated KV cache via LMCache P2P to avoid re-prefill. Requires
  `lmcache.enable` and `lmcache.enable_p2p`.

  | Sub-field | Description |
  |---|---|
  | `enable` | Master switch. **Default:** `False` |
  | `transfer_mode` | `async` (fire-and-forget), `sync` (await, no pin), `pin_sync` (pin→await→unpin). **Default:** `async` |
  | `transfer_timeout_ms` | Timeout for `sync`/`pin_sync` modes before falling back to re-prefill. **Default:** `5000` |

`routing_strategy.cost_model_path`
: Path to a JSON cost model file (required for `throughput_optimal` methods).
  **Default:** `null`

`routing_strategy.request_sort_indicator`
: How to prioritize requests within a routing cycle. Options: `short_length`,
  `long_length`, `small_id`.
  **Default:** `small_id`

`routing_strategy.candidate_sort_indicator`
: How to sort candidate instances. Options: `version`, `reserve_capability`.
  **Default:** `version`

`routing_strategy.enable_multi_priority_queue`
: Use separate queues for different request priorities.
  **Default:** `False`

`routing_strategy.enable_group_sampling_on_multi_instances`
: Allow a single prompt group's responses to be distributed across multiple
  rollout instances for load balance.
  **Default:** `True`

`routing_strategy.delta_throughput_threshold`
: Stop routing new requests to an instance when its marginal throughput contribution
  drops below this fraction.
  **Default:** `0.5`

`routing_strategy.request_budget`
: Estimated token budget per request, used by `throughput_optimal_with_budget` to
  predict response length.
  **Default:** `1024`

`routing_strategy.kv_query_timeout_ms`
: Timeout (ms) for querying KV cache info from each candidate instance in
  `kv_cache_aware` mode. Instances that time out receive score 0 (graceful
  degradation), does not affect correctness.
  **Default:** `30000`

`routing_strategy.check_interval_in_ms`
: Polling interval for the routing loop (ms).
  **Default:** `500`

`routing_strategy.max_concurrent_seqs_per_instance`
: Hard cap on concurrent sequences per instance for routing decisions.
  **Default:** `1024`

```{seealso}
{doc}`../design/flexible_rollout`, Routing strategy design, cost models, and load
balancing benchmarks.
```

### Sync & Migration

Controls when model weights are synchronized and when rollout requests are migrated
between instances for load balancing.

`sync_and_mig_strategy.method`
: Strategy for deciding sync/migration timing.
  - `status_based`: use instance status indicators to decide when to sync
  - `greedy`: sync as soon as a new version is available

  **Default:** `greedy`

`sync_and_mig_strategy.check_interval_in_ms`
: Polling interval for the sync/migration loop (ms).
  **Default:** `100`

`sync_and_mig_strategy.sync.indicator`
: Metric that triggers weight sync. Options: `request_num`, `throughput`, `kv_cache`,
  `hypothesis_test`.
  **Default:** `request_num`

`sync_and_mig_strategy.sync.threshold`
: Workload threshold below which model sync is triggered. Interpretation depends on
  the `indicator` (count, tokens/s, or utilization fraction).
  **Default:** `null`

`sync_and_mig_strategy.sync.check_req_before_sync`
: Before syncing, verify that no routeable requests are pending for this instance.
  **Default:** `True`

`sync_and_mig_strategy.sync.seamless_train_version`
: All model versions ≤ this value are guaranteed to have a ready buffer waiting, so
  training can proceed immediately after weight pull without stalling.
  **Default:** `0`

`sync_and_mig_strategy.mig.enable`
: Whether to enable request migration between rollout instances.
  **Default:** `False`

`sync_and_mig_strategy.mig.indicator`
: Metric used to identify imbalanced instances for migration. Options: `request_num`,
  `throughput`, `kv_cache`.
  **Default:** `request_num`

`sync_and_mig_strategy.mig.threshold`
: Relative imbalance ratio (`max_indicator / min_indicator`) that triggers migration.
  **Default:** `null`

`sync_and_mig_strategy.mig.stop_indicator`
: Metric used to decide when to stop migrating.
  **Default:** `request_num`

`sync_and_mig_strategy.mig.stop_threshold`
: Threshold on `stop_indicator` below which migration halts.
  **Default:** `null`

### Proactive Filter

Handles situations where a buffer is nearly ready but a few remaining requests are
straggling.

`proactive_filter_strategy.method`
: Strategy for handling straggling requests.
  - `null`: disabled (wait indefinitely)
  - `retry`: abort and re-dispatch straggling requests
  - `truncate`: mark buffer as ready with fewer entries

  **Default:** `null`

`proactive_filter_strategy.threshold`
: Number of remaining reserved entries below which the filter strategy activates.
  **Default:** `0`

```{seealso}
{doc}`../design/staleness_control`, How proactive filtering integrates with the
Reserve/Occupy/Consume staleness protocol.
```

### NIXL

Configuration for RDMA-based weight synchronization via NIXL (used when
`ps_mode: nixl_cpu`).

`nixl.server_ip`
: NIXL server IP address.
  **Default:** `${psrl.ps_manager_ip}`

`nixl.server_port`
: NIXL server port.
  **Default:** `23456`

`nixl.max_pinned_temp_memory_slots`
: Number of pinned temporary memory slots for non-contiguous tensor transfers.
  Increase if you hit registration contention with many concurrent PS workers.
  **Default:** `16`

`nixl.enable_tms_for_temp_buffers`
: Manage NIXL temporary buffers with TMS, simplifying re-registration after memory
  is resumed.
  **Default:** `${psrl.tms.enable_nixl}`

### Checkpoint

Controls the Megatron checkpoint save/load strategy. Relevant only when using the
Megatron-LM training backend.

`checkpoint.use_dcp_save`
: Whether to use verl's default DCP (Distributed Checkpointing) for save/load.

  - **`False` (default)**: Use PSRL's per-rank `torch.save` (saves `rank_N.pt` +
    `parallel_config.json` per rank). This path is UCX-safe: it avoids DCP's
    `FullyParallelSaveStrategyWrapper` which calls `all_gather_object` on all shard
    metadata, causing a large temporary allocation that can corrupt NIXL's UCX
    endpoint memory under high memory pressure (manifests as
    `addr_version assertion` SIGABRT). The NIXL background UCX progress thread
    (`enable_prog_thread`) is kept **enabled** in this mode.

  - **`True`**: Use verl's DCP path. Two patches are applied automatically:
    1. **NCCL no-fork patch**: DCP's async writer normally forks child processes
       that inherit NCCL communicators; when the child exits, `ncclCommAbort`
       corrupts the parent's NCCL state → 600-second timeout → SIGABRT. The patch
       replaces the forking multiproc writer with a sequential in-process version.
    2. **NIXL prog_thread disabled**: `enable_prog_thread=False` is passed to the
       NIXL agent to prevent the UCX background thread from racing with DCP's
       `all_gather_object` memory activity.

  **Default:** `False`

### LMCache

KV cache offloading and cross-instance P2P transfer for reducing re-prefill overhead
in multi-turn workloads.

`lmcache.enable`
: Master switch for LMCache KV offloading in vLLM.
  **Default:** `False`

`lmcache.backend`
: Storage backend for offloaded KV blocks.
  - `cpu`: host memory (fast, limited by DRAM)
  - `disk`: filesystem-backed (large capacity, slower)

  **Default:** `cpu`

`lmcache.offload_size_gb`
: Total offload budget in GiB, divided automatically across TP ranks. Do **not**
  set `LMCACHE_MAX_LOCAL_CPU_SIZE` as an env var, that would apply the full budget
  to every rank.
  **Default:** `10.0`

`lmcache.chunk_size`
: Token chunk size for hash-based KV indexing (must divide the block size).
  **Default:** `256`

`lmcache.clear_on_weight_update`
: Evict all cached KV entries after each model weight pull from the PS. Prevents
  stale-weight KV from being reused in the next generation round.
  **Default:** `True`

`lmcache.reserve_local_cpu_size`
: GiB of CPU memory to keep free and never use for KV offloading (headroom for other
  processes on the same node).
  **Default:** `0.0`

`lmcache.save_decode_cache`
: Also cache KV from decode steps (not just prefill). Increases memory usage but
  improves multi-turn prefix reuse.
  **Default:** `True`

`lmcache.cache_policy`
: Eviction policy: `LRU` or `FIFO`.
  **Default:** `LRU`

`lmcache.enable_async_loading`
: Overlap KV cache retrieval with prefill computation to reduce time-to-first-token.
  **Default:** `True`

`lmcache.config_file`
: Path to a full LMCache YAML config. When set, **overrides all individual fields
  above**.
  **Default:** `null`

**Disk backend (when `backend: disk`)**

`lmcache.disk_path`
: Filesystem path for disk-backed KV storage. Required when `backend: disk`.
  **Default:** `null`

`lmcache.max_disk_size_gb`
: Maximum disk usage for KV storage (GiB).
  **Default:** `50.0`

**P2P cross-instance transfer**

`lmcache.enable_p2p`
: Enable cross-instance KV cache transfer via a shared LMCache Controller process.
  Required when `routing_strategy.kv_transfer.enable: True`.
  **Default:** `False`

`lmcache.p2p_transfer_channel`
: Transport for P2P KV transfer.
  - `nixl`: UCX-based (RDMA on multi-node, shared memory on same node). Recommended.
  - `tcp`: fallback when UCX is unavailable.

  **Default:** `nixl`

`lmcache.controller_host`
: Host where the LMCache Controller runs.
  **Default:** `${psrl.ps_manager_ip}`

`lmcache.controller_base_port`
: Base HTTP port for the LMCache Controller's REST API (`/move`, `/lookup`, etc.).
  The actual port is selected via `find_available_port()` starting here.
  **Default:** `9000`

`lmcache.controller_pull_port`
: ZMQ PULL port where the Controller listens for worker registrations and heartbeats.
  **Default:** `8300`

`lmcache.controller_reply_port`
: ZMQ REPLY port for Controller → worker task dispatch.
  **Default:** `8400`

```{seealso}
{doc}`../design/kv_cache`, KV cache management architecture, LMCache Controller
process, and cache eviction behavior.
```

### TMS (torch_memory_saver)

GPU memory management that transparently swaps idle tensors to CPU, enabling colocated
workloads to share GPU memory.

`tms.range`
: Scope of TMS management.
  - `null`: disabled
  - `train`: manage training worker memory only
  - `all`: manage both rollout and training worker memory

  **Default:** `all`

`tms.enable_cuda_graph`
: Release CUDA graphs via TMS when not in use. Requires `range: all`.
  **Default:** `False`

`tms.enable_nixl`
: Manage NIXL temporary buffers with TMS (simplifies re-registration after resume).
  **Default:** `True`

### Agentic RL

Settings for multi-turn agent training loops (tool-use, code generation, SWE-agent).

`agentic_rl.sticky_session`
: Pin all generation calls within a single trajectory to the same rollout instance.
  Improves prefix-cache hit rate for multi-turn conversations.
  **Default:** `False`

`agentic_rl.trajectory_output.enable`
: Write per-trajectory `.txt` files to disk for debugging and analysis.
  **Default:** `True`

`agentic_rl.trajectory_output.dir`
: Output directory for trajectory files. When empty, defaults to
  `${psrl.logging_path}/trajectories`.
  **Default:** `""` (auto)

`agentic_rl.manager_retry_on_error`
: On rollout errors, retry via the manager instead of crashing the worker. On
  validation failure, manager shrinks `val_buffer_size` so the waiter is unblocked.
  **Default:** `True`

### Broadcast Init

When loading a large model checkpoint, each PS worker normally reads from disk
independently. `broadcast_init` instead has rank-0 read the checkpoint and broadcast
weights to other PS workers via NIXL, reducing filesystem load at scale.

`broadcast_init.enabled`
: Enable rank-0 broadcast initialization.
  **Default:** `False`

`broadcast_init.algorithm`
: Broadcast algorithm. Currently only `binary_tree` is supported.
  **Default:** `binary_tree`

### Group & Buffer Post-Processing

Post-processors can filter, re-weight, or transform trajectory groups before they are
submitted to the staleness buffer.

`group_post_process.enable`
: Enable streaming group-level post-processing.
  **Default:** `False`

`group_post_process.name`
: Registered post-processor name. Options: `dynamic_sampling_filter`, `no_filter`.
  When using `dynamic_sampling_filter`, requires `algorithm.filter_groups.metric` to
  be set.
  **Default:** `null`

`buffer_post_process.enable`
: Enable batch-level buffer post-processing (applied when a full buffer is ready).
  **Default:** `False`

`buffer_post_process.name`
: Same options as `group_post_process.name`.
  **Default:** `null`

### Log Probability

`log_prob.enable_rollout_engine_log_prob`
: Whether to request token log-probabilities from the vLLM rollout engine (used for
  importance sampling corrections). Disable to reduce generation overhead when
  log-probs are not needed.
  **Default:** `True`

### Server Rollout

An optional HTTP gateway that exposes PSRL's rollout service externally (useful for
serving agent loops from non-PSRL clients).

`server_rollout.enable`
: Enable the server rollout HTTP gateway.
  **Default:** `False`

`server_rollout.gateway.router_ip`
: Bind address for the gateway process.
  **Default:** `${psrl.ps_manager_ip}`

`server_rollout.gateway.router_port`
: HTTP port for the gateway.
  **Default:** `18080`

`server_rollout.server_concurrency`
: Max concurrent HTTP connections per rollout server.
  **Default:** `64`

### Memory Logger

`memory_logger.enable`
: Enable periodic GPU memory logging for debugging memory pressure.
  **Default:** `False`

`memory_logger.interval_seconds`
: Logging interval (seconds).
  **Default:** `30`

---

## Overrides

Override any parameter from the command line using Hydra syntax:

```bash
python -m psrl.trainer.main_ppo \
    +psrl.staleness=3 \
    psrl.routing_strategy.method=throughput_optimal \
    psrl.deployment.n_rollout_instances=4 \
    psrl.lmcache.enable=True
```

Key syntax rules:

- `key=value`: Override an existing key
- `+key=value`: Add a new key not present in the default config
- `~key`: Remove a key from the config
- Use dot notation for nested keys: `psrl.routing_strategy.method=...`

:::{tip}
For complex experiments, create a separate YAML file with your overrides and pass it
with `--config-path`:

```bash
python -m psrl.trainer.main_ppo \
    --config-path=/path/to/my_overrides \
    --config-name=my_experiment
```
:::

## Example: Advanced 7B FSDP Config

Here is a representative override pattern for 4-node DAPO training from
`examples/dapo_trainer/advanced_qwen2.5_7b_fsdp.sh`:

```bash
python -m psrl.trainer.main_ppo \
    --config-path=./config --config-name='ppo_trainer' \
    psrl.staleness=2 \
    psrl.staleness_buffer_entries=64 \
    psrl.ps_mode=nixl_cpu \
    psrl.rollout_n=8 \
    psrl.deployment.n_rollout_instances=16 \
    psrl.deployment.train_nnodes=2 \
    psrl.deployment.total_nnodes=4 \
    psrl.partial_rollout.enable=True \
    psrl.redundant_rollout.enable=True \
    psrl.routing_strategy.method=throughput_optimal \
    psrl.routing_strategy.enable_multi_priority_queue=True \
    psrl.sync_and_mig_strategy.method=status_based \
    psrl.sync_and_mig_strategy.sync.indicator=kv_cache \
    psrl.sync_and_mig_strategy.mig.enable=True \
    psrl.proactive_filter_strategy.method=retry \
    psrl.proactive_filter_strategy.threshold=4
```

```{seealso}
- {doc}`quickstart`: Minimal working example with DAPO
- {doc}`../design/staleness_control`: Staleness control design
- {doc}`../design/flexible_rollout`: Routing and rollout coordination
- {doc}`../design/kv_cache`: KV cache management
```
