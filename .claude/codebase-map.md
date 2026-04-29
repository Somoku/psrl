# PSRL Codebase Map

> **Purpose**: Fast-lookup reference for the entire PSRL project. Read this file first to orient yourself in any module.
> **Last updated**: 2026-03-31 | **Total**: ~158 source files in `psrl/`, 31 tests, 8 examples, 7 scripts

---

## 1. What Is PSRL

PSRL (Post-training Streaming RL) is a distributed LLM post-training framework. Core innovations:

- **Asynchronous training**: rollout generation and PPO training run concurrently, connected via a parameter server
- **Staleness control**: multi-buffer system tracks model versions; training consumes rollouts within an acceptable staleness window
- **GPU-direct communication (NIXL)**: model weights transfer GPU→GPU bypassing CPU/Ray serialization
- **Pluggable parallelism**: supports FSDP and Megatron-LM for training, vLLM for inference

---

## 2. System Architecture (Data Flow)

```
                         ┌──────────────────┐
                         │   main_ppo.py     │  ← Hydra entry point
                         │   TaskRunner      │  ← Ray driver
                         └────────┬─────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼              ▼
           ┌──────────────┐ ┌──────────┐ ┌────────────────┐
           │ DataProcessor │ │PSManager │ │ RewardManager  │
           │ (Ray actor)   │ │(Ray actor)│ │ (Ray actor)    │
           └──────┬───────┘ └────┬─────┘ └───────┬────────┘
                  │              │                │
    ┌─────────────┼──────────────┼────────────────┤
    ▼             ▼              ▼                ▼
┌────────┐  ┌─────────┐   ┌──────────┐    ┌──────────────┐
│GenWorker│  │GenWorker│   │TrainWorker│   │AgentLoopMgr  │
│(vLLM)  │  │(vLLM)  │   │(FSDP/Meg)│   │(multi-turn)  │
└────────┘  └─────────┘   └──────────┘    └──────────────┘
     │            │              ▲
     └────────────┴──── NIXL ───┘   (GPU-direct model push/pull)
```

**Loop**: TrainWorker pushes model → PSManager bumps version → GenWorkers pull new model → generate rollouts → RewardManager scores → PSManager buffer fills → DataProcessor batches → Trainer calls update → repeat

---

## 3. Directory Tree (Annotated)

```
psrl/
├── __init__.py
├── trainer/                      # ★ Orchestration layer
│   ├── main_ppo.py               # Hydra entry: ray.init → TaskRunner → trainer.fit()
│   ├── constants_ppo.py          # Env vars, Ray runtime config, resource names
│   ├── ppo/
│   │   ├── ray_trainer.py        # PSRL_RayPPOTrainer — main training loop (2500 lines)
│   │   └── utils.py              # Helper functions for trainer
│   └── config/                   # ★ Configuration system
│       ├── config.py             # Base dataclasses: CheckpointConfig, RewardManagerConfig
│       ├── algorithm.py          # AlgoConfig, KLControlConfig, RolloutCorrectionConfig
│       ├── cost_model/analyze.py # Performance cost model analysis
│       ├── *.yaml                # Hydra YAML configs (see §4)
│       └── __init__.py
│
├── workers/                      # ★ Distributed workers (Ray actors)
│   ├── gen/                      # --- Generation / Rollout ---
│   │   ├── gen_worker.py         # PSRL_GenWorker: vLLM inference + NIXL model pull
│   │   ├── verl_gen_worker.py    # verl-native generation worker
│   │   ├── vllm_rollout.py       # vLLM rollout wrapper
│   │   ├── vllm_extension.py     # vLLM engine extensions
│   │   ├── rollout_coordinator.py# Dispatches requests to GenWorker instances
│   │   ├── rollout_gateway.py    # HTTP gateway for rollout requests
│   │   ├── rollout_scheduler.py  # Schedules rollout across workers
│   │   ├── engine_http_server.py # HTTP server wrapping vLLM engine
│   │   └── stats_collector.py    # Rollout latency/throughput stats
│   │
│   ├── train/                    # --- Training ---
│   │   ├── base_train_worker.py  # PSRL_BaseTrainWorker: NIXL push, version mgmt
│   │   ├── fsdp_train_worker.py  # PSRL_FSDPTrainWorker (FSDP2 distributed training)
│   │   └── megatron_train_worker.py # PSRL_MegatronTrainWorker (Megatron-LM)
│   │
│   ├── ps/                       # --- Parameter Server ---
│   │   ├── ps_manager.py         # PSManager: version store + staleness inventory
│   │   ├── ps_storage_worker.py  # PSStorageWorker: holds model shards on GPU
│   │   ├── ps_worker_group.py    # Group of PS storage workers
│   │   ├── staleness_controller.py # StalenessInventory + StalenessBuffer (1264 lines)
│   │   ├── request_status_tracker.py # Tracks request lifecycle states
│   │   └── broadcast.py          # Model broadcast utilities
│   │
│   ├── reward/                   # --- Reward Computation ---
│   │   ├── reward_manager.py     # RewardManager: orchestrates reward pipeline
│   │   └── reward_loop/          # Reward algorithms
│   │       ├── base.py           # BaseRewardLoop
│   │       ├── naive.py          # NaiveRewardLoop (standard)
│   │       ├── dapo.py           # DAPORewardLoop (DAPO algorithm)
│   │       ├── prime.py          # PRIMERewardLoop (PRIME algorithm)
│   │       └── registry.py       # RewardLoop factory registry
│   │
│   ├── agent_loop/               # --- Multi-Turn Agent Loop ---
│   │   ├── manager.py            # PSRL_AgentLoopManager: top-level coordinator
│   │   ├── worker.py             # AgentLoopWorker: single worker process
│   │   ├── router.py             # Routes requests to workers
│   │   ├── route_strategy.py     # Routing strategies (round-robin, etc.)
│   │   ├── gateway_client.py     # Client for rollout gateway
│   │   ├── request_queue.py      # Async request queue
│   │   ├── sticky_session.py     # Session affinity for multi-turn
│   │   ├── prometheus_utils.py   # Monitoring metrics
│   │   ├── agent_data/           # Data structures for agent interactions
│   │   │   ├── base.py           # BaseAgentData
│   │   │   └── tool_agent_data.py# ToolAgentData (tool-use format)
│   │   └── loops/                # Agent loop implementations
│   │       ├── base_agent_loop.py       # BaseAgentLoop
│   │       ├── generate_agent_loop.py   # Single-turn generation
│   │       ├── batch_generate_agent_loop.py # Batched generation
│   │       ├── multi_turn_agent_loop.py # Multi-turn with tool calls
│   │       └── utils.py                 # Agent loop helpers
│   │
│   └── config/                   # Worker-level config dataclasses
│       ├── actor.py              # ActorConfig
│       ├── critic.py             # CriticConfig
│       ├── engine.py             # EngineConfig
│       ├── model.py              # ModelConfig
│       ├── optimizer.py          # OptimizerConfig
│       ├── reward_model.py       # RewardModelConfig
│       └── rollout.py            # RolloutConfig
│
├── tools/                        # ★ Tool-use infrastructure
│   ├── base.py                   # BaseTool abstract class
│   ├── mcp_tool.py               # MCP protocol tool implementation
│   ├── sandbox_fusion_tool.py    # SandboxFusion code execution tool
│   ├── utils.py                  # Tool utilities
│   ├── mcp_clients/              # MCP client management
│   │   ├── manager.py            # MCPClientManager
│   │   ├── schema.py             # MCP message schemas
│   │   └── token_bucket.py       # Rate limiting
│   └── tool_parser/              # Parse tool calls from LLM output
│       ├── base.py               # BaseToolParser
│       └── hermes_tool_parser.py # Hermes format parser
│
├── environments/                 # ★ Training environments
│   ├── base.py                   # BaseEnvironment
│   └── tool_env.py               # ToolEnvironment (tool-use training env)
│
├── utils/                        # ★ Shared utilities
│   ├── nixl/                     # GPU-direct communication (9 files)
│   │   ├── client.py             # NixlClient: send/recv model shards
│   │   ├── server.py             # NixlServer: listen for transfers
│   │   ├── comm_plan.py          # Communication plan optimizer
│   │   ├── nixl_spec.py          # Spec for NIXL transfers
│   │   ├── meta_buffer.py        # Metadata buffer management
│   │   ├── global_vars.py        # Global NIXL state
│   │   ├── network_topology.py   # Network topology discovery
│   │   └── port_scanner.py       # Port availability scanner
│   │
│   ├── converter/                # Model format conversion
│   │   ├── base_converter.py     # BaseConverter interface
│   │   ├── fsdp_converter.py     # HF ↔ FSDP state_dict
│   │   ├── megatron_converter.py # HF ↔ Megatron state_dict
│   │   ├── hf_converter.py       # HuggingFace format
│   │   ├── vllm_converter.py     # HF ↔ vLLM format
│   │   ├── model_mappings.py     # Key name mappings between formats
│   │   └── modeling/             # Per-format model loading
│   │       ├── fsdp_modeling.py
│   │       ├── hf_modeling.py
│   │       ├── megatron_modeling.py
│   │       └── vllm_modeling.py
│   │
│   ├── logger/                   # Logging subsystem
│   │   ├── __init__.py           # psrl_logger setup
│   │   ├── data_logger.py        # Training data logging
│   │   ├── env_logger.py         # Environment interaction logging
│   │   ├── memory_logger.py      # GPU memory usage logging
│   │   ├── ps_logger.py          # Parameter server logging
│   │   ├── ray_logger.py         # Ray cluster logging
│   │   └── deprecated.py         # Legacy logging
│   │
│   ├── dataset/                  # Data pipeline
│   │   ├── data_processor.py     # DataProcessor Ray actor
│   │   └── utils.py              # Dataset helpers
│   │
│   ├── post_processor/           # Post-processing filters
│   │   ├── base.py               # BasePostProcessor
│   │   ├── buffer_post_process/filter.py  # Buffer-level filtering
│   │   └── group_post_process/filter.py   # Group-level filtering
│   │
│   ├── profiling/                # Performance profiling
│   │   ├── collector.py          # Profile data collection
│   │   ├── analyzer.py           # Profile analysis
│   │   └── records.py            # Profile record types
│   │
│   ├── common/                   # Misc utilities
│   │   ├── dynamic_import.py     # Dynamic module loading
│   │   ├── http_utils.py         # HTTP client helpers
│   │   ├── memory_utils.py       # GPU memory management
│   │   ├── nixl_names.py         # NIXL naming conventions
│   │   ├── patch_utils.py        # Runtime monkey-patching
│   │   ├── serialization.py      # Object serialization
│   │   └── worker_naming.py      # Worker name generation
│   │
│   ├── ray/                      # Ray utilities
│   │   ├── lazy_primitives.py    # Lazy Ray object references
│   │   └── lock_context.py       # Distributed locking
│   │
│   ├── reward_score/
│   │   └── sandbox_fusion.py     # SandboxFusion reward scoring
│   │
│   ├── rollout/
│   │   └── rollout_trace.py      # Rollout tracing/debugging
│   │
│   ├── server/
│   │   └── command.py            # Server command utilities
│   │
│   └── visualization/            # Training visualization
│       ├── event_type.py         # Event type definitions
│       ├── log_stats.py          # Statistics aggregation
│       └── log_visualizer.py     # Visualization renderer
│
└── bench/                        # ★ Benchmarking
    ├── basic/cluster_scanner.py  # Cluster topology scanner
    └── rollout/                  # Rollout performance benchmarks
        ├── main_rollout.py
        ├── stats_collector.py
        └── vllm_extension.py
```

### Non-`psrl/` Directories

```
third_party/
├── vllm/           # Vendored vLLM v0.12.0 (4.1 GB) — inference engine, PSRL-patched
├── verl/           # Vendored verl (commit 3824689) — RLHF base framework
├── nixl/           # Vendored NIXL v0.10.1 — GPU-direct RDMA communication
└── torch_memory_saver/ # GPU memory pause/resume utilities

patch/
├── vllm/           # Git patches applied at install time to third_party/vllm
└── (plugin system) # Runtime TMS integration patches via env vars

scripts/
├── install_basic.sh     # Core deps + vLLM + verl (patches applied)
├── install_nixl.sh      # NIXL + UCX
├── install_tms.sh       # torch_memory_saver
├── install_megatron.sh  # Optional Megatron-LM
├── convert_hf_to_mcore.py  # HF→Megatron model conversion
└── docker/              # Docker image management

unit_tests/              # 31 test files across 11 categories
├── workers/ps/          # PS broadcast tests
├── fsdp1/, fsdp2/       # FSDP model loading tests
├── megatron/            # Megatron init tests
├── nixl/                # NIXL e2e, sharding, comm tests (7 files)
├── ray/                 # Ray primitives, locking, GPU sharing tests (9 files)
├── staleness/           # Staleness hash tests
├── state_dict/          # Converter tests (FSDP1, FSDP2, vLLM)
├── tools/               # MCP tool integration tests
├── torch_dist/          # Broadcast tests
└── trainer/             # Role tests

examples/
├── anaylsis/            # Cost model, plotting, regression analysis (7 files)
└── retool/retool.py     # Retool integration
```

---

## 4. Configuration System

**Hydra-based**, compositional YAML with Python dataclass validation.

### Config Hierarchy

```
ppo_trainer.yaml (FSDP template)          ppo_megatron_trainer.yaml (Megatron template)
    │                                          │
    ├── actor/actor.yaml  ──or──  actor/dp_actor.yaml  ──or──  actor/megatron_actor.yaml
    ├── critic/critic.yaml ──or── critic/dp_critic.yaml ──or── critic/megatron_critic.yaml
    ├── ref/ref.yaml                        ref/megatron_ref.yaml
    ├── reward_model/reward_model.yaml      reward_model/megatron_reward_model.yaml
    ├── rollout/rollout.yaml
    ├── data/data.yaml
    ├── model/hf_model.yaml
    ├── optim/fsdp.yaml                     optim/megatron.yaml
    ├── engine/fsdp.yaml                    engine/megatron.yaml
    ├── psrl/psrl.yaml                      (PSRL-specific: staleness, PS mode, deployment)
    ├── algorithm/rollout_correction.yaml
    └── reward_manager.yaml
```

### Launch Command

```bash
source /jizhicfs/lhy/env/psrl.sh
python -m psrl.trainer.main_ppo \
    ++trainer.total_epochs=10 \
    ++psrl.deployment.train_nnodes=2 \
    +train_actor_rollout_ref.model.path=/path/to/model
```

### Key Config Dataclasses (`algorithm.py`)

| Class | Fields | Purpose |
|-------|--------|---------|
| `AlgoConfig` | gamma, lam, adv_estimator, use_kl_in_reward | PPO algorithm hyperparams |
| `KLControlConfig` | type (fixed/adaptive), kl_coef, target_kl | KL divergence control |
| `RolloutCorrectionConfig` | rollout_is, rollout_rs, bypass_mode, use_policy_gradient | Off-policy correction presets |

**RolloutCorrection factory presets**: `.decoupled_token_is()`, `.pg_rs()`, `.decoupled_sequence_is()`, `.pg_sequence_rs()`, `.pg_geometric_rs()`

---

## 5. Key Classes Quick Reference

### Trainer Layer

| Class | File | Role |
|-------|------|------|
| `PSRL_RayPPOTrainer` | `trainer/ppo/ray_trainer.py` | Main training loop orchestrator (2500 lines) |
| `DataProcessor` | `utils/dataset/data_processor.py` | Ray actor: dataset loading, batching |

### Worker Layer

| Class | File | Role |
|-------|------|------|
| `PSRL_GenWorker` | `workers/gen/gen_worker.py` | vLLM inference + async model pulling |
| `PSRL_FSDPTrainWorker` | `workers/train/fsdp_train_worker.py` | FSDP2 PPO gradient updates |
| `PSRL_MegatronTrainWorker` | `workers/train/megatron_train_worker.py` | Megatron-LM PPO updates |
| `PSRL_BaseTrainWorker` | `workers/train/base_train_worker.py` | Shared train logic: NIXL push, version mgmt |
| `PSManager` | `workers/ps/ps_manager.py` | Model version store + staleness inventory |
| `PSStorageWorker` | `workers/ps/ps_storage_worker.py` | Holds model shards on GPU |
| `RewardManager` | `workers/reward/reward_manager.py` | Reward scoring pipeline |
| `PSRL_AgentLoopManager` | `workers/agent_loop/manager.py` | Multi-turn tool-use orchestration |

### Staleness System

| Class | File | Role |
|-------|------|------|
| `StalenessInventory` | `workers/ps/staleness_controller.py` | Collection of versioned buffers |
| `StalenessBuffer` | `workers/ps/staleness_controller.py` | Fixed-size buffer per model version |
| `EntryInfo` | `workers/ps/staleness_controller.py` | Metadata: rollout_id, prompt_id, model_version |
| `RequestStatusTracker` | `workers/ps/request_status_tracker.py` | Request lifecycle state machine |

### Enums

| Enum | Values | Purpose |
|------|--------|---------|
| `EntryCategory` | EMPTY → RESERVED → OCCUPIED | Entry lifecycle in staleness buffer |
| `BufferStatus` | READY, READY_WITH_CAPACITY, STUCK, PENDING | Buffer readiness for training |
| `PSRL_Role` | Actor, Rollout, Critic, RewardModel, RefPolicy, DummyPolicy, Validate | Worker role enum |

---

## 6. Core Flows

### 6.1 Startup

```
main_ppo.py::main()
  → Hydra resolves config
  → run_ppo(config)
    → ray.init(runtime_env=PPO_RAY_RUNTIME_ENV)
    → TaskRunner (Ray actor on head node)
      → add workers (GenWorker, TrainWorker, Critic, etc.)
      → init ResourcePoolManager (GPU allocation)
      → PSRL_RayPPOTrainer(config)
        → _validate_config()
        → _init_ps_manager()        # PSManager Ray actor
        → _init_data_processor()     # DataProcessor Ray actor
      → trainer.fit()               # ← main loop begins
```

### 6.2 Training Loop (`fit()`)

```
for step in range(total_steps):
  ① batch ← DataProcessor.get_batch()           # from staleness buffer (OCCUPIED entries)
  ② scores ← RewardManager.compute(batch)        # LLM / rule-based / sandbox
  ③ Apply KL penalty → token_level_rewards
  ④ advantages ← compute_advantage(GAE/GRPO/REINFORCE++)
  ⑤ critic_wg.update_critic(batch)               # value function update
  ⑥ actor_wg.update_actor(batch)                 # PPO policy gradient
  ⑦ Periodic: validate, checkpoint, log
```

### 6.3 Asynchronous Rollout + Staleness

```
TrainWorker completes update
  → nixl_push_model() [async background thread, GPU-direct]
  → PSManager.push_model_state_dict_nixl(version=N)
  → PSManager creates StalenessBuffer(version=N, size=batch_size)

GenWorker sees new version
  → Pulls model weights via NIXL (GPU→GPU)
  → Reserves entries: EMPTY → RESERVED
  → Generates rollouts
  → Fills entries: RESERVED → OCCUPIED
  → Buffer status: PENDING → READY (when enough entries occupied)

DataProcessor
  → Queries PSManager for READY buffers
  → Consumes entries → clears → EMPTY
  → Old buffers deleted when safe
```

### 6.4 NIXL Model Transfer

```
TrainWorker (rank 0..N):
  ① get_fsdp_full_state_dict() / megatron equivalent
  ② nixl_push_model() → background thread per rank
     ├─ For each key: NixlClient.write(shard → PSStorageWorker GPU)
     ├─ NixlClient.wait() for completion
     └─ PSStorageWorker.transfer_train_to_gen_merged()
  ③ dist.barrier() across ranks
  ④ Rank 0 → PSManager.push_model_state_dict_nixl(version)

GenWorker (on demand):
  ① Await version_ready_event[version]
  ② NixlClient.read(PSStorageWorker GPU → local GPU)
  ③ Load into vLLM engine
```

### 6.5 Multi-Turn Agent Loop

```
AgentLoopManager
  → Spawns AgentLoopWorkers
  → Router distributes requests (round-robin / sticky session)

AgentLoopWorker per request:
  ① Format prompt with tool definitions
  ② Send to GenWorker (via gateway_client)
  ③ Parse LLM output for tool calls (HermesToolParser)
  ④ Execute tool (MCP / SandboxFusion)
  ⑤ Append tool result to conversation
  ⑥ Repeat ②-⑤ until done or max_turns
  ⑦ Return full trajectory for reward scoring
```

---

## 7. Reward System

| Loop | File | Algorithm |
|------|------|-----------|
| `NaiveRewardLoop` | `reward_loop/naive.py` | Standard: single reward per response |
| `DAPORewardLoop` | `reward_loop/dapo.py` | DAPO: dynamic reward with advantage weighting |
| `PRIMERewardLoop` | `reward_loop/prime.py` | PRIME: process reward model integration |

**Registry**: `reward_loop/registry.py` — factory pattern, select by config string.

**Reward sources**: LLM-as-judge, rule-based, SandboxFusion (code execution), custom functions.

---

## 8. Model Format Converter

Converts model state_dict between 4 formats:

```
HuggingFace (HF)  ←→  FSDP  ←→  vLLM
                   ←→  Megatron-LM
```

| Converter | File | Direction |
|-----------|------|-----------|
| `FSDPConverter` | `utils/converter/fsdp_converter.py` | HF ↔ FSDP (flat keys ↔ sharded) |
| `MegatronConverter` | `utils/converter/megatron_converter.py` | HF ↔ Megatron (TP/PP sharding) |
| `VLLMConverter` | `utils/converter/vllm_converter.py` | HF ↔ vLLM |
| `HFConverter` | `utils/converter/hf_converter.py` | HuggingFace canonical format |
| `model_mappings.py` | `utils/converter/model_mappings.py` | Key name mapping tables |

---

## 9. Third-Party Dependencies

| Dependency | Version | Location | Purpose | PSRL Patches |
|-----------|---------|----------|---------|-------------|
| **vLLM** | v0.12.0 | `third_party/vllm/` | LLM inference engine | GPU memory, attention, TMS integration |
| **verl** | commit 3824689 | `third_party/verl/` | RLHF base framework | PPO core, DataProto, worker groups |
| **NIXL** | v0.10.1 | `third_party/nixl/` | GPU-direct RDMA comm | — |
| **torch_memory_saver** | — | `third_party/torch_memory_saver/` | GPU memory pause/resume | vLLM plugin patches |

**Patch system**: Two-tier
1. **Git patches** (`patch/vllm/*.patch`): applied at install time via `scripts/install_basic.sh`
2. **Runtime plugins**: monkey-patches via env vars for TMS integration

---

## 10. Testing Map

```
unit_tests/
├── workers/ps/         → PS broadcast logic
├── fsdp1/, fsdp2/      → FSDP model loading correctness
├── megatron/           → Megatron model init
├── nixl/               → NIXL e2e, sharding, comm planner, send/recv (7 tests)
├── ray/                → Ray primitives, locking, GPU sharing, multi-thread (9 tests)
├── staleness/          → Staleness hash correctness
├── state_dict/         → Converter round-trip tests (FSDP1, FSDP2, vLLM)
├── tools/              → MCP tool integration
├── torch_dist/         → Distributed broadcast
└── trainer/            → Role enum tests
```

---

## 11. Key Conventions

| Convention | Rule |
|-----------|------|
| **Logging** | `psrl_logger = logging.getLogger(__file__)` — never `print()` |
| **Comments** | `# NOTE(author):`, `# TODO(author):`, `# FIXME(author):`, `# HACK(author):` |
| **Docstrings** | Google-style, opening `"""` on its own line |
| **Assertions** | Always include descriptive message with period |
| **Linting** | Ruff, line length 119 |
| **Language** | English only in all code artifacts |
| **Env activation** | `source /jizhicfs/lhy/env/psrl.sh` before any command |

---

## 12. File Size Landmarks (by complexity)

| File | Lines | Why it matters |
|------|-------|---------------|
| `trainer/ppo/ray_trainer.py` | ~2500 | THE main training loop — start here for any training question |
| `workers/ps/staleness_controller.py` | ~1264 | THE staleness system — core async innovation |
| `workers/ps/ps_manager.py` | ~600 | Central coordination point for all workers |
| `workers/gen/gen_worker.py` | ~500 | vLLM generation + model pull logic |
| `workers/train/base_train_worker.py` | ~300 | NIXL push + train-side version management |
| `workers/agent_loop/manager.py` | ~400 | Multi-turn agent orchestration |
| `utils/nixl/client.py` | ~400 | GPU-direct transfer implementation |

---

## 13. Import Dependency Graph (Simplified)

```
main_ppo.py
  → ray_trainer.py (PSRL_RayPPOTrainer)
    → workers/gen/gen_worker.py (PSRL_GenWorker)
    → workers/train/fsdp_train_worker.py (PSRL_FSDPTrainWorker)
    → workers/train/megatron_train_worker.py (PSRL_MegatronTrainWorker)
    → workers/ps/ps_manager.py (PSManager)
    → workers/reward/reward_manager.py (RewardManager)
    → workers/agent_loop/manager.py (PSRL_AgentLoopManager)
    → utils/dataset/data_processor.py (DataProcessor)
    → trainer/config/config.py + algorithm.py

gen_worker.py
  → utils/nixl/client.py (NixlClient)
  → workers/gen/vllm_rollout.py → third_party/vllm

base_train_worker.py
  → utils/nixl/client.py (NixlClient)
  → utils/converter/*.py (model format conversion)
  → third_party/verl (ActorRolloutRefWorker base class)

ps_manager.py
  → workers/ps/staleness_controller.py (StalenessInventory)
  → workers/ps/ps_storage_worker.py (PSStorageWorker)
  → workers/ps/request_status_tracker.py

agent_loop/manager.py
  → agent_loop/worker.py → agent_loop/loops/*.py
  → tools/mcp_tool.py, tools/sandbox_fusion_tool.py
  → environments/tool_env.py
```

---

## 14. Quick "Where Do I Look?" Index

| I want to... | Go to |
|--------------|-------|
| Understand the training loop | `trainer/ppo/ray_trainer.py::fit()` |
| Change PPO hyperparameters | `trainer/config/algorithm.py` + `algorithm/rollout_correction.yaml` |
| Debug model version issues | `workers/ps/staleness_controller.py` |
| Debug model transfer failures | `utils/nixl/client.py`, `workers/train/base_train_worker.py` |
| Add a new reward function | `workers/reward/reward_loop/` + register in `registry.py` |
| Modify vLLM inference | `workers/gen/gen_worker.py`, `workers/gen/vllm_rollout.py` |
| Add a new tool for agent training | `tools/` directory, register in `tools/base.py` |
| Change data loading | `utils/dataset/data_processor.py` |
| Add new model architecture support | `utils/converter/` (add converter + modeling) |
| Modify FSDP training | `workers/train/fsdp_train_worker.py` |
| Modify Megatron training | `workers/train/megatron_train_worker.py` |
| Change GPU resource allocation | `trainer/constants_ppo.py`, `ResourcePoolManager` in `ray_trainer.py` |
| Add monitoring/metrics | `utils/logger/`, `utils/profiling/`, `utils/visualization/` |
| Run benchmarks | `psrl/bench/` |
| Run tests | `unit_tests/` (pytest) |
| Install from scratch | `scripts/install_basic.sh` → `install_nixl.sh` → `install_tms.sh` |