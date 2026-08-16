# veRL Upgrade and Rebase Guide

This document is the source of truth for rebasing PSRL onto a new veRL revision. It replaces the legacy
`CHECKLIST.md`, which describes the retired Python router, `PSRL_GenWorker`, and separate FSDP/Megatron workers.

## 1. Supported architecture and version boundary

PSRL does **not** use veRL's direct HTTP rollout path as its production path. The supported request path is:

```text
PSRL agent/reward client
  -> SMG HTTP API and SessionRouter
  -> SMG Rust gRPC router and PSRL worker selector
  -> smg_grpc_servicer.VllmEngineServicer
  -> PSRL_vLLMHttpServer / vLLM AsyncLLM engine
```

The control plane is:

```text
PSRL RolloutCoordinator
  -> SMG routing-loop pause/resume, worker status, and weight-version endpoints
  -> SMG PSRL selector
  -> PSManager gRPC admission/reservation service
```

Consequently, a veRL rollout change must be classified before it is ported:

1. **Engine construction or worker lifecycle:** port it to PSRL.
2. **Tokenization, multimodal preprocessing, sampling, logprobs, or routed-expert semantics:** port it, preserving the
   SMG gRPC wire contract.
3. **veRL HTTP server, direct `generate()`, PD dispatch, or veRL's own rollout router:** do not copy it mechanically.
   Record why SMG already owns the behavior or explicitly add equivalent SMG support.
4. **Training, metrics, checkpoint, dataset, config, or worker APIs:** rebase normally against veRL.

Every released PSRL revision must pin a compatible tuple, not independent moving branches:

```text
(veRL commit, veRL patch, vLLM version/patch, SMG commit and generated protobuf packages)
```

At the time of this audit, PSRL targets veRL `bf48903d93e4618531d3bbae96551a889007dd8b` and vLLM `v0.22.0`.
The Docker build still uses the floating SMG ref `psrl-dev`; replace that with a tested SMG commit before release.

## 2. Rebase classes

Use these labels in upgrade reviews:

- **Fork:** PSRL copied substantial veRL logic and owns a divergent implementation. Perform a three-way semantic
  merge for every upgrade.
- **Subclass/adapter:** PSRL inherits veRL code. Check constructor signatures, protected methods, mutable fields, and
  lifecycle ordering.
- **Extracted logic:** PSRL moved part of a veRL method into a different actor or pipeline stage. Compare by behavior,
  not by filename.
- **Patched upstream:** behavior is implemented inside `third_party/verl` by `patch/verl/<commit>.patch`.
- **API consumer:** PSRL does not copy the implementation, but relies on the symbol, data schema, or side effects.
- **PSRL/SMG owned:** no code rebase is expected, but its interface must be tested against the upgraded veRL/vLLM
  boundary.

## 3. Core training and orchestration modules

| PSRL module | veRL source to compare | Class | PSRL-owned behavior | Upgrade checks |
| --- | --- | --- | --- | --- |
| `psrl/trainer/main_ppo.py` | `verl/trainer/main_ppo.py`, `verl/trainer/main_ppo_v0.py` | Fork | Separate train/rollout pools, GPU-slot reservation, TransferQueue bootstrap, PSManager, SMG gateway, reward services, and elastic resources | Ray runtime env, logging initialization, device setup, worker-role selection, shutdown, and config validation |
| `psrl/trainer/constants_ppo.py` | `verl/trainer/constants_ppo.py` | Fork | PSRL logging and NIXL/vLLM defaults | Platform-specific env, determinism propagation, Megatron-only CUDA settings, ROCm/Blackwell behavior, and RLInsight flags |
| `psrl/trainer/ppo/ray_trainer.py` | `verl/trainer/ppo/ray_trainer.py` and `verl/trainer/ppo/v1/trainer_base.py` | Fork + subclass | Asynchronous busy-loop training, staleness buffers, TransferQueue metadata, NIXL weight publication, split train/gen workers, SMG lifecycle, multi-reward, fine-grained overlap, and elastic RM | Constructor state, `init_workers`, validation, checkpoint save/load, profiling, logprob/value/advantage/update stages, metrics, and the full `fit()` lifecycle |
| `psrl/trainer/ppo/strategies/` | Stage ordering in veRL `RayPPOTrainer.fit()` / v1 trainer | Extracted logic | Full-batch and mini/micro-batch overlap, with logical-step LR scheduling and delayed model publication | New batch-coupled stages, optimizer-step boundaries, scheduler advancement, rollout correction, distillation, padding, and checkpoint timing |
| `psrl/trainer/ppo/utils.py` | `verl/trainer/ppo/core_algos.py`, `verl/trainer/ppo/v1/utils.py`, and veRL resource-pool management | Fork | `parent_id` grouping, PSRL roles, fractional resource bundles, multi-trajectory final-session advantage propagation, and rollout/RM timing metrics | New advantage estimators and kwargs, `uid`/session-key semantics, padding IDs, response masks, and resource accounting |
| `psrl/utils/dataset/data_processor.py` | `RayPPOTrainer._create_dataloader`, `_save_checkpoint`, `_load_checkpoint`, and the data section of `fit()` | Extracted logic | A Ray actor that independently samples prompts, checkpoints loader state, drives TransferQueue, supports multi-dataset ratios, retries, and streaming batches | Dataloader state format, `gen_batch_size`, curriculum/sampler hooks, over-sampling, validation batching, and new per-sample fields |
| `psrl/utils/dataset/utils.py` | `verl/trainer/ppo/utils.py::{create_rl_dataset,create_rl_sampler}` | Fork | PSRL dataset selection and multi-dataset construction | Keep factory signatures and sampler checkpoint semantics aligned; eliminate the current `print()` while touching this file |
| `psrl/utils/dataset/rl_dataset.py` | `verl/utils/dataset/rl_dataset.py` | Subclass | PSRL tool-schema injection and deferred overlong-prompt filtering | Constructor fields, async filtering, multimodal/tool preprocessing, serialization, `data_source`, and `reward_model_dicts` |

### Trainer invariants that must survive a rebase

- PSRL groups GRPO samples by `parent_id`, not veRL's default `uid`.
- A logical optimizer step may contain several fine-grained chunks. LR schedulers and PS model-version publication advance
  exactly once per logical step.
- `DataProcessor`, not `RayPPOTrainer`, owns train-dataloader state.
- Training consumes `KVBatchMeta`/TransferQueue fields; do not reintroduce whole-batch Ray serialization.
- Padding rows need collision-free IDs and must not affect advantages, metrics, routed experts, or staleness accounting.

## 4. Config and worker API modules

| PSRL module | veRL source to compare | Class | PSRL-owned behavior | Upgrade checks |
| --- | --- | --- | --- | --- |
| `psrl/trainer/config/ppo_trainer.yaml` and `ppo_megatron_trainer.yaml` | veRL trainer templates and generated configs | Fork | Split `train_actor_rollout_ref` / `gen_actor_rollout_ref`, PSRL groups, TransferQueue, and separate deployment topology | Diff defaults lists and all dataclass `_target_` values; compose both DP/FSDP and Megatron configs |
| `psrl/trainer/config/rollout/psrl_rollout.yaml` | `verl/trainer/config/rollout/rollout.yaml` | Overlay | SMG async mode, pooling reward models, PSRL tools/agent data, routed replay, TMS, and rollout coordination | Ensure every inherited field still exists and every PSRL override has the expected type |
| `psrl/trainer/config/__init__.py` | `verl/trainer/config/__init__.py` | API consumer | Re-exports veRL trainer config classes | Detect renamed/deleted exports and avoid creating stale local duplicates |
| `psrl/workers/config/__init__.py` | `verl/workers/config/__init__.py` and submodules | API consumer | Re-exports common veRL configs and PSRL-only rollout/reward configs | Import every exported symbol and instantiate Hydra targets |
| `psrl/workers/config/rollout.py` | `verl/workers/config/rollout.py` | Subclass/adapter | Adds pooling, TMS weight backup, PSRL multi-turn, environment, and agent-data config | Base dataclass field order, `_mutable_fields`, nested default factories, and OmegaConf conversion |
| `psrl/workers/config/model.py` | `verl/workers/config/model.py` | Stale fork | Historical support for file-backed chat templates | This file is not the package-level `HFModelConfig` export. Prefer removing it or making the extension explicit; do not silently maintain two incompatible `HFModelConfig` classes |
| `psrl/workers/config/reward_model.py` | veRL reward/model/engine configs | API consumer + PSRL owned | Multiple heterogeneous reward models and manager selection | Hydra target types, tokenizer/model config fields, and engine strategy names |

The two Hydra search-path plugins, `hydra_plugins/psrl_searchpath.py` and
`psrl/trainer/config/hydra_plugins/psrl_searchpath.py`, must remain identical. They are what allow PSRL overlays to resolve
veRL config groups.

## 5. Training workers and checkpoint integration

| PSRL module | veRL source to compare | Class | PSRL-owned behavior | Upgrade checks |
| --- | --- | --- | --- | --- |
| `psrl/workers/train/engine_train_worker.py` | `verl/workers/engine_workers.py` | Subclass/adapter | Unified actor/ref worker plus PSRL NIXL, TMS sleep/wake, parameter conversion, PS restoration, and model publication | Base constructors, process-group ownership, `init_model`, `update_actor`, return schemas, LR scheduler flag, and worker decorators |
| `PSRL_CriticTrainWorker` in the same file | `verl.workers.engine_workers.TrainingWorker` | Subclass/adapter | Initializes the global process group before the critic engine is constructed | Recheck whenever veRL moves process-group initialization between `__init__` and `init_model` |
| `psrl/workers/train/base_train_worker.py` | No direct veRL equivalent; called from veRL worker lifecycle | PSRL owned/API boundary | NIXL push/pull protocol, async completion/error propagation, PS version management | Engine parameter iteration, optimizer reload, non-persistent buffers, and ordering relative to veRL update/scheduler calls |
| `psrl/utils/checkpoint/megatron_saver.py` | veRL `MegatronCheckpointManager` and Megatron DCP | PSRL owned + patched upstream | UCX-safe synchronous per-rank save/load | Manifest format, topology validation, optimizer/RNG layouts, and round-trip tests |
| `psrl/utils/converter/` | veRL engine state dicts, model merger, Megatron Bridge | API consumer + PSRL owned | FSDP/Megatron/HF/vLLM conversions, optimizer conversion, custom layouts | Parameter names, sharding descriptors, MTP, routed experts, bridge APIs, and supported model architectures |

## 6. Rollout and SMG gRPC boundary

| PSRL module | veRL/SMG source to compare | Class | PSRL-owned behavior | Upgrade checks |
| --- | --- | --- | --- | --- |
| `psrl/workers/gen/vllm_async_server.py` | veRL `vllm_async_server.py`, `vllm_rollout/utils.py`, `vLLMReplica`; SMG `VllmEngineServicer` and proto | Fork + subclass | gRPC serving, PS admission/status, NIXL pulls, LMCache, weight-version gating, abort/drain, routed experts, and SMG registration | Base constructor/protected methods, engine args, MTP, multimodal placeholder handling, sampling defaults, server naming, platform env, and proto fields |
| `psrl/workers/gen/vllm_rollout.py` | `verl/workers/rollout/vllm_rollout/vllm_rollout.py::ServerAdapter` | Subclass | Node/GPU identity for split train/rollout resources | Adapter constructor and sleep/wake/update interfaces |
| `psrl/workers/gen/vllm_extension.py` | `verl/.../vllm_rollout/utils.py::vLLMColocateWorkerExtension` | Subclass | NIXL weight loading, LMCache pin/unpin, local TP/EP identity, and TMS hooks | `monkey_patch_model`, quantization/weight-name resolution, worker-extension RPCs, and vLLM model-loader changes |
| `psrl/workers/gen/smg_adapter.py` | SMG `bindings/python/.../router_args.py` and Python binding constructor | PSRL/SMG owned | Converts Hydra config to SMG args, endpoint constants, worker-registration and weight-version payloads | Every `RouterArgs` field, enum string, endpoint path, and payload key |
| `psrl/workers/gen/rollout_gateway.py` | SMG Python binding and launch lifecycle | PSRL/SMG owned | Launches the native Router and SessionRouter subprocesses | Startup readiness, shutdown, bind addresses, native extension ABI, and error propagation |
| `psrl/workers/gen/rollout_coordination/` | SMG routing-loop controller and worker APIs | PSRL/SMG owned | Pause/resume, sync/migrate strategies, stats, partial rollout, and ThunderAgent scheduling | Endpoint contract, atomic pause/drain/version-update/resume ordering, failure rollback, and instance IDs |
| `psrl/grpc/ps_manager_service.py` | SMG `crates/psrl_state/proto/psrl_manager.proto` and PSRL selector | PSRL/SMG owned | Admission, reservation, request status, aborted-version checks, and instance model versions | Regenerate Python/Rust stubs together and test every RPC in both directions |
| `psrl/utils/tito/training_data.py` and session loop | SMG `crates/tito`, gRPC response processing, and session endpoints | PSRL/SMG owned | Converts SMG TITO records into training tensors, masks, logprobs, and routed experts | Token ownership, prompt override rules, turn boundaries, truncation, contiguous routed-expert coverage, and serialization |

### SMG protocol invariants

- `(base_worker_id, dp_rank)` is the canonical rollout-instance identity.
- `version_tag`/weight-version updates must be applied before a worker becomes routable.
- Pause -> drain/abort -> weight pull -> publish actual version -> resume is a failure-atomic sequence. A failed pull must
  leave the worker unavailable.
- Partial rollout preserves token IDs, logprobs, routed experts, and the sticky/migration hint across loopback.
- Request `priority` is a signed optional value end to end. Unset, zero, and negative values are distinct; lower values
  are scheduled first by vLLM. New gateway APIs must not silently replace it with a transport-local default.
- Session creation carries the exact Hugging Face `model_type` in `x-smg-tito-model-type`; SMG must persist it for the
  session. Families with model-specific boundary semantics must have an exact adapter and fail closed when unsupported;
  model types without special boundary rules use the identity adapter.
- Routed experts use a compact C-contiguous `uint8` or `uint16` three-dimensional tensor on the gRPC wire. PSRL must
  validate shape, dtype, coverage, and compatibility with veRL's signed `int16` training representation before training.
- Direct `PSRL_vLLMHttpServer.generate()` and direct pooling are intentionally rejected. SMG owns inference dispatch.
- veRL PD-disaggregation behavior is not automatically part of PSRL. Supporting it requires an explicit SMG design and
  protocol change.

## 7. Reward, agent, tool, and trace forks

| PSRL module | veRL source to compare | Class | PSRL-owned behavior | Upgrade checks |
| --- | --- | --- | --- | --- |
| `psrl/workers/reward/reward_loop/{base,naive,dapo,gdpo}.py` | `verl/experimental/reward_loop/reward_manager/` | Fork | TensorDict/TransferQueue inputs, merged agent/tool metadata, and PSRL result schemas | Method signatures, padding, score assembly, extra-info fields, and error behavior |
| `psrl/workers/reward/reward_loop/prime.py` | veRL legacy PRIME reward manager | Fork | Async PSRL reward interface | Scorer kwargs, response masking, and metadata |
| `psrl/workers/reward/reward_loop/registry.py` | veRL reward registry/loader | Fork | PSRL registry plus custom reward-loop loading | Module config schema, sync/async callable adapters, and external object loading |
| `psrl/workers/reward/reward_manager.py` and `reward_worker.py` | `verl/experimental/reward_loop/reward_loop.py` | Parallel fork | Distributed dispatch, retries, TransferQueue field updates, command handling, and multiple reward specs | Worker construction, padding/divisibility, result aggregation, cancellation, and reward-model routing |
| `psrl/workers/reward/reward_model/manager.py` | `verl/experimental/reward_loop/reward_model.py` | Parallel fork | SMG-backed reward-model replicas and gateways | Replica config, tokenizer, registration, process groups, and pooling response schema |
| `psrl/utils/reward_score/__init__.py` and `sandbox_fusion.py` | veRL reward-score registry and SandboxFusion helpers | Fork | Async scoring and PSRL sandbox/tool integration | Available scorer modules, timeout/result schema, and dependency moves |
| `psrl/utils/rollout/rollout_trace.py` | `verl/utils/rollout_trace.py` | Fork | PSRL prompt/request IDs, richer Trackio output, dataclass/enum serialization, and PSRL agent output support | Decorator call signatures, token-to-text input/output copying, async behavior, and backend APIs |
| `psrl/tools/function_tool.py` | `verl/tools/function_tool.py` | Fork | Adapts veRL schema inference to PSRL `Tool`/`ToolOutput` and reward/metrics | Decorator forms, schema types, tuple normalization, async calls, and tool file loading |
| `psrl/tools/tool_parser/{gpt_oss,gemma4,qwen3_coder}_tool_parser.py` | `verl/experimental/agent_loop/tool_parser.py` | Fork | Dict-based schemas and PSRL `ToolCall` | Stop tokens, `tool_call_id`, malformed arguments, assistant/tool-message reconstruction, and newly supported model families |
| `psrl/workers/agent_loop/prometheus_utils.py` | `verl/workers/rollout/utils.py::update_prometheus_config` | Fork | Updates Prometheus configs on all Ray nodes | Signature, rollout labels, reload behavior, and logging |
| `psrl/workers/agent_loop/loops/`, `agent_data/`, and `worker.py` | veRL experimental agent loop | Parallel fork | SMG gateway calls, session/TITO mode, PSRL environments, trajectory rewards, retries, and TransferQueue output | Chat-template/token continuity, multimodal expansion, priority, routed experts, empty-token padding, tool-call messages, and output fields |

Do not replace PSRL agent loops with veRL's direct LLM-client implementation. Rebase semantic fixes into the PSRL
gateway/TITO design. In particular, new veRL continuous-token wiring must be evaluated at both prompt construction and
TITO reconstruction boundaries.

## 8. veRL patch inventory

The patch filename must match the exact installed veRL commit so `patch/apply_patch.sh` selects it deterministically.
For `bf48903d`, the patch currently changes these behaviors:

| Patched veRL area | PSRL requirement |
| --- | --- |
| `protocol.py` | Merge arbitrary `*_metrics` from distributed workers without losing per-worker values |
| `single_controller/ray/base.py` | Fractional GPU resources per placement-group bundle for PSRL colocation |
| data/model/engine YAML | No implicit data shuffle, shared-memory model loading, `load_weight`, and per-rank checkpoint switches |
| `trainer/ppo/metric_utils.py` | PSRL max-length metadata, per-data-source original reward metrics, and non-numeric validation metadata |
| `trainer/ppo/padding_utils.py` | Unique numeric padding IDs, `parent_id`, and routed-expert-safe minimal padding |
| `utils/dataset/rl_dataset.py` | Multi-dataset source identity, reward-model dictionaries, and over-sampling |
| `utils/checkpoint/megatron_checkpoint_manager.py` | PSRL per-rank checkpoint backend and manifest integration |
| `utils/megatron/dist_checkpointing.py` | No-fork async filesystem writer to avoid inherited NCCL/UCX corruption |
| `utils/tokenizer/tokenizer.py` | Correct processor method binding for model-specific position-ID methods |
| `workers/config/engine.py` | `load_weight` and `use_per_rank_checkpoint` dataclass fields |
| FSDP/Megatron transformer engines | Dummy/no-weight construction plus TMS weight/optimizer regions |
| `workers/engine_workers.py` | Process-group lifecycle split and logical-step LR scheduler control |

Rules for patch maintenance:

1. Start from a clean checkout of the target veRL commit.
2. Re-implement each requirement against the new upstream structure; do not blindly apply the previous diff.
3. Keep PSRL imports inside veRL patches minimal. They make veRL importability depend on PSRL installation order.
4. Export one patch named after the full or unambiguous short commit.
5. Update the matching Docker patch and verify they are byte-identical.
6. Include focused patch tests in the PSRL repository. If a test remains under `third_party/verl/tests`, explicitly copy
   it into the produced patch or move it to PSRL tests; an untracked test is not part of the deliverable.
7. Validate with `git apply --check` against an archive/clean checkout of the exact commit.

Old commit-named patches are historical artifacts for old pinned revisions. Do not delete them merely because a new
revision is added, and do not apply multiple commit patches in sequence.

## 9. Current `bf48903d` adaptation audit

### Already adapted in the current working tree

- Install scripts and Dockerfiles select veRL `bf48903d`; `patch/verl/bf48903.patch` applies cleanly to that commit.
- `patch/verl/bf48903.patch` and `docker/patch/verl.patch` are byte-identical.
- Tokenizer/chat-template import moves are reflected in active PSRL agent code.
- The unified `ActorRolloutRefWorker`/`TrainingWorker` API is used; PSRL adds a critic process-group wrapper.
- The vLLM server adaptation includes the new platform env source, multimodal banned-token patching, MTP config helper,
  vLLM 0.22 routed-expert guard, and compilation-config field changes.
- Direct vLLM generation/pooling was removed from the PSRL server in favor of the SMG gRPC path.
- Null `data.gen_batch_size` is handled explicitly.
- Fine-grained overlap can suppress intermediate LR scheduler advancement.
- Routed-expert metrics and the SMG/TITO payload path have been added across trainer, agent loop, and TITO conversion.

### Gaps resolved in the current adaptation

1. **Removed replica API:** the obsolete `_validate_launch_requirements()` call was removed from
   `PSRL_vLLMReplica.launch_servers()`; launch validation now remains with the current veRL/vLLM construction path.
2. **Runtime environment and entry point:** PSRL now delegates the platform/config-sensitive baseline to veRL's
   `get_ppo_ray_runtime_env()`, overlays only PSRL defaults, configures full determinism/RLInsight before `ray.init()`,
   and initializes veRL logging in the task runner. PSRL no longer forces `CUDA_DEVICE_MAX_CONNECTIONS` globally.
3. **Agent-loop semantics:** the gateway request carries normalized signed priority; the manager assigns deterministic
   per-batch priorities when absent; multimodal placeholder runs are collapsed before decode; assistant tool calls and
   message identity fields survive reconstruction; empty/short routed-expert outputs are validated without ambiguous
   tensor truth tests. Session/TITO remains the continuous-token source of truth for the production path.
4. **TITO model contract:** PSRL sends the exact Hugging Face `model_type` when creating a session. SMG persists the
   header and selects the exact model adapter, rather than guessing from the served model name.
5. **SMG scheduling contract:** `GenerateRequest` now has an optional signed `priority` field. The Rust HTTP client,
   protobuf, and Python vLLM servicer preserve unset versus zero and pass the value to vLLM's scheduler.
6. **Trace propagation:** all affected SMG unary and streaming gRPC calls inject the current OpenTelemetry context;
   PSRL trace token-to-text conversion is input-aware and copies result containers instead of mutating caller data.
7. **Routed experts:** PSRL rejects `uint16` values outside veRL's signed `int16` training representation and validates
   exact token-row coverage. SMG rejects malformed present payloads and preserves routed experts through partial
   rollout and TITO serialization.
8. **Config composition:** reward-path environment interpolation was corrected. Fully resolved FSDP/vLLM and
   Megatron/vLLM Hydra compositions succeed against the upgraded veRL tree.

### Remaining release blockers (not code-rebase gaps)

1. **Pin the SMG artifact:** the SMG compatibility changes are still in a dirty working tree and Docker uses the
   floating `psrl-dev` ref. Commit the reviewed SMG tree, regenerate/publish all Rust/Python protobuf packages from that
   commit, then pin the resulting SHA in PSRL. Do not pin the current SMG HEAD because it does not contain these changes.
2. **Package the checkpoint test:** the per-rank checkpoint test under `third_party/verl/tests` is not included in the
   produced veRL patch. Move the durable test to PSRL or include it in the patch artifact.
3. **Run distributed acceptance:** local CPU/static validation cannot prove CUDA/NCCL/NIXL/vLLM behavior. Run the exact
   pinned tuple through the GPU matrix in Phase E before release.

Local validation for this adaptation includes Python syntax compilation, focused PSRL contract tests, fully resolved
Hydra FSDP and Megatron compositions, touched-file Rust/Python formatting and linting, generated Python protobuf tests,
gRPC-client tests, TITO tests, partial-rollout tests, and routed-expert router tests.
The repository pytest entry point exits with signal 11 (status 139) during local collection in the available macOS
environment; the same focused test functions pass when invoked directly. Treat the GPU/Linux suite as mandatory rather
than interpreting this environment failure as either a product failure or a successful runtime test. The full SMG
`cargo test` reaches the existing multimodal suite but fails a macOS bit-exact image assertion by one ULP; the affected
PSRL/SMG crates and router filters pass independently. Full-feature Clippy additionally requires system OpenCV and is
blocked by a pre-existing `kv-index` acronym lint, while the changed gRPC/TITO crates pass `-D warnings`.

## 10. Upgrade procedure

### Phase A: establish clean baselines

1. Record old and new veRL commits and obtain a clean checkout of each.
2. Record the exact vLLM and SMG commits and generated protobuf package versions.
3. Save the PSRL and SMG working-tree status. Never generate a release patch from an unexplained dirty dependency tree.
4. Diff old..new veRL for every upstream file listed in this document, including renamed files.

### Phase B: rebuild the veRL patch

1. Apply each requirement in Section 8 to the clean new veRL checkout.
2. Run focused tests for DataProto concat, fractional Ray resources, padding, multi-reward dataset fields, engine
   no-weight initialization, TMS regions, LR scheduling, DCP no-fork behavior, and per-rank checkpoint round trips.
3. Export `patch/verl/<new-commit>.patch` and copy the same bytes to `docker/patch/verl.patch`.
4. Run `git apply --check` against another clean checkout of the new commit.
5. Update `VERL_REF` in install scripts and both Dockerfiles.

### Phase C: rebase PSRL forks

Perform semantic three-way merges in this order:

1. Config classes, YAML defaults, Hydra search path, and runtime env.
2. Dataset creation/filtering and DataProcessor loader/checkpoint behavior.
3. veRL engine workers and PSRL train-worker/NIXL lifecycle.
4. Trainer initialization, stage helpers, checkpointing, metrics, and `fit()`.
5. Reward loop, reward-model, scorer, tool, parser, and trace forks.
6. vLLM server/replica/worker-extension construction, excluding direct routing code owned by SMG.
7. Agent-loop tokenization and output semantics, translated to SMG/TITO.

For each copied method, annotate the upstream file and symbol near the PSRL implementation. A filename-only annotation is
not sufficient when the logic was extracted into a different actor.

### Phase D: rebase the SMG contract

1. Regenerate `vllm_engine.proto` and `psrl_manager.proto` bindings for Rust and Python.
2. Rebuild/install `smg-grpc-proto`, `psrl-state-grpc-proto`, `smg-grpc-servicer`, and the native Python binding from
   the pinned SMG commit.
3. Compare every `RouterArgs` field used by `smg_adapter.py` with the Python binding and Rust config conversion.
4. Contract-test worker registration, discovery, health, admission, reserve, status, weight-version updates, pause/resume,
   abort/drain, partial rollout, pooling, multimodal requests, TITO, and routed experts.
5. Confirm that the SMG Rust/Python implementations and PSRL do not import generated stubs from different builds.

### Phase E: verification gates

At minimum, run:

```bash
python -m compileall -q psrl tests
python -m pytest -q tests/config tests/dataset tests/trainer tests/workers/train
python -m pytest -q tests/workers/reward tests/gen_dplb tests/tito tests/e2e/tito
python -m pytest -q tests/converter tests/state_dict tests/checkpoint
```

In the pinned SMG checkout, run:

```bash
cargo +nightly fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
python -m pytest -q grpc_servicer/tests crates/grpc_client/python/tests
```

Then run GPU/distributed smoke tests for both FSDP and Megatron:

- one training step and one logical fine-grained step;
- actor/ref and actor/critic initialization;
- validation before training and checkpoint resume;
- NIXL push/pull with a version bump;
- SMG gRPC rollout, abort/drain, partial rollout, and weight sync;
- multi-turn TITO with tools and multimodal input;
- reward-model pooling through SMG;
- routed replay end to end, including `uint8` and `uint16` routed experts;
- per-rank and standard DCP checkpoint save/load;
- TMS sleep/wake and elastic resource sharing.

An upgrade is complete only when the exact pinned tuple passes these gates. Import success or a clean patch apply alone is
not sufficient.

## 11. Review checklist

- [ ] Exact veRL, vLLM, and SMG commits are pinned.
- [ ] New veRL patch applies cleanly and matches the Docker patch byte-for-byte.
- [ ] Every row in Sections 3-7 was reviewed or explicitly marked unaffected.
- [ ] No PSRL code calls a removed veRL method or relies on a deleted mutable field.
- [ ] Hydra DP/FSDP and Megatron templates compose and instantiate.
- [ ] Trainer stages and LR/model-version advancement occur once per logical step.
- [ ] `parent_id`, padding, multi-reward, and routed-expert semantics are preserved.
- [ ] SMG protos and all generated packages come from the same pinned source.
- [ ] SMG pause/drain/sync/version/resume failure behavior is tested.
- [ ] Direct veRL routing changes were either intentionally skipped or translated to SMG.
- [ ] Focused CPU tests and distributed GPU smoke tests passed.
- [ ] This document and any upstream-symbol annotations were updated.
