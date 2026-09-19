# veRL Upgrade and Rebase Guide

This directory holds the patch for the veRL revision PSRL is pinned to. This document is the source
of truth for moving PSRL onto a new revision.

`patch/apply_patch.sh verl` selects the patch by the installed veRL commit, then by version, then by
the most recently modified file. Keep only the pinned revision: when the pin moves, delete the
superseded patch in the same change. `git log -- patch/verl` retains the older ones for a rebase that
needs to look back.

## 1. Supported architecture and version boundary

PSRL does not use veRL's direct HTTP rollout path in production. The supported request path is:

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

Classify every upstream rollout change before porting it:

1. **Engine construction or worker lifecycle.** Port it.
2. **Tokenization, multimodal preprocessing, sampling, logprobs, or routed-expert semantics.**
   Port it, preserving the SMG gRPC wire contract.
3. **veRL HTTP server, direct `generate()`, PD dispatch, or veRL's own rollout router.** Do not copy
   it. Record why SMG owns the behavior or add equivalent SMG support.
4. **Training, metrics, checkpoint, dataset, config, or worker APIs.** Rebase normally.

Pin a tuple, not independent moving branches:

```text
(veRL commit, veRL patch, vLLM version and patch, SMG commit and generated protobuf packages)
```

## 2. Rebase classes

Use these labels in upgrade reviews:

- **Fork.** PSRL copied substantial veRL logic and owns a divergent implementation. Do a three-way
  semantic merge.
- **Subclass/adapter.** PSRL inherits veRL code. Check constructor signatures, protected methods,
  mutable fields, and lifecycle ordering.
- **Extracted logic.** PSRL moved part of a veRL method into a different actor or stage. Compare by
  behavior, not by filename.
- **Patched upstream.** The behavior lives in `third_party/verl` through `patch/verl/<commit>.patch`.
- **API consumer.** PSRL relies on a symbol, schema, or side effect without copying the code.
- **PSRL/SMG owned.** No rebase is expected, but test the interface against the upgraded boundary.

## 3. Core training and orchestration modules

| PSRL module | veRL source to compare | Class | PSRL-owned behavior |
| --- | --- | --- | --- |
| `psrl/trainer/main_ppo.py` | `verl/trainer/main_ppo.py` | Fork | Separate train/rollout pools, GPU-slot reservation, TransferQueue bootstrap, PSManager, SMG gateway, reward services, elastic resources |
| `psrl/trainer/constants_ppo.py` | `verl/trainer/constants_ppo.py` | Fork | Overlays PSRL defaults on veRL's runtime environment and forwards proxy and harness variables |
| `psrl/trainer/ppo/ray_trainer.py` | `verl/trainer/ppo/ray_trainer.py` | Fork and subclass | Async busy-loop training, staleness buffers, NIXL weight publication, split train/gen workers, SMG lifecycle, multi-reward, fine-grained overlap |
| `psrl/trainer/ppo/strategies/` | veRL `RayPPOTrainer.fit()` stage ordering | Extracted logic | Full-batch and mini/micro-batch overlap with per-logical-step LR scheduling and delayed model publication |
| `psrl/trainer/ppo/utils.py` | `verl/trainer/ppo/core_algos.py` | Fork | `parent_id` grouping, PSRL roles, fractional resource bundles, session advantage propagation |
| `psrl/utils/dataset/data_processor.py` | `RayPPOTrainer._create_dataloader` and checkpoint helpers | Extracted logic | A Ray actor that owns dataloader state, drives TransferQueue, and streams batches |
| `psrl/utils/dataset/rl_dataset.py` | `verl/utils/dataset/rl_dataset.py` | Subclass | Tool-schema injection and deferred overlong-prompt filtering |

Trainer invariants that must survive a rebase:

- PSRL groups GRPO samples by `parent_id`, not veRL's `uid`.
- One logical optimizer step may contain several fine-grained chunks. LR schedulers and model-version
  publication advance exactly once per logical step.
- `DataProcessor`, not `RayPPOTrainer`, owns train-dataloader state.
- Training consumes `KVBatchMeta` and TransferQueue fields. Do not reintroduce whole-batch Ray
  serialization.
- Padding rows need collision-free ids and must not affect advantages, metrics, routed experts, or
  staleness accounting.

## 4. Config and worker API modules

| PSRL module | veRL source to compare | Class |
| --- | --- | --- |
| `psrl/trainer/config/ppo_trainer.yaml`, `ppo_megatron_trainer.yaml` | veRL trainer templates | Fork |
| `psrl/trainer/config/rollout/psrl_rollout.yaml` | `verl/trainer/config/rollout/rollout.yaml` | Overlay |
| `psrl/trainer/config/__init__.py`, `psrl/workers/config/__init__.py` | veRL config re-exports | API consumer |
| `psrl/workers/config/rollout.py` | `verl/workers/config/rollout.py` | Subclass/adapter |
| `psrl/workers/config/reward_model.py` | veRL reward and engine configs | API consumer and PSRL owned |

The two Hydra search-path plugins, `hydra_plugins/psrl_searchpath.py` and
`psrl/trainer/config/hydra_plugins/psrl_searchpath.py`, must stay identical. They are what let PSRL
overlays resolve veRL config groups.

## 5. Training workers and checkpoint integration

| PSRL module | veRL source to compare | Class |
| --- | --- | --- |
| `psrl/workers/train/engine_train_worker.py` | `verl/workers/engine_workers.py` | Subclass/adapter |
| `psrl/workers/train/base_train_worker.py` | Called from the veRL worker lifecycle | PSRL owned |
| `psrl/utils/checkpoint/megatron_saver.py` | veRL `MegatronCheckpointManager` and Megatron DCP | PSRL owned and patched upstream |
| `psrl/utils/converter/` | veRL engine state dicts, model merger, Megatron Bridge | API consumer and PSRL owned |

Check constructor signatures, process-group ownership, `init_model` ordering, return schemas, the LR
scheduler flag, and worker decorators. `PSRL_CriticTrainWorker` initializes the global process group
before `TrainingWorker.__init__`, because that constructor builds the engine. The per-rank checkpoint
backend validates a versioned `parallel_config.json` that records topology, optimizer mode, and
architecture.

The pinned Megatron-Bridge revision is independent of the veRL pin. Re-check the conversion-task API
boundary (`AutoBridge.get_conversion_tasks`) only when that pin moves.

## 6. Rollout and SMG gRPC boundary

| PSRL module | veRL or SMG source to compare | Class |
| --- | --- | --- |
| `psrl/workers/gen/vllm_async_server.py` | veRL `vllm_async_server.py`, SMG `VllmEngineServicer` and proto | Fork and subclass |
| `psrl/workers/gen/vllm_extension.py` | `vLLMColocateWorkerExtension` | Subclass |
| `psrl/workers/gen/smg_adapter.py`, `rollout_gateway.py`, `rollout_coordination/` | SMG binding and routing-loop controller | PSRL/SMG owned |
| `psrl/grpc/ps_manager_service.py` | SMG `psrl_manager.proto` and PSRL selector | PSRL/SMG owned |
| `psrl/utils/tito/training_data.py` and the session loop | SMG `crates/tito` and session endpoints | PSRL/SMG owned |

SMG protocol invariants:

- `(base_worker_id, dp_rank)` is the canonical rollout-instance identity.
- Pause, drain, weight pull, version publish, and resume form a failure-atomic sequence. A failed pull
  must leave the worker unavailable.
- Request `priority` is a signed optional value end to end. Unset, zero, and negative are distinct,
  and lower values are scheduled first. SMG accepts it on `GenerateRequest`.
- Routed experts use a compact C-contiguous `uint8` or `uint16` tensor on the wire. PSRL validates
  shape, dtype, and row coverage, then converts to veRL's signed `int16` replay tensor.
- Direct `PSRL_vLLMHttpServer.generate()` is intentionally rejected. SMG owns inference dispatch.

**Deferred:** the exact Hugging Face `model_type` session header (`x-smg-tito-model-type`) is not
implemented in SMG. SMG selects the TITO adapter from the served `model_id`. Do not send the header
until SMG persists it.

## 7. Reward, agent, tool, and trace forks

| PSRL module | veRL source to compare | Class |
| --- | --- | --- |
| `psrl/workers/reward/reward_loop/` | `verl/experimental/reward_loop/` | Fork |
| `psrl/workers/reward/reward_manager.py`, `reward_worker.py` | `verl/experimental/reward_loop/reward_loop.py` | Parallel fork |
| `psrl/workers/reward/reward_model/` | `verl/experimental/reward_loop/reward_model.py` | Parallel fork |
| `psrl/utils/reward_score/` | veRL reward-score registry and SandboxFusion helpers | Fork |
| `psrl/utils/rollout/rollout_trace.py` | `verl/utils/rollout_trace.py` | Fork |
| `psrl/tools/function_tool.py`, `psrl/tools/tool_parser/` | veRL tools and agent-loop tool parsers | Fork |
| `psrl/workers/agent_loop/loops/`, `agent_data/`, `worker.py` | veRL experimental agent loop | Parallel fork |

Do not replace PSRL agent loops with veRL's direct LLM client. Rebase semantic fixes into the PSRL
gateway and TITO design.

## 8. veRL patch inventory

The patch filename must match the exact installed veRL commit. `3efe38c7.patch` changes:

| Patched area | PSRL requirement |
| --- | --- |
| `protocol.py` | Merge arbitrary `*_metrics` from distributed workers without losing per-worker values |
| `single_controller/ray/base.py` | Fractional GPU resources per placement-group bundle |
| `trainer/config/data/legacy_data.yaml`, `engine/*.yaml`, `model/hf_model.yaml` | No implicit shuffle, shared-memory model loading, `load_weight`, and per-rank checkpoint switches |
| `trainer/ppo/core_algos.py`, `workers/config/actor.py`, `workers/utils/losses.py` | `session-mean-token-mean` loss, its metric aggregation, and length-corruption diagnostics |
| `trainer/ppo/metric_utils.py` | Metadata-driven max prompt/response lengths, per-source `original_reward_score` metrics, and skipping non-numeric validation variables |
| `trainer/ppo/padding_utils.py` | Unique numeric padding ids, `parent_id`, per-response fields shrunk by length rather than by name, and route-safe padding |
| `trainer/ppo/rollout_corr_helper.py` | Drop zero-response rows from sequence importance-sampling metrics |
| `utils/dataset/rl_dataset.py` | Multi-dataset identity, reward-model dictionaries, and over-sampling |
| `utils/checkpoint/megatron_checkpoint_manager.py` | PSRL per-rank checkpoint backend and versioned metadata |
| `utils/megatron/dist_checkpointing.py` | No-fork filesystem writer to avoid inherited NCCL/UCX corruption |
| `utils/tokenizer/tokenizer.py` | Pickle-safe processor method binding |
| `utils/experimental/torch_functional.py` | Batched logits flattening for the flash-attn cross-entropy kernel |
| `models/transformers/qwen3_5.py` | Pin `lm_head` weights to the activations' device under FSDP offload |
| `workers/engine/fsdp/transformer_impl.py`, `workers/engine/megatron/transformer_impl.py` | Dummy/no-weight construction, TMS regions, and rank-0 micro-batch progress |
| `workers/engine_workers.py` | Process-group lifecycle split and logical-step LR scheduler control |

Patch maintenance rules:

1. Start from a clean checkout of the target veRL commit.
2. Re-implement each requirement against the new upstream structure. Do not blindly apply the
   previous diff.
3. Keep PSRL imports inside the patch minimal.
4. Export one patch named after the commit and delete the superseded patch.
5. Copy the same bytes to `docker/patch/verl.patch` and verify they are identical.
6. Validate with `git apply --check` against a clean checkout of the exact commit.

## 9. Upgrade procedure

### Phase A: establish clean baselines

1. Record the old and new veRL commits and obtain a clean checkout of each.
2. Record the vLLM and SMG commits and the generated protobuf package versions.
3. Save the PSRL and SMG working-tree status.
4. Diff old to new veRL for every upstream file listed here, including renamed files.

### Phase B: rebuild the veRL patch

1. Apply each requirement from Section 8 to the clean new checkout.
2. Run focused tests for DataProto concat, fractional Ray resources, padding, multi-reward dataset
   fields, engine no-weight initialization, TMS regions, LR scheduling, DCP no-fork behavior, and
   per-rank checkpoint round trips.
3. Export `patch/verl/<new-commit>.patch` and copy the same bytes to `docker/patch/verl.patch`.
4. Run `git apply --check` against another clean checkout of the new commit.
5. Update the pin in `scripts/install_basic.sh` and both Dockerfiles.

### Phase C: rebase PSRL forks

Do semantic three-way merges in this order:

1. Config classes, YAML defaults, Hydra search path, and runtime environment.
2. Dataset creation, filtering, and DataProcessor loader state.
3. Engine workers and the PSRL train-worker and NIXL lifecycle.
4. Trainer initialization, stage helpers, checkpointing, metrics, and `fit()`.
5. Reward loop, reward-model, scorer, tool, parser, and trace forks.
6. vLLM server, replica, and worker-extension construction. Skip direct routing owned by SMG.
7. Agent-loop tokenization and output semantics, translated to SMG and TITO.

Annotate the upstream file and symbol near each PSRL implementation. A filename-only annotation is
not enough when the logic moved to a different actor.

### Phase D: rebase the SMG contract

1. Regenerate `vllm_engine.proto` and `psrl_manager.proto` bindings for Rust and Python.
2. Rebuild `smg-grpc-proto`, `psrl-state-grpc-proto`, `smg-grpc-servicer`, and the native Python
   binding from the pinned SMG commit.
3. Compare every `RouterArgs` field used by `smg_adapter.py` with the Python binding.
4. Contract-test worker registration, admission, reserve, status, weight-version updates,
   pause/resume, abort and drain, partial rollout, pooling, multimodal, TITO, and routed experts.
5. Confirm PSRL and SMG do not import generated stubs from different builds.

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

Then run GPU and distributed smoke tests for both FSDP and Megatron:

- one training step and one logical fine-grained step
- actor/ref and actor/critic initialization
- validation before training and checkpoint resume
- NIXL push and pull with a version bump
- SMG gRPC rollout, abort and drain, partial rollout, and weight sync
- multi-turn TITO with tools and multimodal input
- reward-model pooling through SMG
- routed replay end to end, including `uint8` and `uint16` payloads
- per-rank and standard DCP checkpoint save and load
- TMS sleep and wake, and elastic resource sharing

An upgrade is complete only when the pinned tuple passes these gates. A clean patch apply is not
sufficient.

## 10. Review checklist

- [ ] Exact veRL, vLLM, and SMG commits are pinned.
- [ ] The new veRL patch applies cleanly and matches the Docker patch byte for byte.
- [ ] Every row in Sections 3 to 7 was reviewed or marked unaffected.
- [ ] No PSRL code calls a removed veRL method or relies on a deleted field.
- [ ] Hydra DP/FSDP and Megatron templates compose and instantiate.
- [ ] Trainer stages and LR and model-version advancement occur once per logical step.
- [ ] `parent_id`, padding, multi-reward, and routed-expert semantics are preserved.
- [ ] SMG protos and generated packages come from one pinned source.
- [ ] SMG pause, drain, sync, version update, and resume failure behavior is tested.
- [ ] Direct veRL routing changes were skipped on purpose or translated to SMG.
- [ ] Focused CPU tests and distributed GPU smoke tests passed.
