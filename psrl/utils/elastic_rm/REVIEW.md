# Elastic RM Module Deep Review

> Generated: 2026-04-07
> Scope: `psrl/utils/elastic_rm/`, `psrl/trainer/ppo/ray_trainer.py` (`_init_elastic_rm_runtime`), `psrl/trainer/main_ppo.py` (resource pool setup)

---

## 1. Architecture Overview

Elastic RM implements **time-multiplexed GPU sharing between Rollout and Reward Model** instances. The core insight: generation and reward scoring don't need to run at full capacity simultaneously, so they can share the same physical GPUs via `torch_memory_saver` (TMS) weight offload/restore.

### Component Relationship

```
main_ppo.py (TaskRunner.run)
  -> PSRL_RayPPOTrainer.init_workers()
    -> _init_elastic_rm_runtime()
      +-- SubRayResourcePool (shared physical GPUs via placement groups)
      +-- All instances SLEEP initially
      +-- Select non-conflicting instances to WAKE_UP
      +-- Create ElasticExecutor (Ray remote actor)
            +-- _monitor_loop()             -> poll signals -> ScalingPolicy.decide()
            +-- _scale_up_handler_loop()    -> consume scale_up tasks from queue
            +-- _scale_down_handler_loop()  -> consume scale_down tasks from queue

ScalingPolicy.decide()
  +-- force_wake (router backlog > 0 but awake = 0)
  +-- Priority -1: trainer idle -> scale up bottleneck side
  +-- Priority  1: one side full -> free-GPU scale up / cede transfer
  +-- Priority  2: both sides full -> bottleneck transfer (simulated rebalance)
  +-- Priority  3: one side underloaded -> spontaneous scale down
```

### Data Flow per Monitor Tick

```
Coordinators                  ElasticExecutor                ScalingPolicy
    |                              |                              |
    |<-- get_engine_status --------|                              |
    |<-- get_router_backlog -------|                              |
    |                              |                              |
    |     AgentLoopManager         |                              |
    |<-- get_trainer_waiting ------|                              |
    |                              |                              |
    |                              |-- build InstanceSignal[] --->|
    |                              |                     decide() |
    |                              |<-- ScalingDecision ---------|
    |                              |                              |
    |<-- exec_command(SLEEP) ------|  (via handler loops)         |
    |<-- exec_command(WAKE_UP) ----|                              |
    |<-- exec_command(ABORT) ------|                              |
```

---

## 2. Correctness Analysis

### 2.1 Decision Logic (ScalingPolicy)

The 5-priority strategy is **correct and well-structured**:

| Priority | Condition | Action | Verdict |
|----------|-----------|--------|---------|
| Force Wake | awake=0 + backlog>0 | Force wake one instance | Necessary safety net |
| P-1 | trainer idle + waiting on rollout/reward | Scale up bottleneck side | Good trainer utilization |
| P1 | Exactly one side full | Free-GPU first, then cede transfer | Avoids unnecessary cross-role sleep |
| P2 | Both sides full | Bottleneck transfer (gain > hysteresis) | Rebalance simulation prevents blind transfers |
| P3 | One side underloaded | Spontaneous scale down | Reclaims idle resources |

**Correct safety mechanisms:**

- `_decision_execution_in_progress` gate ensures at most one scaling decision executes at a time, with stall-tick abandon timeout.
- `cooldown_ms` prevents oscillation.
- All-stale signal guard skips decisions on outdated data.
- `min_awake_per_role` checked in multiple places (executor sleep, policy candidate selection, cross-role cede budget).

### 2.2 Potential Issues

#### Issue 1: P1 cede transfer emits only `scale_up`, not `scale_down`

In `_make_stepwise_decision` P1 branch:

```python
# P1a: only RM is full -> cede rollout
if rm_up is not None and rollout_down is not None:
    actions.append(ScalingAction(action_type="scale_up", ...rm_up...))
    return actions, "transfer_rollout_to_rm"
```

Only a `scale_up` action is returned. The corresponding `scale_down` (sleep the ceded rollout instance) is performed **implicitly** inside `_scale_up_handler_loop` via `_find_instances_to_scaled_down_for_other_roles`. This is by design but distributes the "cede" semantic across two layers (policy decision vs handler execution), making the flow harder to reason about.

**Risk:** The policy checks `rollout_down is not None` (KV cache < theta_low) at decision time, but the handler selects instances by lowest KV cache at execution time. These may diverge if state changes between decision and execution. The handler's `if not instances_to_scaled_down: continue` guard prevents errors, but can cause a wasted scale_up cycle.

#### Issue 2: P3 waiting queue guard checks single instance, not role total

```python
# scaling_policy.py, P3 branch
if rollout_low and rollout_down is not None and not rm_full
    and rollout_down.waiting_queue_num <= self.max_waiting_queue_for_scale_down:
```

The comment on `max_waiting_queue_for_scale_down` says "per-role as total waiting queues across awake instances", but the code checks **only the single candidate instance's** `waiting_queue_num`. If waiting queues are unevenly distributed, this could allow premature scale-down.

**Suggested fix:** Compute role-level total waiting queue and compare against the threshold, or clarify the comment to match the per-instance semantic.

#### Issue 3: `_scale_down_instance` logs `instance_id` with `%d` format

```python
# elastic_executor.py line 719
psrl_logger.info(
    "Skip scale down for role=%s model=%s instance=%d: keep at least %d awaken instances.",
    instance_role, instance_model_name, instance_id, min_awake_per_role,
)
```

`instance_id` is a `RolloutInstanceId` (tuple[str, int]), not an int. The `%d` format specifier will raise `TypeError` at runtime. Should be `%s`.

---

## 3. OOM Risk Analysis

### 3.1 Sleep Level Asymmetry

| Role | Sleep Level | Weights | KV Cache | GPU Memory Freed |
|------|-------------|---------|----------|------------------|
| Rollout | 2 | Offloaded to CPU | Offloaded to CPU | ~80-95% |
| Reward Model | 1 | **Stays in GPU** | Released | ~10-30% |

### 3.2 The Critical OOM Scenario

Consider this sequence on GPU-0:

1. Reward Model instance A is AWAKEN on GPU-0 (weights ~14GB in VRAM)
2. Policy decides: scale up Rollout, cede RM
3. Handler calls `_scale_down_instance(RM_A)` -> SLEEP level=1 -> **weights remain on GPU-0** (~14GB)
4. Handler calls `_scale_up_instance(Rollout_B)` -> WAKE_UP level=2 -> loads weights (~14GB) + allocates KV cache (~8GB)
5. GPU-0 total demand: 14GB (RM weights, not offloaded) + 14GB (Rollout weights) + 8GB (Rollout KV) = **36GB**

If GPU has 40GB or 80GB this may or may not OOM, but the code has **no memory budget check**.

### 3.3 Current Safeguards

- `_has_other_role_awaken_on_shared_gpu` prevents waking an instance on a GPU where another role is **AWAKEN**. But after SLEEP, the status is ASLEEP, so the guard passes even though level-1 weights are still in GPU memory.
- TMS `torch_memory_saver.pause("weights")` is only called for level >= 2; level 1 raises `NotImplementedError` for weight offload.

### 3.4 Recommendations

1. **Short-term:** Change reward model sleep level to 2 when elastic_rm is enabled, ensuring full GPU release. The wake-up latency increase (~seconds) is acceptable since elastic scaling is already a multi-second operation.
2. **Medium-term:** Add GPU memory budget estimation to `InstanceSignal` (via `torch.cuda.mem_get_info()`) and incorporate it into scaling decisions.
3. **Long-term:** Implement a memory-aware scheduler that models per-instance VRAM footprint (weights + KV cache + activation) and rejects scale-up decisions that would exceed GPU capacity.

---

## 4. Feature Completeness

### What Works

- Rollout <-> Reward Model GPU resource dynamic scheduling
- Multi-dimensional decision signals (KV cache utilization, running/waiting queue, throughput)
- Trainer-idle-aware priority preemption (P-1)
- Router backlog-driven force-wake
- Throughput profile (formula fitting) driven bottleneck transfer simulation (P2)
- GPU affinity-aware free-GPU-first expansion (P1 Step A)
- Post-scale-up ABORT waiting requests for rebalancing

### What's Missing

- No GPU memory budget awareness (see Section 3)
- No priority differentiation between multiple reward models (all RM treated equally)
- No batch scaling (one instance per decision due to execution gate)
- No predictive scaling (purely reactive, based on current signals)
- No graceful drain before sleep (ABORT is post-scale-up only; no pre-sleep drain for the instance being slept)

---

## 5. Code Quality Assessment

### 5.1 Strengths

1. **Decision/Execution separation:** ScalingPolicy is pure decision (no side effects), ElasticExecutor handles execution. Clean separation of concerns.
2. **Async three-loop architecture:** monitor + scale_up_handler + scale_down_handler communicate via `asyncio.Queue`, preventing blocking.
3. **Decision ID lifecycle tracking:** `_decision_pending_action_counts` + `_mark_decision_action_finished` reliably tracks decision completion.
4. **Rich tracing:** Coordinator commands have START/END/EXCEPTION logs with elapsed time, enabling stuck-RPC diagnosis.
5. **Per-ref timeout:** `_await_coordinator_refs_with_per_ref_timeout` prevents a single slow RPC from dropping all results for a tick.

### 5.2 Problems

#### Problem 1: Classes are too large

- `ElasticExecutor`: **1119 lines**, handling instance registration, GPU mapping, 3 async loops, coordinator RPC, scale up/down execution, post-scale-up ABORT, signal collection, and logging.
- `ScalingPolicy`: **1084 lines**, handling throughput profile loading/estimation, lambda EWMA, 5-priority decision, and detailed no-action diagnostics.

Both violate the Single Responsibility Principle.

#### Problem 2: GPU mapping logic is scattered and duplicated

The `(node_id, gpu_id)` concept is processed in at least 6 places across 3 files:

```python
# ray_trainer.py - collection
gpu_ids, node_id = self._collect_instance_gpu_mapping(rollout_wg)
rollout_instance_to_gpu_keys[instance_id] = self._build_gpu_keys(node_id, gpu_ids)

# elastic_executor.py - storage (dict with "gpu_ids" list + "node_id" string)
self.instance_gpu_mappings[role_name][model_name][instance_id] = {
    "gpu_ids": list(gpu_ids or []), "node_id": node_id,
}

# elastic_executor.py - reverse index (set of (node_id, gpu_id) tuples)
self.gpu_to_instances.setdefault(gpu_key, set()).add(target_key)

# elastic_executor.py - conflict query
def _has_other_role_awaken_on_shared_gpu(...)
def _get_instance_gpu_keys(...)

# scaling_policy.py - signal construction (frozenset)
gpu_keys = frozenset((node_id, gid) for gid in gpu_ids)
signal = InstanceSignal(..., gpu_keys=gpu_keys)

# scaling_policy.py - free-GPU candidate selection (set intersection)
def _pick_scale_up_candidate_on_free_gpu(...)
```

The same `(node_id, gpu_id)` concept is converted between `dict`, `set`, `frozenset`, and `list` repeatedly. This is redundant and error-prone.

#### Problem 3: `_init_elastic_rm_runtime` in ray_trainer.py is a ~200-line monolith

This method mixes:
- GPU mapping collection from worker groups
- Instance registration to ElasticExecutor
- Initial SLEEP all instances
- Non-conflicting instance selection
- WAKE_UP selected instances
- ElasticExecutor creation and lifecycle start

These are distinct concerns that should be decomposed.

#### Problem 4: Three-level nested dict pattern

```python
self.instances_status_flags: dict[PSRL_Role, dict[str, dict[int, InstanceStatus]]]
self.instances_engine_stats: dict[PSRL_Role, dict[str, dict[int, dict | None]]]
self.instance_gpu_mappings: dict[PSRL_Role, dict[str, dict[int, dict[str, object]]]]
```

All indexed by `[role_name][model_name][instance_id]`. This triple-nested dict pattern is accessed identically everywhere with `.get(role_name, {}).get(model_name, {}).get(instance_id, ...)`. A single keyed structure would be simpler.

---

## 6. Abstraction Improvement Suggestions

### 6.1 Extract `ClusterTopology` for GPU Mapping

Replace scattered GPU mapping logic with a centralized abstraction:

```python
@dataclass(frozen=True)
class GPUSlot:
    node_id: str | None
    gpu_id: int

@dataclass(frozen=True)
class InstanceKey:
    role: PSRL_Role
    model_name: str
    instance_id: RolloutInstanceId

@dataclass
class InstancePlacement:
    key: InstanceKey
    gpu_slots: frozenset[GPUSlot]
    status: InstanceStatus = InstanceStatus.ASLEEP

class ClusterTopology:
    """Centralized GPU-to-instance mapping with conflict detection."""

    def register(self, placement: InstancePlacement) -> None: ...
    def unregister(self, key: InstanceKey) -> None: ...
    def set_status(self, key: InstanceKey, status: InstanceStatus) -> None: ...

    def get_instances_on_gpu(self, gpu_slot: GPUSlot) -> list[InstancePlacement]: ...
    def has_conflict(self, key: InstanceKey, exclude_role: PSRL_Role | None = None) -> bool: ...
    def find_free_gpu_candidates(self, role: PSRL_Role, model_name: str) -> list[InstancePlacement]: ...
    def get_awake_count(self, role: PSRL_Role, model_name: str) -> int: ...
```

Benefits:
- Single source of truth for GPU mapping
- Conflict detection in one place
- Eliminates `_get_instance_gpu_keys`, `_has_other_role_awaken_on_shared_gpu`, `_add/remove_instance_from_gpu_reverse_index`, `_build_gpu_keys`, `_collect_instance_gpu_mapping`, `_select_non_conflicting_awake_ids`
- Testable in isolation

### 6.2 Flatten Instance State with Composite Key

Replace the triple-nested dicts with a flat dict keyed by `InstanceKey`:

```python
# Before (3 separate triple-nested dicts)
self.instances_status_flags[role_name][model_name][instance_id]
self.instances_engine_stats[role_name][model_name][instance_id]
self.instance_gpu_mappings[role_name][model_name][instance_id]

# After (single flat dict)
@dataclass
class InstanceState:
    status: InstanceStatus
    engine_stats: dict | None
    placement: InstancePlacement

self.instances: dict[InstanceKey, InstanceState]
```

### 6.3 Extract `ThroughputProfileLoader` as Independent Module

`ThroughputProfileLoader` (lines 58-193 in scaling_policy.py) is a self-contained component for loading/evaluating throughput models. It has no dependency on ScalingPolicy internals and should be a separate module:

```
elastic_rm/
  +-- throughput_profile.py    # ThroughputProfileLoader (extracted)
  +-- cluster_topology.py      # ClusterTopology (new)
  +-- scaling_policy.py        # ScalingPolicy (slimmed)
  +-- elastic_executor.py      # ElasticExecutor (slimmed)
  +-- diagnostics.py           # (unchanged)
```

### 6.4 Add Instance State Machine with Transition States

Current: only ASLEEP / AWAKEN.

```python
class InstanceStatus(Enum):
    ASLEEP = auto()
    WAKING_UP = auto()    # WAKE_UP RPC in flight
    AWAKEN = auto()
    SLEEPING = auto()      # SLEEP RPC in flight
```

This would:
- Prevent racing scale decisions on instances mid-transition
- Make conflict detection more accurate (SLEEPING instance still holds GPU memory)
- Enable better logging and diagnostics

### 6.5 Role-Agnostic ScalingPolicy

Current policy hardcodes `"Rollout"` and `"RewardModel"` throughout `_make_stepwise_decision`. For N-role generalization:

```python
@dataclass
class RoleState:
    role_key: str
    signals: list[InstanceSignal]
    is_full_load: bool
    is_low_load: bool
    total_mu: float
    scale_up_candidate: InstanceSignal | None
    scale_down_candidate: InstanceSignal | None

class ScalingPolicy:
    def decide(self, role_states: dict[str, RoleState], ...) -> ScalingDecision:
        # Generic capacity/demand model
        # Identify bottleneck role, surplus role
        # Compute transfer gain without hardcoded role names
```

This would allow elastic scaling among any combination of roles (e.g., Rollout + RM + Teacher Model).

---

## 7. Position in PSRL Architecture

### Module Hierarchy

```
PSRL Architecture
+-- Data Layer (DataProcessor, DatasetType)
+-- Agent Layer (AgentLoopManager, AgentLoopWorker)
+-- Generation Layer
|   +-- RolloutRouter / RolloutGateway
|   +-- RolloutCoordinator  <------------------+
|   +-- vLLM Replicas                          |
+-- Reward Layer                                |
|   +-- RewardLoopManager                       |
|   +-- RewardModelCoordinator  <--------------+|
|   +-- RewardModel Replicas                   ||
+-- Training Layer (Actor, Critic, RefPolicy)  ||
+-- Parameter Server (NIXL)                    ||
+-- Elastic RM Layer  <-----------------------+|
    +-- ElasticExecutor (orchestrator) --------+
    +-- ScalingPolicy (decision maker)
```

Elastic RM is a **cross-cutting layer** that spans Generation and Reward layers, using Coordinators as the bridge to control instance sleep/wake.

### Coupling Analysis

| Direction | Target | Mechanism | Coupling Level |
|-----------|--------|-----------|----------------|
| -> RolloutCoordinator | `exec_command`, `get_instance_engine_status_snapshot`, `get_router_backlog_size` | Ray RPC | Medium |
| -> RewardModelCoordinator | Same as above | Ray RPC | Medium |
| -> AgentLoopManager | `get_trainer_waiting_hint` | Ray RPC | Low |
| -> ScalingPolicy | `InstanceSignal`, `ScalingDecision` | Dataclasses | Low (good) |
| -> PSRL_Role | Hardcoded `Rollout`, `RewardModel` | Enum constants | **High (bad)** |
| -> Command/CommandType | `SLEEP`, `WAKE_UP`, `ABORT` | Command pattern | Medium |
| <- ray_trainer | Init, registration, lifecycle | Direct calls | **High** |

**Key coupling problems:**

1. ScalingPolicy hardcodes two role names as strings, cannot generalize.
2. `_init_elastic_rm_runtime` in ray_trainer.py is a ~200-line method tightly coupled to both Coordinator APIs and ElasticExecutor internals.
3. ElasticExecutor directly depends on Coordinator method signatures.

---

## 8. Improvement Roadmap

### Short-term (low risk, high value)

| Item | Effort | Impact |
|------|--------|--------|
| Fix `%d` format for `instance_id` (tuple) in `_scale_down_instance` log | Trivial | Prevents runtime TypeError |
| Fix P3 waiting queue guard to match per-role-total semantic | Small | Prevents premature scale-down |
| Extract `ThroughputProfileLoader` to separate module | Small | Reduces ScalingPolicy size by ~130 lines |
| Extract `ClusterTopology` class | Medium | Eliminates GPU mapping duplication across 3 files |

### Medium-term (needs test coverage)

| Item | Effort | Impact |
|------|--------|--------|
| Flatten triple-nested dicts to `dict[InstanceKey, InstanceState]` | Medium | Simplifies all state access patterns |
| Add transition states (WAKING_UP, SLEEPING) to InstanceStatus | Medium | Prevents race conditions, improves diagnostics |
| Decompose `_init_elastic_rm_runtime` into `ElasticRuntimeInitializer` | Medium | Decouples ray_trainer from elastic internals |
| Define `ElasticCoordinator` Protocol for coordinator abstraction | Medium | Decouples ElasticExecutor from specific coordinator implementations |

### Long-term (architectural evolution)

| Item | Effort | Impact |
|------|--------|--------|
| Role-agnostic ScalingPolicy (N-role generalization) | Large | Enables multi-role elastic scheduling |
| GPU memory budget-aware scheduling | Large | Eliminates OOM risk from sleep level asymmetry |
| Declarative resource management (desired state reconciliation) | Large | Replaces imperative scale up/down with convergent control |
| Predictive scaling (arrival rate forecasting) | Large | Proactive instead of reactive scaling |

---

## 9. Summary Scorecard

| Dimension | Score | Notes |
|-----------|-------|-------|
| Correctness | 4/5 | Core logic correct; edge cases in P3 waiting guard and P1 implicit cede |
| OOM Safety | 3/5 | GPU conflict detection exists but sleep level 1 leaves weights in GPU |
| Feature Completeness | 4/5 | 5-priority coverage; lacks memory awareness |
| Code Simplicity | 3/5 | Two 1000+ line classes; GPU mapping scattered across 3 files |
| Extensibility | 2/5 | Hardcoded 2 roles; cannot easily add a third |
| Coupling | 3/5 | Coordinator coupling acceptable; ray_trainer init coupling too heavy |

**Top priority:** Extract `ClusterTopology` abstraction, unify GPU mapping management, add GPU memory budget awareness. These three changes yield the highest improvement in maintainability and safety with moderate risk.
