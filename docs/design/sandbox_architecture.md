# Sandbox Architecture

This page maps the sandbox module. It records which components exist, how they
depend on each other, the workflow an episode actually follows, and the
decomposition the module should move to.

It is the structural companion to {doc}`sandbox_lifecycle`, which states the
invariants each component must preserve. Read this page to know where a change
belongs. Read the lifecycle page to know what the change must not break.

:::{note}
The component inventory and the dependency graph below record the layout the
refactor started from, including the file paths of that time. For the current
layout, read the inventory in {doc}`sandbox_lifecycle` and the module map in
`psrl/sandbox/README.md`.
:::

---

## Component inventory

The module splits into five planes. A component is listed with the concern it
owns and the boundary it currently leaks across.

### Contract

| Component | Owns | Boundary note |
|---|---|---|
| `core.py` | Portable specs, results, capabilities, refs, typed errors | `SandboxSession` requires 7 methods for command and file I/O and offers 9 more with default implementations. The optional half mixes state operations (`pause`, `resume`, `snapshot`, `fork`) with consumer accessors (`stats`, `spec`, `command_count`, `resolve_callback_url`) |
| `config.py` | Hydra instantiation of the backend registry | Duplicates backend construction that `docker.py` also performs in its constructor |

### Orchestration

| Component | Owns | Boundary note |
|---|---|---|
| `manager.py` | Backend registry, idempotent create sharing, workflow reservation, lease ownership, delayed cleanup, capacity binding, shutdown | Seven concerns in one class, held together by nine private state fields |
| `capacity.py` | Node CPU and memory accounting, class FIFO queues, owner expiry | The envelope is node local |
| `sync.py` | Thread side facade over the manager | Reaches back into the manager to release a lease that raced close |
| `metrics.py` | Latency and memory observations | Mixed into the backend and session by direct attribute access |

### Docker execution

| Component | Owns | Boundary note |
|---|---|---|
| `docker.py` | Backend facade, policy assembly, image preparation, idempotent recovery | Constructor takes 18 parameters. `_build_container_config` merges security, profile, and spec into one dictionary |
| `docker_session.py` | Command lifecycle, stop detection, exit classification, destruction | Holds `self.backend` and calls into `engine`, `metrics`, `lifecycle.config`, and `forget_session`. The session and its backend are mutually referential |
| `docker_engine.py` | Two bounded HTTP pools, Docker protocol, frame decoding, archive streaming | The cleanest boundary in the module. It depends on `core` errors only |
| `docker_events.py` | One shared container event stream and reconnect gap recovery | Coexists with a polling fallback in `docker_session.py` that detects the same condition |
| `docker_policy.py` | Security, per workload policy, disk admission | Resource configuration is also expressed in `core.ResourceSpec` and `capacity.SandboxCapacityConfig` |
| `docker_lifecycle.py` | Worker heartbeat, node collector supervision | The collector is a node level resource reclaimer, not a property of one backend |

### Crash recovery

| Component | Owns | Boundary note |
|---|---|---|
| `utils/docker_utils.py` | Heartbeat files, label sweeps, CLI force removal, image pruning, and the node GC loop | A grab bag. It also embeds the GC program as a source string that `spawn_node_gc` re-executes through `python -c` |

### Consumer integration

| Component | Owns | Boundary note |
|---|---|---|
| `workers/agent_loop/loops/harness_agent_loop.py` | Episode ownership. Acquire, harness execution, checkpoint, release | Hands `lease.session` to the harness and calls `capabilities` and `stats` on it directly. It therefore bypasses the manager for session semantics |
| `examples/mini_swe/runner.py` | The only production builder of `core.SandboxSpec` | Chooses `workflow_id`, `resource_class`, and `idempotency_key` for rollout and grader phases |

---

## Component dependency graph

Solid arrows are calls. The dashed arrows are the back edges that make the
module hard to reason about.

```text
  consumer plane
  +-----------------------------------------------------------------------+
  |  harness_agent_loop      swebench_grader      mlgym_agent_loop        |
  +----+--------------------------+----------------------+----------------+
       |                          |                      |
       | acquire/release          | raw docker CLI       | SandboxHandle
       v                          v                      v
  +-----------------+      +----------------+    +------------------------+
  | SyncSandbox     |      | (bypasses the  |    |                        |
  | Manager  sync.py|      |  manager)      |    |                        |
  +--------+--------+      +----------------+    |                        |
           |                                     +-----------+------------+
           v                                                 v
  +----------------------------------+                   docker run
  | SandboxManager      manager.py   |                   (own argv)
  |                                  |
  |  _leases            _create_tasks|
  |  _workflow_leases   _provision.. |
  |  _idempotent_leases _pending_rel |
  |  _release_attempts  _pending_cap |
  |  _pending_workflows              |
  +--+-------------+-----------------+
     |             |
     | capacity    | lease
     v             v
  +----------------+     +------------------------------------------+
  | SandboxCapacity|     | DockerBackend            docker.py       |
  | Coordinator    |     |  policy, image prep, idempotent recovery |
  | capacity.py    |     +--+--------+--------+--------+-------------+
  | (Ray actor)    |        |        |        |        |
  +----------------+        |        |        |        |
                            v        v        v        v
              +-------------+  +----------+ +--------+ +-------------+
              | DockerSession| | Container| | Docker | | Docker      |
              | docker_      | | Event    | | Life-  | | Policy      |
              | session.py   | | Watcher  | | cycle  | | docker_     |
              +------+-------+ | docker_  | | docker_| | policy.py   |
                     |         | events.py| | life.. | +-------------+
                     |         +----------+ +---+----+
                     |                         |
                     v                         v
              +----------------+     +---------------------------------+
              | DockerEngine   |     | utils/docker_utils.py           |
              | Client         |     |  heartbeat files, label sweeps  |
              | docker_engine  |     |  CLI force rm, image pruning    |
              +-------+--------+     |  spawn_node_gc -> python -c ... |
                      |              +----------------+----------------+
                      v                               v
                 Docker daemon                   detached GC process
                 (/var/run/docker.sock)          + `docker` CLI

  dashed back edges (each one is a boundary that is currently absent)

  (1) docker_session  -.->  docker_backend   engine, metrics, lifecycle, forget_session
  (2) SyncSandbox     -.->  manager          release a lease that raced close
  (3) harness         -.->  lease.session    capabilities, stats, write_bytes
  (4) capacity        <-.->  manager         remote RPC for acquire, release, renew, release_owner
  (5) collector       -.->  backend          runs outside the worker event loop, reaps by Docker label
```

Two structural facts fall out of the graph.

**The orchestration plane is not a boundary.** Consumers reach through
`SandboxManager` into `lease.session`, and `DockerSession` reaches back into
`DockerBackend`. Any change to session internals can require a change in the
manager, the backend, and the consumer.

**The crash recovery plane is a second root.** `docker_utils.py` depends on the
Docker CLI and on filesystem heartbeats at the same time, so it cannot be
tested or replaced without a daemon and a shared directory.

---

## Workflow

### Node and worker startup

```text
trainer startup
  Detect that a configured backend is a DockerBackend
      |
  Create one SandboxCapacityCoordinator Ray actor per node
      |
  Pass the node's coordinator handle to each agent loop worker on that node

agent loop worker startup
  Derive PSRL_ACTOR_ID from worker id, host, pid, and a random suffix
      |
  build_sandbox_manager(config, coordinator, owner_id)
      |
  DockerBackend reads PSRL_ACTOR_ID for container labels
```

### One episode, two phases

```text
rollout phase
  prepare(spec)                     warm the image, allocate nothing
  acquire(spec)                     -> SandboxLease
      reserve_workflow(workflow_id) reject a concurrent phase for this workflow
      coordinator.acquire           RPC, admits a complete CPU and memory vector
      backend.create                container, heartbeat thread, node collector
      adopt                         lease takes ownership and the capacity charge
  harness runs on lease.session
  release()                         terminate, return capacity, drop the lease

grader phase
  acquire(grader_spec)              same workflow_id, new idempotency_key,
                                    resource_class=grader
  restore from the rollout snapshot, or create a fresh container
  release()
```

### Failure paths

| Trigger | Path |
|---|---|
| A waiter cancels while create is in flight | `_abandon_create` cancels the shared task and releases whatever it returned |
| Every waiter leaves a shared create | The idempotency entry is retained through rollback, then dropped |
| Destruction fails | `_defer_release` keeps the lease and its capacity, and a background reaper retries with backoff |
| Capacity return fails | Only the accounting is retried. Destruction has already succeeded |
| Worker process dies | The node collector reaps containers whose heartbeat aged out |
| Shutdown | Cancel admission, settle provisioning, release leases, close backends, release leases again |

### Independent timing loops

Six loops with separate periods and separate failure handling run during one
acquire and release cycle.

| Loop | Component | Period or bound |
|---|---|---|
| Delayed cleanup reaper | `manager.py` | 1s to 30s backoff, alert at 5 attempts |
| Session lifetime timer | `docker_session.py` | 6 attempts, 1s to 30s |
| Worker heartbeat and owner TTL | `docker_lifecycle.py` | 120s TTL, 30s sweep |
| Capacity owner TTL | `capacity.py` | 180s TTL, 30s heartbeat |
| Stopped container grace | `docker_lifecycle.py` | 300s |
| Event stream reconnect | `docker_events.py` | 1s to 30s |

The ordering constraint between these values is documented in prose and
enforced nowhere in code. `sandbox_lifecycle.md` states it: capacity TTL must
exceed Docker TTL plus one sweep interval.

---

## Structural problems the refactor addressed

This is the critique the decomposition below answers, kept because each item
explains why a boundary sits where it does. Every one is now resolved, and the
module that resolved it is named.

### 1. The ownership state machine existed only in prose

**Resolved by `psrl/sandbox/ownership.py`.** The transitions `prepare -> queued ->
provisioning -> leased -> reclaiming -> released` now live in `LeaseStateMachine`,
which raises on an illegal transition. `holds_resources` is the one predicate that
decides whether an unreclaimable lease keeps its reservation, so a change to a
transition no longer has to be checked against every flag by hand.

### 2. One class owned seven independent concerns

**Partly resolved, and this is the largest remaining item.** `AdmissionGate`,
`NodeReclaimer`, and `PlacementService` were split out of `SandboxManager`, which
still owns the lease registry, workflow reservation, idempotent sharing, deferred
release, idle policy, prefetch, and shutdown. The decomposition below is the target.

### 3. Cancellation and cleanup were reimplemented per layer

**Resolved by `psrl/sandbox/async_utils.py`.** `complete_cleanup` and
`acquire_nowait` are the single implementation, applied at every layer rather than
restated. The two names for one operation are one operation.

### 4. Crash recovery is attached to the wrong owner

**Resolved by `psrl/sandbox/reclaimer.py`.** Heartbeat files and lease helpers moved
out of the backend, and the collector is a real module with an `argparse` entry
point and a `SandboxContainerRuntime` protocol, so the `python -c` program string is
gone and the collector can be imported, typed, and tested.

### 5. The portable contract was both too wide and too leaky

**Resolved by the split in `psrl/sandbox/core.py`.** The required protocol is the
data plane and the optional one is `SandboxStateProtocol`, which a backend implements
only when it declares the matching capability. A backend without state operations now
writes none, and the manager is the only caller of the optional half.

## Target decomposition

Split the module by concern, with each plane owning one responsibility and
depending only downward.

```text
  integration      episode ownership, spec construction from task config
                       |
  ownership        one explicit state machine over the lease lifecycle
                   idempotent sharing, workflow reservation,
                   deferred cleanup, shutdown
                       |
  admission        in-process resource accounting, no runtime knowledge
                       |
  provision        per backend create, connect, restore, destroy
                   owns "name it so a lost response is still reclaimable"
                       |
  runtime          one session and one transport per backend
                   stop detection is an injected strategy
                       |
  api              specs, results, refs, errors, capabilities
                   state operations are a separate optional protocol
```

The node reclaimer is a peer of this stack, not a layer inside it. It takes a
heartbeat directory and a container runtime interface, exposes a real entry
point, and is supervised by the worker rather than embedded in a backend.

### Migration order

Each step is independently verifiable and leaves the module working.

| Step | Change | Why first |
|---|---|---|
| 1 | Extract the node reclaimer into its own subsystem with a real entry point | It has no dependency on ownership or admission, and it is the only part that shells out |
| 2 | Split `api` from state operations. Introduce the explicit lease state machine in `ownership` | Removes the flags and callbacks that make every other change risky |
| 3 | Separate `provision` from `admission`. Turn the capacity coordinator into a library that the manager holds rather than a remote actor reached through the manager | Makes admission testable without Ray |
| 4 | Narrow the session protocol to I/O and move capability checks to the caller boundary | Lets consumers stop reaching into sessions |

---

## Resolved decisions

These shaped the target and are settled in {doc}`sandbox_refactor`.

1. **The MLGym and AIRS-Bench path stays.** The Docker session gains the
   persistent shell and the GPU passthrough under `required_features`, before the
   path moves to the sandbox manager.
2. **State operations are kept, capability gated.** A backend declares what it
   serves. The internal Docker backend serves filesystem snapshot, restore, and a
   cross node resume over a shared snapshot store. Full state snapshot and fork
   stay on the providers that advertise them.
3. **Admission is a library behind an interface.** The node agent owns it per
   node, so the manager does not reach a remote actor directly.

:::{seealso}
- {doc}`sandbox_lifecycle`: invariants, failure semantics, and the verification map
:::
