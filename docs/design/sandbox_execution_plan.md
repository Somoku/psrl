# Sandbox Execution Plan

This page is the executable work breakdown for the sandbox refactor. It turns
the design pages into ordered steps with a deliverable, an assertion set, a test
set, a metric set, an exit criterion, and a rollback note each.

It holds no new decisions. Every decision lives in a design page and is cited
here by id, and the traceability matrix at the end proves that no decision is
left without a step.

Read {doc}`sandbox_refactor` for the target and the reasoning,
{doc}`sandbox_lifecycle` for the invariants a step must not break, and
{doc}`sandbox_architecture` for the component map a step starts from.

---

## How to use this page

- **A step is done when its exit criterion holds**, not when its code is
  written. An exit criterion is always a test, a metric, or an assertion.
- **A phase gate is a hard stop.** The next phase does not start while a gate
  is open, because the ordering is what keeps the module working throughout.
- **Every step removes what it replaces.** No compatibility shim, no dual path,
  no dead flag. A step that cannot delete the old path is not ready.
- **A prose invariant becomes an assertion or a test.** If a rule only exists in
  a document after its step lands, the step is incomplete.

### Ground rules that apply to every step

| Rule | Source | How a reviewer checks it |
|---|---|---|
| No compatibility layer, fallback, or migration shim | Project contract | The diff deletes the replaced path in the same change |
| No knob an operator cannot estimate. Intents in, numbers derived | D14 | Every new config key is an intent, a fraction, or an explicit override |
| Every timeout ordering asserted where configuration is built | D14 | A violating configuration fails at startup, with a test |
| Every plane exposes `snapshot()` and never logs metrics | D13 | The plane has no logger call for metrics, and the trainer hook reads it |
| One session and one transport per backend, stop detection injected | Target shape | No second transport appears inside a backend |
| A capability is declared once, in step 6a | A4 | No second capability table exists anywhere |

---

## Phase map

Order is by dependency, then by risk. The two non obvious constraints are that
placement precedes the retirement, and that GPU accounting precedes the
retirement. Both are explained in {doc}`sandbox_refactor`.

| Phase | Outcome | Gate to open the next phase |
|---|---|---|
| 0 | Planes, one ownership state machine, config assertions, metric hook, recorded baseline | Existing suites pass unchanged, state machine tests pass, baseline recorded |
| 1 | The Docker session and policy own every feature the retired worker has, plus the egress allowlist | MLGym parity on the new session, GPU accounting test, egress test |
| 2 | A sandbox lands on any node, driven by placement, through a remote session | Multi node placement, remote execution, dedicated placement, reservation sweep |
| 3 | The parallel package is deleted, leaving one implementation | MLGym and AIRS-Bench end to end on the manager, including a dedicated env node |
| 4 | Density, image cost, idle reclamation, and the snapshot store | Benchmarks beat the Phase 0 baseline by the stated target |
| 5 | Credential injection and an optional isolation runtime | Credential and runtime policy tests |
| 6 | AgentEnv and OpenSandbox as first class providers | Provider conformance suites, including the two separate resume cases |

Loop side work is tracked separately at the end. It is not a phase, because the
sandbox module does not own it.

---

## Phase 0. Planes, state machine, and the measurement floor

No behavior changes in this phase. That is what makes the existing suites a
valid exit criterion, and it is why the risky structural work goes first.

| Step | Deliverable | Exit criterion | Implements |
|---|---|---|---|
| 0.1 | The node reclaimer becomes its own subsystem with a real entry point, taking a heartbeat directory and a container runtime interface, supervised by the worker | The `python -c` recovery string is gone and the reclaimer has unit tests without a Docker daemon | Architecture step 1 |
| 0.2 | `core.py` splits into a required I/O protocol and an optional state protocol, with one session type carrying a capability set | A backend with no state operations carries no state code, and the state methods keep their raising defaults | A1, D1 |
| 0.3 | One explicit ownership state machine over `prepare`, `queued`, `provisioning`, `leased`, `reclaiming`, `released` | An illegal transition raises, and every transition has a test. The flags and callbacks it replaces are deleted | D1 |
| 0.4 | Admission becomes a library behind an interface, held by its owner rather than reached as a remote actor through the manager | Admission is testable without Ray, and the existing fairness tests pass against the library | D7, architecture resolved decision 3 |
| 0.5 | The session protocol narrows to I/O, and capability checks move to the caller boundary | No consumer reaches into a session for a capability decision | D1 |
| 0.6 | Configuration builds intents into derived values and asserts every timeout ordering | A configuration that inverts an ordering fails at startup, with a test per ordering | D14, S11 |
| 0.7 | The trainer metric hook reads a `snapshot()` from every plane under the `sandbox/` prefix, bounded and failure tolerant | A plane that raises or hangs cannot stall a training step, and `meta/collect_failures` counts it | D13 |
| 0.8 | The performance baseline is recorded with the existing Docker benchmark | Create, exec, and release at p50 and p95, plus sandboxes per node at the envelope, are committed as the reference | Observability section |

### The orderings that step 0.6 must assert

These exist as prose in several design pages, and this is where they become
executable.

- The capacity lease TTL outlives the container TTL plus the sweep period.
- The cross node reservation TTL sits between the provision deadline and the
  sandbox lease TTL.
- The idle pause window is shorter than the inactivity reap window, which is
  shorter than the absolute lifetime.
- A member queue deadline is a fraction of the episode deadline.

**Rollback.** Every step in this phase is a structural change with the existing
suites as the contract, so a revert is a single change with no data or
deployment consequence.

---

## Phase 1. Docker session and policy feature absorption

Everything the retired worker does next to a container lands here, because
Phase 3 deletes that worker and Phase 2 needs the node side complete.

| Step | Deliverable | Exit criterion | Implements |
|---|---|---|---|
| 1.1 | An exec strategy interface with a persistent shell strategy and a one shot strategy, the strategy selected by the spec | MLGym parity on the persistent strategy, and a harness episode on the one shot strategy. Commands stay serialized on both | D4, C1 |
| 1.2 | Silence timeout, readiness probe, and head and tail observation truncation on the session exec contract | A silent command fails on its own timeout without failing the task, a container is ready only after the probe, and a long observation keeps both ends | Retirement audit |
| 1.3 | GPU passthrough without the NVIDIA container toolkit, a GPU dimension in the admission envelope, and `CUDA_VISIBLE_DEVICES` partitioning over the granted devices | Two concurrent sandboxes cannot be admitted for one device, and each sees only its partition | Retirement audit, architecture resolved decision 1 |
| 1.4 | A disk dimension in the admission envelope, implementing the existing `ResourceSpec.disk_mb` | A disk request is accounted and enforced, and an over request is rejected at admission | Phase 1 scope |
| 1.5 | A per sandbox egress allowlist | A sandbox reaches an allowed destination and fails a denied one, with a test per direction | D5, D12 |
| 1.6 | The idle primitives on the session, an in flight signal from the exec lock and an activity stamp at both command start and command return | A long running command reports in flight for its whole duration, with a test | D10 |

### Why 1.6 is here and not in Phase 4

Phase 4 owns the idle **policy**, the pause and the reap. This step owns the
**signal**, because the signal belongs to the session and the session is open in
this phase. Splitting them this way means the Phase 4 policy is a decision over
a trustworthy input, rather than a timestamp that lies for the duration of every
long command.

**Rollback.** Each feature is behind the strategy interface or a policy field,
so a revert disables the feature and leaves the session working. Nothing in this
phase deletes a consumer path.

---

## Phase 2. Placement and the node agent

This is the phase that lets the internal Docker backend use machines across
nodes, and it is the phase that gives the retired worker's dedicated placement
mode a target owner.

| Step | Deliverable | Exit criterion | Implements |
|---|---|---|---|
| 2.1 | A node agent service per sandbox node, composing the local manager, admission, backends, and reclaimer, owning the daemon, exposing the session protocol over Ray RPC | A caller on another node runs commands and files against a sandbox it does not host. No caller opens a remote daemon socket | D7, D9 |
| 2.2 | A placement service as a per job Ray actor, with a node registry, capability match, then image locality, then least loaded, and a watcher that rebuilds the fleet view | A stale node is drained and not selected, and placement restarts without persisted state | D6 |
| 2.3 | The reservation protocol, a lease with an id, an owner, a TTL, and a sweeper, with cancel separate from release | A rejected provision cancels explicitly, a caller that dies is swept, and `placement/reservations_open` returns to zero in both tests | D11 |
| 2.4 | The caller side `SandboxBackend` becomes a remote client returning a session that speaks the local protocol | `harness_agent_loop` and the MiniSWE runner are unchanged, verified by running them against a remote node | D1, D2, D9 |
| 2.5 | Reachability as a placement constraint, and a callback URL resolved for the target node rather than the caller | A sandbox on another node records TITO tokens, and a node behind a boundary is drained | Reachability section |
| 2.6 | Colocated and dedicated placement policies | A dedicated request lands on an env node and never on a trainer node, with a test | Retirement audit |
| 2.7 | A read only diagnostics surface on the node agent, addressed by sandbox id, serving the log tail, inspect, stats, and a four way exit reason. Placement retains the id to node mapping for a bounded window | A post mortem for a finished sandbox needs no knowledge of its node, and no diagnostic call can mutate a sandbox | Diagnostics section |
| 2.8 | The placement metric family | `placement/decision_s_p95`, rejections, `no_candidate`, reservations open, and reservations swept report per step | D13 |

### Status

Steps 2.1 to 2.3 and 2.5 are wired: the trainer creates the placement service and one
node agent per node in `gen_actor_rollout_ref.rollout.agent.node_ips`, each worker builds
its backend over those handles, and the plane is exercised on a real cluster by
`python -m tests.sandbox.smoke_ray_plane`.

Two exit criteria are not met yet, so the steps that carry them stay open:

- **2.4** asks that `harness_agent_loop` and the MiniSWE runner be unchanged while running
  against a remote node. They are unchanged, but no run has been made against a remote
  node: the smoke script covers the plane, not a rollout. A remote rollout needs a Docker
  daemon on a second host.
- **2.6** asks for colocated and dedicated placement policies. `required_label` pins a
  request to a labelled node, which is the mechanism, but there is no policy that keeps a
  sandbox off a trainer node beyond naming the fleet in `node_ips`.

Step 2.7's diagnostics surface exists on the node agent, and 2.8's metrics family reaches
the trainer's per-step hook, so both are done.

### One node stays the degenerate case

A single node deployment must be the same code path with one registry entry, and
this is verified rather than assumed. The Phase 1 suites run unchanged against
the node agent with one node, which is what proves the multi node design did not
fork the single node path.

**Rollback.** Placement selection is a policy with a node local mode, so a
revert selects the hosting node and the module behaves as it did in Phase 1. The
node agent stays, because by then it owns the daemon.

---

## Phase 3. One sandbox implementation

The audit table in {doc}`sandbox_refactor` is the gate. A feature without a
target owner blocks deletion, and so does a target owner that does not exist
yet.

| Step | Deliverable | Exit criterion | Implements |
|---|---|---|---|
| 3.1 | `psrl/environments/mlgym_env.py` moves from coordinator handles to manager leases, and the MLGym loop moves from `SandboxHandle` to `SandboxLease` and `SandboxSession` | MLGym end to end on the persistent shell strategy, with the same results as the baseline run | D3 |
| 3.2 | The AIRS-Bench path builds a sandbox manager at startup and requests dedicated placement | AIRS-Bench end to end on a dedicated env node, with GPU accounting enforced | D3 |
| 3.3 | The retired config keys fold into the sandbox config group, slot and routing keys becoming admission and placement settings | No key is left without a target | D3, D14 |
| 3.4 | The retired package's test cases move into `tests/sandbox/` | Every case that covered an invariant still covers it against the main stack | D3 |
| 3.5 | The parallel package and its design page are deleted, and the design index entry is removed | The tree has one sandbox implementation, and no import references the package | D3 |

**Rollback.** This is the only irreversible phase, because 3.5 deletes code.
Tag the commit before 3.5, and do not land 3.5 in the same change as 3.1 or 3.2.
A consumer problem found after 3.5 is fixed forward in the main stack, which is
the point of having the audit table as a gate.

---

## Phase 4. Efficiency, density, and the snapshot store

This is the first phase whose exit criterion is a number rather than a
behavior, which is why Phase 0 recorded a baseline.

| Step | Deliverable | Exit criterion | Implements |
|---|---|---|---|
| 4.1 | The idle policy, an idle pause and an inactivity reaper, both requiring no command in flight and no activity for a derived window | A container running a long command is never paused or reaped, with fault injection. `outcome/reaped_idle` and `state/pause_s_p95` report | D10, retirement audit |
| 4.2 | Configurable utilization and overcommit, with actual usage read from cgroup v2 rather than the static request | Sandboxes per node at the envelope beats the baseline by the stated target, with no OOM regression in `outcome/oom` | Phase 4 scope |
| 4.3 | Image aware placement by resolved digest, and a per run prefetch plan over the run's working set | `image/locality_hit_ratio` and `image/prefetch_coverage` report, and cold start p95 beats the baseline target | S3 |
| 4.4 | A warm container pool with its own budget and TTL, excluded from the reaper | A claim adopts a pool entry instead of pulling and creating, and a pool entry is never reaped as abandoned | Phase 4 scope |
| 4.5 | The shared snapshot store, a writable OCI registry on object storage, publishing on demand, keyed by digest, namespaced per run, with TTL garbage collection and a local cache budget with LRU eviction | A checkpoint on one node restores on another, a store outage fails the checkpoint rather than returning a node local reference, and a node under disk pressure evicts snapshots before base images | D8, S1, S2, S4, S5, S6, S7, S9, S10, S11 |
| 4.6 | Per task snapshot reuse, capturing a task's prepared environment once and reusing it across steps and groups | The second visit to a task pays no setup, verified by a setup duration metric across two steps | D17 |
| 4.7 | Optional, on metric evidence only. A node local peer proxy for base image blobs, with the Docker daemon pointed at it and push excluded | Adopted only when `image/pull_s_p95` and `image/locality_hit_ratio` show pull cost dominates. No image reference changes | S12 |

### The performance plan for this phase

The levers are not equal, and the order matters more than the individual
tuning.

1. **Image materialization.** The largest single cost on a cold cluster.
   Locality and prefetch address it without a cache service, and the peer proxy
   is the upgrade if the metrics ask for it. On demand layer loading is not
   reachable here and is not attempted, for the reason in S8.
2. **Setup reuse.** A per task snapshot moves setup from once per group per step
   to once per task per run. On a task set visited across many steps this is
   larger than any tuning in the next item.
3. **Density.** Idle pause, overcommit, and real usage accounting. Bounded by
   what a Docker daemon exposes, so the ceiling is lower than a provider's.
4. **Scheduling.** Placement and warm pools. Smallest lever, and only effective
   after the first three, because a scheduler cannot recover a cost that was
   already paid.

What this phase explicitly does not promise is provider grade density or cold
start. That lives in Phase 6, and the internal backend is the test path (D2, S8).

**Rollback.** Every step is a policy or a configuration that defaults to the
Phase 3 behavior. The store is the exception, because a published snapshot
outlives a revert, so its garbage collection must be able to expire snapshots
whose writer is gone. That is already the TTL design (S4).

---

## Phase 5. Remaining security

| Step | Deliverable | Exit criterion | Implements |
|---|---|---|---|
| 5.1 | Credential injection, so a task secret never enters an image or a spec that is logged | A task reaches its credential, and no sanitized surface contains the value | D5, E3 |
| 5.2 | An optional gVisor or Kata runtime selected by a policy profile | A profile that requires a runtime fails at admission when the node cannot provide it, rather than falling back | D5 |

The egress allowlist is not here. It landed in Phase 1 (D12).

**Rollback.** Both steps are policy profiles, so a revert selects the default
profile.

---

## Phase 6. External backends

Three ordered steps, each with its own design page. 6a changes no provider
behavior, which is what lets 6b and 6c be independently testable.

### Step 6a. Capability model and the shared layer

| Step | Deliverable | Exit criterion | Implements |
|---|---|---|---|
| 6a.1 | The new `SandboxFeature` members, and `ResumeLevel` with `resume_level` on capabilities and `required_resume_level` on the spec | A `full_state` requirement is rejected on a `filesystem` backend at admission, never downgraded | A2, A5, S10 |
| 6a.2 | State operations routed through the manager with the existing guards, and the cross node intent added to the restore guard | No consumer calls a backend state method directly | A1, D1 |
| 6a.3 | A forked or restored session always takes the sanitization path, refreshing transport and reseeding entropy | Two children of one fork draw different random streams, with a test | A6, B6 |
| 6a.4 | `acquire_group` over one spec per member, with source and resource agreement validated, a workflow free parent terminated after adoption, and no member holding capacity while waiting for a sibling | Two concurrent groups on a node that fits one and a half groups both complete, and neither stalls to its deadline | A7, D15, D16 |
| 6a.5 | Error mapping and provisioning ownership, reclaiming a lost create by listing on the idempotency metadata | The lost create and lost parent fault injection cases pass, and no duplicate group can exist | Error mapping section |
| 6a.6 | The shared four part backend skeleton and the provider conformance suite, including the two separate resume cases | The suite runs against a provider behind its environment flag, and the internal Docker backend fails the `full_state` resume case | A4, conformance table |

The conformance suite is a Phase 6 deliverable rather than a per provider one,
because a suite written twice is a suite that diverges.

### Step 6b. AgentEnv

| Step | Deliverable | Exit criterion | Implements |
|---|---|---|---|
| 6b.1 | The full feature set declared on the AgentEnv state driver, including the resume level, with the E2B compatible data plane kept and the control plane native | The declaration matches the normative table, and no capability is silently absent | E2 |
| 6b.2 | `fork(count)` batched at the provider maximum, driven by the member spec list, failing the group on a partial fork | A group of the configured size is produced in one provider call per batch, and a partial fork leaves no orphan | B1, B2, B6 |
| 6b.3 | A snapshot name sent on capture, and a restore addressable by id or name | A snapshot is found by name and restores | Gap 2 |
| 6b.4 | The workflow TTL sent on resume | A resumed sandbox does not expire against its pre pause deadline | Gap 3 |
| 6b.5 | The template pulled or built in `prepare`, with the watch status polled to ready inside a build deadline | A failed build fails admission, not a rollout, and the first acquire is a warm claim | A3, B3, B4 |
| 6b.6 | Volume support behind the volume feature, as a spec field distinct from a host mount | A task that needs provider storage gets it, and a host mount request is still rejected on this backend | B5 |
| 6b.7 | Lost create reclamation by listing on the create metadata | The fault injection case passes | Gap 6 |

### Step 6c. OpenSandbox

| Step | Deliverable | Exit criterion | Implements |
|---|---|---|---|
| 6c.1 | The control client and the backend, with create, connect, terminate, and shutdown, never hardcoding a data endpoint | A sandbox is created and terminated, and the provider state machine maps to `SandboxStatus` | Provider shape |
| 6c.2 | The exec client and the provider native persistent bash session, with the one shot mode selected by the spec | A harness keeps shell state across turns, and a grader step runs one shot | C1 |
| 6c.3 | The state driver, with an asynchronous pause polled to settle, a resume, a snapshot, a restore, and a data endpoint re-resolved after every resume | A resume that moved the sandbox still accepts commands | C2 |
| 6c.4 | `prepare` for the template build and the warm pool claim, and the egress policy and credential vault wired through | The provider enforces the policy, and PSRL reads only sanitized metadata | A3, C3, C4, E3 |

### Step 6d. Selection by capability

| Step | Deliverable | Exit criterion | Implements |
|---|---|---|---|
| 6d | A recipe declares a capability profile with an optional backend pin, and the manager selects and rejects at admission | A recipe runs unchanged on two backends that both satisfy its profile, and fails fast on one that does not | E1 |

**Rollback.** A provider backend is additive. A revert removes a backend from
the registry and the recipes that pinned it fail at startup rather than
silently changing behavior, which is the intended failure.

---

## Loop side work, tracked but not owned here

These are agent loop and session concerns. The backend contract enables them,
and none of them blocks a sandbox phase.

| Id | Work | Depends on | Why it is not a sandbox phase |
|---|---|---|---|
| L1 | The worker builds one spec per group member, sets the member queue deadline from the episode deadline, and bounds groups in flight | 6a.4 | The loop owns group identity and the episode deadline |
| L2 | A resumable agent loop, a control layer outside the preemptible pool, a durable episode checkpoint keyed by the trajectory id, TITO continuity, and idempotent commit and reward | 6a, 6b, 4.5 | These are loop and session state, not sandbox state |
| L3 | A thin workflow pause hook, so a preempted trainer pauses its workflow's sandboxes | 6a.2 | The trainer decides when, the provider does the pause |

L2 is what turns `RESUME_ANYWHERE` into a surviving rollout. Until it lands, the
capability is verified by conformance rather than used in production, and that
is an acceptable state because the alternative is a backend built after the loop
that needs it.

---

## Observability rollout

The hook is built once in step 0.7 and each later phase adds its family. A phase
that adds a mechanism without its metric is not done, because the next phase's
exit criterion may depend on reading it.

| Phase | Families added |
|---|---|
| 0 | Admission, utilization, collection meta |
| 1 | Exec, lifecycle outcome |
| 2 | Placement, provision |
| 4 | Image, state, snapshot store |
| 6 | The same families, reported per backend |

Three signals are the ones an operator acts on, and they are the alerting set.
An oldest lease age growing without bound, reservations open growing while
create latency is flat, and any snapshot push failure. The rest is diagnostic.

---

## Risk register

| Risk | Phase | Early signal | Response |
|---|---|---|---|
| The state machine rewrite changes behavior silently | 0 | An existing suite fails for a reason that looks cosmetic | Treat any behavior delta in Phase 0 as a defect, not an improvement. Phase 0 is defined as behavior preserving |
| The persistent shell diverges from the retired sentinel protocol | 1 | MLGym results differ from the baseline run | Parity against a recorded MLGym run is the exit criterion, not a passing test |
| GPU double admission | 1, 3 | Two sandboxes report the same device | 1.3 lands before 3.1, and the accounting test is a phase gate |
| A dedicated placement regression hides until AIRS-Bench runs | 2, 3 | A GPU sandbox appears on a trainer node | 2.6 lands before 3.2, and placement asserts the node class |
| Reservation leak starves the cluster slowly | 2 | Reservations open grows while create latency is flat | The sweeper plus the explicit cancel, both tested, and the metric in the alerting set |
| An idle pause stops a running test suite | 4 | A task fails with a truncated test run | The in flight condition from 1.6, with fault injection in 4.1 |
| Overcommit causes OOM kills | 4 | The OOM outcome counter rises | Overcommit is off by default, raised against the OOM counter, and the counter is an exit criterion |
| The snapshot store fills node disks | 4 | The local disk used ratio approaches its budget | The separate budget with LRU eviction, and base images protected from eviction |
| A published snapshot outlives its writer | 4 | Store objects with no live run | TTL garbage collection plus an object store lifecycle rule |
| A conformance pass on the internal backend is read as harness resumability | 6 | A resume works in test and fails in production | The resume level, and two separate conformance cases, one of which the internal backend must fail |
| A provider capability drifts from its declaration | 6 | A required feature is satisfied but the behavior is absent | One normative table, and conformance runs per declared feature |

---

## Decision traceability

Every decision in every sandbox design page maps to a step. A decision with no
step is either not designed or not needed, and neither is acceptable.

| Id | Step |
|---|---|
| D1 | 0.2, 0.3, 0.5, 2.4, 6a.2 |
| D2 | 2.4, 4.5, and the Phase 4 performance plan |
| D3 | 3.1 to 3.5 |
| D4 | 1.1 |
| D5 | 1.5, 5.1, 5.2, 6c.4 |
| D6 | 2.2 |
| D7 | 0.4, 2.1 |
| D8 | 4.5, 4.6, 6a.1 |
| D9 | 2.1, 2.4 |
| D10 | 1.6, 4.1 |
| D11 | 2.3 |
| D12 | 1.5 |
| D13 | 0.7, 2.8, and the observability rollout |
| D14 | 0.6, 3.3 |
| D15 | 6a.4, L1 |
| D16 | 6a.4, 6b.2 |
| D17 | 4.6 |
| A1 | 0.2, 6a.2 |
| A2 | 6a.1 |
| A3 | 6b.5, 6c.4 |
| A4 | 6a.6, and the ground rules |
| A5 | 6a.1 |
| A6 | 6a.3 |
| A7 | 6a.4 |
| B1 | 6b.2 |
| B2 | 6b.2 |
| B3 | 6b.5 |
| B4 | 6b.5 |
| B5 | 6b.6 |
| B6 | 6a.3, 6b.2 |
| C1 | 1.1, 6c.2 |
| C2 | 6c.3 |
| C3 | 6c.4 |
| C4 | 6c.4 |
| E1 | 6d |
| E2 | 6b.1 |
| E3 | 5.1, 6c.4 |
| E4 | L3 |
| S1 | 4.5 |
| S2 | 4.5 |
| S3 | 4.3, 4.5 |
| S4 | 4.5 |
| S5 | 4.5 |
| S6 | 4.5 |
| S7 | 4.5 |
| S8 | 4.7 gating, and the Phase 4 performance plan |
| S9 | 4.5 |
| S10 | 4.5, 6a.1 |
| S11 | 0.6, 4.5 |
| S12 | 4.7 |

The three resolved decisions on {doc}`sandbox_architecture` map to 1.1 and 1.3
for the retained MLGym path, 6a.1 for capability gated state operations, and 0.4
for admission as a library.

---

## Definition of done

The refactor is complete when all of the following hold at once.

- One sandbox implementation exists in the tree, and every consumer reaches it
  through the top level API.
- The internal Docker backend and both providers pass the same conformance
  suite, differing only where a capability declaration says they differ.
- A sandbox runs on any node in the cluster, and a caller cannot tell a local
  session from a remote one.
- Every invariant in {doc}`sandbox_lifecycle` has coverage, and every timeout
  ordering fails at startup when violated.
- Every metric family reports, and the Phase 4 targets hold against the Phase 0
  baseline.
- No decision in the traceability matrix is unimplemented.

:::{seealso}
- {doc}`sandbox_refactor`: the target, the reasoning, and the decisions
- {doc}`sandbox_lifecycle`: the invariants and the verification map
- {doc}`sandbox_architecture`: the component map each step starts from
- {doc}`sandbox_backend_integration`: the normative capability model for 6a
- {doc}`sandbox_agentenv`: the AgentEnv design for 6b
- {doc}`sandbox_opensandbox`: the OpenSandbox design for 6c
- {doc}`sandbox_snapshot_store`: the store design for 4.5
:::
