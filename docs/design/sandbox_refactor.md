# Sandbox Refactor Plan

This page is the working plan for turning the PSRL sandbox into a thin control
plane in front of pluggable sandbox services. It records the decisions taken,
the order of migration, and the open questions.

Read it with {doc}`sandbox_architecture` for the current component map and with
{doc}`sandbox_lifecycle` for the invariants a change must not break. This page
says where the module is going. Those two say where it is and what it preserves.
{doc}`sandbox_external_backends` designs the two external providers in detail.

---

## Target shape

PSRL keeps only what it alone can own. Everything a sandbox provider can own
moves behind a backend.

- The PSRL side is a **thin control plane**. It admits a request, owns the
  lease, reserves the workflow phase, and returns a session handle. It holds no
  runtime knowledge.
- Every backend, **including the internal Docker backend**, is a service reached
  through the same top-level API. No component outside the sandbox module may
  treat the Docker path as special.
- The internal Docker backend is the local distributed test backend. It needs no
  external platform, runs on a plain cluster with a local daemon per node, and
  exercises the same contract the external providers implement.

### Planes

Each plane owns one responsibility and depends only downward. The current files
are shown so a change has an obvious home.

| Plane | Owns | Starts from |
|---|---|---|
| `api` | Specs, refs, results, errors, capabilities. A required I/O protocol and a separate optional state protocol | `core.py` |
| `ownership` | One explicit lease state machine, idempotent sharing, workflow reservation, deferred cleanup, shutdown | `manager.py` |
| `admission` | CPU, memory, GPU, and disk accounting with class guarantees. No runtime knowledge | `capacity.py` |
| `placement` | Cluster wide node selection by capability, image locality, and load | new |
| `provision` | Per backend create, connect, restore, destroy, and the naming that survives a lost response | `backends/` |
| `runtime` | One session and one transport per backend. Stop detection is an injected strategy | `backends/docker_*` |
| `reclaimer` | Node level orphan reclamation by heartbeat age, with a real entry point | `backends/docker/cli.py` |
| `integration` | Episode ownership and spec construction from task config | `harness_agent_loop.py`, `examples/mini_swe/runner.py` |

---

## Decisions

| Id | Decision | Consequence |
|---|---|---|
| D1 | PSRL keeps a thin control plane. Sandboxes are services, the internal Docker backend included | The manager stops growing runtime features. Feature work moves behind the backend contract |
| D2 | The internal Docker backend is the local distributed test backend | It must need no external platform and stay contract complete. It must not become the production scale path. It keeps a filesystem state path over a shared snapshot store (D8) |
| D3 | One sandbox implementation. Every backend, internal or external, is reached through the same manager | Every sandbox invariant lives in one place, and a workload cannot reach a container by a second path |
| D4 | Efficiency and density are fixed in the main stack. Exec is one interface with an injectable strategy, and the default strategy is a persistent shell | One exec interface, not two. A harness that runs a single long foreground command and a benchmark that needs shell state across turns are two strategies behind one contract, not two contracts |
| D5 | Security controls are added and, where a provider offers them, delegated to the provider | Egress and secret handling are not reimplemented twice |
| D6 | The placement service is a per job Ray actor | It reuses the retired coordinator's deploy path. The node agent boundary stays stable if it later becomes a standalone service |
| D7 | The node agent reuses `SandboxManager` | One ownership implementation instead of two |
| D8 | The internal Docker backend keeps a filesystem state path over a shared snapshot store, and `RESUME_ANYWHERE` carries a semantic level | It exercises pause, resume, and cross node resume at the `filesystem` level. Process and memory state do not move, and a caller that needs them requires the `full_state` level instead |
| D9 | The remote session protocol is Ray RPC | It matches the deployment. A documented protocol can follow for a non Ray caller |
| D10 | Idle means no command in flight and no activity for the window, not an aged timestamp | A pause or a reap acquires the session exec lock without waiting. A long command can never look idle |
| D11 | A cross node reservation is a lease with an id, an owner, a TTL, and a sweeper, reusing the single node model | A rejected or lost provision cannot leak a slot. Placement holds a cache and rebuilds it from the nodes |
| D12 | The per sandbox egress allowlist is a Phase 1 data integrity control, not a Phase 5 security feature | An unbounded egress makes a reward untrustworthy, so it is fixed before the efficiency work |
| D13 | Every plane exposes `snapshot()` and the trainer's existing per step hook is the only reader | Sandbox metrics share the training step axis and the tracking logger. No plane logs, and a collection failure cannot stall a step |
| D14 | An operator configures an intent and the module derives the dependent timeouts, and every timeout ordering is asserted at configuration build | A deployment cannot misconfigure an ordering, and a knob that only an author could set does not exist |
| D15 | A group is admitted member by member and no member holds capacity while waiting for a sibling. Atomic group reservation is rejected | A deadlock between two groups is structurally impossible, and admission keeps the non reserving queue it has. Group members can start at different times |
| D16 | A group is N specs differing only in identity, and a fork parent belongs to no workflow | The per trajectory `workflow_id` and `idempotency_key` survive a group, and the one phase per workflow reservation needs no new semantics |
| D17 | A task's setup is captured once as a snapshot and reused across steps and groups, with fork as the fallback | Setup cost drops from once per group per step to once per task per run, and the group stops being a scheduling unit at all |

---

## Migration order

Each phase is independently verifiable and leaves the module working. Order is
by dependency, then by risk.

| Phase | Change | Depends on | Verification |
|---|---|---|---|
| 0 | Make the planes and the ownership state machine explicit, with no behavior change. Record the performance baseline | none | existing unit and contract tests pass, state machine tests added, baseline recorded |
| 1 | Docker session and policy absorb every retired feature, including GPU accounting and the egress allowlist | 0 | live Docker tests, MLGym parity on the new session, egress policy tests |
| 2 | Cluster placement service and a remote session endpoint, so a sandbox can land on any node | 0, 1 | multi node placement and remote execution |
| 3 | MLGym and AIRS-Bench move to the sandbox manager, and the parallel package is deleted | 1, 2 | MLGym and AIRS-Bench end to end on the manager, including a dedicated env node |
| 4 | Efficiency and density. Envelope gains overcommit, image locality, idle pause, warm containers | 2 | density and cold start benchmarks against the Phase 0 baseline |
| 5 | Remaining security. Credential injection and an optional isolation runtime | 1 | credential and runtime policy tests |
| 6 | External backends. AgentEnv and OpenSandbox as first class providers | 0, 2 | provider conformance suites |

Two ordering constraints are not obvious and are the reason this order is not
the order the phases were first written in.

- **Placement comes before a consumer switches.** A dedicated environment node is a
  placement decision, so a workload that needs one can only move to the manager once
  placement exists. Switching first would leave its sandboxes on the agent loop node
  and put device-using sandboxes on the trainer's nodes.
- **Device accounting comes before a consumer that needs devices switches.** If the
  switch happens before the admission envelope has a device dimension, two sandboxes
  can be admitted for the same device.

### Phase 0. Planes and the state machine

Split `core.py` into a required I/O protocol and an optional state protocol.
Move `manager.py` concerns into the ownership plane with one explicit machine
over `prepare -> queued -> provisioning -> leased -> reclaiming -> released`.
The state machine replaces the flags and callbacks that make every other change
risky. This phase changes structure only, so the existing tests are the
verification.

It also records the baseline the later phases are measured against, described in
[Observability](#observability).

### Phase 1. Docker session and policy feature absorption

Everything the retired worker does next to a container lands here, because
Phase 3 deletes that worker and Phase 2 needs the node side to be complete.

- A persistent shell exec strategy, a readiness probe, a silence timeout, and
  head and tail observation truncation.
- GPU passthrough without the NVIDIA container toolkit, a GPU dimension in the
  admission envelope, and `CUDA_VISIBLE_DEVICES` partitioning over the admitted
  devices. The envelope gains GPU here rather than in Phase 4, because a
  consumer switch without it can over admit a device.
- Disk in the admission envelope. `ResourceSpec.disk_mb` already exists and
  Docker does not implement it yet.
- The per sandbox egress allowlist. It lands here, not in Phase 5, because an
  agent that reaches an unintended network is a training data problem before it
  is a security problem. The DeepSeek sandbox report documents agents reading
  runtime logs, forging control calls, and bypassing access control to raise
  their reward, so an unbounded egress makes a reward untrustworthy. The
  allowlist only depends on Phase 0, so nothing forces it to wait.

Details of what each feature carries and where it lands are in the retirement
audit below. The session contract keeps commands serialized, so the exec
strategies share one interface.

### Phase 2. Cluster placement

Add the placement service and the node agent described in the multi node section
below. This is the change that lets the internal Docker backend use machines
across nodes, and it is what gives the retired worker's dedicated placement mode
a target owner.

### Phase 4. Efficiency and density

- Make `utilization` and overcommit configurable, and reclaim memory for idle
  sandboxes.
- Add an inactivity reaper and an idle pause, so a sandbox that is waiting for
  the model releases compute instead of pinning memory. See
  [Idle is not the absence of a recent command](#idle-is-not-the-absence-of-a-recent-command).
- Add image locality to placement and a per run prefetch plan. See
  [Image strategy](#image-strategy).
- Add a warm container pool, so a claim adopts a prepared container instead of
  pulling and creating. See [Image strategy](#image-strategy).

### Phase 5. Remaining security

- Credential injection, so task secrets never enter the sandbox image.
- Optional gVisor or Kata via `RuntimeClass`, selected by a policy profile.

The egress allowlist is in Phase 1, for the reason given there.

### Phase 6. External backends

Add the capability model and the two provider backends, in order.

- 6a. Capability model and shared backend layer. {doc}`sandbox_backend_integration`
- 6b. AgentEnv backend. {doc}`sandbox_agentenv`
- 6c. OpenSandbox backend. {doc}`sandbox_opensandbox`

The capability and best-practice rationale is in
{doc}`sandbox_external_backends`.

---

## Multi node internal Docker backend

Today `SandboxManager` runs on the agent loop worker, and the trainer creates one
`SandboxCapacityCoordinator` per node and gives each worker its own node's
coordinator. A sandbox therefore always lands on the node that runs the agent
loop. Capacity is correct per node and cannot address another node.

To use machines across nodes, add the two planes the target shape names.

### Placement service

One per training job, it holds a registry of sandbox nodes and selects one.

- Each node advertises its capabilities. Backend kinds, supported features, GPU
  count, and host mount support.
- Selection is capability match first, then image locality, then least loaded.
  This generalizes the node selection the module already needed, and adds image
  locality to it.
- It reserves before the node agent provisions, and it enforces per node and per
  class ceilings so placement can never over admit a node.
- A node that misses its heartbeat is drained, not selected. Placement refuses a
  stale node rather than trusting a cached load.

### The reservation protocol

Placement reserves and the node enforces, so a reservation exists in two places
at once. Without a release protocol every rejected or lost request leaks a
reservation, and a cluster starves one slot at a time. `capacity.py` already
solves this for one node, and the cross node version reuses the same model
rather than inventing a second one.

- **A reservation is a lease.** It has an id, an owner, and a TTL, exactly like
  a capacity lease. Placement returns the id with the node choice.
- **The caller cancels what it does not use.** A node that rejects the
  provision, or a provision that fails, makes the caller cancel the reservation
  by id. Cancel is separate from release, the same split `capacity.py` already
  makes, because cancelling a never-provisioned reservation and releasing a
  finished sandbox are different events and the metrics must tell them apart.
- **A sweeper is the backstop.** A reservation whose owner stops renewing past
  its TTL is swept. This covers a caller that died between the reserve and the
  provision, which is the case an explicit cancel cannot cover.
- **The caller renews on a cadence.** A worker renews every reservation it still
  holds a handle for at a third of the TTL, so one lost call does not cost the
  reservation. A released sandbox leaves the worker's handle index, so the worker
  stops renewing it the moment it ends rather than keeping a dead slot charged.
- **The node is authoritative and rebuilds the truth.** Placement holds a cache,
  not a ledger. On restart it rebuilds its view from the node agents rather than
  from persisted state, the same way the retired coordinator's watcher rebuilt a
  fleet view. A reservation that no node knows about does not survive the
  rebuild.
- **The reservation TTL is shorter than the sandbox lease TTL, and longer than
  the provision deadline.** Any other ordering either sweeps a reservation while
  its provision is still in flight, or holds a slot long after the caller is
  gone. This ordering is an assertion in the code, not a sentence in a document,
  for the reason given in [Invariants are code](#invariants-are-code).

### Node agent

One per sandbox node, it owns everything that must run next to the daemon.

- The local manager, admission, backends, and the reclaimer, as one service.
- It exposes the session protocol over RPC. A caller on another node can run
  commands, read and write files, pause, resume, snapshot, and terminate.
- It owns the daemon. The caller never talks to a remote daemon directly.
- It exposes a read only diagnostics surface, described below.

### Diagnostics after the sandbox moves off the node

Today a failing sandbox is debugged on the worker's own node, with the daemon a
socket away. Once a sandbox can land anywhere, the operator no longer knows
which node to log into, and the container may already be gone. The node agent
has to answer that, because it is the only process that can.

The node agent serves, read only and per sandbox id:

- The container's recent log tail and its last exec results, bounded the same
  way an observation is bounded.
- The inspect output and the resource stats, which are what distinguish an OOM
  kill from a non zero exit.
- The reason a sandbox left, which is the field a post mortem starts from.
  Reclaimed by heartbeat age, reaped for inactivity, terminated by the caller,
  and killed for memory are different events and must not collapse into one.

Two rules keep this from becoming a second control plane. It is read only, so no
diagnostic call can change a sandbox. And it is addressed by sandbox id, so the
operator never needs to know the node, which is the whole point of placement.

The sandbox to node mapping is what makes that lookup possible, so placement
retains it for a bounded window after a sandbox ends.

**Do not proxy a remote Docker daemon.** Connecting to another node's daemon
over TCP would create containers whose memory the caller's admission does not
account for. Admission must live with the daemon, so the node agent owns the
daemon and the caller holds only a remote session handle.

### Caller side

The `SandboxBackend` on the caller side becomes a remote client.

- `create` asks the placement service for a node, then asks that node agent to
  provision.
- The returned session is a remote handle that speaks the same protocol as a
  local session, so `harness_agent_loop` and the runner do not change.
- The lease and the workflow reservation stay on the caller side. Capacity is
  charged on the node that hosts the sandbox.

This is what makes "the internal Docker backend is external to the other
components" concrete. The caller cannot tell a local session from a remote one.

### Reachability

A sandbox on another node must still reach the session server that records TITO
tokens. The single node case relies on `SandboxSession.resolve_callback_url`,
which rewrites a worker loopback URL through the Docker host gateway. Cross node
placement breaks that assumption, because the sandbox is no longer a sibling of
the worker.

- **The node forwards.** The worker passes its session server as a `host:port` with
  the acquire, and the node opens one `CallbackForwarder` per caller: a listener on
  a node-chosen port that proxies to the worker. One forwarder serves every sandbox
  that worker places on that node.
- **The sandbox is handed the same URL either way.** `resolve_callback_url` rewrites
  the loopback host to the node's gateway alias and the port to the forwarded port,
  so a harness does not know or care which node it landed on. The node chooses the
  port rather than reusing the caller's, because a caller's port may already be taken
  on the node.
- A node is only a placement candidate if it can reach the session server
  endpoint. Advertise that reachability as a placement constraint, not as a
  session detail: a node whose forwarder cannot reach the worker leaves a sandbox
  unable to record its tokens.
- A node on the wrong side of a network boundary is drained, not selected.

### Which side owns what

Multi node moves some components to the RL side and some to the node side. This
table is the boundary. The plane column refers to the planes in the target shape.

| Component | Side | Plane |
|---|---|---|
| Placement service | Thin control plane | `placement` |
| Node registry, capability and health, image index aggregation | Thin control plane | `placement` |
| Lease, workflow reservation, idempotent sharing | Thin control plane | `ownership` |
| Remote session client, the caller's `SandboxBackend` | Thin control plane | `provision`, `runtime` |
| Spec and result contract, capability gating | Thin control plane | `api` |
| Node agent process | Docker backend side | deployment |
| Local admission and capacity envelope | Docker backend side | `admission` |
| Docker daemon adapter, session, transport | Docker backend side | `provision`, `runtime` |
| Node reclaimer and heartbeat collector | Docker backend side | `reclaimer` |
| Local image index and pull | Docker backend side | `runtime` |

The Docker backend side is the node agent. It is not the thin control plane, even
though it reuses the same ownership and admission code. The line is what must run
next to the daemon, and that is the runtime and the admission that guards it.

Admission appears on both sides. The caller reserves through placement, and the
node enforces the same reservation locally. The node is authoritative, because
only the node can see its own daemon.

### Failure and reclamation

- The lease stays on the caller. The container is owned by the node agent.
- A node agent that dies leaves its containers to that node's reclaimer, which
  reclaims by heartbeat age, the same way the single node case already does.
- A node that cannot confirm cleanup rejects new work until its daemon and
  collector recover, so a healthy caller cannot be admitted against a node that
  still holds memory.

### Cross node resume

Placement alone does not move a running sandbox. Cross node resume needs a
snapshot on shared storage, which is a state operation and not a placement one.

The internal Docker backend keeps this state path. A node agent writes a
filesystem snapshot to a shared snapshot store, and a restore on another node
pulls it. The snapshot is filesystem only, because Docker commits the writable
layer and has no supported memory checkpoint. Process and memory state do not
move, so a resumed container starts fresh processes over the restored
filesystem. The harness must therefore be restartable from the filesystem, which
is the constraint the resumable rollout section already names.

The external providers do the full state version natively in Phase 6, where a
resume restores memory as well.

### Resume has two semantic levels

These two resumes are not the same operation, and a single boolean capability
would say they are. A backend that resumes a filesystem elsewhere and a backend
that resumes a live process elsewhere both satisfy "the state moved", but only
one of them keeps the harness running.

That difference is exactly the risk in D8. The internal backend is the test
path, so a conformance run that passes on it would otherwise be read as evidence
that a harness survives a resume. It is not, because on Docker every process is
new. The capability has to carry the level so the test cannot make that claim.

| Level | What survives | Backend | What a caller must tolerate |
|---|---|---|---|
| `filesystem` | The workspace and anything written to disk | Internal Docker over the snapshot store | Every process is new. The harness restarts and rereads its own state from disk |
| `full_state` | Memory, processes, and the filesystem | AgentEnv, OpenSandbox | Nothing. The harness continues where it stopped |

The rules that follow from it.

- A spec requires a level, not a flag. Requiring `full_state` on the internal
  Docker backend is rejected at admission rather than downgraded.
- The internal Docker backend is used to verify lifecycle, ownership,
  admission, and the store round trip. It is not used to verify that a harness
  is resumable, because it cannot.
- Harness resumability is verified on a `full_state` backend, and separately the
  harness is verified to be restartable from a filesystem, which is the weaker
  and portable property.

The capability encoding is in {doc}`sandbox_backend_integration`.

The store design, including the transport, the key, the flows, and the GC, is in
{doc}`sandbox_snapshot_store`.

### One node is the degenerate case

Multi node must be a configuration change, not a code fork. With the placement
service and node agents, a single node cluster is the same code path with one
entry in the registry.

---

## Group scheduling

A GRPO group is `rollout.n` trajectories of one prompt. The backend can fan a
group out from one prepared sandbox, which is the largest efficiency win in
{doc}`sandbox_agentenv`. This section is about what that does to admission,
because a group of sandboxes admitted as a unit is a gang, and a gang is how a
scheduler deadlocks.

### A group is a completion unit, not a scheduling unit

The algorithm needs a group's trajectories to **finish** before it can compute
an advantage. It does not need them to **start** together. Nothing in the
rollout requires simultaneity, so nothing in admission has to provide it.

One structural fact makes that conclusive.

- The backend that has a native fork is a provider backend, and a provider
  sandbox consumes no worker node capacity, so `uses_node_capacity` is false and
  local admission is skipped entirely.
- The backend that uses local admission is the internal Docker backend, and it
  has no native fork, so a group there is `fanout` independent creates with no
  shared parent.

So the fork path never touches the local envelope, and the local envelope path
never has a shared parent. Gang admission is required by neither.

### A group is N identities, not one spec

`workflow_id` and `idempotency_key` are per trajectory today. The MiniSWE runner
sets `workflow_id` from the sample's session id and derives the idempotency key
from the same prefix, so a group carries `fanout` distinct values of each. A
single spec cannot hold them.

- The caller builds one spec per member, differing only in identity. The manager
  validates that the members share a source and a resource request, which is the
  precondition a fork needs anyway.
- The parent of a fork belongs to no trajectory. It is provisioned with no
  `workflow_id` and its own resource class, so it cannot collide with a member's
  workflow reservation and cannot starve a grader.
- The parent is terminated as soon as the children are adopted. Its slot is held
  for the fork, not for the episode.

This is why the workflow reservation does not need new semantics. One active
phase per workflow stays true, because each member is its own workflow and the
parent is not a workflow at all.

### Independent does not mean unshared

Independent describes admission, not content. A reader who takes it to mean the
members share nothing would remove the reason the group path exists at all, so
the axes are worth separating.

| Dimension | Shared across a group | Mechanism |
|---|---|---|
| Task environment and setup | yes | The fork parent, or a per task snapshot |
| Image layers | yes | One digest, shared overlay lower directories, and locality that prefers one node |
| Prompt prefix KV | yes | Session affinity in the router, which is not a sandbox concern |
| The sandbox instance | no, and must not be | Each trajectory writes its own workspace |
| Admission and capacity | no | One lease per member |
| Workflow reservation | no | One workflow per member |
| Idempotent sharing | must be prevented | One key would put two trajectories in one sandbox |

The last row is a correctness constraint, not a preference. Idempotent sharing
exists so a retried acquire adopts an existing sandbox, and inside a group that
would make two trajectories write the same workspace and correlate the
advantage. Distinct member identity is what prevents it, which is a second
reason the group is N specs and not one.

The sharing that matters happens where it costs nothing. On the fork path it
happens inside one provider call, and that path consumes no node capacity at
all. On the internal Docker path there is no parent to share, so nothing is
lost by admitting members independently.

A per task snapshot is the stronger form, because its reuse is not time
coupled. A fork requires the parent to be alive while the group fans out. A
snapshot is already durable, so any member arriving at any time in any later
step reuses it. The reuse surface grows from one group in one step to one task
for the whole run, and the requirement for simultaneity disappears with it.

### No member holds capacity while waiting for a sibling

That single rule is what makes a deadlock impossible, and it is worth stating as
a rule because the obvious implementation breaks it. Two groups that each admit
half of a node and then wait for the other half starve each other until the
queue deadline expires, and the node sits half idle for the whole window.

Three properties keep the rule true.

- **Members are admitted independently.** `capacity.py` grants one resource
  vector per lease through a per class queue that deliberately does not reserve,
  and that stays as it is. A member that cannot be admitted fails on its own,
  holding nothing.
- **The member deadline is short and derived.** A member queues against a
  fraction of the episode deadline, not against the default queue timeout, so a
  shortage costs an episode retry rather than a long stall. An operator sets the
  episode deadline and never this number.
- **Demand is bounded at the source.** `AgentLoopManager` already governs how
  many groups are in flight. The concurrent episode count times the per episode
  footprint is what must fit the cluster envelope. A queue is a smoothing
  mechanism, not the place to resolve oversubscription.

### Atomic group reservation is rejected

An all-or-nothing `acquire_many` would remove the deadlock by making the group
one waiter. It is rejected because it moves the failure rather than removing it.
A gang queue head-of-line blocks, and a large group can sit behind a stream of
small ones forever. The current admission avoids exactly this, and says so: an
oldest borrower that cannot fit does not block a younger one. Adding gang
semantics would need backfill and aging to be fair again, which is real
complexity added to the one component that is currently both correct and simple,
in service of a guarantee the algorithm never asked for.

### Per task snapshots remove the group from scheduling

Fork exists to pay a task's setup once per group instead of once per sample. But
the setup is a property of the task, not of the group, and an RL run visits the
same task in many steps. Capturing it once and reusing it is strictly better
than forking it once per group per step.

- Prepare the task environment once, snapshot it, and key the snapshot by the
  task and the setup inputs.
- Every member of every group, in every later step, is created from that
  snapshot. No parent exists at rollout time, no fork happens, and no group is a
  unit of anything.
- This is what the DeepSeek sandbox report calls turning an interactive session
  into a reusable environment, and it is what a snapshot backed template is on
  the provider backends.

It works at three levels of fidelity, and the level is what a caller declares.
A provider captures memory and processes. The internal Docker backend captures
the filesystem, which is enough for a setup step whose result is on disk. Cost
is snapshot storage per task plus an eviction policy, which the snapshot store
already needs for its local cache.

Fork remains the fallback for the first visit to a task and for a setup that
cannot be captured.

---

## Idle reclamation and pause

Phase 4 adds two things that both act on an idle sandbox. An inactivity reaper
destroys it, and an idle pause releases its compute and keeps its state. Both
need the same definition of idle, and that definition is not the obvious one.

### Idle is not the absence of a recent command

The retired worker stamps its last-used time when a command **returns**, not
when it starts. A command that runs for forty minutes leaves that stamp forty
minutes old for its whole duration, so a reaper that only reads the stamp sees a
busy sandbox as idle. Today this is safe only because the idle window is larger
than the command timeout, which is a margin and not a rule.

An idle pause turns that margin into a fault. Pausing a container that is
running a test suite stops the suite, and on the internal Docker backend the
processes do not come back, because the snapshot is filesystem only.

So idle is two conditions, not one.

- **No command in flight.** `DockerSession` already serializes commands behind
  an exec lock, so the lock is the in-flight signal. A pause or a reap must
  acquire it, and must not wait on it.
- **No activity for the idle window.** The window is measured from the last
  command boundary, either its start or its return, whichever is later.

A sandbox that fails the first condition is not idle, regardless of the second.
This applies to the reaper, to the idle pause, and to a preemption pause, since
all three suspend or destroy a sandbox that a command may be using.

### What each action is for

| Action | Trigger | Keeps state | Who calls it |
|---|---|---|---|
| Idle pause | Idle for the pause window | yes | The node agent, for density |
| Inactivity reap | Idle for the reap window, which is longer | no | The node reclaimer, for leaks |
| Absolute lifetime reap | Wall clock lifetime exceeded | no | The node reclaimer, as the backstop |
| Preemption pause | The trainer loses its GPU allocation | yes | The workflow, not a timer |

The windows are ordered, so a sandbox pauses before it is reaped, and the
absolute lifetime is the last line. An operator sets one intent, and the phases
derive the rest. See [Configuration ergonomics](#configuration-ergonomics).

---

## Image strategy

The internal Docker backend pulls a whole image on every node that needs it. A
provider backend loads layers on demand, or delivers blocks from a cache and from
peers. Several mechanisms cut the internal backend's cost, in order of how much
they buy.

### Image aware placement

- Each node agent indexes the images on its daemon and reports the index to the
  placement service. An index entry is keyed by the resolved digest, not the tag,
  so a mutable tag cannot claim a hit it does not have.
- A spec carries its image reference as a placement hint. The service prefers a
  node that already holds the digest, then the least loaded node.
- A cold node is a fallback, not a preference. When no node holds the digest, the
  service picks by load and the pull happens before the container starts.
- `prepare` stays what it is today, a warm that allocates no capacity. Placement
  decides where a sandbox runs, and `prepare` decides when the pull happens.

### Per run prefetch

- Before rollout starts, one planning step reads the task set and produces the
  working set of the run: `PrefetchPlan.from_specs`, deduplicated by image
  reference, with a template or snapshot source left out because its own backend
  materializes it.
- The working set is warmed by `SandboxManager.prefetch` on the nodes a task is
  likely to land on — `PlacementService.candidates`, best first, bounded to a
  configured number of nodes per image. Warming every node instead would pay a
  pull per node for images most nodes will never serve. The local Docker backend
  warms through the same pull path a create uses, so the deployment's own pull
  concurrency still bounds it.
- The task set rarely visits every image, so the working set is far smaller than
  the corpus. Warming the working set is what makes a cold cluster fast.
- `image/prefetch_coverage` reports the share of the working set that is now
  local, and `image/locality_hit_ratio` reports whether tasks are landing where
  their image already is.
- Prefetch is an optimization, not a correctness requirement. A task whose digest
  missed the prefetch still runs, it just pays the pull, so a reference that
  cannot be warmed is counted rather than raised.

### Layer sharing

- A Docker pull materializes a whole image, but a container reuses the lower
  layers of every other image that shares them, because overlay2 mounts shared
  lower directories. A task set built on a few base images pays for a base layer
  once per node.
- This is automatic and needs no design. It is why caching the base images
  matters more than the number of task images.

### Warm containers

- The internal Docker backend can keep a pool of prepared containers off the
  critical path. A pool entry is started on the keepalive command, with its image
  materialized and its setup done, so a claim adopts it instead of pulling and
  creating.
- A warm pool trades idle memory for a lower cold start. Because the backend
  already reaps idle sandboxes, a pool needs its own budget and TTL, or the
  reaper treats a pool entry as an abandoned sandbox.
- This mirrors the warm pool the external providers offer, so a recipe can
  express the same intent on either path.

### On demand layered loading

- The internal Docker backend cannot do this. A Docker pull materializes the
  whole image, and PSRL does not own a layer store.
- A backend that declares `IMAGE_ON_DEMAND` loads layers lazily. A backend that
  declares `IMAGE_BLOCK_DELIVERY` fetches blocks from a node cache and from
  peers instead of the origin. For either, the caller passes an image or template
  reference and PSRL does no pre pull at all.
- This is why the external backends are the scale path and the internal Docker
  backend is the test path. See {doc}`sandbox_external_backends`.
- The internal Docker backend gets no cache service in the baseline. A Docker
  daemon mirror redirects only Docker Hub, and a `distribution` proxy caches one
  upstream, so a cache that covers a real task corpus costs either rewritten
  image references or a proxy on every node. Placement and prefetch carry the
  baseline instead, and the upgrade is a node local peer proxy that intercepts
  the pull rather than renaming it. See {doc}`sandbox_snapshot_store`.
- DSec takes the shared filesystem plus EROFS route rather than a registry,
  precisely so it loads only the accessed image blocks. PSRL does not follow that
  in the internal backend, by design, because that backend is the test path. The
  comparison is in {doc}`sandbox_snapshot_store`.

---

## Observability

The single node stack already reports admission per training step.
`RayPPOTrainer._sandbox_capacity_metrics` reads each node coordinator's
`snapshot()` and emits `sandbox_capacity/<node>/<key>`, which the trainer logs
through the existing tracking logger at the training step. Every new plane
reports through that same hook, so sandbox metrics land next to reward and
timing on the same step axis and need no second pipeline.

### Principles

- **One collection point.** The trainer's per step metric hook is the only
  reader. A plane exposes a `snapshot()` and holds counters. It never logs.
- **Collection cannot stall training.** The read is bounded by a timeout and a
  failure returns nothing, which is what the existing hook does. A collection
  failure increments a counter instead of raising.
- **Cluster aggregates first, per node second.** A per node key set multiplies
  by the node count, and a large cluster turns a dashboard into noise. Every
  family reports the cluster aggregate and the worst node. The full per node
  breakdown is opt in.
- **Report latency as fixed quantiles, not raw samples.** Each plane keeps a
  bounded window and reports p50, p95, and max for the step. A step axis cannot
  carry a histogram, and unbounded samples would grow with the run.

### Metric families

Prefix `sandbox/`, so the group is one panel. Existing capacity keys keep their
prefix.

| Family | Keys | Answers |
|---|---|---|
| Admission | `admission/waiters`, `admission/wait_s_p50`, `admission/wait_s_p95`, `admission/timeouts`, `admission/borrow_bypasses`, `admission/oldest_lease_age_s` | Is the queue the bottleneck, and is a lease leaking |
| Utilization | `capacity/cpu_used_ratio`, `capacity/memory_used_ratio`, `capacity/gpu_used_ratio`, `capacity/nodes_drained` | Is the envelope the limit, or is placement leaving capacity idle |
| Placement | `placement/decision_s_p95`, `placement/rejections`, `placement/no_candidate`, `placement/reservations_open`, `placement/reservations_swept` | Is a reservation leaking, and is any node eligible |
| Image | `image/locality_hit_ratio`, `image/pull_s_p50`, `image/pull_s_p95`, `image/prefetch_coverage`, `image/warm_pool_hit_ratio` | Is the cache working, and is prefetch covering the working set |
| Provision | `provision/create_s_p50`, `provision/create_s_p95`, `provision/failures`, `provision/orphans_reclaimed` | Is cold start improving, and is cleanup keeping up |
| Exec | `exec/duration_s_p95`, `exec/timeouts`, `exec/silence_timeouts`, `exec/output_truncated_ratio` | Is the sandbox or the harness the slow half |
| Lifecycle outcome | `outcome/released`, `outcome/reaped_idle`, `outcome/reaped_lifetime`, `outcome/oom`, `outcome/reclaimed_orphan` | Why sandboxes ended, which is the first question in a post mortem |
| State | `state/pause_s_p95`, `state/resume_s_p95`, `state/checkpoint_s_p95`, `state/snapshot_bytes`, `state/restore_failures` | Is the state path affordable, and is cross node resume healthy |
| Snapshot store | `store/push_s_p95`, `store/pull_s_p95`, `store/push_failures`, `store/local_disk_used_ratio`, `store/gc_deleted` | Is the store keeping up, and is snapshot disk growing |
| Collection | `meta/collect_failures`, `meta/collect_s` | Is the metric path itself healthy |

The outcome family is a counter set, not a gauge, because the question is always
"how many ended this way this step". The rest are gauges or quantiles.

### Baseline and targets

Phase 0 records a baseline with `tests/sandbox/benchmark_docker_backend.py`, so
Phase 4 has something to compare against. Without it, "density and cold start
benchmarks" is not a verification, it is a measurement with no pass condition.

The baseline is create, exec, and release latency at p50 and p95, plus the
sandboxes a node holds at its envelope. Phase 4 targets are set from that
baseline, and the corresponding `sandbox/` keys are how a live run shows whether
the target holds.

### Alerting on what an operator acts on

Three signals mean an operator has to act, and the rest are diagnostic.

- `admission/oldest_lease_age_s` growing without bound. Nothing reclaims a
  lease by age, so a lease charged to a container nobody deletes only shows up
  here.
- `placement/reservations_open` growing while `provision/create_s_p95` is flat.
  That is a reservation leak, not load.
- `store/push_failures` above zero. A checkpoint that cannot publish makes a
  cross node resume impossible, and the run keeps going as if it could.

---

## Configuration ergonomics

Every mechanism in this plan has a knob, and a plan with fifty knobs is a plan
nobody can deploy. The rule is that an operator sets an intent, and the module
derives the numbers that follow from it.

- **Never ask for a number the operator cannot estimate.** A pause window, a
  reservation TTL, and a sweeper period are derived from the deadlines the
  operator already sets. A warm pool depth expressed in containers is not
  estimable, and one expressed as a fraction of the rollout batch is.
- **One intent, several derived values.** The idle windows are the clearest
  case. The operator sets the episode deadline, and the pause window, the reap
  window, and the absolute lifetime are derived from it in that order, so the
  ordering invariant cannot be misconfigured. An explicit override stays
  possible and is not the path a normal deployment takes.
- **Defaults must be safe, not fast.** Publishing a snapshot on demand, an
  empty warm pool, and no overcommit are the defaults, because each one is the
  behavior the module has today. A deployment opts into density, it does not
  opt out of a surprise.
- **A capability is declared, never tuned.** A recipe names the level of resume
  it needs. It never configures how a provider achieves it.
- **A key that only an author can set is a design smell.** If a value can only
  be chosen by reading the implementation, it belongs in the implementation.

The full per node metric breakdown, the overcommit ratio, and the warm pool
budget are the three knobs that stay explicit, because each trades a resource
an operator owns.

---

## Invariants are code

Several constraints in this plan are orderings between timeouts. The reservation
TTL sits between the provision deadline and the lease TTL. The capacity lease
TTL outlives the container TTL and the sweep period. The pause window is shorter
than the reap window, which is shorter than the absolute lifetime.

An ordering written only in prose is not enforced. A configuration that violates
it produces a slow leak or a container reclaimed while in use, and neither
failure points at the configuration. Every ordering in this plan is asserted
where the configuration is built, so a violation fails at startup rather than in
the middle of a run. The same applies to the ownership state machine, which
replaces flags precisely so an illegal transition is a raise and not a
possibility.

---

## Resumable rollout state

The DeepSeek sandbox report holds rollout state in an agent sandbox and a
scaffold neutral worker container, both outside the preemptible GPU pool. This
section maps that onto PSRL and states what a resumable agent loop needs.

Note that `gen/rollout_coordination/sync_and_migrate/` migrates in-flight
generation between rollout engines. It is not sandbox migration, and the two are
unrelated.

### The two holders in DSec

- **Agent sandbox.** Hosts the scaffold and its tools, and owns the task state.
- **Worker container.** Manages the sandbox and provides a scaffold neutral
  control layer for the rollout.

Together they are the single source of truth. A preempted trainer reconnects to
them instead of replaying a command log.

### The corresponding PSRL components

| DSec holder | PSRL component | What it holds |
|---|---|---|
| Agent sandbox | The sandbox session, leased by `HarnessAgentLoop` | Task workspace, container process state, and the mounted harness runtime tree |
| Scaffold and tools | The `Harness` and `HarnessRuntime` | The coding CLI running inside the sandbox |
| Worker container | `PSRL_AgentLoopWorker` plus the per-episode `AgentLoopBase` instance | Turn count, episode budget, compaction budget, retry attempt, terminate reason |
| Rollout state as a whole | The TITO session in the session router and SMG | The conversation prefix tree and the per-turn token records |
| Trainer side bookkeeping | `AgentLoopManager` | In-flight groups, refill, stall detection, retries |

Two gaps follow.

- In DSec the control layer runs outside the preemptible pool. In PSRL the agent
  loop worker is a Ray actor launched by the trainer, so a preemption takes the
  loop with it.
- In DSec the two holders are the source of truth. In PSRL the rollout state is
  spread across the sandbox, an in-memory loop instance, and the TITO session,
  and the loop instance is discarded at the end of each episode.

### What a resumable agent loop needs

1. **A sandbox that outlives the trainer.** The sandbox must pause instead of
   being reclaimed when the loop dies. Today the node collector reaps a sandbox
   whose owner heartbeat has aged out, so a preemption destroys it. This needs
   `HIBERNATE`, and for a resume on another node, `RESUME_ANYWHERE`.
2. **A control layer outside the preemptible pool.** The agent loop worker must
   be long lived, the way DSec's worker container is, or it must be cheap to
   rebuild and re-adopt an episode.
3. **A durable episode checkpoint.** The loop state, the turn count, the budgets,
   the retry attempt, and the session and sandbox identifiers must reach a store
   a new loop instance can read. The trajectory identifier that already flows
   through the pipeline is the stable key.
4. **TITO continuity.** The session router and SMG hold the conversation tree.
   Either they stay alive across the preemption, or a resumed loop rebuilds the
   session from the sandbox message history and the TITO checkpoints.
5. **Idempotent commit and reward.** A resumed episode must not double commit to
   the TransferQueue or double score. DSec used a command log where it could not
   preserve the sandbox. PSRL has none, so pause and resume is the primary
   mechanism and replay is the fallback.
6. **A harness that survives or restarts.** With `FREEZE` the harness process
   survives in place. With `HIBERNATE` a microVM snapshot carries it. On a plain
   Docker container the process is gone, so the harness must be restartable from
   the sandbox filesystem.

Items 1 and 6 are backend work and belong to Phases 4 and 6. Items 2 through 5
are loop and session work and are out of scope for the sandbox refactor, but the
backend contract is what makes them possible.

:::{seealso}
- {doc}`sandbox_execution_plan`: the ordered work breakdown that implements this plan
- {doc}`sandbox_architecture`: the current component map and the boundaries that are missing
- {doc}`sandbox_lifecycle`: invariants, failure semantics, and the verification map
- {doc}`sandbox_external_backends`: the capability and best-practice rationale
- {doc}`sandbox_backend_integration`: the shared capability model for step 6a
- {doc}`sandbox_agentenv`: the AgentEnv backend design for step 6b
- {doc}`sandbox_opensandbox`: the OpenSandbox backend design for step 6c
- {doc}`sandbox_snapshot_store`: the internal Docker backend's durable snapshot store
:::
