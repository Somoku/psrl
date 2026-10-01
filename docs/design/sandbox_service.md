# Sandbox Service

This page designs `psrl/sandbox` as a **standalone service** rather than a library
inside a Ray worker: one control plane for the fleet, one agent per node, a Python
SDK for callers, and the sandbox runtimes supplied by existing open-source projects
rather than reimplemented here.

It supersedes the deployment shape in {doc}`sandbox_refactor`, which assumed the
module lives in-process. The capability model stays where it is: this page adds no
feature table and cites {doc}`sandbox_backend_integration` for every capability
question (A4). {doc}`sandbox_external_backends` keeps the provider semantics, and
{doc}`sandbox_lifecycle` keeps the ownership invariants a node must preserve.

---

## Why a service

Three requirements drive the shape, and each one is a requirement the in-process
library cannot meet.

- **One deployment, many backends.** A user names a backend at init and the service
  provisions it. Today every backend is deployed by hand, per node.
- **Cross-backend resource management.** A quota and a placement decision must span
  backends. A library that lives in one worker sees only that worker's leases.
- **High request throughput from any caller.** A service is addressed over a socket,
  so a caller needs no Python import of PSRL and no Ray.

What does **not** change: PSRL remains a caller. The SDK is the only surface it uses.

---

## The two levels

The design follows the cluster/node split, because that split is forced by one fact:
a sandbox's state lives on the node that holds it, so only that node can admit
against its own pressure, and only a fleet-wide component can compare nodes.

```
CLUSTER-LEVEL   sandboxd-control
  Gateway       ingress, no per-sandbox state
  Quota         cross-backend admission ledger by resource class
  Router        spec -> backend
  Placement     filter, then rank; two-sided reservation
  Monitor       pulls one fleet view; Placement reads it
  Provisioner   backend deployment and preflight

NODE-LEVEL      sandboxd-node (one per node)
  Admission     final admission authority, against live local pressure
  Lifecycle     create, pause, resume, snapshot, release
  Reclaimer     idle pause, idle reap, lifetime, orphan
  LoadReporter  reports this node's view to Monitor

DATA PLANE      reused, never reimplemented
  Docker daemon, AgentEnv node runtime, Cubelet, OpenSandbox docker server
  execd / envd inside each sandbox
```

### The invariant that makes it affordable

**Command and file traffic never traverses the control plane.** An episode issues one
create and dozens of commands, so a control plane in the command path would add a hop
and a serialization per command for no decision. Every backend already runs an agent
inside the sandbox, and `create` therefore returns the resolved agent endpoint with the
handle. The SDK talks to that endpoint directly for the rest of the session.

This is not new: the OpenSandbox backend already resolves `exec_endpoint` once and then
speaks to the sandbox. The service generalizes it into a rule.

| Operation | Calls per episode | Path |
|---|---|---|
| `create` | 1 | SDK, control, node, backend |
| `exec`, file read and write | dozens | **SDK, sandbox agent** |
| `snapshot`, `pause`, `release` | 0 to 2 | SDK, control, node |

---

## Monitor and Placement

Monitor collects; Placement decides. Keeping them apart is what lets several Placement
replicas share one view, and what stops the view from being a side effect of whatever
heartbeat happened to arrive.

```go
type NodeView struct {
    NodeID        string
    Backend       string
    SeenAt        time.Time

    // What each class could be granted now. A request is compared against its own
    // class, never against the envelope remainder: a borrower may not take another
    // class's unmet guarantee, so the remainder overstates what a class can get.
    ClassHeadroom map[string]Headroom
    Envelope      Headroom

    // Measured, not inferred. Placement ranks on utilisation as a fraction, so two
    // nodes of different sizes order by load rather than by size.
    LiveSandboxes int
    CPUUsedPct    float64
    MemUsedPct    float64

    ImageDigests  []string
    GPUFree       int
    Labels        []string
    Draining      bool
}

type Monitor interface {
    Fleet(ctx context.Context) ([]NodeView, error)       // Placement pulls
    Report(ctx context.Context, view NodeView) error     // nodes push
}
```

Placement ranks **capability, then room, then balance, then image locality**. The order
is the design: a capability and room are requirements, locality is an optimisation.
Ranking locality first lets one per-task image draw a whole batch onto the node that
happens to hold it.

A Placement replica overlays its own recent decisions on the pulled view, because a
reservation is charged here before the node can report it. Without the overlay, a burst
of concurrent requests all read the same view and all choose the same node.

Neither component holds durable state. Monitor rebuilds by polling nodes; Placement
holds a cache and rebuilds from Monitor.

---

## Node-level authority

The node is the last word on admission, and it has information the cluster does not:
Placement compares a periodically reported headroom, while the node knows its pressure
now.

```go
func (a *Admission) Admit(ctx context.Context, spec Spec) (*Grant, error) {
    if !a.fitsClassGuarantee(spec) { return nil, ErrClassFull }
    if !a.underClassCeiling(spec)  { return nil, ErrClassCeiling }

    // Live pressure, which a reported headroom cannot express.
    if a.liveCPUPct() > a.cfg.LocalCPUCeiling { return nil, ErrNodePressure }
    if a.liveMemPct() > a.cfg.LocalMemCeiling { return nil, ErrNodePressure }

    gpus, err := a.takeGPUs(spec.GPUCount)   // indices, not a count
    if err != nil { return nil, err }
    return &Grant{LeaseID: newLeaseID(), GPUIndices: gpus}, nil
}
```

A node rejection is not a failure. Gateway treats it as a signal to ask Placement for
another node, which is what keeps a stale fleet view from overriding a local limit.

### Reclamation is node-level, and it is one loop

Reclamation is a property of the node, not of the caller: a caller that exits must not
leave its sandboxes until some other clock notices. One loop, one set of windows, four
reasons.

```go
func (r *Reclaimer) sweep(now time.Time) {
    for _, sb := range r.local.All() {
        switch {
        case sb.Busy():
            continue                                   // in flight, decide next pass
        case sb.Idle(now) > r.pauseWindow && !sb.Paused():
            r.pause(sb)                                // release compute, keep state
        case sb.Idle(now) > r.reapWindow:
            r.release(sb, ReapedIdle)
        case sb.Age(now) > r.lifetime:
            r.release(sb, ReapedLifetime)              // backstop for a busy sandbox
        case sb.OwnerGone(now):
            r.release(sb, ReclaimedOrphan)
        }
    }
}
```

Idle is two conditions, not one. A command must not be in flight, and no command
boundary may be more recent than the window: a backend that stamps activity when a
command *returns* leaves the stamp stale for the whole of a long command, so a sweep
reading only the stamp pauses a running test suite.

The lifetime is the backstop behind the windows. A sandbox that keeps running commands
is never idle and still has to end, or one stuck episode holds a node slot for the rest
of the run.

---

## Timing: three knobs

An operator can estimate how long an episode takes. Everything else follows from it, so
everything else is derived and asserted rather than configured.

```yaml
timing:
  episode_deadline_s: 1800   # required: one episode, including grading
  node_ttl_s: 120            # optional: silence before a node is drained
  rpc_timeout_s: 60          # optional: coordination-call deadline
```

| Derived | From | Why that ratio |
|---|---|---|
| `pause_window` | `episode_deadline` x 2 | two episodes of silence is idle, not slow |
| `reap_window` | `pause_window` x 3 | paused before destroyed, always |
| `lifetime` | `reap_window` x 4 | backstop, never the first thing to fire |
| `acquire_timeout` | `episode_deadline` | a queue wait longer than an episode is a fault |
| `load_report_interval` | `node_ttl` / 4 | several missed reports fit inside one TTL |
| `monitor_pull_interval` | `node_ttl` / 4 | pull no faster than nodes report |
| `capacity_lease_ttl` | `pause_window` | capacity outlives a pause |
| `owner_heartbeat` | `capacity_lease_ttl` / 3 | one lost round trip keeps the lease |
| `reservation_ttl` | `node_ttl` / 2 | a reservation expires before its node drains |
| `reservation_renew` | `reservation_ttl` / 3 | same reason |
| `sweep_interval` | shortest TTL / 4 | a sweep enforces a TTL, so it must be faster |
| `lifecycle_lease`, `lifecycle_gc` | `capacity_lease_ttl` / 2, then / 2 | crash recovery completes inside the lease |

One `Validate()` asserts every ordering, including the cross-level ones that no
assertion covers today.

```go
func (t TimingContract) Validate() error {
    must(t.PauseWindow < t.ReapWindow,          "a sandbox is paused before it is destroyed")
    must(t.ReapWindow < t.Lifetime,             "the backstop fires last")
    must(t.CapacityLease >= t.PauseWindow,      "capacity is not reclaimed under a pause")
    must(t.LoadReportInterval*3 < t.NodeTTL,    "a node is drained on silence, not on one miss")
    must(t.ReservationTTL < t.NodeTTL,          "no reservation outlives its node record")
    must(t.SweepInterval < t.ShortestTTL(),     "a sweep is faster than the TTL it enforces")
    must(t.LifecycleLease+t.LifecycleGC < t.CapacityLease,
         "a dead node's containers do not outlive the reservation protecting the node")
    return nil
}
```

An override is explicit and still validated, so a configuration that inverts an
ordering fails at startup rather than leaking slowly:

```yaml
timing:
  episode_deadline_s: 1800
  overrides:
    reap_window_s: 60        # still runs Validate()
```

---

## The Python SDK

The SDK is the whole public surface. Its discipline is what keeps the service from
growing a second control plane: **the SDK serializes, remembers an endpoint, and
forwards a deadline. It makes no decisions.** No caching of placement, no retry policy
of its own, no lifecycle state machine. Anything that decides lives in the service.

### Client and session

```python
class SandboxClient:
    """Connection to one sandboxd-control.

    A Unix socket when the control plane is co-located, a TCP address otherwise.
    """
    def __init__(self, endpoint: str = "unix:///run/sandboxd.sock") -> None: ...

    async def create(self, spec: SandboxSpec) -> Sandbox: ...
    async def create_group(self, specs: Sequence[SandboxSpec]) -> list[Sandbox]: ...
    async def restore(self, snapshot: SnapshotRef, spec: SandboxSpec | None = None) -> Sandbox: ...
    async def delete_snapshot(self, snapshot: SnapshotRef) -> None: ...

    async def fleet(self) -> FleetReport: ...      # what Monitor sees, for metrics
    async def quota(self) -> QuotaReport: ...      # per-class ledger, for metrics
    async def close(self) -> None: ...


class Sandbox:
    """One sandbox. Carries the agent endpoint resolved at create time."""

    ref: SandboxRef
    capabilities: SandboxCapabilities      # what this backend actually granted
    node_id: str
    backend: str

    # Data plane: straight to the sandbox agent, never through the control plane.
    async def exec(self, command: str, *, cwd: str | None = None,
                   env: Mapping[str, str] | None = None,
                   timeout_s: float | None = None,
                   silence_timeout_s: float | None = None) -> ExecResult: ...
    async def read_bytes(self, path: str) -> bytes: ...
    async def write_bytes(self, path: str, data: bytes) -> None: ...

    # Control plane.
    async def status(self) -> SandboxStatus: ...
    async def diagnostics(self) -> SandboxDiagnostics: ...
    async def snapshot(self, kind: SnapshotKind) -> SnapshotRef: ...
    async def pause(self, mode: PauseMode) -> None: ...
    async def resume(self) -> None: ...
    async def release(self) -> None: ...

    async def __aenter__(self) -> Sandbox: ...
    async def __aexit__(self, *exc) -> None: ...   # releases
```

`exec` keeps `silence_timeout_s` beside `timeout_s` because a command that prints
nothing for a long stretch is stuck rather than slow, and the two need different
answers.

### What a caller passes

`SandboxSpec` is the portable request. It already exists in `psrl/sandbox/core.py` and
crosses the wire unchanged, with two additions.

```python
@dataclass(frozen=True)
class SandboxSpec:
    source: SandboxSource                      # image or template reference
    resources: ResourceSpec                    # cpu_count, memory_mb, disk_mb, gpu_count
    resource_class: str = "default"            # which quota share pays for it
    workflow_id: str | None = None             # one sandbox phase per workflow
    idempotency_key: str | None = None         # a retried create adopts, never duplicates

    env: Mapping[str, str] = ...
    metadata: Mapping[str, str] = ...
    workdir: str | None = None
    exec_mode: ExecMode | None = None          # persistent shell, or one process per command

    required_features: frozenset[SandboxFeature] = frozenset()
    required_resume_level: ResumeLevel | None = None
    backend: str | None = None                 # a pin; None selects by capability

    egress: EgressPolicy | None = None
    credentials: tuple[CredentialRef, ...] = ()
    credential_bindings: tuple[CredentialBinding, ...] = ()
    mounts: tuple[MountSpec, ...] = ()
    volumes: tuple[VolumeSpec, ...] = ()
    state_policy: SandboxStatePolicy = ...

    # New: tuning that only one backend understands, namespaced by backend name.
    # A key naming a backend this deployment does not run is an error, not a no-op,
    # so a typo is visible instead of silently dropped.
    backend_options: Mapping[str, Mapping[str, Any]] = ...
```

Two rules keep `backend_options` from eroding the portable contract.

- **It never participates in routing.** Only the portable fields decide which backend
  serves a spec, so two specs differing in `backend_options` are still comparable.
- **It is outside the spec's identity.** Otherwise one environment tuned two ways would
  read as two different idempotent requests.

### How a backend gets chosen

A caller states requirements; the service refuses rather than degrades. That is the
whole selection contract, and it is why the capability model exists.

```python
# Capability-based: any backend that can resume a live process elsewhere.
spec = SandboxSpec(
    source=SandboxSource.template("swe-task"),
    resources=ResourceSpec(cpu_count=4, memory_mb=8192),
    required_features=frozenset({SandboxFeature.RESUME_ANYWHERE}),
    required_resume_level=ResumeLevel.FULL_STATE,
    resource_class="rollout",
)

# Pinned, for an ablation that must hold the backend fixed.
spec = replace(spec, backend="agentenv")
```

A spec requiring `FULL_STATE` is rejected on a filesystem-resume backend at admission,
which is the point: a conformance run must not read a workspace restore as proof that a
harness survived a resume. The feature names and which backend declares each one are in
{doc}`sandbox_backend_integration`.

### Errors the SDK raises

Failures are typed by **attribution**, because the caller's response differs and because
an infrastructure fault must never be recorded as a model or harness failure.

| Error | Meaning | Caller's response |
|---|---|---|
| `SandboxCapacityTimeout` | never admitted; no sandbox existed | capacity planning, retry later |
| `SandboxCapabilityError` | no backend satisfies the spec | fix the spec or the deployment |
| `SandboxSessionLostError` | the sandbox stopped unasked | replace the rollout, do not blame the harness |
| `SandboxOomError` | the kernel killed it for memory | resource fault, raise the request |
| `SandboxCommandTimeout` | a command exceeded its deadline; `sandbox_preserved` says whether the sandbox survived | continue with a fresh shell, or replace |
| `SandboxSetupError` | the task's preparation failed | task fault, nothing was captured |
| `SandboxTransportError` | the control channel failed | says nothing about the workload |

---

## How PSRL adapts

PSRL stops building a `SandboxManager` and holds a `SandboxClient` instead. The trainer
configures the service; the worker uses it.

### Configuration

The deployment names its backends once. `mode` selects which scheduler runs: `psrl`
uses the service's own Placement and Admission and does not start the backend's control
plane; `provider` defers to the backend's own scheduler and leaves the service doing
cross-backend quota only.

```yaml
# psrl/trainer/config/psrl/deployment.yaml
sandbox_service:
  endpoint: unix:///run/sandboxd.sock

  timing:
    episode_deadline_s: 1800

  quota:                       # cross-backend shares, by resource class
    classes:
      rollout:  {guaranteed_share: 0.70}
      grader:   {guaranteed_share: 0.20}
      prepare:  {guaranteed_share: 0.10}

  backends:
    - type: docker
      mode: psrl
      nodes: [192.168.1.21, 192.168.1.22]
      capacity: {memory_mb: 480000, cpu_cores: 90, gpu_count: 0, utilization: 0.5}

    - type: agentenv
      mode: psrl               # direct to each node runtime; no gateway, no scheduler
      nodes: [192.168.1.31, 192.168.1.32]

    - type: cubesandbox
      mode: provider           # its filter/score framework decides the node
      endpoint: 192.168.1.40:8089
```

A `resource_class` in a spec selects a share. The shares are a priority and a ceiling,
not a partition: a class with no queued request reserves nothing, so an idle grader share
is borrowable by rollout and is reclaimed as soon as a grader queues.

### Worker side

```python
# psrl/workers/agent_loop/worker.py
self.sandbox = SandboxClient(config.sandbox_service.endpoint)

# psrl/workers/agent_loop/context.py
@dataclass(frozen=True)
class AgentLoopContext:
    ...
    sandbox: SandboxClient          # replaces sandbox_manager: SandboxManager
```

An agent loop reads the same way it does now, with the lease replaced by the context
manager:

```python
async with await self.sandbox.create(task.sandbox_spec) as sandbox:
    harness = create_harness(self.harness_config, sandbox)
    result = await sandbox.exec(command, timeout_s=budget)
    snapshot = await sandbox.snapshot(SnapshotKind.FILESYSTEM)
```

`harness/base.py` and the graders already depend only on `exec`, `read_bytes`, and
`write_bytes`, so they take `Sandbox` in place of `SandboxSession` with no other change.

### Trainer side

The trainer stops building a Ray actor topology for sandboxes. It reads the service's
reports for the per-step metric hook, which keeps every sandbox metric on the training
step axis:

```python
fleet = await self.sandbox.fleet()      # Monitor's view: nodes, headroom, locality
quota = await self.sandbox.quota()      # per-class grants, queue depth, waits
```

The planes report and never log a metric, and the trainer is the only reader.

---

## Languages

| Component | Language | Why |
|---|---|---|
| `sandboxd-control`, `sandboxd-node` | Go | a standalone service needs one static binary, predictable GC with many resident records, real parallelism, and compile-time contracts across contributors |
| Backend adapters | Go, in-process | a sidecar would add a hop to every create |
| `pysandbox` SDK | Python | the caller's language, and PSRL's |
| Sandbox runtimes, in-sandbox agents | reused | the node-local performance work is already done, in Rust, by the backends |

The node-local mechanisms that make density possible, on-demand image loading, page-cache
sharing, free-page reporting, CPU QoS, are backend territory. The service selects a
backend that has them and never reimplements one.

---

## Phases

Each phase is independently useful, so none of them is a bet on the next.

| Phase | Work | Exit criterion |
|---|---|---|
| 0 | On the current Python: collapse timing to three knobs, move reclamation to node level, trim the feature set, split `manager.py` by responsibility, freeze the `.proto` contract | the optimisations land and the contract is generated, not hand-written twice |
| **Gate** | **Run a real RL workload against the "commands direct, control plane by RPC" shape** | measured episode latency and throughput are acceptable; otherwise the Go rewrite does not start |
| 1 | Go `sandboxd-control` and `sandboxd-node`, Docker adapter, `pysandbox` | one node, one backend, end to end through the SDK |
| 2 | AgentEnv adapter in `psrl` mode; **delete the Python control plane** | two backends; no Python control-plane code remains |
| 3 | Provisioner and preflight | a named backend deploys with no per-node manual step |
| 4 | CubeSandbox and OpenSandbox adapters | a mixed-backend run, and an ablation that holds everything but the backend fixed |

Phase 0 pays for itself whether or not phase 1 happens. The Python implementation is a
second implementation of one contract, not a compatibility layer, and phase 2 deletes it
rather than leaving a switch.

---

## Decisions

| Id | Decision | Consequence |
|---|---|---|
| F1 | The module becomes a service with a Python SDK, and PSRL is one caller | PSRL holds a `SandboxClient`, not a `SandboxManager`; no Ray actor topology for sandboxes |
| F2 | Command and file traffic go straight to the sandbox agent | `create` returns the resolved endpoint; a control plane in the command path is a defect, not a tuning choice |
| F3 | Monitor collects one fleet view and Placement pulls it | several Placement replicas share a view; each overlays its own in-flight decisions |
| F4 | The node is the final admission authority | a rejection asks Placement for another node, so a stale fleet view cannot override a local limit |
| F5 | Reclamation is node-level, one loop, four reasons | a caller that exits leaves nothing behind; idle pause and orphan reclaim stop being two mechanisms on two clocks |
| F6 | Three timing knobs; everything else derived and asserted in one place | a configuration that inverts an ordering fails at startup instead of leaking |
| F7 | `mode` is a per-backend deployment property, `psrl` or `provider` | the same backend can run both ways in one fleet, which is what makes a scheduler ablation possible |
| F8 | `backend_options` is namespaced, validated, outside routing and outside spec identity | a provider's own tuning is reachable without making the portable spec unportable |
| F9 | The SDK serializes, remembers an endpoint, and forwards a deadline | no second control plane, so there is no state machine to drift |
| F10 | The control plane is Go; the data plane is reused | the deployment boundary decides the language, not the algorithm |
