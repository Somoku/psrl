# Sandbox lifecycle and resource accounting

The sandbox boundary must preserve trajectory isolation and keep physical
resource ownership aligned with admission accounting. Runtime failures must not
silently become successful cleanup or model-quality observations.

## Component boundaries

| Component | Responsibility |
|---|---|
| `core.py` | Portable specs, results, capabilities, and provisioning failure ownership |
| `manager.py` | Admission, workflow reservations, lease delivery, rollback, reclamation, shutdown |
| `capacity.py` | Node CPU and memory accounting, class FIFO queues, admission deadlines |
| `backends/docker/backend.py` | Docker policy assembly, image preparation, named provisioning, backend ownership |
| `backends/docker/session.py` | Command lifecycle, stop detection, exit classification, session destruction |
| `backends/docker/engine.py` | HTTP pools, Docker protocol, incremental output decoding, file archives |
| `backends/docker/exec.py` | The one-shot and persistent-shell command strategies |
| `backends/docker/events.py` | One shared container event stream and its reconnect gap recovery |
| `backends/docker/lifecycle.py` | Worker heartbeat and detached collector lifecycle |
| `backends/docker/policy.py` | Typed security, workload policy, and disk admission configuration |
| `backends/docker/cli.py` | Docker CLI housekeeping: force removal, label sweeps, image pruning |
| `backends/e2b.py` | The E2B, AgentEnv, and CubeSandbox backends over the provider SDK |
| `backends/opensandbox.py` | The OpenSandbox backend over its lifecycle, execd, and egress planes |
| `async_utils.py` | Repeated-cancellation protection for ownership transitions |
| `placement.py`, `node_agent.py`, `remote.py` | The cross-node reservation and remote-session protocol |
| `reclaimer.py` | Node-level orphan reclamation by heartbeat age |
| `snapshot_store.py`, `task_snapshot.py` | The durable snapshot store and per-task capture reuse |

The synchronous facade submits work to the owning worker event loop. It does
not instantiate another backend, connection pool, or capacity manager.

## Ownership state machine

```text
prepare -> queued -> provisioning -> leased -> reclaiming -> released
                cancellation |          failure |
                             +-----> reclaiming <-+
```

Preparation owns only reusable artifacts. Queued admission owns a request ID.
Provisioning owns a capacity reservation and responsibility for any runtime
created by the backend. A lease takes ownership immediately after creation,
before capability validation or restored-state sanitation can fail.

Cancellation before admission withdraws the request by ID, including an RPC
whose successful grant might have raced the cancellation. Cancellation after
provisioning begins waits for the backend result and reclaims it. Shared
idempotent creation counts interested waiters, so one cancellation cannot
abandon another caller's resource.

The last waiter retains the idempotency entry through rollback. This prevents
a new retry from adopting a container that the abandoned request is deleting.
Session destruction and capacity return are individually idempotent. Their
completion and the manager's removal of the lease form one cancellation-protected
transition.

A failed destruction stays in the manager's reclamation set with its capacity
charged. A single worker reaper retries pending leases with bounded backoff.
A failed capacity return retries only accounting after destruction succeeds.
An episode can finish while its cleanup remains owned by the manager.

Backend `create` must either return a session, fail without a runtime allocation,
or raise `SandboxProvisionError` with the session requiring cleanup. Docker uses
a unique container name to retain cleanup responsibility after a lost create
response. Failed starts preserve the actual container ID for reclamation.

## Admission and fairness

Every request supplies a complete CPU and memory vector. A grant subtracts both
atomically. Release adds the recorded vector exactly once. Class ceilings are
hard caps, and individually impossible requests fail before queueing.

Within each resource class, requests are FIFO. A request inside its class
guarantee is admitted whenever the envelope has room, independent of what any
other class is waiting for: the guarantee is a contract, not a preference. Slack
left over by guarantees below their share is reserved for a guaranteed request
that does not yet fit, and otherwise lent to borrowers in arrival order. Borrowing
is best-effort FIFO, so an older borrower that cannot fit does not block a younger
one that can. Blocking it would idle the node for the length of the longest
running episode, and the admission deadline already reports genuine starvation as
a capacity fault. Skipped borrowers are counted, never given a reservation.

`workflow_id` is a reservation across queued, provisioning, leased, and
reclaiming states. The manager rejects concurrent phase acquisition for the
same workflow before admission. This prevents a rollout from holding the
resources its grader needs. Independent workflows remain concurrent.

An allocation's age is diagnostic information, not evidence that its runtime
has stopped. Explicit cleanup owns normal reclamation. Owner TTL provides
crash recovery under the node collector's operational assumptions.

## Docker execution and failure semantics

Docker control requests and command streams use independent bounded connection
pools. Command saturation therefore does not occupy the lifecycle pool. The
exec deadline spans serialization, setup, output draining, and final status.
Commands in one session are serialized, which `SandboxSession.exec` states as a
contract so no caller depends on the opposite.

The stream decoder retains at most the configured payload budget, plus a partial
eight-byte frame header and the current transport chunk. It validates frame
boundaries even after exhausting the output budget. A cut frame is an error.
An intact but oversized stream returns bounded output and a truncation flag.
A successful command result also requires a final Engine exit status. File reads
stream their archive through a spooled temporary file, so peak memory is one copy
of the file rather than three.

Cancellation cancels and joins the command and watcher tasks before destruction.
A container stop is a runtime failure rather than caller cancellation, and it is
observed on one shared event stream rather than per-session polling, so the stop
that leaves an exec stream open forever is detected as it happens. A reconnect
replays from the last event seen and re-inspects every container still being
waited on, because a restarted daemon has no history to replay. A daemon that
refuses the event stream degrades to polling. The backend retains exit state
until diagnosis and explicit removal. A proven `OOMKilled` state produces
`SandboxOomError`, while a normal nonzero command exit remains a task result.

Container OOM priority is left at the Docker default. `OomScoreAdj` is inherited
by every process in the container, so it cannot change the ranking the cgroup
killer uses inside one memory limit. What it does change is the host ranking,
where a negative value makes the kernel prefer the trainer, the rollout engine,
and the parameter server over a disposable sandbox. Bounding a runaway command is
the memory limit's job, which is also why a limited sandbox gets no extra swap.
Deprioritizing a sandbox against its own node is opt-in through
`DockerSecurityConfig.oom_score_adj`.

Each new Docker session has a fresh writable layer, an explicit keepalive
entrypoint, init-based process reaping, CPU and memory limits, and no additional
swap allowance when memory is limited. Snapshot restores retain the requested
resource, environment, mount, and policy configuration.

## Crash boundary

The detached collector reaps expired Docker owners independently of the worker
event loop. Heartbeat namespace and Docker endpoint must agree across workers.
A live owner's stopped container is retained for one grace period so the owning
session can classify its exit, then reaped, because a stopped container can never
serve another command and nothing else would reclaim its writable layer.

Reclamation is driven by heartbeat age, not owner identity, so a restarted run
reclaims an abandoned run's containers even though every owner id is unique. Each
worker also sweeps synchronously before its first sandbox, so admission never runs
against an envelope that omits memory an earlier run still holds.

Capacity owner expiry is time based. A Docker outage can delay physical deletion
past that expiry, so the two TTLs do not provide a distributed fencing guarantee.
An unhealthy node must stop accepting sandbox work until its daemon and collector
recover. Explicit cleanup failure never authorizes early capacity release by a
live manager. Provider-managed backends retain their provider's crash semantics.

## Verification map

| Invariant | Regression coverage |
|---|---|
| Repeated cancellation completes destruction and accounting | `tests/sandbox/test_manager.py` |
| Shared create survives one waiter leaving | `tests/sandbox/test_manager.py` |
| A removal error for an absent container returns its capacity | `tests/sandbox/test_docker_reliability.py` |
| An undeletable container keeps its reservation and escalates | `tests/sandbox/test_docker_reliability.py`, `test_manager.py` |
| Deferred cleanup does not block the next workflow phase | `tests/sandbox/test_manager.py` |
| Lost create response retains a named cleanup target | `tests/sandbox/test_docker_reliability.py` |
| A guaranteed request outranks a waiting borrower | `tests/sandbox/test_sandbox_capacity.py` |
| Borrowers progress under competing classes | `tests/sandbox/test_sandbox_capacity.py` |
| Incomplete frames fail and oversized frames stay bounded | `tests/sandbox/test_docker_engine.py` |
| A file read costs one copy of the file | `tests/sandbox/test_docker_engine.py` |
| Control and stream pools have separate connectors | `tests/sandbox/test_docker_reliability.py` |
| Event stops, gap recovery, and the polling fallback | `tests/sandbox/test_docker_reliability.py` |
| OOM diagnosis precedes container deletion | `tests/sandbox/test_docker_oom.py`, `test_docker_cli.py` |
| A stopped container outlives diagnosis, then is reaped | `tests/sandbox/test_docker_cli.py` |
| A restarted run reclaims an earlier run's containers | `tests/sandbox/test_docker_lifecycle.py` |

Real Docker conformance and performance commands are in the
[sandbox README](../../psrl/sandbox/README.md#verify). Remote RL acceptance
must also exercise sustained rollout and grading concurrency, worker termination,
daemon interruption, and full resource recovery after the run.
