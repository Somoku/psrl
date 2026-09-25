# Sandbox operations

Ownership, execution, image, recovery, and state semantics for the sandbox
module. Read [README.md](README.md) first for install, configuration, and
deployment. Invariants are in
[sandbox_lifecycle](../../docs/design/sandbox_lifecycle.md).

## Where things live

`psrl/sandbox/` holds the control plane. A backend is one module or one package under
`backends/`, and a backend's own modules stay inside that package.

| Path | Owns |
|---|---|
| `core.py` | The portable contract: specs, features, sessions, errors, and the helpers a backend needs |
| `manager.py` | Admission, leases, groups, idle policy, per-task snapshots, shutdown |
| `capacity.py` | The node envelope, class shares, and reservations |
| `ownership.py` | The lease state machine |
| `config.py` | Configuration into a manager, with every timeout ordering asserted |
| `metrics.py` | Latency, memory, and quantile observations |
| `snapshot_store.py`, `task_snapshot.py` | The durable store, and per-task capture reuse |
| `placement.py`, `node_agent.py`, `remote.py` | The cross-node reservation and remote-session protocol |
| `reclaimer.py` | Node-level orphan reclamation, with its own entry point |
| `backends/docker/` | The Docker backend, one module per concern |
| `backends/e2b.py` | E2B, AgentEnv, and CubeSandbox |
| `backends/opensandbox.py` | OpenSandbox |

`backends/docker/` splits by concern rather than by layer, and its package docstring
carries the module map, so a change lands in the module that owns the concern.

## Backends and what they accept

| Backend | Source | Resource request | Authentication |
|---|---|---|---|
| Docker | OCI image | CPU, memory, and GPU accounted on the node envelope | Docker socket or endpoint permissions |
| E2B | Template | Fixed by the template | E2B API key |
| AgentEnv | Template, or an image that resolves to one | Fixed by the template, so a request is refused | Provider API key |
| CubeSandbox | Template or snapshot | CPU, memory, and disk | Provider API key |
| OpenSandbox | Image, snapshot, or template | Kubernetes-style quantities under `resourceLimits` | Provider API key |

For E2B or AgentEnv runs, leave the per-sandbox resource request unset, because
the template owns the shape. When selecting a microVM, set the Docker-only
`policy_profile` to null and name the backend's template or image instead.

Portable `ResourceSpec.disk_mb` reaches Docker as a host headroom check rather than a
per-container quota, because the Engine API has no disk limit. OpenSandbox documents no
disk quantity at all, so it refuses a spec that names one. Only the providers that
expose a disk dimension with a contract accept it.

## Ownership and cleanup

A lease is released only after confirmed sandbox destruction and capacity return.
Repeated cancellation waits for the complete ownership transition. If deletion or
capacity return fails, the manager retains the lease and retries with exponential
backoff up to 30 seconds. Completed episodes can return while cleanup is deferred,
but the reservation remains charged.

A partial creation failure can carry a session in `SandboxProvisionError`.
Backend authors must use it when a runtime object might still exist. Ordinary
creation failures must leave no allocated runtime behind. `terminate()` must
raise when destruction cannot be confirmed.

Identical idempotent requests share one provisioning task and lease. Cancelling
one waiter leaves other waiters intact. When every waiter leaves, queued
admission is withdrawn and any in-flight creation is settled and reclaimed.
An idempotency key cannot be reused while its previous session is being reclaimed.

A group is a completion unit rather than a scheduling unit. A native fork asks the
provider for the whole group in one call, and a partial result destroys every child
that started before it raises, so a short group is never returned and nothing is
left running. A backend without a native fork falls back to one create per member.

Shutdown rejects new work, cancels queued admission, settles provisioning,
reclaims leases, and closes backend transports. Unconfirmed cleanup is reported
as a shutdown error. Allocation age alone never releases capacity.

## Docker execution

Command streams and lifecycle operations use separate persistent HTTP connection
pools, each bounded by `connection_limit`. A full set of long-running commands
therefore cannot consume the connections required for inspection and deletion.
Commands in one session are serialized. The command deadline covers waiting for
that session, exec creation, streaming, and final exit-status inspection.

`max_exec_output_bytes` bounds the combined stdout and stderr payload. Docker
frame headers do not consume this budget. Excess output is drained and reported
with `ExecResult.truncated` plus a text marker. Incomplete or malformed framing
is a transport failure. Store complete results in files and read them with
`read_bytes` when truncation would lose task data.

Timeout, cancellation, and transport failure destroy the disposable session, because
closing an exec connection does not establish that its process stopped. Container
stops come from one shared Docker event stream, so an OOM kill interrupts its hung
command within a beat instead of at the next poll. A daemon that will not serve
`/events` falls back to inspect polling on `container_watch_interval_s`.
Explicitly retained exit state distinguishes a proven OOM failure through
`SandboxOomError`, and the container is removed after diagnostics. Exit code 137
alone is not proof of a container OOM.

`idle_timeout_s` is an absolute Docker lifetime limit rather than an inactivity
timer. `request_timeout_s` bounds control-plane requests. Long command streams use
the command deadline, so a silent command does not inherit a shorter HTTP total
timeout. Binary file reads stream the archive through a spooled temporary file, so
peak memory is one copy of the file rather than three.

## Image preparation, locality, and caching

`await manager.prepare(spec)` checks the daemon cache and pulls missing images when
`auto_pull` is enabled. Acquisition also prepares the image before reserving CPU or
memory. Concurrent requests share each image download, and `image_pull_concurrency`
bounds distinct downloads per backend instance. Cancelling one waiter does not
cancel a shared pull. Shutdown joins remaining preparation tasks.

Use immutable image digests for reproducibility, because preparation does not
refresh an already cached mutable tag. Private registries use the backend's
`registry_auth` mapping with secret-backed Hydra interpolation.

Configure `disk_admission.path` to the daemon's data filesystem as the worker sees
it, together with `min_free_mb` and `wait_timeout_s`, to reject creation when space
runs low. This is a host headroom check rather than a container disk quota.

Three opt-in keys bound work the daemon would otherwise repeat:

| Key | What it does |
|---|---|
| `warm_pool` | Keeps `depth_fraction x batch_size` prepared containers, capped by `max_entries`, each with its own `ttl_s`. A create adopts one instead of pulling and starting |
| `snapshot_store` | Publishes a committed filesystem snapshot to an OCI registry so another node can restore it, and is what makes the backend declare `RESUME_ANYWHERE`. `retention` is an intent such as `one_run` or `one_day`, not a TTL |
| `snapshot_local_cache_fraction` | The share of the node's sandbox disk the local snapshot cache may hold, evicted least-recently-used first |

`manager.prefetch(plan)` warms a run's working set ahead of rollout. The plan is
`PrefetchPlan.from_specs(specs)`, deduplicated by reference. `image/prefetch_coverage`
reports the share that became local, and `image/locality_hit_ratio` reports whether
tasks are landing where their image already is. A task that missed the prefetch still
runs.

**No caller invokes it yet.** The mechanism and its metrics are implemented and tested,
but nothing in a trainer or worker calls `manager.prefetch`, so coverage stays at zero
in a real run. Only the remote path narrows the nodes a task is likely to land on, and
it narrows them for a prefetch nobody triggers. The step is tracked in the execution
plan.

Keep a local Docker daemon on each Ray worker node, because a daemon reached over
TCP has resources that are not in the caller's node-capacity accounting.

## Placing sandboxes on other nodes

Turning on `psrl.deployment.sandbox_placement.enabled` moves sandboxes off the agent
loop worker's own node. The trainer creates one placement service for the job and one
node agent per node in `gen_actor_rollout_ref.rollout.agent.node_ips`, and each worker
builds one remote backend over those handles. A worker then holds **no local backend
and no local capacity accounting**, because a sandbox it places consumes another node's
envelope, and a leftover local backend would let a task fall back to an unaccounted
daemon.

| Setting | Meaning |
|---|---|
| `enabled` | Off means every worker keeps running its own sandboxes |
| `backend_name` | The backend the fleet runs, which is also the placement filter and the name a task asks for |
| `node_ttl_s` | A node that has missed this long is drained rather than trusted |
| `reservation_ttl_s` | How long a reservation survives without renewal. The caller renews at a third of it |
| `sweep_interval_s` | The sweep enforces the reservation TTL, so it has to be the shorter of the two |
| `heartbeat_interval_s` | How often a node reports it is still there. Keep it well inside the node TTL |
| `rpc_timeout_s` | Deadline for one call to a node agent or the placement service |
| `required_label` | A node label a request pins itself to, or null to accept any node |
| `callback_target` | A `host:port` a node forwards to so a sandbox reaches this worker's session server, or null to leave callback URLs alone |

The orderings between the TTLs are asserted when the placement service is built, so an
inverted configuration fails at startup rather than by draining the fleet later.

What the plane requires of a deployment:

- **`node_ips` must be set.** It is the sandbox fleet, and defaulting to every alive
  node would put sandboxes on the nodes the trainer is training on.
- **A cross-node resume needs a shared snapshot store**, because the internal backend
  publishes a snapshot to a registry and re-creates from the digest reference. A node
  that cannot reach the store cannot restore a snapshot taken elsewhere. Without one
  the backend does not declare `RESUME_ANYWHERE`, so such a spec is refused at
  admission rather than failing when it checkpoints.
- **Each node agent derives its envelope from `capacity.*`.** The node builds it from
  the same sandbox config a worker reads, and an agent whose backend consumes the node
  refuses to start without one, because an unbounded node admits every request against
  a daemon nobody is accounting for and advertises no devices.
- **A sandbox that calls back into the worker needs `callback_target`.** Without it the
  node does not open a forwarder, so a loopback URL inside the sandbox reaches the
  sandbox rather than the worker that owns the trajectory.

Two failure modes are worth knowing, because both are silent otherwise:

- **Liveness is per node, not per worker.** A node reports on its own cadence from its
  own agent, so one worker dying does not drain a node other workers are using. A node
  whose report reaches a placement that has forgotten it registers again, which is what
  makes a placement restart survivable: heartbeating an unknown node is ignored.
- **A node keeps a partially provisioned sandbox.** When a create fails after the node
  has already started one, the node's own manager owns the cleanup, and the caller is
  told what happened rather than handed a session it cannot reach.

Verify a plane on a real cluster, which is the only check that covers serialization and
node affinity:

```bash
python -m tests.sandbox.smoke_ray_plane
```

It creates real actors and drives registration, placement, the reservation protocol,
liveness, re-registration, and a failing create. It does not create a container, because
that needs a Docker daemon: `tests/sandbox/test_docker_live.py` covers that behind
`PSRL_RUN_DOCKER_INTEGRATION`.

## Crash recovery

`DockerLifecycle` maintains one heartbeat thread per worker and one detached
collector per heartbeat directory. Workers using one daemon must share the same
absolute directory path and policy. Use a separate directory for each daemon, and
mount the directory from the daemon host in containerized deployments. The
collector filters by the directory's `psrl.lease_store` namespace.

The default Docker owner TTL is 120 seconds with a 30-second sweep interval.
Capacity owner TTL is 180 seconds. Keep the capacity TTL above the Docker TTL plus
a sweep interval. Unreadable heartbeat files do not prove owner death. A live
owner's stopped container is reaped once it has stayed stopped for `stopped_grace_s`
(300 seconds), which leaves the owning session time to read `OOMKilled` off it
first.

Each worker sweeps once at startup, before it creates its first sandbox, so a
restarted run reclaims the previous run's containers instead of being admitted
against an envelope that does not know their memory is still spoken for.
Reclamation is driven by heartbeat age rather than by owner identity, so the new
run's different owner id does not matter.

**Owner expiry assumes the node collector and daemon are healthy.** TTL ordering
is not a physical fence during daemon failure or a prolonged cleanup backlog. Stop
admission on an unhealthy node before resuming workloads. Shutdown errors and
deferred-cleanup warnings must be monitored rather than treated as successful
resource reclamation.

## State operations

Snapshot, restore, and branch require backend capabilities and an enabled
`SandboxStatePolicy`. Docker snapshots include only the writable filesystem. They
do not include process memory, bind-mounted data, or remote side effects. Snapshot
tags are unique, and temporary branch snapshots are deleted after use.

Secret-bearing environment variables and post-command capture are rejected by
default. Restored microVMs refresh transports and mix host entropy into the guest.
A filesystem branch reuses the source spec with a fresh idempotency key. A workflow
restricted to one active sandbox cannot branch while retaining its parent, so use
separate workflow identities for intentionally concurrent branches.

A provider restore is a new sandbox with a new id, and the provider rejects
overrides that the snapshot already fixes. Environment variables, volumes, and
mounts are carried by the snapshot, so a restore clears them rather than sending
them for the provider to refuse. A capture therefore carries the environment it was
taken with.
