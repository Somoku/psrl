# PSRL sandboxes

`SandboxManager` owns execution environments for one worker. Build a
`SandboxSpec`, acquire a lease, and use its session for commands, binary file
I/O, resource statistics, and capability-gated state operations. Release the
lease when the trajectory or grading phase ends.

Each trajectory gets a fresh writable filesystem. Cached image layers are
shared, but mutable containers are never pooled between unrelated trajectories.
For lifecycle invariants and implementation boundaries, see
[Sandbox lifecycle](../../docs/design/sandbox_lifecycle.md).

## Configuration

Configure Hydra targets under
`gen_actor_rollout_ref.rollout.agent.sandbox.backends`. For example:

```yaml
sandbox:
  default_backend: docker
  capacity:
    memory_mb: null
    cpu_cores: null
    utilization: 0.5
    classes:
      rollout:
        guaranteed_share: 0.60
      grader:
        guaranteed_share: 0.25
    acquire_timeout_s: 1800
  backends:
    docker:
      _target_: psrl.sandbox.backends.DockerBackend
      image_pull_concurrency: 2
      connection_limit: 128
      max_exec_output_bytes: 16777216
      security:
        require_rootless: false
        pids_limit: 4096
        cap_drop: [ALL]
        no_new_privileges: true
```

Declare a sandbox's `resource_class` at the workload call site. The node
coordinator admits its complete CPU and memory request atomically. Unset node
limits use cgroup or machine detection, with `utilization` applied to both.
Provider-managed microVMs do not consume the worker node's Docker envelope.

Class guarantees set admission priority, and optional `max_share` sets a hard
ceiling. A request inside its class guarantee is admitted as soon as the envelope
has room, whatever the other classes are waiting for. When a guaranteed request
does not fit, the remaining slack is held for it instead of being lent out. Other
requests borrow the slack in arrival order, and an older borrower that does not
fit does not block a younger one that does. A request larger than its class
ceiling or the node envelope fails immediately. Watch
`max_borrow_bypasses`: a borrower that keeps being skipped means the class shares
no longer match the workload, and `acquire_timeout_s` will report it as a
capacity fault rather than hanging.

Use the same `workflow_id` for sequential rollout and grading phases. Release
one phase before acquiring the next. The manager rejects a second reservation
while the workflow still owns a sandbox or has provisioning in progress.
`acquire_timeout_s` bounds admission independently of the episode's work budget.

Docker enforces CPU and memory limits and disables swap when memory is limited.
It uses an init process, drops capabilities, and retains the daemon's seccomp
profile. Rootless mode and a read-only root filesystem require compatible
images and are opt-in. Kernel OOM priorities are left alone, because a negative
`OomScoreAdj` is inherited by the whole container and only shifts the *host*
ranking, which would make the kernel kill the trainer before a disposable
sandbox. Set `security.oom_score_adj` when you want that trade.
Image entrypoints are cleared so the configured keepalive command owns startup.

Use typed `policy_profiles` for workload overrides. MiniSWE's configured bridge
profile maps `host.docker.internal` to the Docker host and rewrites loopback
proxy URLs to that alias. Remote proxy URLs retain their original destination.

## Ownership and cleanup

A lease is released only after confirmed sandbox destruction and capacity
return. Repeated cancellation waits for the complete ownership transition.
If deletion or capacity return fails, the manager retains the lease and retries
with exponential backoff up to 30 seconds. Completed episodes can return while
cleanup is deferred, but the reservation remains charged.

A partial creation failure can carry a session in `SandboxProvisionError`.
Backend authors must use it when a runtime object might still exist. Ordinary
creation failures must leave no allocated runtime behind. `terminate()` must
raise when destruction cannot be confirmed.

Identical idempotent requests share one provisioning task and lease. Cancelling
one waiter leaves other waiters intact. When every waiter leaves, queued
admission is withdrawn and any in-flight creation is settled and reclaimed.
An idempotency key cannot be reused while its previous session is being reclaimed.

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

Timeout, cancellation, and transport failure destroy the disposable session:
closing an exec connection does not establish that its process stopped.
Container stops come from one shared Docker event stream, so an OOM kill
interrupts its hung command within a beat instead of at the next poll. A daemon
that will not serve `/events` falls back to inspect polling on
`container_watch_interval_s`. Explicitly retained exit state distinguishes proven
OOM failures through `SandboxOomError`. The container is removed after
diagnostics. Exit code 137 alone is not proof of a container OOM.

`idle_timeout_s` is an absolute Docker lifetime limit, not an inactivity timer.
`request_timeout_s` bounds control-plane requests. Long command streams use the
command deadline, so silent commands do not inherit a shorter HTTP total timeout.
Binary file reads stream the archive through a spooled temporary file, so peak
memory is one copy of the file rather than three.

## Image preparation and locality

`await manager.prepare(spec)` checks the daemon cache and pulls missing images
when `auto_pull` is enabled. Acquisition also prepares the image before reserving
CPU or memory. Concurrent requests share each image download, and
`image_pull_concurrency` bounds distinct downloads per backend instance.
Cancelling one waiter does not cancel a shared pull. Shutdown joins remaining
preparation tasks.

Use immutable image digests for reproducibility. Preparation does not refresh an
already cached mutable tag. Private registries use the backend's `registry_auth`
mapping with secret-backed Hydra interpolation.

Configure `disk_admission.path` to the daemon's data filesystem as visible to the
worker, together with `min_free_mb` and `wait_timeout_s`, to reject creation when
space remains low. This is a host headroom check, not a container disk quota.
Portable `ResourceSpec.disk_mb` is unsupported by Docker.

Keep a local Docker daemon on each Ray worker node. A remote daemon's resources
are not represented by the worker's node-capacity accounting. Prefetch the exact
image subset for a run and place trajectories on nodes with warm images when
image loading dominates setup time.

## Crash recovery

`DockerLifecycle` maintains one heartbeat thread per worker and one detached
collector per heartbeat directory. Workers using one daemon must share the same
absolute directory path and policy. Use a separate directory for each daemon,
and mount the directory from the daemon host in containerized deployments.
The collector filters by the directory's `psrl.lease_store` namespace.

The default Docker owner TTL is 120 seconds with a 30-second sweep interval.
Capacity owner TTL is 180 seconds. Keep the capacity TTL above the Docker TTL
plus a sweep interval. Unreadable heartbeat files do not prove owner death.
A live owner's stopped container is reaped once it has stayed stopped for
`stopped_grace_s` (300 seconds), which leaves the owning session time to read
`OOMKilled` off it first.

Each worker sweeps once at startup, before it creates its first sandbox, so a
restarted run reclaims the previous run's containers instead of being admitted
against an envelope that does not know their memory is still spoken for.
Reclamation is driven by heartbeat age, not by owner identity, so the new run's
different owner id does not matter.

**Owner expiry assumes the node collector and daemon are healthy.** TTL ordering
is not a physical fence during daemon failure or a prolonged cleanup backlog.
Stop admission on an unhealthy node before resuming workloads. Shutdown errors
and deferred-cleanup warnings must be monitored rather than treated as successful
resource reclamation.

## State operations

Snapshot, restore, and branch require backend capabilities and an enabled
`SandboxStatePolicy`. Docker snapshots include only the writable filesystem.
They do not include process memory, bind-mounted data, or remote side effects.
Snapshot tags are unique, and temporary branch snapshots are deleted after use.

Secret-bearing environment variables and post-command capture are rejected by
default. Restored microVMs refresh transports and mix host entropy into the guest.
A filesystem branch reuses the source spec, with a fresh idempotency key.
A workflow restricted to one active sandbox cannot branch while retaining its
parent. Use separate workflow identities for intentionally concurrent branches.

| Backend | Source | Resource mapping | Authentication |
|---|---|---|---|
| Docker | OCI image | CPU and memory | Docker socket or endpoint permissions |
| E2B | Template | Fixed by template | E2B API key |
| AgentEnv | Template or OCI image | Cold images accept CPU, memory, disk | Provider API key |
| CubeSandbox | Template or snapshot | CPU, memory, disk | Provider API key |

For template-backed E2B or AgentEnv runs, set MiniSWE memory overrides to null.
When selecting a microVM, set Docker-only `policy_profile` to null and provide
the backend's template or image settings.

## Verification

Run unit and contract tests without a Docker daemon:

```bash
python -m pytest tests/sandbox
ruff check psrl/sandbox tests/sandbox
python scripts/audit_prose_style.py psrl/sandbox tests/sandbox
git diff --check
```

Run real Docker conformance and collect reproducible latency samples:

```bash
PSRL_RUN_DOCKER_INTEGRATION=1 PSRL_DOCKER_TEST_IMAGE=python:3.11-slim \
  python -m pytest tests/sandbox/test_docker_live.py
python tests/sandbox/benchmark_docker_backend.py \
  --image python:3.11-slim --iterations 100 --concurrency 16
```

Compare runs on the same idle node with pre-pulled images. Report create and
exec p50, p95, maximum latency, and peak session count. Unit tests establish
failure semantics, not production throughput or GPU training convergence.
