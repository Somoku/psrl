# PSRL sandbox abstraction

`SandboxManager` is worker-scoped and `SandboxLease` is trajectory-scoped.
Agent loops and environments treat the runtime as a black box: they build a
`SandboxSpec`, then use `exec`, binary file I/O, status, resource stats and
lifecycle methods without importing Docker, E2B, AgentEnv or CubeSandbox code.

The required data plane is deliberately small. State operations are optional
semantic capabilities; a filesystem snapshot is never treated as a full-state
snapshot. Docker supports filesystem snapshot/restore through image commits but
does not advertise process-memory snapshots or native fork.

## Runtime boundary

- `SandboxManager`: backend registry, idempotent create, lease ownership,
  capability/state-policy checks, node-capacity RPC and shutdown.
- `SandboxCapacityCoordinator`: one Ray actor per Docker node; owns only the
  shared CPU/memory envelope and weighted admission queue.
- `SyncSandboxManager`: thread-safe synchronous facade over the worker event
  loop; it does not create a second backend client or event loop.
- `DockerBackend`: persistent asynchronous Docker Engine API pool. The Docker
  CLI remains only in the crash-recovery path, which must survive interpreter
  failure.
- `DockerLifecycle`: one worker ownership lease and graceful cleanup. A single
  detached collector per lease directory reclaims containers after the owner
  lease expires.
- `E2BBackend`: E2B SDK data plane and generic E2B native state driver.
- `AgentEnvBackend` / `CubeSandboxBackend`: provider-specific control-plane
  factories and state drivers with an E2B-compatible command/file data plane.

MiniSWE keeps two separate adapters. `MiniSWEEnvironment` only translates
dataset rows to observations. `MiniSWEAgentAdapter` implements the synchronous
third-party harness protocol and delegates all runtime work to
`SyncSandboxSession`.

## Configuration

Backends are Hydra targets under
`gen_actor_rollout_ref.rollout.agent.sandbox.backends`:

```yaml
sandbox:
  default_backend: docker
  capacity:
    memory_mb: null    # detect the node/cgroup limit
    cpu_cores: null    # detect the node/cgroup limit
    utilization: 0.5   # the only node-envelope safety margin
  backends:
    docker:
      _target_: psrl.sandbox.backends.DockerBackend
      image_pull_concurrency: 2  # per backend instance, not per node
      max_exec_output_bytes: 16777216  # truncated with a marker, never fatal
      lifecycle:
        # Must be shared by all workers that use the same Docker daemon.
        heartbeat_dir: /tmp/psrl-sandbox-heartbeats
      disk_admission:
        # Optional and disabled by default. The path must be the daemon data
        # filesystem as mounted on the worker host.
        path: /dockerdata
        min_free_mb: 51200
        wait_timeout_s: 300
      security:
        require_rootless: false  # turn on after every node uses rootless dockerd
        pids_limit: 4096
        cap_drop: [ALL]
        cap_add: []
        no_new_privileges: true
        read_only_rootfs: false
      policy_profiles:
        # Docker's bridge network remains isolated from host loopback services.
        mini_swe: {}
    agentenv:
      _target_: psrl.sandbox.backends.AgentEnvBackend
      api_url: ${oc.env:AGENTENV_E2B_API_URL}
      api_key: ${oc.env:AGENTENV_API_KEY,null}
    cubesandbox:
      _target_: psrl.sandbox.backends.CubeSandboxBackend
      api_url: ${oc.env:CUBE_E2B_API_URL}
      api_key: ${oc.env:CUBE_API_KEY,null}
```

Docker uses the daemon's default seccomp profile unless `seccomp_profile` is
set. `seccomp=unconfined` is rejected. Read-only rootfs and rootless mode are
opt-in because they require compatible images/daemons; capability drop,
no-new-privileges, PID limiting and init/reaping are enabled by default.
MiniSWE uses Docker's bridge network. Its default policy maps
`host.docker.internal` to the daemon host and rewrites loopback HTTP(S)/ALL
proxy URLs to that alias, so node-local proxies remain reachable without host
networking. Remote proxy URLs are unchanged.

Crash recovery uses one heartbeat thread per worker rather than one asyncio
task per container, so long-running commands and a busy event loop do not make
healthy leases appear stale. The heartbeat directory is the coordination
boundary: workers connected to one daemon must see the same directory. In a
containerized deployment, mount it from the daemon host. The collector uses an
advisory file lock, which the kernel releases automatically if the collector
dies; a backend restarts a collector that previously exited after an idle
period.

Collectors filter containers by `psrl.lease_store`, derived from the absolute
heartbeat directory. Use the same absolute mount path and lease policy for all
workers in one namespace, and a separate directory for each Docker endpoint.
Deploy this change with old collectors stopped: older collectors do not honor
the namespace label. Containers created before this label was added need their
original owner cleanup or an explicit administrative cleanup. Unreadable lease
files are treated as an infrastructure error, not proof that their owners died.

Docker image preparation is asynchronous and does not reserve container CPU or
memory. `await manager.prepare(spec, backend=...)` checks the daemon image cache
and, when `auto_pull` is enabled, pulls missing images. Concurrent requests for
one image share a download; distinct downloads are bounded by
`image_pull_concurrency`. Cancellation of one waiter does not cancel another
task's shared download. Backend shutdown cancels and joins remaining downloads.
Completed requests are not cached in Python, so external image deletion is
observed by the next preparation request. Use immutable image digests when
reproducibility matters; preparation does not refresh an already cached tag.

The generic harness loop overlaps rollout image preparation with TITO session
creation and grader image preparation with rollout. MiniSWE's threaded agent
loop also starts grader image preparation before dispatching its runner.
Container setup and harness preparation remain ordered within a trajectory:
the first command requires a ready container. These operations can overlap
other trajectories, but this is not a ready-container pool or a dataset-wide
prefetch scheduler.

Grading always uses a separate container. Docker shares the cached immutable
image layers while allocating a fresh writable layer; the rollout lease is
released before the grader requests capacity. The generic loop no longer
commits an unprepared Docker container merely to recreate the same baseline.
Explicit filesystem snapshots remain supported and have unique tags. Full-state
snapshots on capable microVM backends retain their existing workflow.

Exec output is bounded by `max_exec_output_bytes`, including Docker frame
headers. A command that exceeds it is drained and its output is truncated with a
marker, so the command finishes, the container stays healthy, and the caller gets a
bounded answer instead of losing its work. That budget is a diagnostic guard, not a
limit on results: data that must survive intact is written to a file and read back
with `read_bytes`, which has no such budget. Command timeout, cancellation, and
transport failure still destroy the disposable container, because the container can
no longer be trusted to have finished the command. The command deadline includes exec
creation, stream attachment, and exit-status inspection. The underlying per-request
transport timeout still applies.

`cgroup_parent`, when configured on `DockerBackend`, is passed through as an
opaque Docker setting so both cgroupfs paths and systemd slice names work. PSRL
does not create cgroups or write version-specific control files. Each container
receives Docker CPU and memory limits from `ResourceSpec`; the same request is
charged atomically against the node envelope before creation.

## Node resource planning

For node-local Docker, the trainer pins one `SandboxCapacityCoordinator` to each
node and gives every worker on that node the same actor handle. The coordinator
owns one CPU-plus-memory envelope. `SandboxManager.acquire()` submits the actual
container request, waits until both dimensions fit, and returns capacity only
after the sandbox is terminated. Rollouts and graders therefore share spare
capacity naturally; there are no fixed pools or per-worker estimates.

Every request names the resource class of the sandbox it will create, such as
`rollout` or `grader`. Each class owns a FIFO queue and a guaranteed share of the
envelope, so the order classes arrive in cannot starve one of them: a request
inside its own guarantee is admitted as soon as the envelope has room, whatever
the other classes are doing. Guarantees are floors rather than partitions, and
they are meant to sum to less than one:

- the remainder is one elastic pool, borrowed by a class only while no other
  class has a request waiting;
- summing to less than one is what keeps every guarantee satisfiable at the same
  time without preempting a running sandbox.

Size the shares from measurement. A class needs enough of the envelope to cover
its phase's footprint times the fraction of an episode spent in that phase, both
of which are recorded in `reward_info.timing` and, with
`sandbox_config.collect_resource_metrics`, in `sandbox_peak_memory_mib`. A class
that is never declared has no guarantee and can only use the elastic pool, which
the coordinator warns about once.

A request that stays queued for `acquire_timeout_s` fails with a capacity fault
instead of waiting forever, because a sandbox that was never admitted says
nothing about the model or the harness and must not be reported as one. It must
leave room for the episode inside the rollout's derived child deadline, which the
agent loop validates at startup. Admission reports per-class guarantees, usage,
queue depth, and wait time in `snapshot()`, which the trainer logs once per node at
startup.

A multi-phase job must hold one sandbox at a time: release the rollout sandbox
before requesting the grader. Holding one while waiting for another cannot be
fixed by any admission order, because each phase occupies capacity the other
needs. `SandboxManager` reports and counts such a job when the specs share a
`workflow_id`, so the supported ordering is a checked contract rather than a
convention.

The coordinator actor's Ray `max_concurrency` is derived from the workers
assigned to that node and their real agent-loop concurrency. It is an internal
scheduler bound, not a resource-budget knob and therefore is not user
configurable. The user-facing envelope is therefore optional CPU, optional
memory, one utilization margin, the `classes` guarantees, and the
`acquire_timeout_s` deadline. Lease TTL and heartbeat interval retain typed
defaults and are advanced failure-recovery settings.

The defaults reclaim orphan Docker containers after 120 seconds with a
30-second sweep interval, before the capacity owner can expire after 180
seconds. If these advanced values are overridden, keep the capacity TTL larger
than the Docker lease TTL plus one GC interval.

`memory_mb` and `cpu_cores` may be explicit or detected from the coordinator's
cgroup/node. `utilization` is the single safety margin for co-located services.
Requests larger than the envelope fail immediately. Worker heartbeats renew all
of their allocations, while lease expiry recovers capacity after a killed Ray
actor, and reports the wait as a capacity fault rather than as a cancellation.
Remote/provider-managed backends do not consume this node envelope.

A Docker exec timeout destroys that disposable session. Once an Engine exec
start request times out, this client cannot safely kill only that exec process;
retaining the container could leave a runaway process racing later commands.

When MiniSWE selects a microVM backend, set its Docker-only policy to null and
provide a template (Cube) or template/image (AgentEnv):

```yaml
sandbox_config:
  backend: cubesandbox
  policy_profile: null
  snapshot_verifier: true
  environment:
    template: my-swe-template
```

AgentEnv/E2B templates have fixed resources, so set `memory: null` in both
MiniSWE `rollout_environment` and `grader_environment` for template-backed
runs. AgentEnv cold images and Cube templates accept explicit resource
requests.

## Source, resource and auth mapping

| Backend | Source | CPU / memory / disk | Timeout | Auth |
|---|---|---|---|---|
| Docker | OCI image | CPU and memory; portable disk limit rejected | local lifetime task | Docker socket permissions |
| E2B | template | fixed by template; overrides rejected | SDK `timeout` | E2B API key |
| AgentEnv | template or cold OCI image | cold image maps all three; template is fixed | `timeout` | `X-API-Key` plus SDK auth |
| CubeSandbox | template/snapshot | maps `cpuCount`, `memoryMB`, `diskSizeMB` | `timeout` | `X-API-Key` plus SDK auth |

Private Docker pulls can pass Engine `X-Registry-Auth` fields through the
backend's `registry_auth` mapping. Prefer secret-backed Hydra environment
interpolation and never place registry passwords directly in checked-in YAML.

Provider requests carry `psrl.idempotency_key` in metadata. Docker additionally
uses a deterministic container name and a canonical spec hash, so a retry
reuses only an exactly matching running container. Concurrent calls inside one
worker share the same create task and lease.

## State operations in RL

Snapshot/restore/branch require both backend capabilities and an explicitly
enabled `SandboxStatePolicy`. PSRL takes these precautions:

- secret-looking environment variables and credential-bearing proxy URLs are
  rejected by default;
- capturing after a user command is rejected unless the caller explicitly
  accepts non-rollbackable external side effects;
- the live parent reconnects after snapshot because provider command streams
  can be invalidated;
- restored/forked sessions use fresh client connections and mix 64 bytes of
  host entropy into `/dev/urandom` before use;
- temporary snapshots used to emulate branch are deleted automatically;
- MiniSWE uses a clean verifier snapshot only when source and CPU/memory/disk
  exactly match the grader spec, otherwise it creates a fresh verifier.

Snapshotting cannot roll back remote APIs, queues, databases or already-sent
TCP traffic. The default pre-command checkpoint rule is the enforceable safety
boundary; `allow_external_side_effects=true` is an explicit research-mode
escape hatch, not an exactly-once guarantee.

## Metrics

The node coordinator snapshot reports total and available CPU/memory,
allocations, waiters, grants, releases, expirations and mean wait time. Backends
aggregate count, failures, total/mean/max latency for create, exec,
file I/O, stats, pause/resume, snapshot/restore/fork and terminate. They also
track active/peak sessions and sampled current/peak memory. Metrics do constant
work per operation and do not sample the command hot path automatically.

For MiniSWE, set `sandbox_config.collect_resource_metrics=true` to add one
cgroup stats sample per trajectory (`sandbox_memory_mib`,
`sandbox_peak_memory_mib`, `sandbox_cpu_total_s`) and exact
`sandbox_create_s` to trajectory timing. The worker logs final aggregate
lifecycle metrics on shutdown.

## Local and remote verification

Fast checks, with no daemon/provider required:

```bash
python -c "import psrl.sandbox; from psrl.sandbox.backends import DockerBackend, AgentEnvBackend, CubeSandboxBackend"
ruff check .
pytest -q tests/sandbox
git diff --check
```

Real Docker conformance and repeatable performance samples:

```bash
PSRL_RUN_DOCKER_INTEGRATION=1 \
PSRL_DOCKER_TEST_IMAGE=python:3.11-slim \
pytest -q -s tests/sandbox/test_docker_live.py

python tests/sandbox/benchmark_docker_backend.py \
  --image python:3.11-slim --iterations 100 --concurrency 16 \
  | tee docker-sandbox-benchmark.json
```

Set `PSRL_REQUIRE_ROOTLESS=1` in the conformance command to make rootless mode
an assertion rather than a recommendation. Compare branches on identical idle
nodes, with images pre-pulled, at least three runs, and report p50/p95/max create
and exec latency plus peak container and dockerd RSS.

Live microVM snapshot/restore conformance:

```bash
PSRL_LIVE_MICROVM_BACKEND=agentenv \
PSRL_LIVE_MICROVM_API_URL=http://agentenv-api:8080 \
PSRL_LIVE_MICROVM_API_KEY=... \
PSRL_LIVE_MICROVM_SOURCE_KIND=image \
PSRL_LIVE_MICROVM_SOURCE=registry/swe:tag \
pytest -q -s tests/sandbox/test_microvm_live.py

PSRL_LIVE_MICROVM_BACKEND=cubesandbox \
PSRL_LIVE_MICROVM_API_URL=http://cube-api:8080 \
PSRL_LIVE_MICROVM_API_KEY=... \
PSRL_LIVE_MICROVM_SOURCE_KIND=template \
PSRL_LIVE_MICROVM_SOURCE=my-template \
pytest -q -s tests/sandbox/test_microvm_live.py
```

## Multi-node Docker policy

Keep one node-local Docker daemon per Ray worker node. A centralized remote
daemon adds a network hop to every exec/file operation, creates a shared failure
and scheduling bottleneck, and loses data locality. Ray remains the distributed
scheduler; the sandbox manager remains worker-local.

For large image sets, loading every image on every node is simple but expensive.
Prefer, in order: precompute the run's exact image subset, prefetch by digest,
use a registry mirror/cache close to the cluster, and schedule tasks only onto
nodes where the image is warm. A P2P image distributor or containerd lazy-pull
snapshotter becomes worthwhile only when measured image warm-up dominates
rollout time. Replacing Docker with Kubernetes solely for per-episode lifecycle
is not recommended: control-plane latency and object overhead are larger than
node-local Engine API calls.

## Remaining work

1. Export metrics to the repository's production metrics backend and add
   per-node daemon RSS/disk/image-cache gauges.
2. Add streaming file transfers and optional command-output sinks. Docker exec
   output is bounded, but `ExecResult` still materializes the retained output.
3. Add image-locality scheduling and digest manifests for very large SWE image
   corpora.
4. Evaluate warm pools only with a backend-specific, proven clean reset; never
   lease mutable Docker state across unrelated trajectories by default.
