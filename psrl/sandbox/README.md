# PSRL sandboxes

`SandboxManager` owns execution environments for one agent loop worker. Build a
`SandboxSpec`, acquire a lease, and use its session for commands, binary file
I/O, resource statistics, and capability-gated state operations. Release the
lease when the trajectory or grading phase ends.

Each trajectory gets a fresh writable filesystem. Cached image layers are
shared, and mutable containers are never pooled between unrelated trajectories.

- Lifecycle invariants and internal boundaries:
  [sandbox_lifecycle](../../docs/design/sandbox_lifecycle.md)
- Ownership, execution, and recovery behavior: [OPERATIONS.md](OPERATIONS.md)
- Standing up a provider: [DEPLOY_AGENTENV.md](DEPLOY_AGENTENV.md) and
  [DEPLOY_OPENSANDBOX.md](DEPLOY_OPENSANDBOX.md)
- Why the module is shaped this way:
  [sandbox_refactor](../../docs/design/sandbox_refactor.md)

## Pick a backend

The three backends differ in who owns the machine that runs the sandbox, which
decides where capacity is accounted and how much you deploy.

| Backend | Runs on | Admission | Needs |
|---|---|---|---|
| `docker` | The worker's own node, one local daemon | PSRL, inside that node's envelope | A daemon per node, and a host firewall for egress |
| `agentenv` | A provider, through its E2B-compatible SDK | The provider | The `sandbox-e2b` extra, and a template per task image |
| `opensandbox` | A provider, through its lifecycle and sidecar HTTP APIs | The provider | Nothing extra |

Choose `docker` to keep one deployment and run close to the trainer. Choose a
provider when the task image is heavy, the workload wants a microVM boundary, or
a sandbox must not consume trainer CPU and memory.

Placeholders used below:

| Placeholder | Meaning |
|---|---|
| `${SANDBOX_IMAGE}` | A task image reference, ideally pinned by digest |
| `${PROVIDER_API_URL}` | A provider API base URL |
| `${PROVIDER_API_KEY}` | A provider API key, read from the environment |
| `${HEARTBEAT_DIR}` | A directory every worker sharing one daemon can see |
| `${ENV_NODE_IP}` | The node that should host sandboxes |
| `${TASK_ID}` | One AIRS-Bench task id |
| `${AIRS_REPO}` | A clone of the AIRS-Bench repository |

## Install

```bash
python -m pip install -e .
```

Only AgentEnv adds a dependency, because it drives the provider's own SDK:

```bash
python -m pip install -e ".[sandbox-e2b]"
```

OpenSandbox needs no package. Its lifecycle, `execd`, and egress planes are
reached through the provider's endpoint resolution, so PSRL is not coupled to an
SDK version.

A provider backend also needs the provider itself, and each one has hard
prerequisites that PSRL cannot work around. Read the deployment page before
configuring one:

| Backend | Read |
|---|---|
| `agentenv` | [DEPLOY_AGENTENV.md](DEPLOY_AGENTENV.md), which starts with the kernel and `/dev/kvm` the runtime needs |
| `opensandbox` | [DEPLOY_OPENSANDBOX.md](DEPLOY_OPENSANDBOX.md), which covers the server, its runtime choice, and the two keys the egress features need |

## Configure

Declare backends under `gen_actor_rollout_ref.rollout.agent.sandbox.backends`.
The manager instantiates each one with Hydra, so a block carries a `_target_` and
stays declarative until a worker builds its manager. The shipped defaults are in
`psrl/trainer/config/rollout/psrl_rollout.yaml`.

### Docker

The shipped block in `psrl_rollout.yaml` is the starting point. Five things in it
are worth deciding rather than inheriting:

```yaml
sandbox:
  default_backend: docker
  capacity:
    utilization: 0.5
    classes:
      rollout: {guaranteed_share: 0.60}
      grader: {guaranteed_share: 0.25}
  timing:
    episode_deadline_s: 1800
  backends:
    docker:
      _target_: psrl.sandbox.backends.DockerBackend
      default_exec_mode: persistent
      lifecycle:
        heartbeat_dir: ${HEARTBEAT_DIR}
      security:
        cap_drop: [ALL]
        no_new_privileges: true
```

`capacity.utilization` is the only safety margin on the node envelope, and a null
limit uses the node's own cgroup or machine limit. `timing.episode_deadline_s`
derives the idle pause window, the reap window, and the absolute lifetime in that
order, so an inverted ordering cannot be configured. `heartbeat_dir` is what makes
crash recovery work, and it has to be visible to every worker on the node.

Set `security.require_rootless: true` only after the daemon is rootless, and add
`egress_firewall_command` where a host firewall can program an `egress` allowlist.
The backend declares `EGRESS_POLICY` only where that command works, so a deployment
without it refuses a policy it cannot enforce rather than starting a sandbox open.

### AgentEnv

```yaml
sandbox:
  default_backend: agentenv
  backends:
    agentenv:
      _target_: psrl.sandbox.backends.AgentEnvBackend
      api_url: ${oc.env:AGENTENV_API_URL}
      api_key: ${oc.env:AGENTENV_API_KEY}
      template_build_timeout_s: 1800
      snapshot_request_timeout_s: 300
```

AgentEnv creates a sandbox from a **template**, and an image becomes a template
rather than a create-time source. So a rollout prepares one template per task
image before it acquires anything, and a spec whose image was never prepared
fails with the provider's missing-template error.

The backend refuses a resource override, a host bind mount, and a provider volume,
because the template's build fixes the first and a provider has no host for the
second. Keep `lifetime_timeout_s` inside the provider's 24-hour cap on a sandbox's
life. `snapshot_request_timeout_s` covers a snapshot, which the provider can take
minutes over. [DEPLOY_AGENTENV.md](DEPLOY_AGENTENV.md) gives the reason for each
refusal and the host prerequisites the runtime needs.

### OpenSandbox

```yaml
sandbox:
  default_backend: opensandbox
  backends:
    opensandbox:
      _target_: psrl.sandbox.backends.OpenSandboxBackend
      config:
        api_url: ${oc.env:OPENSANDBOX_API_URL}
        api_key: ${oc.env:OPENSANDBOX_API_KEY}
        namespace: psrl
        egress_port: 18080
        ready_timeout_s: 120
        template_timeout_s: 1800
        entrypoint: [tail, -f, /dev/null]
        default_resources: {cpu: "1", memory: "2Gi"}
        default_timeout_s: 3600
```

OpenSandbox takes one typed `config` block rather than flat keyword arguments,
because it has more deployment settings than fit on one line. AgentEnv takes
flat keyword arguments.

Three of these keys are off by default because the provider does not serve what
they unlock on every runtime, so each one also declares a capability:
`isolation_runtime`, `template_publish`, and `warm_pool_ref`. Read
[DEPLOY_OPENSANDBOX.md](DEPLOY_OPENSANDBOX.md) for what each one requires, and for
which of the two runtimes can serve it. A spec requiring a capability the
deployment does not declare is refused at admission rather than served without it.

## Deploy across nodes

**A sandbox runs on the node that hosts its agent loop worker.** The trainer places
`agent.num_workers` workers across the allowed nodes, and each worker builds its own
manager, so a `docker` sandbox lands on that worker's node and a provider sandbox is
somewhere that node can reach.

Start Ray across the nodes, then keep four things true:

| Concern | What to do |
|---|---|
| Where sandboxes run | `gen_actor_rollout_ref.rollout.agent.node_ips="['${ENV_NODE_IP}']"` restricts worker placement, and the sandboxes follow. Empty allows every alive node |
| The env node's resources | Fence it out of the Ray pool with `psrl.deployment.total_nnodes`, so rollout and training workers cannot land on it |
| One daemon per node | Keep a local Docker daemon on every node that hosts a worker, because a remote daemon's containers are not in the worker's envelope |
| Crash recovery | Point `lifecycle.heartbeat_dir` at `${HEARTBEAT_DIR}`, visible to every worker using that daemon, and mount it in containerized deployments. Use one directory per daemon |

Run the sandboxes on the training nodes instead by clearing the pin, which is
simpler to operate and competes with the trainer for CPU and memory:

```bash
bash examples/airs_bench/run_qwen3-4b.sh \
    gen_actor_rollout_ref.rollout.agent.node_ips=[] \
    gen_actor_rollout_ref.rollout.agent.sandbox.capacity.utilization=0.5
```

Every sandbox key hangs off `gen_actor_rollout_ref.rollout.agent.sandbox` in the
composed config, because `psrl_rollout.yaml` is included there. A short path such as
`sandbox.capacity.utilization=0.5` is not an alias for it: Hydra either rejects the
override or, with `+`, writes a second top-level `sandbox` key that nothing reads.

A provider backend ignores the node envelope, because the provider schedules the
sandbox. `capacity.*` then bounds only the `docker` backend, so a mixed deployment
sizes the envelope for Docker alone.

**To place sandboxes on other nodes**, turn on
`psrl.deployment.sandbox_placement.enabled` and name the fleet in
`gen_actor_rollout_ref.rollout.agent.node_ips`, which is required because defaulting
to every alive node would put sandboxes on the GPU nodes. The trainer then creates one
placement service and one node agent per node, and every worker places through them, so
`capacity.*` is enforced where the containers run. See [OPERATIONS.md](OPERATIONS.md).

## Run end to end

The AIRS-Bench recipe pins its sandboxes to an env node and trains on a compute
node, and [its README](../../examples/airs_bench/README.md) covers the data it
needs first:

```bash
bash examples/airs_bench/run_qwen3-4b.sh
```

Check a deployment with one sandboxed episode, against a running Ray cluster,
before committing to a full run. The script reads a prepared task from
`${AIRS_DATA_ROOT}`, which has to hold `airs_prepared/${TASK_ID}` and
`airs_configs/data/${TASK_ID}`:

```bash
python -m examples.airs_bench.scripted_episode --task-id "${TASK_ID}"
```

Point a recipe at a provider by adding its block and switching the default. Edit the
config for a permanent change, or pass the same keys as overrides. Hydra adds a
missing key with `+`:

```bash
python -m psrl.trainer.main_ppo \
    +gen_actor_rollout_ref.rollout.agent.sandbox.backends.agentenv._target_=psrl.sandbox.backends.AgentEnvBackend \
    +gen_actor_rollout_ref.rollout.agent.sandbox.backends.agentenv.api_url="${PROVIDER_API_URL}" \
    +gen_actor_rollout_ref.rollout.agent.sandbox.backends.agentenv.api_key="${PROVIDER_API_KEY}" \
    gen_actor_rollout_ref.rollout.agent.sandbox.default_backend=agentenv
```

An AgentEnv image source needs its template first, and the agent loop does that per
task before it acquires anything, so no extra step is needed for a rollout. To check
the import by hand, run the live conformance test below.

## Gotchas

- **A sandbox is admitted against the node envelope only on `docker`.** A provider
  schedules its own sandbox, so `capacity.*` does not bound it and an overcommitted
  provider quota shows up as a provider error, not a capacity timeout.
- **A cross-node run reads a stale `/tmp`.** `/tmp` is node-local, so put
  `${HEARTBEAT_DIR}` and any dataset on a path every worker can see, or a restarted
  run reclaims nothing.
- **A provider has prerequisites PSRL cannot work around.** An AgentEnv deployment
  needs kernel 6.8+ and `/dev/kvm`, and an OpenSandbox template needs a fast-sandbox
  runtime. Both are in the deployment pages.
- **Sandboxes stop with the worker.** A preempted worker takes its local sandboxes
  with it, and nothing resumes them. A resumable rollout is loop work, tracked as
  L2 in the execution plan.

## Verify

Unit and contract tests need no daemon and no provider:

```bash
python -m pytest tests/sandbox
ruff check psrl/sandbox tests/sandbox
python scripts/audit_prose_style.py psrl/sandbox tests/sandbox
git diff --check
```

Exercise a real Docker daemon and record reproducible latency:

```bash
PSRL_RUN_DOCKER_INTEGRATION=1 PSRL_DOCKER_TEST_IMAGE=${SANDBOX_IMAGE} \
  python -m pytest tests/sandbox/test_docker_live.py
python -m tests.sandbox.benchmark_docker_backend \
  --image ${SANDBOX_IMAGE} --iterations 100 --concurrency 16
```

Exercise a real Ray cluster, which is the only check that covers actor serialization,
node affinity, and per-node liveness. It needs no Docker daemon:

```bash
python -m tests.sandbox.smoke_ray_plane
```

Exercise a real provider, which is the only check that the deployment is reachable
and a template exists. Each deployment page carries that command with the environment
variables its provider needs.

Compare latency runs on the same idle node with pre-pulled images. Unit tests
establish failure semantics, not production throughput or GPU convergence.
