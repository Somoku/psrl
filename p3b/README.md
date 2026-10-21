# sandboxd

The sandbox service: one control plane for a fleet, one agent per node, and a
Python SDK as the only surface a caller uses. Backends supply the sandbox
runtimes; nothing here reimplements one.

Design and rationale: [sandbox_service](../../docs/design/sandbox_service.md).

| Placeholder | Meaning |
|---|---|
| `${REPO}` | this repository's root |
| `${SOCK}` | where the SDK reaches the service, default `/run/sandboxd.sock` |

## Install

Two artifacts, installed separately because they are deployed to different
places: the service runs on every sandbox node, the SDK wherever a caller runs.

```bash
cd ${REPO}/p3b
make install          # builds and installs ${PREFIX}/bin/sandboxd, PREFIX=/usr/local
make sdk-install      # pip install -e sdk, the surface a caller imports
```

`PREFIX`, `DESTDIR`, `GO`, and `PIP` are overridable. `make help` lists every
target.

## Quickstart

The default deployment is a **control plane over node agents**, and every plane
runs the Docker backend. `make cluster` starts the example fleet on this host:
two node agents on ports 9090 and 9091 (with distinct cgroup subtrees), then the
control plane on port 9000. It preflights all three configs first, so a
misconfigured node fails before anything is started.

```bash
make cluster          # nodes first, then the control plane; logs in bin/logs
make cluster-stop     # stop what it started
```

A preflight failure names what it could not reach. `docker daemon at
/var/run/docker.sock is not answering` means the daemon, not the service. The
control plane's preflight does not require a local daemon: it declares the
backend for routing but drives none of it.

Then, from Python, against the control plane:

```python
from sandboxd import Resources, SandboxClient, SandboxSpec, Source

async with SandboxClient("127.0.0.1:9000") as client:
    async with await client.create(
        SandboxSpec(
            source=Source.image("alpine:latest"),
            resources=Resources(memory_mb=512, cpu_count=1),
            resource_class="rollout",
        )
    ) as sandbox:
        print((await sandbox.exec("echo hello")).stdout)
```

For a real fleet, install the binary on every machine and start the node agent
there, then the control plane on its own host:

```bash
# on each node
sandboxd -preflight -config node.json
sandboxd -config node.json
# on the control host
sandboxd -preflight -config control.json
sandboxd -config control.json
```

`example.json` and `make run-combined` remain the single-process development
shape. They are not the deployment default: they bypass placement across
machines and keep one worker's fleet to itself.

## Layout

| Path | What it holds |
|---|---|
| `api/v1/` | generated from `psrl/sandbox/api/v1/sandbox.proto`, the single contract |
| `cmd/sandboxd/` | the binary; one process serves control, node, or both |
| `internal/quota/` | the cross-backend ledger: class shares over the whole fleet |
| `internal/placement/` | node selection and the two-sided reservation protocol |
| `internal/monitor/` | the fleet view placement reads |
| `internal/node/` | per-node admission, lifecycle, and the reclamation sweep |
| `internal/backend/` | the backend contract and the registry that selects one |
| `internal/backend/dockerbackend/` | containers over the Engine API |
| `internal/server/` | gRPC and the SDK's JSON listener |
| `internal/timing/` | the fleet-liveness spans (node TTL, RPC timeout) and the mechanism's own constants |

## Configuration

A deployment states how fast its nodes answer; everything else is a property of
the mechanism, not of the deployment, so it is a constant rather than a knob.

```json
{
  "listen": "unix:///run/sandboxd.sock",
  "node_id": "node-a",
  "timing": { "node_ttl_s": 120, "rpc_timeout_s": 60 },
  "fleet":  { "memory_mb": 960000, "cpu_millis": 180000, "disk_mb": 4000000 },
  "node":   { "memory_mb": 480000, "cpu_millis": 90000, "disk_mb": 2000000,
              "gpu_indices": [], "local_cpu_ceiling": 0.95, "local_mem_ceiling": 0.9 },
  "classes": {
    "rollout": { "guaranteed_share": 0.7 },
    "grader":  { "guaranteed_share": 0.2 },
    "prepare": { "guaranteed_share": 0.1 }
  },
  "backends": [
    { "type": "docker", "mode": "direct", "socket": "/var/run/docker.sock", "api_version": "v1.40", "max_create_concurrency": 16 }
  ],
  "default_backend": "docker"
}
```

| Field | Default | Why it matters |
|---|---|---|
| `timing.node_ttl_s` | `120` | silence before a node is drained |
| `timing.rpc_timeout_s` | `60` | bounds a coordination call, never a command |
| `fleet.*` | required | the cross-backend ceiling, which no single backend can enforce. Memory is the sum of what the nodes will grant: each node's declared envelope times its `overcommit`. Stating the declared envelopes alone caps the fleet below what the machines will admit, and a burst then queues behind the ledger rather than behind the machines |
| `node.*` | required | this machine's envelope. Memory is what this node lends to the fleet, so keep `memory_mb × overcommit` at or below the host's memory: that keeps the configured density inside the machine, and leaves the measured ceiling below a second line rather than the only one |
| `node.local_*_ceiling` | `0` (off) | measured pressure above which the node refuses regardless of its envelope. The memory ceiling is a fraction of the **sandbox subtree**, not of the host: a node that shares its cgroup with a trainer must not measure the trainer's page cache |
| `node.cgroup_parent` | `psrl-sandboxes` | the subtree Docker creates sandboxes into and the sampler measures. The service creates it, bounds it at `memory_mb x overcommit` (capped by what the ancestor cgroup can hold), and moves itself into a sibling leaf. Empty takes the default; setting it explicitly is how a deployment states that its ceilings are about sandboxes |
| `node.sample_interval_s` | `5` | how often the domain is measured, and therefore the window a CPU rate is averaged over |
| `node.sandbox_memory_safety` | `0.05` | share of the ancestor's memory limit kept free for page cache growth and non-sandbox work when the subtree's limit is resolved |
| `node.elastic` | off | prices a sandbox from what its class actually costs instead of its declared maximum, in both dimensions a burst is bounded by. Memory: `min_reserve_mb` for a class with no history, then a trimmed maximum of observed resident memory plus `growth_margin`, grown immediately and lowered only after `shrink_after` quiet samples. CPU: `min_reserve_cpu_millis`, then the same estimator over each sandbox's `cpuacct` rate. While a dimension is above `shrink_above`, an idle sandbox is asked for it back — its kernel memory limit lowered to its resident memory plus `release_margin_mb`, its CPU quota to what it measured plus `release_margin_cpu_millis` (floors `min_limit_mb` and `min_cpu_millis`) — and both are restored below `restore_below`. Requires the fleet quota and placement to charge the same reserve, or it changes nothing end to end |
| `classes.*.guaranteed_share` | — | must sum below one, or a guarantee is unsatisfiable |
| `classes.*.max_share` | unset | a ceiling, even on an idle fleet |
| `backends[].warm_pool` | unset (off) | pre-built containers a matching create claims instead of building. The template (`image`, `memory_mb`, `cpu_count`, `disk_mb`, `env`, `workdir`, `resource_class`, `exec_mode`, `lifetime_s`) is what entries are built from and is matched **exactly**, because a label or a cgroup limit written at create cannot be changed afterwards. `lifetime_s` is the exception: it is a backstop rather than a charge, so it is the longest lifetime the deployment runs and any request at or below it is served. Entries are held against `memory_mb` of the declared envelope, not against a class, and refill stops at `pause_above` (default `0.9`) while `reclaim_above` also destroys idle entries |
| `backends[].mount_allowlist` | empty (shares nothing) | host trees a sandbox may be given. A mount is the one request in the contract that is about the machine, so the node states what it will share instead of honouring any path a caller names: a source is resolved (a symlink out of an allowed tree is refused), must already exist (the daemon would otherwise create it silently), and must sit under a listed tree. `host_mount` is advertised only when this is non-empty |
| `backends[].mode` | `direct` | `direct`: this service places and drives the runtime. `provider`: the backend’s own scheduler does |
| `backends[].max_create_concurrency` | unset, which is unbounded | bounds concurrent container creates. A daemon serializes parts of container setup in the kernel, so an unbounded burst turns latency into timeouts rather than throughput. Size it from the daemon, not the machine: `16` is the value the OpenSandbox deployment documents |

A sandbox's lifetime is not a service setting: the caller states it on the create
(`lifetime_s`), because the caller knows what the sandbox is for. The service
reclaims the sandbox by it, so a caller that dies or forgets to release leaves
nothing running.

## One control plane over several nodes

The default deployment splits the planes: one control process, and one node
process per machine. `make cluster` is the single-host example of that shape;
`run-combined` exists only for development and must not be used for a fleet.

| Field | Role `control` | Role `node` |
|---|---|---|
| `role` | `"control"` | `"node"` |
| `listen` | where the SDK connects, `host:port` so every worker can reach it | unused |
| `node_listen` | unused | where the control plane reaches this node |
| `fleet_nodes` | every node, as `node_id` and `address` | unused |
| `node` | unused | this machine's envelope |
| `fleet` | the whole fleet's ceiling | unused |
| `backends` | declared, for capability routing | declared, and driven |

Both roles declare `backends` and `classes`, and the two declarations have to
agree: the control plane routes a spec by capability, and the node it picks is
the one that actually creates. A class named in a spec must exist in both.

Start the nodes first, then the control plane; a control process whose nodes are
absent starts, but places nothing until they report.

```bash
# on each node
sandboxd -config node.json
# on the control host
sandboxd -config control.json
```

Then point the rollout at `host:port` rather than at a socket. Confirm the fleet
is whole before a run — a control plane reports every node it has heard from,
and the count is the answer:

```python
print(await client.fleet())
```

## Scheduling modes

`mode` is a property of one backend's deployment, not of the backend. The same
backend can run both ways in one fleet, which is what makes a scheduler
ablation hold everything else fixed.

| Mode | Who picks the node | When to use it |
|---|---|---|
| `direct` | this service, with in-flight reservation and image locality | the backend’s own scheduler adds nothing |
| `provider` | the backend's control plane | the backend has a real scheduler whose decisions would be lost |

## Tests

```bash
cd ${REPO}/p3b
make test             # unit, no daemon needed
make test-race        # the concurrency invariants
make bench            # the create path and the port allocator
make test-rollout     # end-to-end, needs a Docker daemon (build tag)
```

The SDK's live suite drives the real SDK against a real service and a real
daemon, so it needs a built binary:

```bash
SANDBOXD_BINARY=${PREFIX}/bin/sandboxd pytest ${REPO}/tests/sandbox/test_sdk_live.py
```

Every live suite skips itself where no Docker socket is present, so the default
run works on a machine that has none.

Three drivers exercise a deployed service rather than a fake:

```bash
# placement spread and direct exec on every node
python verify_cluster.py <control-host:port> 12

# stdout/stderr, missing path, container death, owner silence, episode-shaped load
python tests/sandbox/fault_injection.py --endpoint <control-host:port> \
    --node-ssh 'node-a=root@host-a,node-b=root@host-b' --image alpine:latest

# node process outage and out-of-band container removal
python tests/sandbox/fleet_faults.py --endpoint <control-host:port> \
    --nodes 'node-a=root@host-a,node-b=root@host-b' --target node-b \
    --image alpine:latest --restart-cmd '<binary> -config <node-config> > /tmp/node.log 2>&1 & sleep 3'

# a held load that samples live sandboxes, renewal failures, fds, and containers
python tests/sandbox/soak.py --endpoint <control-host:port> \
    --nodes 'node-a=root@host-a,node-b=root@host-b' --image alpine:latest \
    --seconds 900 --concurrency 128
```

The fleet drivers SSH to the nodes to kill a container or a node process; the
addresses are arguments, never configuration in the repository.

## Using it from PSRL

PSRL reaches the service through the `p3b` backend, which is a client and
nothing more: placement, admission, and reclamation stay here. Two steps.

1. Start the default deployment: one control plane and one node agent per
   machine, both on the Docker backend. `listen` is what the worker reaches.
2. Point the workers at the control plane. The shipped rollout config already
   selects `p3b` and names a placeholder control host; override it per
   deployment (or set `PSRL_P3B_ENDPOINT`):

```yaml
sandbox:
  default_backend: p3b
  backends:
    p3b:
      endpoint: ${oc.env:PSRL_P3B_ENDPOINT,p3b-control:9000}
      resource_class: rollout   # must name a class this service declares
```

The `resource_class` has to exist in this service's own `classes` block, or
every create is refused for a class it was never given a share of.

The default is `p3b` because the deployment is a control plane over node agents:
one quota ledger and one placement view span every worker and machine. The
in-process `docker` backend remains for single-worker development and for
ablations that must hold the backend fixed.

The service must be listening before a worker starts. The backend connects on
its first create, and a missing endpoint is a startup error rather than
something it retries.

## Gotchas

- **A node refusal is not a failure.** The control plane asks placement for
  another node. A fleet with one node turns the refusal into an error, which is
  why a single-node deployment reads `node_pressure` as a hard stop.
- **A stopped container and a missing file are different 404s.** The node asks
  the runtime which one it got: `runtime gone` releases the sandbox and its
  charge, `path not found` leaves a healthy sandbox usable. A caller that sees
  `SandboxTransportError` for an exec on a dead node is seeing a stale node, not
  a task result; the node's own prober will release the charge when it returns.
- **`fleet.memory_mb` is what the service may admit in total, not the sum of
  your nodes' declared envelopes.** State the sum of what the nodes *will
  grant* — each node's declared envelope times its `overcommit` — or the ledger
  caps the fleet below what the machines would take and a burst queues behind
  the ledger instead of behind the machines. Setting it above that moves the
  refusal from the ledger to the node, where it costs a round trip first.
- **Guarantees that sum to one leave no elastic pool.** A class can then never
  borrow, so an idle fleet still refuses a burst. Keep the sum below one.
- **A stale socket blocks startup.** The service removes its own socket at
  start; a socket owned by another process makes bind fail with `address
  already in use`. Check for a second `sandboxd` before assuming a bug.
- **Commands do not go through the control plane.** If `exec` fails while
  `create` works, the fault is on the node or in the sandbox, not in placement.
- **Renewals target the nodes that hold the owner's sandboxes, not every node
  that ever reported.** A node that missed its reports is stale, and a heartbeat
  that paid its timeout would turn one dead machine into a permanent beat
  delay. `client.diagnostics()` reports renewal failures and the last fan-out
  duration; `client.contract()` reports the timing spans a client should derive
  its cadence from.
- **The quota ledger follows the node's inventory.** A sandbox removed out of
  band is released from the node in one probe interval and from the control
  plane's ledger in two eligible reports; `owner_lease` is only the backstop for
  a partitioned node.
