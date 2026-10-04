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

Validate a configuration before serving it. The exit code is the whole answer,
which is what a deployment pipeline reads:

```bash
make preflight                        # validates example.json, reaches each backend
sandboxd -config ${REPO}/p3b/example.json
```

A preflight failure names what it could not reach. `docker daemon at
/var/run/docker.sock is not answering` means the daemon, not the service.

Then, from Python:

```python
from sandboxd import Resources, SandboxClient, SandboxSpec, Source

async with SandboxClient("unix:///run/sandboxd.sock") as client:
    async with await client.create(
        SandboxSpec(
            source=Source.image("alpine:latest"),
            resources=Resources(memory_mb=512, cpu_count=1),
            resource_class="rollout",
        )
    ) as sandbox:
        print((await sandbox.exec("echo hello")).stdout)
```

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
| `internal/timing/` | every deadline, derived from three declared values |

## Configuration

An operator declares intent; the service derives the rest. A value that can be
derived is not a knob.

```json
{
  "listen": "unix:///run/sandboxd.sock",
  "node_id": "node-a",
  "timing": { "episode_deadline_s": 1800, "node_ttl_s": 120, "rpc_timeout_s": 60 },
  "fleet":  { "memory_mb": 960000, "cpu_millis": 180000, "disk_mb": 4000000 },
  "node":   { "memory_mb": 480000, "cpu_millis": 90000, "disk_mb": 2000000,
              "gpu_indices": [], "local_cpu_ceiling": 0.95, "local_mem_ceiling": 0.9 },
  "classes": {
    "rollout": { "guaranteed_share": 0.7 },
    "grader":  { "guaranteed_share": 0.2 },
    "prepare": { "guaranteed_share": 0.1 }
  },
  "backends": [
    { "type": "docker", "mode": "direct", "socket": "/var/run/docker.sock", "api_version": "v1.40" }
  ],
  "default_backend": "docker"
}
```

| Field | Default | Why it matters |
|---|---|---|
| `timing.episode_deadline_s` | required | the one number an operator can estimate; every window follows from it |
| `timing.node_ttl_s` | `120` | silence before a node is drained |
| `timing.rpc_timeout_s` | `60` | bounds a coordination call, never a command |
| `fleet.*` | required | the cross-backend ceiling, which no single backend can enforce |
| `node.*` | required | this machine's envelope |
| `node.local_*_ceiling` | `0` (off) | measured pressure above which the node refuses regardless of its envelope |
| `classes.*.guaranteed_share` | — | must sum below one, or a guarantee is unsatisfiable |
| `classes.*.max_share` | unset | a ceiling, even on an idle fleet |
| `backends[].mode` | `direct` | `direct`: this service places and drives the runtime. `provider`: the backend’s own scheduler does |

Override a derived span only when a deployment genuinely needs a different one.
The orderings are still asserted against it:

```json
"timing": { "episode_deadline_s": 1800, "overrides": { "reap_window_s": 600 } }
```

## One control plane over several nodes

The quickstart runs both planes in one process, which serves one machine. A
fleet splits them: one control process, and one node process per machine.

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

## Using it from PSRL

PSRL reaches the service through the `p3b` backend, which is a client and
nothing more: placement, admission, and reclamation stay here. Two steps.

1. Start a service where the sandboxes should run. One combined process per
   node is the simplest shape; a fleet runs one control plane over several node
   processes instead. `listen` is what the worker reaches.
2. Point the rollout at it, in
   `psrl/trainer/config/rollout/psrl_rollout.yaml`:

```yaml
sandbox:
  default_backend: p3b          # was: docker
  backends:
    p3b:
      endpoint: unix:///run/sandboxd.sock
      resource_class: rollout   # must name a class this service declares
```

The `resource_class` has to exist in this service's own `classes` block, or
every create is refused for a class it was never given a share of.

Choose `p3b` over the in-process `docker` backend when more than one worker
shares a fleet: the service holds one quota ledger across every node and
backend it drives, so a class guarantee bounds the whole run rather than each
worker's own node. A single-worker deployment gains nothing from the hop.

The service must be listening before a worker starts. The backend connects on
its first create, and a missing socket is a startup error rather than something
it retries.

## Gotchas

- **A node refusal is not a failure.** The control plane asks placement for
  another node. A fleet with one node turns the refusal into an error, which is
  why a single-node deployment reads `node_pressure` as a hard stop.
- **`fleet.memory_mb` is not the sum of your nodes.** It is what the service may
  admit in total. Setting it above the nodes' real capacity moves the refusal
  from the ledger to the node, where it costs a round trip first.
- **Guarantees that sum to one leave no elastic pool.** A class can then never
  borrow, so an idle fleet still refuses a burst. Keep the sum below one.
- **A stale socket blocks startup.** The service removes its own socket at
  start; a socket owned by another process makes bind fail with `address
  already in use`. Check for a second `sandboxd` before assuming a bug.
- **Commands do not go through the control plane.** If `exec` fails while
  `create` works, the fault is on the node or in the sandbox, not in placement.
