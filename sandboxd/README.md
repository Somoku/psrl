# sandboxd

The sandbox service: one control plane for a fleet, one agent per node, and a
Python SDK as the only surface a caller uses. Backends supply the sandbox
runtimes; nothing here reimplements one.

Design and rationale: [sandbox_service](../../docs/design/sandbox_service.md).

| Placeholder | Meaning |
|---|---|
| `${REPO}` | this repository's root |
| `${SOCK}` | where the SDK reaches the service, default `/run/sandboxd.sock` |

## Quickstart

Build the service and run it against the local Docker daemon:

```bash
cd ${REPO}/sandboxd
go build -o /usr/local/bin/sandboxd ./cmd/sandboxd
sandboxd -config ${REPO}/sandboxd/example.json
```

Then, from Python:

```python
from psrl.pysandbox import Resources, SandboxClient, SandboxSpec, Source

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
    { "type": "docker", "mode": "psrl", "socket": "/var/run/docker.sock", "api_version": "v1.40" }
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
| `backends[].mode` | `psrl` | `psrl`: this service places. `provider`: the backend's own scheduler does |

Override a derived span only when a deployment genuinely needs a different one.
The orderings are still asserted against it:

```json
"timing": { "episode_deadline_s": 1800, "overrides": { "reap_window_s": 600 } }
```

## Scheduling modes

`mode` is a property of one backend's deployment, not of the backend. The same
backend can run both ways in one fleet, which is what makes a scheduler
ablation hold everything else fixed.

| Mode | Who picks the node | When to use it |
|---|---|---|
| `psrl` | this service, with in-flight reservation and image locality | the backend's own scheduler adds nothing |
| `provider` | the backend's control plane | the backend has a real scheduler whose decisions would be lost |

## Tests

```bash
cd ${REPO}/sandboxd
go test ./internal/...                 # unit, no daemon needed
go test -race ./internal/...           # the concurrency invariants
go test ./internal/backend/dockerbackend/ ./internal/server/   # needs a docker socket
```

The Python SDK's live suite needs a built binary:

```bash
SANDBOXD_BINARY=/usr/local/bin/sandboxd pytest ${REPO}/tests/sandbox/test_pysandbox_live.py
```

Both live suites skip themselves where no Docker socket is present.

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
