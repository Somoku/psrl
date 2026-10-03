# Deploy OpenSandbox

OpenSandbox separates into two parts, and a deployment chooses how much of it to
run. Its data plane is `execd`, a Go agent that runs inside every sandbox and
serves commands, files, PTYs, code contexts, and metrics. Its control plane is a
Python lifecycle server that creates containers and tracks them.

The two modes differ in how much of that you deploy.

| | `direct` | `provider` |
|---|---|---|
| OpenSandbox server | not deployed | required |
| Who places the sandbox | p3b | OpenSandbox |
| What you install | the agent image | the server, or a cluster |
| Create concurrency | bounded by p3b and the daemon | bounded by the server |
| Warm pools, template builds, block delivery | no | yes, on Kubernetes |
| Cross-node snapshot restore | no | yes, on Kubernetes |

Both modes reach the agent directly for every command, so the data plane is the
same either way. The difference is in the create path.

---

## direct mode

p3b drives the container runtime itself and stages the agent into each sandbox.
No OpenSandbox server runs. This is the shape to use when p3b should own
placement and the create path should be as short as it can be.

### 1. Pull the agent image

```bash
docker pull opensandbox/execd:latest
```

That image carries `execd`, `bootstrap.sh`, and `bwrap`. They are extracted once
when the service starts and then bind-mounted read-only into every sandbox, so
the per-sandbox cost is a mount rather than a copy.

### 2. Declare the backend

```json
{
  "type": "opensandbox",
  "mode": "direct",
  "execd_image": "opensandbox/execd:latest",
  "socket": "unix:///var/run/docker.sock",
  "stage_dir": "/var/lib/sandboxd/opensandbox-agent",
  "network_mode": "host",
  "max_create_concurrency": 16
}
```

`network_mode: host` removes the per-sandbox veth, netns, and iptables work from
the create path, and lets the agent be reached on a host port with no mapping to
allocate. On a cgroup v1 kernel that is the difference between a create path that
scales and one that does not.

`max_create_concurrency` bounds concurrent container creates. The daemon
serializes parts of container setup in the kernel, so an unbounded burst turns
latency into timeouts rather than throughput.

Set `port_min` and `port_max` only if you are not using host networking; they
bound where the agent is published when a mapping is needed.

### 3. Verify before serving

```bash
sandboxd -preflight -config deploy.json
```

Preflight checks the runtime, the agent image, and the stage directory, and names
what to run if any of the three is missing. It installs nothing.

### 4. What direct mode cannot serve

An outbound network policy and the credential vault both need OpenSandbox's
egress sidecar, which direct mode does not compose. They are also mutually
exclusive with host networking in OpenSandbox itself. A workload that needs
either should use `provider` mode.

A snapshot in direct mode is a container commit: a local image, no registry push.
It restores on the node that took it and nowhere else, so `resume_anywhere` is
not declared and the node travels on the handle.

---

## provider mode

OpenSandbox's own control plane places and runs the sandbox; p3b does
cross-backend quota only. Use this for a Kubernetes-backed cluster, or when you
need warm pools, template builds, block-level image delivery, or a snapshot that
restores on any node.

### Steps

#### 1. Install the server

Pin the version. The `1.1.0` wheel does not import, because it ships without
`opensandbox_server.services.fast_sandbox.generated`, so a plain install fails at
startup rather than at install time:

```bash
python -m pip install "opensandbox-server==0.2.3"
```

Requirements, from the provider's own installation page:

| Requirement | Version |
|---|---|
| Python | 3.10 or newer |
| Docker Engine, for the Docker runtime | 20.10 or newer |
| Kubernetes, for the Kubernetes runtime | 1.21.1 or newer |
| Host OS | Linux, macOS, or Windows with WSL2 |

#### 1b. Pre-pull the execution sidecar

Every sandbox runs an `execd` sidecar, and the server pulls its image on the first
create. Pull it ahead of the run, or the first acquire pays the pull inside its own
readiness deadline and times out:

```bash
docker pull "$(grep -oP '(?<=^execd_image = ")[^"]+' ~/.sandbox.toml)"
```

The image reference is `[runtime].execd_image` in the config written by step 2, so
run this after that step.

#### 2. Write a configuration

```bash
opensandbox-server init-config ~/.sandbox.toml --example docker
```

Use `--example k8s` for the Kubernetes runtime. The server reads `~/.sandbox.toml`
by default, and `SANDBOX_CONFIG_PATH` or `--config` overrides the path.

Two keys decide whether this deployment can serve what a recipe asks for. The
example config sets `[egress].image` already and defaults `[egress].mode` to `dns`,
so the one to change is the mode:

| Key | What it enables | Without it |
|---|---|---|
| `[egress].image` | `networkPolicy` enforcement, and the credential vault | A create that carries a policy is not enforced |
| `[egress].mode = "dns+nft"` | Credential injection | The vault refuses to activate. The default `dns` mode cannot stop a direct-IP connection from bypassing DNS policy |

Set `[server].api_key`. With it empty the server runs unauthenticated, and a
non-interactive environment needs `OPENSANDBOX_INSECURE_SERVER=YES` to acknowledge
that rather than refusing to start.

#### 3. Start the server

```bash
opensandbox-server --config ~/.sandbox.toml
```

It listens on `[server].host` and `[server].port`, and serves the lifecycle API under
`/v1`. Point `DOCKER_HOST` at a non-default daemon if the Docker runtime needs one.

#### 4. Verify it answers

```bash
curl --fail http://127.0.0.1:8080/health
# → {"status": "healthy"}

curl --fail -H "OPEN-SANDBOX-API-KEY: ${OPENSANDBOX_API_KEY}" \
    "${OPENSANDBOX_URL}/v1/sandboxes"
```

The second call is the one that matters: `/health` needs no key, so it says nothing
about whether the key and the URL a worker will use are correct.

#### 5. Point PSRL at it

```yaml
sandbox:
  default_backend: opensandbox
  backends:
    opensandbox:
      _target_: psrl.sandbox.backends.OpenSandboxBackend
      config:
        api_url: ${oc.env:OPENSANDBOX_API_URL}
        api_key: ${oc.env:OPENSANDBOX_API_KEY}
```

`api_url` is the lifecycle base URL without the `/v1` suffix. `execd` is resolved per
sandbox from that URL, so no execution-plane address is configured.

#### 6. Declare what the server actually provides

Each of these is a property of the deployment rather than of the API, so the backend
declares the matching capability only where the deployment sets it. A recipe that
requires a capability the deployment does not declare is refused at admission rather
than served without it.

| `config` key | Declares | Requires |
|---|---|---|
| `isolation_runtime` | `ISOLATION_RUNTIME` | A `[secure_runtime]` (gVisor or Kata) configured on the server |
| `template_publish` | `TEMPLATE_BUILD` | A Kubernetes or fast-sandbox runtime **and** `${PUBLISH_TARGET}`. The template API answers 501 on the Docker runtime, so a Docker deployment can never build a golden image |
| `warm_pool_ref` | `WARM_POOL` | A pool the operator pre-created as a Kubernetes CRD. A create sends `extensions.poolRef` and the pool's pod fixes the workload shape, so the provider rejects a network policy, a credential proxy, a volume, and a snapshot beside it |

#### 7. Run one sandboxed episode

```bash
python -m examples.airs_bench.scripted_episode --task-id "${TASK_ID}"
```

Or exercise the backend against the live provider, which is the only check that the
URL, the key, and a task image all work together:

```bash
PSRL_LIVE_MICROVM_BACKEND=opensandbox \
PSRL_LIVE_MICROVM_API_URL="${OPENSANDBOX_URL}" \
PSRL_LIVE_MICROVM_API_KEY="${OPENSANDBOX_API_KEY}" \
PSRL_LIVE_MICROVM_SOURCE="${SANDBOX_IMAGE}" \
PSRL_LIVE_MICROVM_SOURCE_KIND=image \
  python -m pytest tests/sandbox/test_microvm_live.py
```

---

## What each runtime can serve

| Capability | Docker runtime | Kubernetes runtime |
|---|---|---|
| Image create, commands, files | Yes | Yes |
| Pause and resume | Yes | Yes |
| Credential vault | Needs `[egress].image` and `mode = "dns+nft"` | Same |
| Snapshot and cross-node resume | No | Yes, and it needs an OCI registry plus push and pull secrets on the controller |
| Golden-image template | **No**, the template API answers 501 | Yes, with a fast-sandbox runtime and a publish target |
| Warm pool | **No** | Yes, as a Pool CRD |

A snapshot resumes at the **filesystem** level: the workspace comes back and every
process starts fresh. A recipe that needs a live process to survive a move has to ask
for `FULL_STATE` and be served by a provider that offers it, which this one does not.

---

## Multi-node

A worker only needs to reach `${OPENSANDBOX_URL}`, so the provider backend ignores
`agent.node_ips` and the node envelope. `capacity.*` then bounds only the `docker`
backend, and a mixed deployment sizes the envelope for Docker alone.

Server-side facts that constrain a multi-node deployment:

- **Server HA is not supported yet.** The Helm chart's `server.replicaCount` defaults
  to 1, and multi-replica coordination is not implemented.
- **The Kubernetes chart serves the server on port 80** behind a `ClusterIP` Service.
  Reach it from outside the cluster with `server.service.type: NodePort` or
  `LoadBalancer`, or with `kubectl port-forward`.
- **Keep `[server].port = 80`** when overriding `configToml`, unless the chart
  templates are changed with it.
- **A namespace `LimitRange` applies to the egress sidecar.** Set `[egress].requests`
  and `[egress].limits`, or the default reservation can be far larger than DNS and nft
  enforcement needs.

---

## Gotchas

- **A non-zero exit code is reported as 1.** The `execd` stream carries no exit code,
  so a grader that branches on one has to write the status to a file and read it back.
- **`stats()` reports nothing.** `GET /metrics` on `execd` describes the host rather
  than the sandbox, so every field is left unknown instead of reporting the node's
  memory as one sandbox's footprint. Size an envelope from the `docker` backend.
- **An `execd` sidecar answers a moment after the sandbox is `Running`.** The backend
  waits for the plane itself rather than for the state, so a create returns a sandbox
  that can take a command. A caller driving the lifecycle API by hand has to wait too.
- **The URL a worker uses is not the URL a laptop uses.** `127.0.0.1` reaches the
  server only from the node it runs on. Give every worker a routable
  `${OPENSANDBOX_URL}`, or run a node-local server per node.
- **A credential binding needs three things at once.** An outbound policy whose
  `default_action` is `deny` and which allows every host a binding names, a server
  with `[egress].mode = "dns+nft"`, and `credentialProxy.enabled` on the create, which
  the backend sets for a spec that has bindings. A vault that never activates leaves the
  workload making unauthenticated requests whose rewards still look valid.
- **Do not put a service-mesh sidecar in the sandbox pod.** Credential injection
  assumes the OpenSandbox egress sidecar is the only transparent outbound interception
  layer in the network namespace.
- **An isolation runtime and an egress policy can conflict.** A deployment that
  declares `isolation_runtime` should confirm the provider supports policy enforcement
  on that runtime before relying on both.
- **A pool cannot carry a policy.** A spec that needs egress or a brokered credential
  is created from its image even when `warm_pool_ref` is set, so it pays a cold start.
- **A provider error is not a capacity timeout.** The provider schedules the sandbox,
  so an exhausted quota arrives as a provider error. Fall back to
  `gen_actor_rollout_ref.rollout.agent.sandbox.default_backend=docker` when the
  provider is unavailable.

---

## Verify

The backend's unit and conformance tests need no server:

```bash
python -m pytest tests/sandbox/test_opensandbox_backend.py tests/sandbox/test_backend_conformance.py
```

The live check above is the only one that proves a deployment is reachable.
