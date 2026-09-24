# Deploy OpenSandbox

OpenSandbox is an external sandbox platform: the provider runs and schedules the
sandbox, so a deployment that uses it needs no Docker daemon on the worker's node
and consumes none of the node's envelope. What it needs instead is a reachable
server, and a runtime that serves the features the recipe asks for.

This page is the deployment path. For the backend's request mapping, read
[sandbox_opensandbox](../../docs/design/sandbox_opensandbox.md).

Placeholders:

| Placeholder | Meaning |
|---|---|
| `${OPENSANDBOX_URL}` | The server's base URL, for example `http://sandbox-api.internal:8080` |
| `${OPENSANDBOX_API_KEY}` | The key set as `[server].api_key` in the server's TOML |
| `${SANDBOX_IMAGE}` | A task image reference, ideally pinned by digest |
| `${PUBLISH_TARGET}` | An S3-compatible target for golden images, for example `s3://bucket/psrl` |

---

## Steps

### 1. Install the server

```bash
python -m pip install opensandbox-server
```

Requirements, from the provider's own installation page:

| Requirement | Version |
|---|---|
| Python | 3.10 or newer |
| Docker Engine, for the Docker runtime | 20.10 or newer |
| Kubernetes, for the Kubernetes runtime | 1.21.1 or newer |
| Host OS | Linux, macOS, or Windows with WSL2 |

### 2. Write a configuration

```bash
opensandbox-server init-config ~/.sandbox.toml --example docker
```

Use `--example k8s` for the Kubernetes runtime. The server reads `~/.sandbox.toml`
by default, and `SANDBOX_CONFIG_PATH` or `--config` overrides the path.

Two keys decide whether this deployment can serve what a recipe asks for, and an
unset one makes the corresponding feature impossible rather than merely slow:

| Key | What it enables | Without it |
|---|---|---|
| `[egress].image` | `networkPolicy` enforcement, and the credential vault | A create that carries a policy is not enforced |
| `[egress].mode = "dns+nft"` | Credential injection | The vault refuses to activate. DNS-only mode cannot stop a direct-IP connection from bypassing DNS policy |

Set `[server].api_key`. With it empty the server runs unauthenticated, and a
non-interactive environment needs `OPENSANDBOX_INSECURE_SERVER=YES` to acknowledge
that rather than refusing to start.

### 3. Start the server

```bash
opensandbox-server --config ~/.sandbox.toml
```

It listens on `[server].host` and `[server].port`, and serves the lifecycle API under
`/v1`. Point `DOCKER_HOST` at a non-default daemon if the Docker runtime needs one.

### 4. Verify it answers

```bash
curl --fail http://127.0.0.1:8080/health
# → {"status": "healthy"}

curl --fail -H "OPEN-SANDBOX-API-KEY: ${OPENSANDBOX_API_KEY}" \
    "${OPENSANDBOX_URL}/v1/sandboxes"
```

The second call is the one that matters: `/health` needs no key, so it says nothing
about whether the key and the URL a worker will use are correct.

### 5. Point PSRL at it

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

### 6. Declare what the server actually provides

Each of these is a property of the deployment rather than of the API, so the backend
declares the matching capability only where the deployment sets it. A recipe that
requires a capability the deployment does not declare is refused at admission rather
than served without it.

| `config` key | Declares | Requires |
|---|---|---|
| `isolation_runtime` | `ISOLATION_RUNTIME` | A `[secure_runtime]` (gVisor or Kata) configured on the server |
| `template_publish` | `TEMPLATE_BUILD` | A Kubernetes or fast-sandbox runtime **and** `${PUBLISH_TARGET}`. The template API answers 501 on the Docker runtime, so a Docker deployment can never build a golden image |
| `warm_pool_ref` | `WARM_POOL` | A pool the operator pre-created as a Kubernetes CRD. A create sends `extensions.poolRef` and the pool's pod fixes the workload shape, so the provider rejects a network policy, a credential proxy, a volume, and a snapshot beside it |

### 7. Run one sandboxed episode

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
