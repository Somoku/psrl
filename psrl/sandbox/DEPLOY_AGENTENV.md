# Deploy AgentEnv

AgentEnv (AENV) is an external sandbox platform built on Firecracker microVMs. The
provider runs and schedules the sandbox, so a deployment that uses it needs no Docker
daemon on the worker's node and consumes none of the node's envelope. What it needs
instead is a reachable server, and a host that can run microVMs at all.

This page is the deployment path. For the backend's SDK mapping, read
[sandbox_agentenv](../../docs/design/sandbox_agentenv.md).

Placeholders:

| Placeholder | Meaning |
|---|---|
| `${AENV_HOME}` | The provider's data directory, `/var/lib/aenv` unless `AENV_HOME_PATH` says otherwise |
| `${AENV_URL}` | The API base URL a worker reaches, for example `http://sandbox-api.internal:8000` |
| `${AENV_API_KEY}` | The key the server generated, or the shared value `AENV_API_KEY` sets |
| `${SANDBOX_IMAGE}` | A task image reference, ideally pinned by digest |
| `${SHARED_STORE}` | A path every node can see, for the snapshot repository |

---

## Steps

### 1. Check the host can run a microVM

Two prerequisites decide whether AgentEnv works at all, and neither is something
PSRL can work around:

| Requirement | Why |
|---|---|
| Linux kernel 6.8 or newer | The runtime's device and network features need it |
| `/dev/kvm` access | Firecracker executes the guest on KVM. A host without standard KVM needs the provider's PVM path instead |

The install script also loads the `ublk_drv` kernel module, which is how the guest
root filesystem is served from an OverlayBD-backed userspace block device.

### 2. Install the server

Installation needs root, but the service does not run as root. It uses a dedicated
`aenv` account with `CAP_NET_ADMIN` and `CAP_SYS_ADMIN`, plus group access to
`/dev/kvm` and the ublk devices.

```bash
curl -fsSL https://raw.githubusercontent.com/kvcache-ai/AgentENV/main/scripts/install.sh \
  | sudo AENV_HOME_PATH="${AENV_HOME}" bash
sudo systemctl start aenv
```

`AENV_HOME_PATH` chooses the data directory. Persistent state lives under
`${AENV_HOME}`, and transient namespace and daemon-socket state lives under
`/run/aenv`.

Docker is the alternative, and it still needs `/dev/kvm`:

```bash
docker pull ghcr.io/kvcache-ai/aenv-server:latest
docker run --rm -it --name aenv-server \
  --device /dev/kvm --privileged -v /dev:/dev \
  -p 8000:8000 \
  ghcr.io/kvcache-ai/aenv-server:latest
```

`--privileged` is required for Firecracker's network namespace operations. The
server downloads its runtime assets on first start.

### 3. Read the API key

The server generates it on first start:

```bash
sudo cat "${AENV_HOME}/secrets/api-key"
# in the container:
docker exec aenv-server cat /workspace/env/secrets/api-key
```

A multi-node deployment should set `AENV_API_KEY` to one shared value on every
gateway and runtime node, instead of letting each generate its own. Look in
`/run/secrets/api-key` if the environment variable is not what a node is using.

### 4. Point the port at what a worker will use

| Deployment | Listen address | Notes |
|---|---|---|
| Single node | `:8000` | Set `API_ADDR` in `/etc/default/aenv` to change it |
| Gateway and scheduler, multi-node | `:8080` for the gateway | `GATEWAY_HTTP_LISTEN_ADDR`; the scheduler listens on `:9090` for gRPC |

### 5. Verify it answers

```bash
curl --fail http://127.0.0.1:8000/health

curl --fail -H "X-API-Key: ${AENV_API_KEY}" \
    "${AENV_URL}/templates"
```

`X-API-Key` is the control-plane credential, and it is never a data-plane
credential. The second call is the one that matters: `/health` needs no key, so it
says nothing about whether the key a worker will send is correct.

### 6. Mount a shared snapshot store for a multi-node run

A snapshot resumes a sandbox on another node, so the repository has to be visible
from every node. `[snapshot].repository_backend` selects it:

| Value | Backing |
|---|---|
| `posix_fs`, the default | A directory, which must be shared storage across nodes |
| `oss` | An OSS or S3-compatible bucket |

`[snapshot].local_cache_path` is the node-local cache, which does not need to be
shared. A deployment that keeps the default `posix_fs` repository in per-node state
can snapshot and restore on one node and nowhere else.

### 7. Install the PSRL extra and prepare a template

```bash
python -m pip install -e ".[sandbox-e2b]"
```

AgentEnv creates a sandbox from a **template**, and an image becomes a template. So
a rollout prepares one template per task image before it acquires anything, and a
spec whose image was never prepared fails on the provider's missing-template error.
The agent loop does this per task, so a run needs no extra step.

### 8. Point PSRL at it

```yaml
sandbox:
  default_backend: agentenv
  backends:
    agentenv:
      _target_: psrl.sandbox.backends.AgentEnvBackend
      api_url: ${oc.env:AGENTENV_API_URL}
      api_key: ${oc.env:AGENTENV_API_KEY}
```

`api_url` is the API base URL, and the backend authenticates with `X-API-Key`.

The SDK also has a **data-plane** URL, for the process interaction a command and a
file call use. It resolves from the server for a hosted deployment. A self-hosted
deployment whose sandboxes are reached through the provider's own proxy should set
`E2B_SANDBOX_URL` in the worker's environment to the same base URL, which is what
the provider documents for both its single-node and its gateway topologies. The
backend configures `api_url` and `api_key` only, so this one is an environment
variable:

```bash
export E2B_API_URL="${AENV_URL}"
export E2B_SANDBOX_URL="${AENV_URL}"
export E2B_API_KEY="${AENV_API_KEY}"
```

### 9. Run one sandboxed episode

```bash
python -m examples.airs_bench.scripted_episode --task-id "${TASK_ID}"
```

Or exercise the backend against the live provider, which is the only check that the
URL, the key, the data-plane route, and a task image all work together:

```bash
PSRL_LIVE_MICROVM_BACKEND=agentenv \
PSRL_LIVE_MICROVM_API_URL="${AENV_URL}" \
PSRL_LIVE_MICROVM_API_KEY="${AENV_API_KEY}" \
PSRL_LIVE_MICROVM_SOURCE="${SANDBOX_IMAGE}" \
PSRL_LIVE_MICROVM_SOURCE_KIND=image \
  python -m pytest tests/sandbox/test_microvm_live.py
```

---

## What this backend refuses, and why

Each of these is the provider fixing a thing PSRL would otherwise have to guess, so
the refusal is the honest answer rather than a missing feature:

| Refused | Reason |
|---|---|
| A resource override | The template's build fixes CPU, memory, and disk |
| A host bind mount | A provider has no host to bind |
| A provider volume | The SDK's volume-mount value has no published shape, so `VOLUME` is undeclared and admission refuses a spec that needs one |

Keep `lifetime_timeout_s` inside the provider's 24-hour cap on a sandbox's life. The
backend refuses a longer one locally rather than letting the server reject it.

---

## Multi-node

A worker only needs to reach `${AENV_URL}`, so the provider backend ignores
`agent.node_ips` and the node envelope. `capacity.*` then bounds only the `docker`
backend, and a mixed deployment sizes the envelope for Docker alone.

Provider-side facts that constrain a multi-node deployment:

- **Topology is a gateway plus a scheduler plus per-node runtimes.** The gateway takes
  API calls, the scheduler picks a node, and each runtime node executes the microVMs.
- **Set one shared `AENV_API_KEY`** on every node, or the gateway and its runtimes will
  not agree on the credential.
- **A scheduler without Redis loses its bindings on restart.** `SCHEDULER_REDIS_ADDR`
  unset keeps sandbox-to-node bindings in memory, and unset also means a restarted
  scheduler forgets which node holds what.
- **The snapshot repository must be shared** for a cross-node resume, as in step 6.
- **TLS is not the server's job.** The server authenticates but does not encrypt, so a
  deployment that crosses a network boundary terminates TLS at a proxy.

---

## Gotchas

- **A host without KVM cannot silently fall back.** AgentEnv needs `/dev/kvm` or the
  provider's PVM path. A worker on a host without either fails at create rather than
  degrading.
- **An image source needs its template first.** The agent loop prepares one per task, so
  a hand-written spec that skips `prepare` fails on a missing template.
- **No GPU.** The runtime is a Firecracker microVM, which cannot pass a PCI device
  through, so a task that needs CUDA cannot run here.
- **A provider error is not a capacity timeout.** The provider schedules the sandbox, so
  an exhausted quota arrives as a provider error. Fall back to
  `gen_actor_rollout_ref.rollout.agent.sandbox.default_backend=docker` when the provider
  is unavailable.
- **The template and recovery endpoints are the least verified part of this backend.**
  The provider's SDK is the contract for the sandbox surface and PSRL follows it, but
  the template import and the lost-create listing go through REST paths the provider
  does not publish. Treat them as unverified against a new provider version.

---

## Verify

The backend's unit and conformance tests need no server:

```bash
python -m pytest tests/sandbox/test_e2b_backend.py tests/sandbox/test_backend_conformance.py
```

The live check above is the only one that proves a deployment is reachable.
