# OpenSandbox Backend (6c)

This page is step 6c of {doc}`sandbox_refactor`. It designs the OpenSandbox
backend as a new first class provider. Read {doc}`sandbox_backend_integration`
for the shared capability model this step builds on.

OpenSandbox is not E2B compatible, so unlike AgentEnv it is not a client factory
under `E2BBackend`. It is a backend with its own session type, because it has a
different execution plane and a different state model.

Its value to PSRL is that the same API serves a laptop and a cluster, so the
local dev loop and the cluster run share one code path. It is also the provider
that offers warm pools and provider native egress and secret controls.

---

## Provider shape

| Plane | Base | Owns |
|---|---|---|
| Lifecycle | `/v1` on the sandbox server | Create, delete, pause, resume, snapshots, templates, endpoint resolution |
| Execution, `execd` | A per sandbox endpoint from endpoint resolution, container port 44772 | Commands, bash sessions, code contexts, files, metrics |
| Egress sidecar | A per sandbox endpoint from endpoint resolution, port 18080 | The runtime network policy and the credential vault |

Authentication is the `OPEN-SANDBOX-API-KEY` header on the lifecycle plane. The other
two planes are reached the same way and by the same rule: **resolve the sandbox
endpoint for the plane's port, then forward the headers the resolution returned.** The
provider decides which headers it needs — `X-EXECD-ACCESS-TOKEN` is one of them, not a
constant this backend constructs — so a backend must never hardcode an address or
assume the lifecycle key authorizes a direct plane call. Endpoint resolution returns
the address and the headers together, because both are transport state. The session
re-resolves its data endpoint at connect and after every resume, because a checkpoint
and resume can move the sandbox.

Lifecycle states are `Pending`, `Running`, `Pausing`, `Paused`, `Stopping`,
`Terminated`, and `Failed`.

| Provider state | `SandboxStatus` | Why |
|---|---|---|
| `Running` | `RUNNING` | Accepts commands |
| `Paused` | `PAUSED` | Exists, needs a resume |
| `Terminated` | `TERMINATED` | Gone |
| `Pending`, `Pausing`, `Stopping`, `Failed` | `UNKNOWN` | No stable state a caller can command |

---

## Class design

Following the shared skeleton in {doc}`sandbox_backend_integration`.

| Class | Owns |
|---|---|
| `OpenSandboxControlClient` | Lifecycle HTTP. Create, get, delete, pause, resume, snapshot, template, endpoint resolution. One bounded connection pool |
| `OpenSandboxExecClient` | `execd` HTTP for one sandbox. Command, session, files, metrics. Resolved per sandbox |
| `OpenSandboxEgressClient` | Egress sidecar HTTP for one sandbox. The credential vault. Resolved per sandbox, like `execd` |
| `OpenSandboxSession` | One sandbox. `exec`, `read_bytes`, `write_bytes`, `status`, `terminate`, and the state calls. Holds the resolved `execd` endpoint |
| `OpenSandboxBackend` | Capabilities, `create`, `connect`, `restore`, `delete_snapshot`, `prepare`, `shutdown`. Holds the control client and a state driver |

The three clients use independent connection pools, so a saturated command stream
cannot starve lifecycle calls, the same separation the Docker backend uses.

---

## Lifecycle endpoint map

| PSRL operation | Request |
|---|---|
| Create | `POST /v1/sandboxes` from an image, a snapshot, or a template |
| Inspect | `GET /v1/sandboxes/{sandboxId}` |
| Delete | `DELETE /v1/sandboxes/{sandboxId}` |
| Pause | `POST /v1/sandboxes/{sandboxId}/pause`, asynchronous |
| Resume | `POST /v1/sandboxes/{sandboxId}/resume` |
| Snapshot full state | `POST /v1/sandboxes/{sandboxId}/snapshots` |
| List snapshots | `GET /v1/snapshots` |
| Get and delete snapshot | `GET /v1/snapshots/{snapshotId}`, `DELETE /v1/snapshots/{snapshotId}` |
| Restore | `POST /v1/sandboxes` with the snapshot as the source |
| Renew TTL | `POST /v1/sandboxes/{sandboxId}/renew-expiration` |
| Build a template | `POST /v1/templates` with `{image, publish}`, then poll `GET /v1/templates/{templateId}` until `status.phase` is `Succeeded` |
| Endpoint resolution | `GET /v1/sandboxes/{sandboxId}/endpoints/{port}` → an address and the headers to forward |

The outbound policy is **not** a lifecycle call. It travels on the create request as
`networkPolicy`, because the provider configures it during provisioning, and the
lifecycle `networkpolicy` endpoints are the fsb path for a sandbox that has no sidecar.

### The egress plane

Reached by resolving the sandbox endpoint for the egress port and forwarding those
headers, exactly like `execd`.

| PSRL operation | Request |
|---|---|
| Inspect the runtime policy | `GET /policy` |
| Mutate the runtime policy | `PATCH /policy` with a rule list, merge semantics |
| Create the credential vault | `POST /credential-vault` with credentials and bindings |
| Read sanitized vault state | `GET /credential-vault`, `/credential-vault/credentials[/{name}]`, `/credential-vault/bindings[/{name}]` |

Resource requests travel as `resourceLimits`, a map of Kubernetes-style quantities, so
`cpu_count` becomes `cpu` (a whole core as `"2"`, a fraction as millicores), `memory_mb`
becomes `memory` (`"4096Mi"`), and a GPU count becomes `gpu`. The provider documents no
disk quantity, so `disk_mb` is refused rather than dropped. Because the provider requires
the field on every create it does not reject outright, a spec that asks for nothing gets
the deployment's configured default.

### What each create shape allows

The three shapes are not interchangeable, and the provider rejects a request that mixes
them:

| Shape | Required | Rejected |
|---|---|---|
| Image | `entrypoint`, `resourceLimits` | — |
| Template (`templateId`) | `timeout` | `entrypoint`, `env`, `resourceLimits`, `volumes`, `platform`, `credentialProxy`, `secureAccess`, `lifecycle` |
| Snapshot (`snapshotId`) | `resourceLimits` | `entrypoint` is optional; the captured process is kept |

`timeout` is in seconds, has a floor of 60, and **omitting it disables automatic
expiry** — which is what PSRL does for an image create unless the spec declares a
lifetime, because a sandbox that expires mid-episode looks like a workload failure. A
template create is the exception: the provider requires a timeout there, so a spec with
no declared lifetime falls back to the deployment's `default_timeout_s`.

There is no per-sandbox workdir field, so `spec.workdir` reaches the sandbox as the
`cwd` of each command. There is no per-sandbox isolation-runtime field either: the
boundary is a property of the deployment, so it is declared in configuration rather than
requested per sandbox, and a spec that requires one is refused when the deployment has
none. Metadata keys under `opensandbox.io/` are reserved and are refused here, because
the server would reject the whole create for a reason the caller could not see.

A create is not usable the moment it returns: the provider publishes the execution
endpoint asynchronously. The backend waits for the sandbox to be running *and* for that
endpoint to resolve, so a startup failure lands in the create rather than in a confusing
first command.

---

## Session and exec model

The `execd` plane offers two command modes, and the choice matches the persistent
shell decision in {doc}`sandbox_refactor`.

- **Persistent bash session.** `POST /session` opens one and returns
  `{"session_id": …}`. The body is optional and carries only a `cwd`: there is no shell
  to choose, so a working directory is applied per command rather than faked with a
  `cd` the caller cannot see. `POST /session/{sessionId}/run` takes
  `{command, cwd?, timeout}` and streams the run; `DELETE /session/{sessionId}` ends it.
  Commands in one session are serialized, matching the `SandboxSession.exec` contract.
- **One shot.** `POST /command` takes `{command}` or `{argv}`, plus `cwd`, `envs`,
  `timeout`, `background`, and `uid`/`gid`. It has no shell state, and it is the only
  mode that can carry an environment, so a command that needs one uses it.
- **Timeouts are milliseconds and server enforced.** Both run endpoints take `timeout`
  in milliseconds, and the provider terminates the process when it is reached. That is
  the only deadline either side of the call can honour, so `timeout_s` is converted
  rather than dropped.

### The stream carries no exit code

`ServerStreamEvent` is `{type, text, results, execution_complete?, error?}`, with types
`init`, `status`, `stdout`, `stderr`, `result`, `execution_complete`,
`execution_count`, `ping`, and `error`. Output arrives in `text`, a code-interpreter
result arrives under `results` as a MIME map, and a failure arrives as an `error` with
`ename`, `evalue`, and `traceback`.

**There is no exit code anywhere in the stream**, and the only endpoint that has one is
`GET /command/status/{id}` — which needs a command id that the streaming response does
not document how to convey. So a run that reports an error is reported as a failure, and
a non-zero exit that produces no error event is not distinguishable from success. A
caller that must know an exit code has to make it observable itself, for example by
having the command write it to a file the caller then downloads.

Files use `GET /files/download?path=`, `POST /files/upload`, `GET /files/info?path=`,
and `DELETE /files?path=`. Upload is multipart: a JSON `metadata` part naming the
destination, then the file. Download supports range requests and line-based reads, so
`read_bytes` streams instead of holding copies.

`GET /metrics` returns `cpu_count`, `cpu_used_pct`, `mem_total_mib`, `mem_used_mib`,
and `timestamp`. It reports neither a memory peak nor a cumulative CPU time, so `stats`
leaves both unknown rather than reporting a measured zero.

A spec field selects the exec mode, persistent session or one shot. Persistent is
the default for an agent harness, because the harness expects shell state across
turns.

---

## Image delivery

Images are content-addressed artifacts, not a registry pull. A template build
compiles an OCI image into a golden-image artifact and records its manifest
digest. Golden images, pause checkpoints, and snapshots all live in one
S3-compatible artifact store, and a resume never trusts a mutable tag.

A node keeps a private reflink-backed cache. A node-local delivery daemon fetches
4 MiB blocks in the order cache, then peer, then origin, so each block leaves the
origin roughly once for the cluster.

This is block-level delivery, not overlaybd style layer loading. Declare
`IMAGE_BLOCK_DELIVERY`, not `IMAGE_ON_DEMAND`, so a spec that needs true lazy
layers does not select this backend by mistake.

---

## State operations

- **Pause and hibernate.** `POST /pause` and `POST /resume`. Pause is
  asynchronous, so the backend polls `GET /v1/sandboxes/{sandboxId}` until the
  state settles to `Paused` or `Failed` before it returns.
- **Filesystem snapshot.** `POST /sandboxes/{sandboxId}/snapshots` commits the
  sandbox's root filesystem and the artifact persists to the store, so it restores on
  another host. Running processes and memory are **not** captured, so the backend
  declares `FILESYSTEM_SNAPSHOT` with `RESUME_ANYWHERE` at the `FILESYSTEM` level and
  refuses a spec that requires `FULL_STATE_SNAPSHOT`. Claiming full state here would
  let a caller read a workspace resume as proof that its harness survives a move.
  The create is accepted with a `Creating` snapshot and the artifact lands later, so
  the backend polls `GET /v1/snapshots/{snapshotId}` to `Ready` before returning a
  reference a caller could restore.
- **Restore.** `POST /v1/sandboxes` with the snapshot source, then resolve a
  fresh data endpoint. A restore is a new sandbox with a new id, so a connect on
  the old id must fail and the manager adopts the new ref.
- **Fork.** Not a provider feature. A caller that needs a branch uses the
  manager's checkpoint and restore path, which already falls back when
  `NATIVE_FORK` is absent.

---

## Warm pool and templates

Two settings decide whether the warm and golden-image paths exist at all, because
neither is served on every runtime:

- `warm_pool_ref` names a pool the operator pre-created. With it set, a create sends
  `extensions.poolRef` and the runtime answers with the `allocation` field that
  confirms the claim, which the backend counts as a warm claim. The provider rejects
  a network policy, a credential proxy, a volume, and a snapshot beside a pool
  reference, so a spec that carries any of them is created from its image instead.
- `template_publish` is an S3-compatible target for golden-image builds. Without it
  the backend never touches the template API, which matters because that API answers
  **501 on a runtime that cannot build golden images** (the Docker runtime among them).
  A build is keyed by the `templateId` the server mints, and there is no name-keyed
  lookup, so a built template is found by listing `GET /v1/templates` and matching its
  source image.

A template-backed sandbox carries no egress sidecar, so a spec that brokers a
credential cannot use the template path: the provider rejects `credentialProxy` in
template mode outright. The two are alternatives, not layers, so a spec with bindings is
created from its image even when a prepared template exists.

Template mode also fixes the workload shape, so the warm path costs a resource request
and an environment: those belong to the template's golden image and cannot be stated per
sandbox. A caller that needs either is better served by the image path.

Template builds are asynchronous. The recipe builds the template in `prepare`,
polls `GET /v1/templates/{templateId}` until `Succeeded`, and fails admission if
the build did not finish. This keeps a slow build off the rollout critical path
and turns a build failure into an admission failure rather than a create failure.

---

## Egress and credentials

The egress sidecar is the only outbound interception layer, which is what makes both
of these features provider-enforced rather than reimplemented.

**The policy** is a default action and an ordered rule list, and it travels on the
create request as `networkPolicy` — the same document the sidecar's own `/policy`
serves, so a policy applied at create and one applied at runtime cannot disagree.
`PSRL`'s `EgressPolicy` is the same shape: `default_action` plus `EgressRule`s. The
provider resolves names rather than addresses and derives a rule's port from its
scheme, so a rule that names a port is refused here rather than sent for the provider
to reject. The default action is deny, because a policy that has to deny explicitly
is one a caller forgets, and the omission is invisible until a reward looks wrong.

**The vault brokers credentials; it does not inject them into the environment.** This
is the part a reading of the API surface alone gets wrong. A value is written to the
sidecar, and the sandbox process receives a *placeholder* instead. When the sandbox
makes an outbound request, the sidecar matches it against the bindings and attaches
the real credential on the way out. So a secret is never in the sandbox's
environment, command line, filesystem, or logs, which is the whole point: a prompt
injection that can read the environment gets a string that says where the secret
went rather than what it is.

- A spec names its credentials (`CredentialRef`) and its bindings
  (`CredentialBinding`). A binding says which hosts the credential may be sent to and
  how it is attached — `bearer`, `basic`, `apiKey`, `customHeaders`, or `passthrough`
  with placeholder substitutions for an upstream that takes a secret in a path, a
  query, or a body.
- A binding must name a host. A binding that matched every host would hand the
  credential to whatever the workload asked for.
- The policy must allow every host a binding names, and a spec that fails that is
  refused before anything is created. The sidecar injects only into a request it is
  allowed to forward, so a blocked host would otherwise fail with a network error that
  says nothing about the missing allow rule.
- The vault is written with `POST /credential-vault` on the resolved **egress**
  endpoint, after the sandbox exists and its policy is active — a vault cannot be
  created before there is a policy. Values are write-only: even the vault's own read
  endpoints return names, source types, and revisions, and never a value.
- The vault is **sandbox-local**, so a restore writes it again on the new sandbox
  rather than assuming it travelled with the snapshot.
- **A template-backed sandbox has no sidecar and therefore no vault.** The warm path
  and credential brokering are mutually exclusive on this backend, so a spec with
  bindings is created from its image even when a prepared template exists.

This is why `EGRESS_POLICY` and `CREDENTIAL_INJECTION` are declared and then
delegated. It is also the Phase 5 security path for any workload on this backend,
because the provider enforces the policy and PSRL does not reimplement it.

---

## Capability declaration

The declaration lives in {doc}`sandbox_backend_integration`, which is the only
normative capability table. What matters here is why the two image features and
the resume level read the way they do.

- `RESUME_ANYWHERE` at the `full_state` level, because a checkpoint captures
  memory and the artifact store makes it reachable from any host.
- `IMAGE_BLOCK_DELIVERY` and not `IMAGE_ON_DEMAND`, for the reason in the image
  delivery section above.
- No `NATIVE_FORK`, so a branch goes through the manager's checkpoint and
  restore fallback.
- No `FREEZE`, because pause releases compute rather than keeping memory
  resident.
- `ISOLATION_RUNTIME` is declared from configuration and not by default, because
  the provider has no per-sandbox runtime field: the boundary is a property of the
  server the deployment points at.

---

## Migration steps

| Step | Change |
|---|---|
| 6c.1 | Add `OpenSandboxControlClient` and `OpenSandboxBackend` with `create`, `connect`, `terminate`, and `shutdown` |
| 6c.2 | Add `OpenSandboxExecClient` and the persistent bash session exec model |
| 6c.3 | Add the state driver. Pause, resume, snapshot, restore, and endpoint re-resolution |
| 6c.4 | Add `prepare` for the template build and the warm pool claim, and wire the egress and credential policy |

Each step is independently testable. Step 6c.1 and 6c.2 cover the data plane and
can land before the state work.

---

## Decisions

| Id | Decision | Consequence |
|---|---|---|
| C1 | The spec selects the exec mode, default persistent bash session | The harness gets shell state. A grader step uses one shot |
| C2 | The session re-resolves the `execd` endpoint on every resume | Correct after a move. A cached endpoint with retry is a later optimization |
| C3 | The warm pool depth starts at the provider default | Pool sizing from the rollout batch size is a later input |
| C4 | PSRL writes the egress policy and reads sanitized metadata only | The provider enforces it, and PSRL never reads a secret value |

:::{seealso}
- {doc}`sandbox_backend_integration`: the capability model and shared layer this step builds on
- {doc}`sandbox_agentenv`: step 6b, the AgentEnv backend
- {doc}`sandbox_external_backends`: the capability and best-practice rationale
- {doc}`sandbox_refactor`: the persistent shell and the phases
- {doc}`sandbox_lifecycle`: the invariants this backend must preserve
:::
