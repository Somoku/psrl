# AgentEnv Backend (6b)

This page is step 6b of {doc}`sandbox_refactor`. It designs the AgentEnv backend
so it uses the provider's full functionality, following AgentEnv's own best
practices. Read {doc}`sandbox_backend_integration` for the shared capability
model this step builds on.

AgentEnv (AENV) is the sandbox runtime that powers agentic RL for Kimi K3 and is
already the backend Miles uses to train Terminal-Bench-2. PSRL should use it as
the primary scale backend for stateful coding agents.

---

## Why AgentEnv

| Property | What it gives PSRL |
|---|---|
| Firecracker microVM per sandbox | A kernel boundary around untrusted model code, not a shared host kernel |
| Snapshot-backed boot and resume under 50 ms, pause under 100 ms | Idle sandboxes release CPU and memory, and a resume is cheap |
| Incremental memory and filesystem snapshots | A full state checkpoint without a cold boot |
| Native fork, one call per group | One prepared environment branches into a whole GRPO group |
| Snapshots persisted to object storage or a shared filesystem | A restore lands on any node, which is `RESUME_ANYWHERE` |
| Overlaybd and ublk on-demand image loading | A node does not pre-pull a large image corpus |
| Memory ballooning with high overcommit | Density holds as sandboxes diverge |
| Multi-node gateway and scheduler | The provider scales across machines and PSRL does not |

All of these are provider features PSRL should expose, not reimplement.

---

## What is verified, and what is not

AgentEnv is an E2B-compatible provider, and the provider's own SDK is the published
contract for it. That matters: it means most of this page is checkable against source
rather than inferred.

- **The SDK owns the sandbox surface, including create.** `create(template=…, timeout=…,
  envs=…, metadata=…)`, `pause()`, `connect()` (which resumes), `create_snapshot(name=…)`,
  `fork(count=…, timeout=…)`, `delete_snapshot(…)`, `set_timeout()`, `get_metrics()`, and
  `files`/`commands` are public SDK methods with published semantics, and the backend
  calls them rather than reaching past them to a REST shape nobody publishes.
- **An image resolves to the template `prepare` imported.** This provider creates a
  sandbox from a template, and an image becomes a template rather than being a create-time
  source, so an image source resolves to the template name derived from its reference. A
  spec whose image was never prepared fails on the provider's missing-template error
  rather than on a request shape PSRL invented.
- **Fork is one SDK call.** It returns **one entry per requested fork**, each a sandbox
  or the exception that fork failed with, and it documents **no upper bound on the
  count**. There is no batch size to negotiate, so PSRL does not invent one. The source
  is checkpointed in place: briefly paused, snapshotted with its memory, then resumed,
  with its id and expiry untouched.
- **A resume is a connect.** There is no separate resume method: connecting to a paused
  sandbox resumes it, and the connection has to be rebuilt anyway because the pause
  invalidated it. The connect carries the sandbox's new life, and the provider applies it
  only when it is longer than the remaining one — which is how a resumed sandbox avoids
  expiring against its pre-pause deadline.
- **A snapshot can be named**, and the name matters: a qualified `<team>/<name>:<tag>`
  survives a re-import where a bare id does not. Creation takes minutes, so the SDK
  request timeout is raised for it.
- **A sandbox's life has a documented ceiling** (24 hours on the higher plan), so a longer
  request is refused here with a reason instead of at the server.

:::{warning}
**Two things stay outside the SDK, and one capability is missing.** Importing or verifying
the template an image resolves to, and the listing that reclaims a create whose reply was
lost, are provider-side flows with no public API reference, so their request shapes could
not be verified. Volumes are **not wired**: the SDK takes a `volume_mounts` value whose
shape is not published, so the backend refuses a spec that asks for provider storage and
does not declare the `VOLUME` capability — admission refuses it rather than sending a
guessed request. Everything else on this page is verified against the SDK's source.
:::

So the split is the SDK for everything sandbox-scoped, and a native path only for
template management. The work in 6b is to declare the full feature set, expose the
parameters PSRL currently hardcodes, and add the provider's best-practice flows.

---

## Full native surface

### Sandboxes

| Operation | Request | Notes |
|---|---|---|
| Create | `Sandbox.create(template=…, timeout=…, envs=…, metadata=…)` | From a template, an image, or a snapshot id |
| Get | `get_info()` | State and readiness |
| Kill | `kill()` | Snapshots and templates survive deletion |
| Pause | `pause()` | Saves runtime state and stops the microVM, releasing CPU and memory |
| Resume | `connect(sandbox_id, timeout=…)` | Connecting resumes a paused sandbox; the timeout is the new life |
| Renew | `set_timeout(timeout)` | Extends or reduces the life set at create or by the last renewal |
| Fork | `fork(count=…, timeout=…)` | One call, one entry per child, each a sandbox or an exception |
| Metrics | `get_metrics()` | CPU, memory, and disk usage, as a time series |
| TTL and auto-eviction | Provider policy on the sandbox | By default a sandbox pauses at its TTL, and it can be configured to delete instead |

Fork details that matter to the design. The source is briefly paused while its state is
captured, then returns to Running. Each child inherits the source filesystem, memory,
network policy, security mode, and CPU, memory, and disk configuration. What PSRL does
with a partial result is a group policy, stated under the fork interface below.

### Snapshots

| Operation | Request | Notes |
|---|---|---|
| Create | `create_snapshot(name=…)` | The source is paused while it captures, and the id and names come back together |
| List | `list_snapshots(sandbox_id=…, name=…)` | A paginator |
| Delete | `delete_snapshot(id or qualified name)` | |
| Start from a snapshot | `Sandbox.create(template=<id or qualified name>)` | A new sandbox with a new id that inherits filesystem, processes, memory, and environment. Env, volumes, and mounts are **rejected** as overrides |

A snapshot is the `RESUME_ANYWHERE` primitive. Because it persists to object
storage or a shared filesystem, a restore can land on a node that never held the
source. This is the cross-node resume path.

### Templates

| Operation | Form | Notes |
|---|---|---|
| Pull an OCI image | `aenv pull <image>` with `--name`, `--cpu`, `--memory` | Imports an existing image as a template |
| Build a Dockerfile | `aenv build <context>` with `--start-cmd`, `--ready-cmd`, `--probe` | Builds with BuildKit inside a microVM, then captures a snapshot |
| Watch a build | `aenv template watch <name>` | Statuses are `waiting`, `building`, `ready`, and `error` |
| HTTP management | `/templates` create, list, get, delete | |

A template is snapshot backed. Publishing boots the image, runs the startup
command, waits for readiness, and captures a snapshot, so a sandbox created from
a template resumes a captured state instead of booting a fresh guest. This is why
a template cold start is under 50 ms. A template name is accepted anywhere a
template id is.

### Other surfaces

- **Volumes.** Create and mount provider-managed storage, which survives a
  sandbox and can attach to forks and snapshots. This is the provider form of
  `MountSpec`, and it is not a host bind mount.
- **Networking and proxy.** A reverse proxy reaches services inside a sandbox
  over HTTP and WebSocket. This is the mechanism a harness inside a sandbox uses
  to reach the session server, so a placement decision depends on it.
- **Authentication.** The lifecycle API uses the `X-API-Key` header, which
  `ProviderControlClient` already sends.
- **Lifecycle hooks.** Provider extension points for setup at lifecycle
  transitions.

---

## Capability declaration

The declaration lives in {doc}`sandbox_backend_integration`, which is the only
normative capability table. This page states why AgentEnv earns each entry, and
what the current code is missing.

- `HIBERNATE` because pause saves runtime state and releases CPU and memory.
- `FULL_STATE_SNAPSHOT` because a snapshot carries memory and filesystem.
- `RESUME_ANYWHERE` at the `full_state` level, because a snapshot persists to
  object storage or a shared filesystem and a restore resumes the captured
  processes, not just the disk.
- `NATIVE_FORK` because fork is one provider call that reports per child.
- `WARM_POOL` because a snapshot-backed template is a warm claim.
- `IMAGE_ON_DEMAND` because overlaybd layers load lazily, which is true layer
  level laziness and not block delivery.
- No `VOLUME`: the provider offers provider-managed storage, but the SDK's
  `volume_mounts` value has no published shape, so the backend refuses a spec that needs
  it and does not declare the capability.
- No `FREEZE`, because pause writes state to storage rather than keeping host
  memory resident.

`AgentEnvStateDriver` declares the whole surface: `HIBERNATE`,
`FULL_STATE_SNAPSHOT` with a `FULL_STATE` resume level, `RESTORE`, `NATIVE_FORK`,
`RESUME_ANYWHERE`, `WARM_POOL`, `TEMPLATE_BUILD`, `IMAGE_ON_DEMAND`, `EGRESS_POLICY`,
and `CREDENTIAL_INJECTION`. `VOLUME` is withheld, because the SDK's volume-mount
value has no published shape.

---

## Fork interface (6b.2)

The backend forks the group. The group size comes from rollout config and is the
same for every prompt, so the fork count is one config value and never a per
prompt choice. In PSRL the group size is `gen_actor_rollout_ref.rollout.n`, read
by the harness loop as `rollout_n`, and validation uses
`train_actor_rollout_ref.rollout.val_kwargs.n`.

The interface has four parts.

### Specs

A group is one spec per member, differing only in identity, because
`workflow_id` and `idempotency_key` are per trajectory. The group size is a
config value and never a per prompt choice, so the caller knows how many specs
to build before it sees the prompt.

There is no `fanout` field on the spec. A count on a single spec cannot carry
`fanout` workflow ids, and it would invite the group to be admitted as a gang,
which {doc}`sandbox_refactor` rejects.

### Manager

```python
async def acquire_group(
    self, specs: Sequence[SandboxSpec], backend: str | None = None
) -> list[SandboxLease]:
    ...
```

- One spec returns one lease, the same as `acquire`.
- The manager validates that the members agree on their source and resource
  request, and rejects a list that does not, because a fork cannot serve members
  that want different environments.
- With `NATIVE_FORK` it provisions a parent that has no `workflow_id` and its own
  resource class, forks one fewer child than the member count, adopts each child
  under one member's identity, creates the remaining member directly, and
  terminates the parent. The parent's slot is held for the fork, not for the
  episode.
- Without `NATIVE_FORK` it creates one sandbox per spec. Each member is an
  ordinary idempotent acquire, so the portable fallback is today's path repeated.
- No member holds capacity while waiting for a sibling, on either path. That is
  the rule that makes two concurrent groups unable to starve each other.
- A partial fork destroys **every** child that started and raises, so a short group is
  never returned and nothing is leaked. The group size stays uniform, which is what the
  config promises. The children that started after the failing entry count too: the
  provider created them, so the backend must destroy them.
- The provider documents no maximum count, so one call asks for the whole group.
- Every child goes through the restore sanitization path, so it does not inherit
  the parent's random stream or transport state. A fork clones memory, so
  without this the whole group shares one seeded generator. The rule and its
  consequences are in {doc}`sandbox_backend_integration`.

### Session and AgentEnv

The session fork takes a count and returns the children.

```python
async def fork(self, count: int) -> list[SandboxSession]:
    ...
```

`AgentEnvStateDriver.fork` calls the SDK's `fork(count=count, timeout=ttl)`, where `ttl`
is the workflow's timeout, and the SDK returns one entry per child: a sandbox, or the
exception that child failed with.

- Adopt every entry that is a sandbox, sanitize it as a restored session, and return the
  children in the order the provider returned them.
- On any `error` entry, terminate the adopted children and raise.
- The source is briefly paused during capture, so a fork must not overlap a
  command on the parent. The parent's exec lock is the in-flight signal, the
  same one the idle pause uses, so a fork takes it and does not wait on it.

### Loop wiring

`AgentLoopManager.get_dispatch_plan` already co-locates a prompt's children on
one worker, so the worker holds a whole group and can build the member specs
itself. It calls `acquire_group` once and gives one lease to each sample's run.

Two details belong to the loop, not to this backend step.

- The member queue deadline is a fraction of the episode deadline, so a capacity
  shortage costs a retry rather than a long stall.
- The number of groups in flight is what bounds demand, and the loop manager
  already governs it.

Until the loop change lands, callers pass a single spec and the fork path stays
dormant.

### When the fork is not the best answer

Fork pays a task's setup once per group per step. A snapshot of the prepared
task pays it once per task per run, and AgentEnv's snapshot backed template is
exactly that object. Where a task is visited in more than one step, prepare and
snapshot beats fork, and the group stops being a unit of anything. Fork stays
the answer for the first visit to a task and for setup that cannot be captured.
The reasoning is in {doc}`sandbox_refactor`.

---

## Best-practice flows

These are the flows AgentEnv is built for, mapped to PSRL calls.

### Warm once, acquire fast

Build or pull one template per task image before rollout, then acquire every
sandbox from the template. A template acquire resumes a captured snapshot, which
is the under 50 ms path. A cold acquire from the task's own image is the
fallback for an image that has no template, and it pays a full guest boot.

### Fork the group

Prepare one sandbox, then fork the GRPO group from it through `acquire_group`.
The group size is `rollout.n` from config, so it is the same for every prompt.
Setup runs once for the group, not once per sample. A partial fork fails the
group rather than training a short one, because the config fixes the size.

### Pause idle, resume on demand

A sandbox waiting on the model or the next turn is paused, which releases CPU and
memory and keeps state. The next command or the resume call returns it. This is
the AgentEnv best practice for idle environments, and it is what keeps density
high across a long rollout.

### Snapshot for cross-node resume

When a workflow must survive a preemption or move to another node, snapshot it
and let the restore land wherever there is capacity. This is `RESUME_ANYWHERE`,
and it is the backend half of the resumable rollout design in
{doc}`sandbox_refactor`.

A capture is named as well as identified, and a restore addresses the name it was
captured under, falling back to the id when no name was recorded. A name survives
a re-import of the snapshot and the provider accepts one anywhere it accepts an
id, so the name is the more durable of the two.

### Align the TTL with the episode

AgentEnv auto-pauses a sandbox at its TTL, and can be configured to delete
instead. PSRL's `idle_timeout_s` must map to the provider TTL, and the pause
versus delete choice must match the workflow. A rollout that expects a resume
sets pause. A grader sandbox that is disposable sets delete. A resume renews the
TTL, so a relocked sandbox does not immediately expire again.

## What the provider's documentation pins down

The data plane is E2B-compatible and the provider documents it, so these are checked
against the provider rather than inferred:

- **Create** takes `template` (a name, an id, or a snapshot id), `timeout` in seconds,
  `envs`, and `metadata`. The provider's `timeout` is the sandbox's own life, so a spec's
  absolute lifetime wins over its idle window when both are set.
- **Commands and files** are the E2B SDK surface: `commands.run(...)`, `files.read(...)`,
  `files.write(path, data)`, `kill()`, `is_running()`, `pause()`, `create_snapshot(...)`.
  A snapshot is created with a raised *request* timeout, because the provider documents
  that creation can take minutes while the SDK's own request timeout defaults far lower.
  The AgentEnv control client already uses a five-minute request timeout for this reason.
- **A snapshot restore accepts a narrow override list**: a lifetime, user metadata, the
  network policy, and the provider's secure-access and auto-pause switches. Environment
  variables, volumes, and mounts are fixed by the snapshot and the provider **rejects** a
  restore that passes them. The backend therefore clears those fields on the restore
  path — that is the provider's semantics, not a dropped request, because a capture
  already carries the environment it was taken with.
- **Snapshot lifetime is independent** of the source sandbox, and a restored sandbox
  occupies the snapshot, so deleting it while a restore is live fails.
- **Pause is resumed by connecting**, which the provider documents as auto-resuming a
  paused sandbox, and it asks the caller to re-confirm its long connections and process
  state afterwards — which is what `refresh_transport` does.
- **A snapshot can only be taken from a running sandbox**, whose state becomes
  `snapshotting` while it is captured.

:::{warning}
**The template management paths are not publicly documented.** Template import, Dockerfile
build, and build watching are provider-side flows reached outside the SDK, so
`/templates` and the cold-image create shapes could not be checked against a public API
reference and remain the least verified part of this backend. Everything sandbox-scoped
— create, kill, pause, resume, fork, snapshot, restore, metrics, files, commands — now
goes through the provider's own SDK and is verified against its source.
:::

## Known limitation

A lost create response on the **generic** `E2BBackend` reclaims nothing: the SDK
failure modes that leave a provider sandbox behind cannot be enumerated without a
live endpoint, so that sandbox leaks until the provider's own idle timeout.
`CubeSandboxBackend` inherits that behaviour, because it subclasses `E2BBackend`
without overriding `create`.

`AgentEnvBackend` does not share the limitation. A create that fails with an
idempotency key on the spec lists `GET /sandboxes?metadata.psrl.idempotency_key=…`,
destroys every sandbox the provider reports, and counts the reclaim — so a retry
cannot build a second sandbox for one trajectory. A create with no idempotency key
has nothing to filter on and is left to the provider's idle timeout, which is the
boundary of the promise rather than an oversight.

---

## Data plane decision

Keep the E2B compatible SDK for commands and files. AgentEnv exposes an E2B
compatible API for exactly this, and the Miles integration uses it, so the
compatibility path is a supported path and not a fallback. The native control
plane stays native, because state operations are not in the E2B subset.

---

## Migration steps

| Step | Change |
|---|---|
| 6b.1 | Declare the full feature set on `AgentEnvStateDriver`, after 6a adds the members |
| 6b.2 | Implement `fork(count)` batched at the provider maximum, and `acquire_group` over a member spec list |
| 6b.3 | Send a snapshot name, and address a restore by id or name |
| 6b.4 | Send the TTL on resume |
| 6b.5 | Build or pull the template in `prepare`, and poll the watch status |
| 6b.6 | Add volume support behind the `VOLUME` feature |
| 6b.7 | Reclaim a lost create by listing sandboxes filtered on the idempotency metadata |

Each step is independently testable against a live AgentEnv server. Steps 6b.1
through 6b.4 need only the existing integration test fixture.

---

## Decisions

| Id | Decision | Consequence |
|---|---|---|
| B1 | The backend forks the group, and the group size is `rollout.n` from config | The size is uniform across prompts. `acquire_group` takes one spec per member and `fork(count)` fans out the rest |
| B2 | A partial fork fails the group | The group size stays uniform. A run that wants a short group needs a loop change |
| B3 | A recipe owns its template lifecycle | A shared template cache is possible later, but not in this step |
| B4 | A template stays a `SandboxSource` | `prepare` builds the source. A named asset can come later |
| B5 | A volume is a new spec field, not a `MountSpec` | A host bind mount stays distinct from provider storage |
| B6 | A forked child is sanitized as a restored session and carries no idempotency key | The group does not share one random stream, and a retry adopts the parent rather than a sibling |

:::{seealso}
- {doc}`sandbox_backend_integration`: the capability model and shared layer this step builds on
- {doc}`sandbox_opensandbox`: step 6c, the OpenSandbox backend
- {doc}`sandbox_external_backends`: the capability and best-practice rationale
- {doc}`sandbox_refactor`: the resumable rollout state and the phases
- {doc}`sandbox_lifecycle`: the invariants this backend must preserve
:::
