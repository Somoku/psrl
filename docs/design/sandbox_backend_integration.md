# Sandbox Backend Integration (6a)

This page is step 6a of {doc}`sandbox_refactor`. It defines the capability model
and the shared backend layer that the two provider designs reuse. The provider
designs are {doc}`sandbox_agentenv` (step 6b) and {doc}`sandbox_opensandbox`
(step 6c).

Read {doc}`sandbox_external_backends` for the capability and best-practice
rationale, and {doc}`sandbox_lifecycle` for the invariants a backend must
preserve.

---

## Step 6a scope

1. Extend `SandboxFeature` with the provider capabilities the two backends need.
2. Split the portable contract into a required I/O protocol and an optional state
   protocol, so a backend that has no state operations carries no state code.
3. Route every state operation through `SandboxManager` with capability gating,
   so a caller asks the manager and never a backend directly.
4. Fix the shared backend skeleton and the error and provisioning rules, so 6b
   and 6c only add a control client, a data client, and a driver.

6a changes no provider behavior. It makes the contract able to express the
provider features, then 6b and 6c declare and implement them.

---

## Capability model

:::{important}
This page is the only normative source for what a feature means and which
backend declares it. The provider pages describe provider specific semantics and
endpoints, and they reference this page rather than restating the declaration. A
capability table copied into a provider page drifts, which has already happened
once, so there is exactly one table per question and it is here.
:::

`SandboxFeature` in `psrl/sandbox/core.py` today names `FREEZE`, `HIBERNATE`,
`FILESYSTEM_SNAPSHOT`, `FULL_STATE_SNAPSHOT`, `RESTORE`, `NATIVE_FORK`, and
`HOST_MOUNT`. Add the following.

| New feature | Meaning | Used by |
|---|---|---|
| `RESUME_ANYWHERE` | A snapshot restores on a different host, at a declared semantic level | Internal Docker, AgentEnv, OpenSandbox |
| `WARM_POOL` | A pre-warmed sandbox can be claimed instead of created | OpenSandbox, AgentEnv |
| `IMAGE_ON_DEMAND` | Layers load lazily, so a node does not pre-pull the image | AgentEnv |
| `IMAGE_BLOCK_DELIVERY` | Blocks are delivered from a cache and from peers rather than pulled whole from the origin | OpenSandbox |
| `TEMPLATE_BUILD` | The provider can build a reusable template from an image or a build context | AgentEnv, OpenSandbox |
| `VOLUME` | The provider can create and mount provider-managed storage | OpenSandbox |
| `EGRESS_POLICY` | The provider enforces a per sandbox outbound policy: a default action and ordered rules | OpenSandbox |
| `CREDENTIAL_INJECTION` | The provider brokers secrets at its egress boundary, attaching them to outbound requests instead of placing them in the sandbox | OpenSandbox |
| `ISOLATION_RUNTIME` | The provider selects gVisor, Kata, or a microVM boundary | OpenSandbox |

### Resume is a level, not a flag

A backend that restores a filesystem on another host and a backend that restores
a live process on another host both satisfy "the state moved", and only one of
them keeps the harness running. A single boolean would let a conformance run on
the internal Docker backend be read as proof that a harness survives a resume,
which it is not, because on Docker every process is new. The reasoning is in
{doc}`sandbox_refactor`, and this is how the contract encodes it.

```python
class ResumeLevel(StrEnum):
    FILESYSTEM = "filesystem"   # disk survives, every process is new
    FULL_STATE = "full_state"   # memory and processes survive
```

- `SandboxCapabilities` carries `resume_level: ResumeLevel | None`. It is set
  only when `RESUME_ANYWHERE` is declared, so one field cannot contradict the
  feature set.
- A spec requires a level, not the bare feature. `required_resume_level` is
  checked at admission, and `FULL_STATE` on a `FILESYSTEM` backend is rejected
  rather than downgraded.
- `FILESYSTEM` satisfies a `FILESYSTEM` requirement and nothing more. Levels are
  ordered, so the check is one comparison and not a capability matrix.

| Backend | `RESUME_ANYWHERE` | `resume_level` |
|---|---|---|
| Internal Docker over the snapshot store | yes | `filesystem` |
| AgentEnv | yes | `full_state` |
| OpenSandbox | yes | `full_state` |

### Enforcement

A spec declares `required_features`, and the manager enforces them, so a
deployment fails fast rather than losing a guarantee quietly.

- `SandboxManager._acquire_new` calls `capabilities.require` on
  `spec.required_features` before provisioning. It already does this.
- `SandboxManager.checkpoint` and `SandboxManager.branch` require the snapshot
  and restore features for the requested `SnapshotKind`. They already do this.
- `SandboxManager.restore` requires `RESTORE` and an enabled
  `SandboxStatePolicy`. It already does this.

The 6a work is to add the new members, and to require `RESUME_ANYWHERE` at the
requested level on a restore whose caller intends a cross-node outcome.

---

## State protocol and manager wiring

Split `SandboxSession` so the required protocol is small and the state operations
are a separate optional protocol.

- **Required I/O.** `exec`, `read_bytes`, `write_bytes`, `status`, `terminate`.
- **Optional state.** `pause`, `resume`, `snapshot`, `fork(count)`, plus `stats`
  and `refresh_transport` as supporting calls. `fork` takes a count and returns
  the children, so a provider with `NATIVE_FORK` can fan out a group in one call.

A backend declares its state protocol through `SandboxCapabilities`, and the
manager is the only caller of the state protocol. A consumer reaches a session
for I/O and for `capabilities` and `stats`, and asks the manager for everything
else.

| Manager call | Provider path | Guard |
|---|---|---|
| `acquire(spec)` | `backend.create(spec)` | `required_features`, host mount check |
| `acquire_group(specs)` | `backend.create` for a workflow-less parent then `session.fork` for one fewer than the member count when `NATIVE_FORK`, else one create per spec | Members share a source and a resource request, `NATIVE_FORK` for the shared path, entropy reseed per child, no idempotency key on a fork child |
| `connect(ref)` | `backend.connect(id)`, which resumes a paused sandbox | none |
| `checkpoint(session, kind)` | `session.snapshot(kind)` then `session.refresh_transport()` | `SandboxStatePolicy` enabled, secret env check |
| `restore(snapshot, spec)` | `backend.restore(snapshot, spec)` | `RESTORE`, enabled policy, entropy reseed on restore |
| `branch(session)` | `session.fork(1)` when `NATIVE_FORK`, else checkpoint then restore | snapshot feature check |
| `delete_snapshot(snapshot)` | `backend.delete_snapshot(snapshot)` | none |
| `release(lease)` | `session.terminate()` then capacity return | confirmed destruction |

The guards already live in `manager.py`. 6a keeps them and adds the
cross-node intent to the restore guard.

### A forked child is a restored sandbox, not a copy

A fork clones the source filesystem **and its memory**, so every child starts
from one captured process image. That has two consequences the group path must
handle, and both already have precedent in the restore path.

- **Entropy.** Every child resumes the same seeded random state, the same
  process ids, and the same pending temporary names. Model sampling happens in
  the rollout engine, so trajectory diversity is unaffected, but sandbox side
  randomness collides. Test shuffling, `mktemp` names, and ephemeral port
  choices repeat across the group. `SandboxStatePolicy` already reseeds a
  restored guest with host entropy through `_sanitize_restored_session`, and a
  fork child takes the same path. A child that skipped it would be the one case
  where a resumed guest keeps stale entropy on purpose.
- **Transport.** The captured image contains the source's transport state, so a
  child refreshes its transport exactly as a restore does.

So `fork` reuses the restore sanitization rather than adding a second path. The
rule is that any session which begins from captured state is sanitized, whether
it came from a restore or from a fork.

### Identity within a group

`workflow_id` and `idempotency_key` are per trajectory, so a group of `fanout`
members carries `fanout` distinct values of each. `acquire_group` therefore
takes a list of specs that differ only in identity, not one spec and a count.
The reasoning is in {doc}`sandbox_refactor`, and the contract is:

- **The caller builds one spec per member.** The manager validates that the
  members agree on the source and the resource request, which is the
  precondition a fork needs, and rejects a list that does not.
- **The fork parent belongs to no trajectory.** It is created with no
  `workflow_id`, no `idempotency_key`, and its own resource class, so it cannot
  collide with a member's workflow reservation. It is terminated as soon as the
  children are adopted.
- **A forked child carries no idempotency key.** The key exists so a retried
  acquire adopts the sandbox a lost response created. A child is not
  individually recoverable and is cheap to recreate, so it carries none, which
  is what `branch` already does.
- **A retried group acquire rebuilds the group.** Idempotency lives on the
  member specs, so a retry adopts whatever members already exist and creates or
  re-forks the rest. No path can produce two groups for one prompt.
- **On the create path each member keeps its own key.** Without a fork there is
  no parent, so every member is an ordinary idempotent acquire.

### State safety policy

`SandboxStatePolicy` gates every state operation. It already rejects a capture
after user commands unless `allow_external_side_effects` is set, rejects
secret-bearing environment variables unless `allow_secret_capture` is set, and
reseeds a restored microVM with host entropy. A provider restore path must call
`_sanitize_restored_session`, which refreshes the transport and reseeds entropy,
because a checkpoint resumes a guest that may have stale transport state and
repeated entropy.

---

## Shared backend skeleton

Both provider backends split into four parts, following
`psrl/sandbox/backends/e2b.py`.

| Part | Owns | Depends on |
|---|---|---|
| Control client | Lifecycle HTTP calls. Create, pause, resume, snapshot, restore, delete | The provider lifecycle base URL and key |
| Data client | Commands and files inside one sandbox | The per sandbox execution endpoint |
| Session | One live sandbox. I/O and the state calls | Both clients |
| Backend | Capabilities, `create`, `connect`, `restore`, `delete_snapshot`, `shutdown` | A client factory and a state driver |

The split exists because the two planes fail differently and keep independent
connection pools. A create failure is a provisioning fault. A lost command
stream is a transport fault. Keeping them apart is what lets the manager classify
a failure without catching every provider error.

### Capability profiles

This is the normative declaration for all three backends. The provider pages
explain how a provider implements a feature, and they do not restate this table.

| Feature | Internal Docker | AgentEnv | OpenSandbox |
|---|---|---|---|
| `FREEZE` | yes, `docker pause` | no | no |
| `HIBERNATE` | no | yes | yes |
| `FILESYSTEM_SNAPSHOT` | yes | no | no |
| `FULL_STATE_SNAPSHOT` | no | yes | yes |
| `RESTORE` | yes | yes | yes |
| `RESUME_ANYWHERE` | yes | yes | yes |
| `resume_level` | `filesystem` | `full_state` | `full_state` |
| `NATIVE_FORK` | no | yes, one call for the whole group | no |
| `WARM_POOL` | yes, when a pool is configured | yes | yes |
| `IMAGE_ON_DEMAND` | no | yes | no |
| `IMAGE_BLOCK_DELIVERY` | no | no | yes |
| `TEMPLATE_BUILD` | no | yes | yes |
| `VOLUME` | no | no, the SDK's volume-mount shape is unpublished | yes |
| `HOST_MOUNT` | yes | no | no |
| `EGRESS_POLICY` | yes, when a host firewall is available | no | yes |
| `CREDENTIAL_INJECTION` | yes, into the sandbox environment | no | yes, brokered on outbound requests |
| `ISOLATION_RUNTIME` | yes, when a runtime is configured | no, Firecracker only | yes |

Each of the conditional entries is a node capability rather than a backend one, and
it is declared only where the node can honour it. A node without a host firewall
must not advertise `EGRESS_POLICY`, a node that cannot provide a gVisor or Kata
runtime must not advertise `ISOLATION_RUNTIME`, and a deployment that configured
no pool must not advertise `WARM_POOL`, because a spec that requires the feature
would otherwise be admitted into a sandbox that lacks it.

Three entries need their reason stated, because a reader would otherwise assume
the opposite.

- `FREEZE` is absent on both providers, because each provider's pause writes
  state to storage and releases compute. That is a hibernation. Only the
  internal backend has a freeze that keeps host memory resident, and that freeze
  does not survive a node loss.
- The internal backend declares no `FULL_STATE_SNAPSHOT`, because Docker commits
  the writable layer and has no supported memory checkpoint. This is why its
  `resume_level` is `filesystem` (D8).
- `HOST_MOUNT` is the internal backend only. A provider exposes storage as a
  volume, which is provider managed and not a host bind mount, so a spec that
  needs a host path is not portable and should say so at admission.

---

## Error mapping and provisioning ownership

| Provider condition | PSRL error |
|---|---|
| Create failed and left no sandbox | The backend's own error |
| Create failed but a sandbox may exist | `SandboxProvisionError` carrying the session |
| Command transport lost | `SandboxTransportError` subclass |
| Command exceeded its deadline | `TimeoutError` |
| Sandbox killed for memory | `SandboxOomError` when the provider proves it |
| Admission never granted | `SandboxCapacityTimeout`, raised by admission not the backend |

Both providers return a sandbox id from create, so a lost create response must
retain a cleanup target. Both accept metadata on create, so the idempotency key is
stamped into the metadata and a recovery lists by it.

:::{warning}
**Known limitation: the reclaim is AgentEnv only.** `AgentEnvBackend` has an HTTP
control plane, so `GET /sandboxes?metadata.psrl.idempotency_key=…` is a real call and
a lost create is reclaimed before the failure surfaces. The generic `E2BBackend` —
and `CubeSandboxBackend`, which subclasses it without overriding `create` — have only
the provider SDK: which SDK failures leave a sandbox behind cannot be enumerated
without a live endpoint, and the SDK's listing and pagination surface is version
specific. A reclaim written against a guess could miss orphans or delete a sandbox
that is not this worker's, and the second outcome is worse than the leak. Until a live
endpoint pins those failure modes, a lost create on those two backends leaks until the
provider's own idle timeout, so `no duplicate group can exist` is not yet true of
them. See {doc}`sandbox_agentenv`.
:::

A provider sandbox consumes no worker node capacity, so
`SandboxBackend.uses_node_capacity` stays false and admission is skipped. The
provider owns its own scheduling, which is the point of an external backend.

---

## Configuration

Register both backends declaratively under the existing sandbox config group, so
`build_sandbox_manager` instantiates them the same way it does Docker.

```yaml
sandbox:
  default_backend: docker
  backends:
    agentenv:
      _target_: psrl.sandbox.backends.AgentEnvBackend
      api_url: ${oc.env:AGENTENV_API_URL}
      api_key: ${oc.env:AGENTENV_API_KEY}
    opensandbox:
      _target_: psrl.sandbox.backends.OpenSandboxBackend
      config:
        api_url: ${oc.env:OPENSANDBOX_API_URL}
        api_key: ${oc.env:OPENSANDBOX_API_KEY}
```

Keep credentials in environment interpolation, never in a committed file. A
token is never a default, matching the repository path and secret rules.

---

## Backend selection by capability

A recipe declares requirements, and the manager rejects a backend that cannot
meet them.

```python
spec = SandboxSpec(
    source=SandboxSource.template("swe-task"),
    resources=ResourceSpec(cpu_count=4, memory_mb=8192, disk_mb=16384),
    required_features=frozenset({SandboxFeature.RESUME_ANYWHERE}),
    required_resume_level=ResumeLevel.FULL_STATE,
)
```

A recipe declares the level it needs and nothing about how a backend reaches it.
A run that only needs its workspace to survive asks for `FILESYSTEM` and works
on every backend. A run whose harness must keep running asks for `FULL_STATE`
and is rejected on the internal Docker backend at admission, which is the point.

---

## Conformance tests

Run the same suite against both providers, under their own environment flags.

| Case | Asserts | Runs on |
|---|---|---|
| Create, exec, read, write, terminate | The required data plane | every backend |
| Pause then resume then exec | State survives a hibernation cycle | `HIBERNATE` or `FREEZE` |
| Resume on a second host, workspace intact | `RESUME_ANYWHERE` at the `filesystem` level | every backend that declares it |
| Resume on a second host, a running process continues | `RESUME_ANYWHERE` at the `full_state` level | `full_state` backends only |
| Snapshot, restore, exec | Full state survives a restore | `FULL_STATE_SNAPSHOT` |
| Fork a prepared parent | `NATIVE_FORK` children are independent | `NATIVE_FORK` |
| Fork then sample entropy in each child | Children do not share a random stream | `NATIVE_FORK` |
| Retry a group acquire after a lost parent response | The parent is adopted and the children are re-forked, with no duplicate group | `NATIVE_FORK` |
| Lost create response | Provisioning ownership reclaims the sandbox by metadata | every backend |
| Command timeout and transport loss | Error classification is correct | every backend |
| Snapshot delete is idempotent | A second delete succeeds | `RESTORE` |
| Require a level the backend lacks | Admission rejects rather than downgrades | every backend |

The two resume cases are deliberately separate. The internal Docker backend
passes the first and must fail the second, and a suite that only had one case
would report the internal backend as evidence for a guarantee it does not
provide.

The lost create and lost parent cases need a fault injection point in the
control client, because neither can be triggered against a healthy provider.

---

## Decisions

| Id | Decision | Consequence |
|---|---|---|
| A1 | One session type with a capability set, not a separate stateful base class | The state methods keep their default raising implementations, and 6b and 6c add features rather than types |
| A2 | `RESUME_ANYWHERE` gates a restore | A restore that requires cross node resume is rejected at admission when the backend cannot provide it |
| A3 | The template build runs in `prepare`, with a build deadline | A failed build is an admission failure, not a rollout failure |
| A4 | This page is the only normative capability declaration | A provider page explains semantics and endpoints, and cannot drift from the declaration because it does not hold one |
| A5 | `RESUME_ANYWHERE` carries an ordered `resume_level`, and a spec requires a level | A `full_state` requirement is rejected on a `filesystem` backend instead of being satisfied by a weaker resume |
| A6 | A forked child takes the restore sanitization path | A child cannot reuse the parent's entropy, random stream, or transport state |
| A7 | `acquire_group` takes one spec per member and the fork parent belongs to no workflow | Per trajectory identity survives a group, and no member is admitted while holding capacity for a sibling |

:::{seealso}
- {doc}`sandbox_agentenv`: step 6b, the AgentEnv backend design
- {doc}`sandbox_opensandbox`: step 6c, the OpenSandbox backend design
- {doc}`sandbox_external_backends`: the capability and best-practice rationale
- {doc}`sandbox_refactor`: the thin control plane and the phases
- {doc}`sandbox_lifecycle`: the invariants every backend must preserve
:::
