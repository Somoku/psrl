# External Sandbox Backends

This page designs AgentEnv and OpenSandbox as first class PSRL sandbox backends.
The goal is to expose each provider's strongest capabilities through the shared
contract, and to stop PSRL from reimplementing what the provider already does.

It is the provider half of {doc}`sandbox_refactor`, which sets the thin control
plane and the migration order. Read {doc}`sandbox_lifecycle` for the invariants
these backends must preserve. {doc}`sandbox_backend_integration` holds the shared
capability model, and {doc}`sandbox_agentenv` and {doc}`sandbox_opensandbox` hold
the two backend designs.

---

## What the contract must express

PSRL must be able to ask for a provider's best mode and to refuse a backend that
cannot deliver it. `SandboxFeature` therefore grows the provider capabilities,
so a plan or a recipe declares requirements and the manager rejects a mismatch
instead of silently degrading.

The feature names, their meanings, and which backend declares each one are in
{doc}`sandbox_backend_integration`, which is the only normative capability
table. This page explains what each capability buys an RL run.

Two properties drive the rest of the design.

- **`RESUME_ANYWHERE` is what makes preemption survivable**, so a caller must be
  able to require it. It carries an ordered level, because a resume that
  restores a disk and a resume that restores a live process are different
  guarantees.
- **The image capabilities are not interchangeable.** Lazy layer loading and
  block delivery from a cache and peers both avoid a whole origin pull on every
  node, and only the first is lazy at layer granularity. A spec that needs one
  must not select the other by accident.

---

## Native operation map

| PSRL operation | AgentEnv | OpenSandbox |
|---|---|---|
| Create | Template or image, snapshot backed | `Sandbox.create`, or a warm pool claim |
| Exec and files | Native exec and file API | `commands.run` and the files API |
| `pause(FREEZE)` | In place pause | Not native, use the FastSandbox path |
| `pause(HIBERNATE)` | Snapshot then release compute | FastSandbox checkpoint |
| `resume` | In place, or on another host | On any host |
| `snapshot(FULL_STATE)` | Memory and filesystem snapshot to a shared store | Artifact store checkpoint |
| `restore` | On any host | On any host |
| `fork(NATIVE_FORK)` | Native fork | Not native |
| Image | Overlaybd layers load on demand | Content-addressed artifacts with block-level peer delivery and a node cache |
| Egress | Provider policy | Egress controls |
| Secrets | Provider | Credential vault |
| Isolation | Firecracker | gVisor, Kata, or Firecracker |

---

## AgentEnv backend

`backends/e2b.py` already drives AgentEnv with an E2B compatible data plane. The
sandbox surface, create included, goes through the provider's own SDK:
`AgentEnvClientFactory` calls `Sandbox.create` and `Sandbox.connect`, and
`AgentEnvStateDriver` pauses, snapshots, and forks through the same SDK. The data
plane stays E2B compatible because AgentEnv exposes that API for commands and files.

So the move to the native API is mostly done. The work is to declare the extended
features and close the gaps in {doc}`sandbox_backend_integration`, not to rewrite
the backend. Keep the E2B path only for true E2B servers.

### Capabilities to expose

- **Snapshot backed templates.** A task image plus a warm snapshot as the source,
  so a cold start restores a snapshot instead of booting a fresh guest.
- **Pause and resume.** Fast in place pause and resume, with a separate
  hibernate that checkpoints and releases compute.
- **Snapshot to shared storage.** The snapshot persists to object storage or a
  shared filesystem, so a restore can land on any node. Declare
  `FULL_STATE_SNAPSHOT`, `RESTORE`, and `RESUME_ANYWHERE` together.
- **Native fork.** One prepared sandbox branches into independent sandboxes.
  Declare `NATIVE_FORK`.
- **On demand image loading.** Overlaybd layers load lazily, so a node does not
  pre-pull the whole image set. Declare `IMAGE_ON_DEMAND`.
- **Memory overcommit.** The provider returns reclaimable guest memory to the
  host, so density does not collapse as sandboxes diverge.

### How the RL workflow uses it

- **Fork for the group.** Prepare the environment once, then fork one child per
  GRPO group member. Setup runs once, not once per sample. This is the largest
  win the backend enables and it needs `NATIVE_FORK`.
- **Pause idle sandboxes.** A sandbox waiting on the model or the next turn is
  paused, so compute is not pinned while the agent thinks.
- **Resume on any host.** After a preemption, the trajectory resumes on whatever
  node has capacity. This is the mechanism that lets a stateful rollout outlive
  the preemptible trainer.
- **Do not pre-pull.** Rely on `IMAGE_ON_DEMAND` instead of seeding every node.

---

## OpenSandbox backend

OpenSandbox is the provider that gives the same API on a laptop and on a
cluster. Add it as a backend against its published OpenAPI lifecycle and
execution contracts.

### Capabilities to expose

- **Docker and Kubernetes runtimes behind one API.** The same backend serves both
  deployments, chosen by provider configuration.
- **FastSandbox checkpoint and resume.** State checkpoints to the artifact store
  and all compute is released. A resume lands on any host. Declare
  `HIBERNATE`, `FULL_STATE_SNAPSHOT`, `RESTORE`, and `RESUME_ANYWHERE`.
- **Warm pool.** Pre-warmed Firecracker sandboxes are claimed with a near
  constant admission time. Declare `WARM_POOL`.
- **Isolation runtimes.** gVisor, Kata, and Firecracker, selected per sandbox.
  Declare `ISOLATION_RUNTIME`.
- **Egress and credential vault.** Per sandbox outbound controls and a credential
  broker that keeps real secrets out of the workload: the value is written to the
  provider's sidecar and the sandbox holds a placeholder, so the secret is attached on
  the way out rather than placed in the environment. Declare `EGRESS_POLICY` and
  `CREDENTIAL_INJECTION`.
- **Open protocol.** The lifecycle and execution APIs are public contracts, so a
  future runtime can plug in without a client change.

### How the RL workflow uses it

- **One backend for both loops.** The developer runs the recipe locally against
  Docker and the cluster run uses Kubernetes, with no code change.
- **Warm pool on the critical path.** Admission claims a warm sandbox instead of
  creating one, which removes cold start from the rollout step.
- **FastSandbox for preemption.** A preempted job checkpoints its sandboxes and
  resumes them later, possibly on other nodes.
- **Delegate security.** Egress and secret handling come from the provider, so
  PSRL does not implement Phase 5 twice.

---

## Image distribution

For external backends, PSRL stops owning image strategy.

- The backend advertises `IMAGE_ON_DEMAND` or `IMAGE_BLOCK_DELIVERY`, and the
  caller passes an image or template reference.
- PSRL keeps its local cache and locality strategy only for the internal Docker
  backend, where it still matters.
- A recipe that needs a fast cold start declares `WARM_POOL` and fails at
  admission when the backend cannot provide it, rather than degrading quietly.

---

## RL workflow integration

The lesson from the DeepSeek sandbox report is that rollout state must outlive
the trainer. A preempted GPU job must be able to reconnect and continue, with
the sandbox holding the state.

With `RESUME_ANYWHERE`, PSRL can reach that shape.

- On preemption, the trainer sends pause to the sandboxes of the preempted
  workflow. The provider releases compute and keeps state.
- The next request to a paused sandbox resumes it, on any node.
- The sandbox plus a scaffold neutral worker hold the rollout state as the single
  source of truth, so the trainer holds no recovery logic.

This needs a resumable agent loop and harness, which is a separate concern from
the backend. The backend contract enables it. The loop change is not part of this
page.

---

## Migration

| Step | Change | Depends on |
|---|---|---|
| 6a | Add the capability model and wire the state operations through `SandboxManager` | Phase 0 |
| 6b | Declare the AgentEnv extended feature set, add the fork count, and own provisioning recovery | 6a |
| 6c | Add the OpenSandbox backend against its OpenAPI contract | 6a |
| 6d | Let a recipe select a backend by capability profile, and reject a mismatch at admission | 6b, 6c |

The state drivers in `backends/e2b.py` are the starting point for 6a. The work is
to make every provider capability a declared feature and to make the manager
enforce the declaration.

---

## Decisions

| Id | Decision | Consequence |
|---|---|---|
| E1 | A recipe declares a capability profile, with an optional backend pin | The manager selects by capability and fails fast on a mismatch. A pin overrides the choice |
| E2 | The E2B compatible data plane is kept, and the control plane is native | AgentEnv keeps the supported Miles path, and state operations stay native |
| E3 | A credential vault credential comes from environment interpolation | The value is read from this process and written to the provider's vault, and only sanitized metadata is ever read back. The sandbox holds a placeholder, so the secret is brokered on outbound requests and never lives in the workload |
| E4 | The provider owns pause and resume, and PSRL adds a thin workflow pause hook later | The preemption protocol lands with the loop work, not the backend |

:::{seealso}
- {doc}`sandbox_backend_integration`: the shared capability model for step 6a
- {doc}`sandbox_agentenv`: the AgentEnv backend design for step 6b
- {doc}`sandbox_opensandbox`: the OpenSandbox backend design for step 6c
- {doc}`sandbox_refactor`: the thin control plane, the phases, and the multi node design
- {doc}`sandbox_lifecycle`: the invariants these backends must preserve
- {doc}`sandbox_architecture`: the current backend and state driver layout
:::
