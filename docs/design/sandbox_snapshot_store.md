# Shared Snapshot Store

This page designs the durable store behind the internal Docker backend's state
path, which is D8 in {doc}`sandbox_refactor`. It makes a filesystem snapshot
restorable on another node, so the internal backend can exercise pause, resume,
and cross node resume over the same contract the external providers serve. Base
image distribution is a separate question, answered here only because an earlier
version of this design put it in the same registry.

Read {doc}`sandbox_refactor` for the multi node design and {doc}`sandbox_lifecycle`
for the invariants.

---

## Scope

In scope.

- Make a `docker commit` image reachable from any sandbox node.
- Let `SandboxManager.restore` land on another node.
- Bound the lifetime of a snapshot and clean it up, both in the store and on the
  node.
- State what base image distribution does and does not get from this registry.

Out of scope.

- A base image cache service. The reasoning is below, and the mechanisms that
  replace it belong to placement and prefetch in {doc}`sandbox_refactor`.
- Memory and process state. Docker commits the writable layer and has no
  supported memory checkpoint.
- The external providers. Each persists its own snapshots natively, and this
  store does not replace that.
- A general artifact store for training data or checkpoints.

---

## Why node local is not enough

`DockerSession.snapshot` commits to the node's daemon and
`DockerBackend.restore` creates from that local image. The image lives on one
daemon, so a restore on another node has nothing to create from. That single gap
is what the store closes. Everything else about the current Docker state path
stays as it is.

---

## Store options

| Option | Transport | Dedup | Extra tools on a node | Service |
|---|---|---|---|---|
| OCI registry | `docker push` and `docker pull` | By digest. Shared base layers upload once | none, Docker suffices | one registry |
| Shared FS OCI layout | `skopeo copy` | By digest blobs | skopeo | no service |
| Shared FS image tar | `docker save` and `docker load` | none. Each snapshot is a full tar | none | no service |

A registry is the recommended option. Docker pushes and pulls it natively, so a
sandbox node needs no extra tool. Content addressing dedups the shared base
layers, so the first snapshot of an image uploads the base once and every later
snapshot uploads only its diff. Integrity is a digest check, and garbage
collection is a solved problem.

The shared filesystem options avoid a service, which suits the no external
platform rule in D2, at the cost of a tool on every node or a full tar per
snapshot.

---

## The store is a pull source, not a new protocol

The cheapest design reuses the create path. A snapshot is pushed to the registry
and the registry reference becomes the sandbox source. `DockerBackend.create`
already pulls a missing image when `auto_pull` is set, and
`_create_or_recover` already handles the 404 path, so a restore on another node
pulls through code that exists. No separate materialize step is needed.

The shared filesystem tar option does need an explicit materialize step, because
a create cannot load a tar. That is a second reason to prefer the registry.

---

## Key and naming

- Namespace per sandbox run, so one run cannot see or delete another run's
  snapshots.
- Repository `psrl-snapshots/<run-id>/<workflow-id>`.
- Tag the unique checkpoint id, and pin the digest in the ref.

`SnapshotRef` fields:

- `backend` is `docker`.
- `snapshot_id` is the digest reference `<registry>/<repo>@<digest>`.
- `metadata` carries `psrl.docker.image` (the pushed reference),
  `psrl.snapshot.namespace`, and the source spec fields the manager already
  records.

The digest, not the tag, is the durable key, so a tag move cannot change what a
restore gets.

---

## Checkpoint flow

`DockerSession.snapshot` with `SnapshotKind.FILESYSTEM`:

1. `docker commit <container> psrl/snapshot/<id>:<uuid>`. This exists today.
2. Tag the commit for the store repository.
3. Push it.
4. Read the digest from the push result.
5. Return a `SnapshotRef` whose id is the digest reference.
6. Leave the local tag as a cache entry, inside the local snapshot budget
   described under [Delete and GC](#delete-and-gc).

Idempotency. The checkpoint id is unique, so a retry pushes a new tag. A partial
push leaves an unreferenced manifest, which the store GC removes. Do not record a
snapshot until the push confirms a digest, so a checkpoint never returns a ref
that only one node can use.

Publish timing is an open question below. The default keeps the current node
local commit unless the spec requires `RESUME_ANYWHERE`, so the common case pays
no push cost.

---

## Restore flow

`DockerBackend.restore(snapshot, spec)`:

1. Read the registry digest reference from the snapshot.
2. Replace the spec source with `SandboxSource.image(digest_reference)`.
3. Call `create`, which pulls the image when it is absent.

The manager already adopts the new session and charges capacity on the node that
hosts it. A restore is a new sandbox with a new ref, so the caller must treat it
as new.

---

## Delete and GC

- Explicit delete removes the digest from the store and prunes the local image.
  `SandboxManager.delete_snapshot` already routes here.
- The store records the **intent to publish before the push**, as a provisional
  record keyed by the tag, and replaces it with the digest record when the push
  reports one. A push whose response was lost then leaves a record rather than an
  invisible manifest, so a store GC can still expire it.
- A store GC removes records past their retention, provisional ones included.
  That covers a crash between a push and its record, because there is no window
  in which a manifest exists with nothing pointing at it.
- A **run-scoped retention has no clock**, so the run ending is what expires it.
  `SandboxManager.end_run` is the signal, and the agent-loop worker calls it when
  it stops: its runs end with it.
- The record journal is **append only**, compacted when it outgrows the live
  records. A whole-file rewrite per publish made publishing n snapshots cost
  O(n²) bytes on the checkpoint path, which is work a checkpoint cannot afford.
  A crash can truncate the final line, and everything before it is intact.
- The node reclaimer already prunes local `psrl/snapshot` images by age, and it
  can carry the digest namespace.

### The local snapshot cache needs its own budget

A commit writes a full image layer to the node's local disk, and a pulled
snapshot writes another. Both land in the same Docker storage the base image
cache uses, and both are larger than a base image because they carry the
workspace. Leaving them to an age based prune means two things compete for one
disk with no policy, and the loser is whichever the prune reaches first. A node
that evicts its base images to hold snapshots then pays a pull on every create,
which is the opposite of what the cache is for.

So the local snapshot images get a budget of their own.

- A byte budget for the snapshot namespace, separate from the base image cache.
- Eviction by least recently used within that budget, not by age. A snapshot
  that a restore keeps reading is the one worth keeping, and its age says
  nothing about that.
- A restore that misses the local cache pulls from the store, so eviction is
  never a correctness event. That is what makes LRU safe here.
- Age remains the backstop for an orphan, since an unreferenced local commit
  from a crashed checkpoint has no access pattern to rank.

The budget is expressed as a fraction of the node's sandbox disk allowance, not
as an absolute size, because an operator can estimate the split between base
images and snapshots and cannot estimate the bytes.

The store side TTL and this local budget are independent. The store bounds how
long a snapshot is restorable, and the budget bounds how much disk a node spends
caching one.

---

## Failure modes

| Condition | Behavior |
|---|---|
| Store unreachable at checkpoint | Fail the checkpoint. Never return a node local ref that a restore elsewhere cannot use |
| Push fails partway | Leave an unreferenced manifest for GC, and do not record a snapshot |
| Pull fails at restore | Raise. Use `SandboxProvisionError` when a container may exist |
| Digest mismatch | Integrity failure. Do not create from the image |
| Snapshot missing at restore | Fail the restore, not the workflow |

---

## Security

- Reuse the backend's `registry_auth` for the store registry.
- `SandboxStatePolicy` still gates capture. A snapshot persists whatever the
  writable layer holds, so the existing secret environment and post-command
  checks stay the guard.
- Namespace snapshots per run, so a restore cannot cross runs.

---

## Base images are not in this registry

An earlier version of this design ran the same registry as a pull through cache
for base images. Two facts make that not work as stated, and they are the reason
base image distribution is a separate question from snapshot storage.

- **A Docker daemon mirror only covers Docker Hub.** `registry-mirrors` does not
  redirect a pull from another registry, which the Dragonfly project states
  plainly as the reason it offers a proxy instead: Docker does not support
  private registries with `registry-mirrors`. A task corpus hosted on `ghcr.io`
  or a private registry is therefore untouched by a mirror, and that is most of
  a coding agent corpus.
- **A `distribution` proxy caches exactly one upstream.** `proxy.remoteurl` is a
  single endpoint, and the proxied namespace is read only, so it can neither
  cover several upstreams nor accept a snapshot push.

Covering every registry therefore costs either a rewrite of every image
reference or a proxy on every node. Neither is justified before a measurement
says image pull is the bottleneck, and the internal Docker backend is the test
path (S8). So the baseline is no base image cache service at all.

### Baseline, no cache service

- The snapshot registry serves snapshots only, under `psrl-snapshots`.
- Base images are pulled from their canonical references, and the two mechanisms
  that keep that cheap are already in the plan. Image aware placement sends a
  task to a node that already holds the digest, and a per run prefetch warms the
  run's working set on the nodes placement is likely to choose. See
  {doc}`sandbox_refactor`.
- The honest cost is that an upstream is hit once per node per image rather than
  once per cluster. Locality is what keeps the node count small, and
  `image/locality_hit_ratio` and `image/pull_s_p95` are what show whether it is
  working.

### The upgrade, a node local peer proxy

When those metrics show pull cost matters, the upgrade is a node local peer
proxy such as Dragonfly, not a registry role.

- Each node runs the peer daemon, and the Docker daemon is pointed at it as an
  HTTP and HTTPS proxy with a rule scoped to layer blobs. Registries served over
  HTTPS need the proxy's certificate, which is why this is a node side change
  and not a configuration line.
- It covers **every** registry, because it intercepts the pull rather than
  redirecting a name. No image reference changes anywhere in PSRL, so placement
  keeps indexing canonical digests and provenance stays intact.
- It delivers blobs from peers, so a layer crosses the origin roughly once for
  the cluster. That is block level peer delivery, the same shape OpenSandbox
  gets from its own delivery daemon, and it is the closest thing to DSec's
  efficiency that plain Docker Engine allows.
- Push must be excluded from the proxy rule, so a snapshot push still reaches
  the snapshot registry directly.

The alternative, a proxy cache with rewritten references, is rejected. One
Harbor with a proxy cache project per upstream does cover every registry, but
every image reference becomes a mirror reference, so the canonical digest has to
be carried separately for locality and provenance, and one more thing can be
wrong in the reference PSRL hands a task.

---

## Deployment topology

The store has one role, snapshots. They are written on a checkpoint and read on
a restore, and they want mutable, TTL scoped storage.

One writable `distribution/registry` serves it (S6). There is no proxy role to
separate it from, because base images are not in this registry. A registry in
pull through cache mode is read only for the proxied namespace and could not
accept a push anyway, which is the second reason the two roles were never going
to be one deployment.

Harbor is the alternative when the operator already runs it and wants its GC,
retention, and UI in one place. It is a heavier stack for one writable
repository namespace.

### Storage backend

Object storage, S3 or MinIO compatible, backs the registry (S7). A replica is
then stateless and can be added or replaced freely, which is what makes the N
replica shape work. The external providers already persist snapshots to object
storage, so the internal and external paths share one storage idiom.

An object store lifecycle rule is the backstop GC, so a bucket that outlives its
registry still expires old data.

### Replicas and reachability

- Run N replicas behind one endpoint, a VIP or a load balancer. With object
  storage the replicas are interchangeable.
- Every sandbox node's Docker daemon must reach the endpoint, and a node that
  cannot is not a placement candidate for a spec that requires
  `RESUME_ANYWHERE`. Add the registry to `insecure-registries` only if it serves
  plain HTTP on a trusted network.
- A cloud burst node needs the same reachability, or it cannot participate in a
  cross node restore.

### Placement

- Put the registry on the sandbox cluster's fast network, next to the daemons. A
  snapshot push and pull sits on the checkpoint and restore path, so it wants
  low latency and high bandwidth to the object store.
- Do not place it on the GPU training nodes. It is I/O heavy and would compete
  with the trainer.

### Authentication

- Reuse the backend's `registry_auth`, with a per run credential that can push
  and pull.
- Namespace snapshots per run, so a credential cannot read another run's
  snapshots.

### Garbage collection

- The TTL GC deletes a snapshot manifest by digest, which is cheap and does not
  need the registry offline.
- A mark and sweep GC reclaims orphaned blobs. The `distribution` GC runs with
  the registry in read only mode, so schedule it on the registry host with a
  short read only window, or use Harbor's job when Harbor is the deployment.
- Add an object store lifecycle rule as a backstop, so a bucket that outlives its
  registry still expires old data.

### Failure modes

| Condition | Behavior |
|---|---|
| Snapshot registry unreachable | A checkpoint that needs publish fails, and a cross node restore fails. Never fall back to a node local snapshot |
| Object store unreachable | The registry reports unhealthy. Checkpoints fail and restores fail |
| A base image upstream unreachable | Only a node that has not cached the digest is affected, and locality is what keeps that set small. There is no cache service to fail |

---

## Configuration

```yaml
sandbox:
  backends:
    docker:
      _target_: psrl.sandbox.backends.DockerBackend
      snapshot_store:
        registry: 192.168.1.x:5000
        snapshot_namespace: psrl-snapshots
        publish: on_demand
        retention: one_run
        local_cache_fraction: 0.3
```

Base images have no key here, because the baseline has no cache service. The
peer proxy upgrade is a node side deployment and a daemon setting, not a PSRL
config group, which is part of why it can be adopted without touching a recipe.

Credentials come from environment interpolation, matching the repository secret
rules.

Two of these keys are deliberately not numbers.

- `retention` is an intent, and the TTL follows from it. `one_run` expires a
  snapshot when its run ends, `one_day` and `one_week` are the longer
  choices, and an explicit `ttl_s` stays available for a deployment that needs
  it. An operator knows how long a snapshot should outlive its run and does not
  know the right number of seconds.
- `local_cache_fraction` is the share of the node's sandbox disk allowance the
  snapshot cache may hold, so the remainder is the base image cache. A fraction
  is estimable and a byte count is not.

---

## Capability declaration

The declaration is in {doc}`sandbox_backend_integration`, which is the only
normative capability table. The store is what lets the internal backend declare
`RESUME_ANYWHERE` at all, and the level it declares is `filesystem`.

Docker commits the writable layer and has no supported memory checkpoint, so a
restore on another node brings back the workspace and nothing else. A resumed
container starts new processes over the restored filesystem, which is why the
level exists and why a caller that needs a running process to survive must
require `full_state` and land on a provider instead. `FREEZE` remains the in
place `docker pause`, which keeps host memory resident and does not survive a
node loss, so it is not a cross node mechanism.

---

## Relationship to DSec

DSec does not use a registry. Its container backend stores base images,
workspaces, and toolkits on a shared distributed filesystem, converts images to
EROFS offline, separates metadata from data so metadata stays local, and loads
image blocks on demand. It patches dockerd to compose overlayfs layers from
independent EROFS layers, and it keeps snapshots as incremental disk snapshots.
The report rejects the registry and peer to peer approach on purpose, because
sandboxes access only a small fraction of their image data.

This design differs on purpose. The internal Docker backend is the local
distributed test backend (D2), so it optimizes for contract conformance and
operational simplicity, not for DSec's density and cold start. A registry is the
established, Docker native way to make an image reachable cluster wide, and it
reuses the create and auto pull path that already exists.

The DSec capabilities are not lost. They live in the external providers that PSRL
integrates. AgentEnv loads images on demand through overlaybd and ublk, and
returns memory to the host with ballooning for density. It is also the project
that hosts the storage component DSec open sourced. OpenSandbox delivers images
as content-addressed artifacts with block-level peer delivery and a node cache,
which avoids a whole origin pull on every node. So the DSec style efficiency
path in PSRL is AgentEnv and OpenSandbox, and the internal Docker backend is not
expected to reach it (S8).

---

## Decisions

| Id | Decision | Consequence |
|---|---|---|
| S1 | The store is an OCI registry | Docker native push and pull, digest dedup, and no per node tool |
| S2 | Publish on demand, only when a spec requires `RESUME_ANYWHERE` | A node local checkpoint with no cross node intent pays no push |
| S3 | The registry serves snapshots only. Base images rely on image aware placement and per run prefetch, with no cache service | A daemon mirror covers only Docker Hub and a `distribution` proxy covers one upstream, so a real cache costs either rewritten references or a node side proxy. Neither is taken before a measurement asks for it |
| S4 | GC is TTL driven | A crash between a push and its record cannot leak a snapshot forever |
| S5 | Snapshots are namespaced per run | Coarse and simple to GC, and a restore cannot cross runs |
| S6 | The topology is one writable registry, replicated, on one object store | There is no proxy role to separate, because base images are not in this registry |
| S7 | The registry storage backend is object storage, S3 or MinIO compatible | A replica is stateless, so the N replica shape holds, and the internal and external paths share one storage idiom |
| S8 | The registry is the internal Docker backend's end state, not a bridge to on demand layers | The internal backend is the test path. DSec style on demand efficiency lives in the external providers, not here |
| S9 | The local snapshot cache has its own byte budget with LRU eviction, separate from the base image cache | A node cannot evict its base images to hold snapshots. A local miss pulls from the store, so eviction is never a correctness event |
| S10 | The store declares `RESUME_ANYWHERE` at the `filesystem` level | A caller that needs a live process to survive a move is rejected here and lands on a provider instead |
| S11 | Retention is an intent and the local budget is a fraction | An operator sets how long a snapshot outlives its run and what share of disk it may hold, not a TTL in seconds or a size in bytes |
| S12 | The base image upgrade path is a node local peer proxy, not a mirror registry, and it is adopted on metric evidence | It covers every registry without rewriting a reference, and it adds peer block delivery. Rewritten mirror references are rejected because the canonical digest would have to be carried separately |

:::{seealso}
- {doc}`sandbox_refactor`: the multi node design, D8, and the resumable rollout
- {doc}`sandbox_backend_integration`: the capability model the internal backend also follows
- {doc}`sandbox_lifecycle`: the invariants a snapshot and restore must preserve
:::
