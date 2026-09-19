# Docker sandbox review and bounded preparation

Review date: 2026-09-17. Scope: the current working tree, including existing
uncommitted Docker lifecycle, capacity, and harness changes. This is a source
review and CPU regression result, not a production capacity certification.

## Architecture decision

Keep rollout and grading in separate containers. Reuse immutable image layers
and prepared images, and overlap bounded image preparation with rollout. Do not
reuse the rollout writable layer, processes, or resource lease for grading.
The user explicitly selected this design during review.

The boundaries remain:

- `DockerEngineClient`: asynchronous Engine transport and wire validation.
- `DockerBackend` / `DockerSession`: Docker policy, image preparation, command
  lifetime, snapshots, and container lifecycle.
- `DockerLifecycle` / `docker_utils`: worker leases and independent crash GC.
- `SandboxManager` / `SandboxLease`: backend selection and CPU/memory admission.
- Agent loops: task dependencies, preparation overlap, grading, and final cleanup.

The transport protocol is a useful fake-engine testing seam. Typed policy
dataclasses are useful configuration boundaries. Moving node admission into
Docker utils, or adding a generic warm-container pool, would obscure ownership
and make safe grading harder rather than simplify this design.

## Findings addressed

| Priority | Finding | Change |
| --- | --- | --- |
| P1 | Pull treated HTTP 200 as success even when Docker emitted an error inside its JSON stream. | Parse progress incrementally and raise on `error` or `errorDetail.message`. |
| P1 | Exec output could grow without bound in the worker process. | Add a 16 MiB default frame budget; close the response and terminate the sandbox on overflow. |
| P1 | Lost exec streams could leave container processes running; absent exit codes defaulted to success. | Terminate on transport failure and require a final exit status. |
| P1 | Command timeout covered only stream attachment, excluding exec setup and inspection. | Apply a deadline around the complete engine exec operation. |
| P1 | GC scanned all PSRL containers, even when collectors used different heartbeat directories. | Label and filter by lease-store namespace. |
| P1 | A heartbeat permission or I/O error was treated as a dead owner. | Only missing files mean missing leases; other errors log and skip deletion. |
| P2 | CLI cleanup omitted anonymous volumes. | Use `docker rm -f -v`, matching Engine cleanup's volume behavior. |
| P2 | Every snapshot of one container reused `latest`, changing older snapshot references. | Generate a unique snapshot tag per capture. |
| P2 | Missing-image creates and speculative preparation could duplicate pulls. | Share concurrent preparation per image and bound distinct pulls per backend. |
| P2 | Generic harness committed a Docker container before any preparation work. | Reuse the original cached image; avoid per-trajectory empty baseline commits. |
| P2 | Filesystem capacity checks and lifecycle startup ran directly on the event loop. | Run these blocking operations in the thread executor and bound the polling sleep by its deadline. |
| P2 | Cleanup failure could replace a container-start failure. | Preserve the original exception and log cleanup failure. |
| P2 | Lifecycle callbacks could retain closed backend objects. | Unregister the atexit callback after close. |
| P2 | GC lock metadata failure could leak the lock file descriptor. | Close the handle on acquisition/metadata exceptions. |

GC now stats one heartbeat per distinct owner instead of once per container.
Shutdown drains image tasks and closes the Engine client even if lifecycle
cleanup fails. Completed preparation entries are discarded, so image deletion
outside PSRL does not leave an indefinitely stale Python cache.

## Style and unnecessary abstraction

PSRL's Python rules apply here: Ruff, 119 columns, snake_case, explicit typed
interfaces, and component logging. SMG's inspected local contribution guides
support the same ownership and error-context principles; its Rust-specific
100-column formatter, `tracing`, and `thiserror` rules do not apply literally to
these Python files.

The original modules are not fully compliant with PSRL's prose conventions:
many existing docstrings open on the summary line rather than their own line.
This review avoids a large unrelated formatting rewrite. New explanatory
docstrings follow the local guide, and modified files pass Ruff.

The owner-hash helper existed only to wrap one expression and was inlined.
Repeated exec cleanup paths were combined. The new preparation completion
callback is retained because it owns task-map removal and retrieves exceptions
after a caller cancels. The GC sweep/list/lock boundaries are useful fault and
testing boundaries, not simply excessive small functions.

Defensive checks for immutable idempotency specs, snapshot secret capture,
unsupported hibernation, and explicit resource limits are substantive safety
contracts and should stay. Catching cleanup errors is appropriate only when the
original failure remains visible or a later cleanup path still owns the resource.

The superseded `psrl/sandbox/budget.py` and its resolver tests were deleted.
Searches found that module imported only by its own test file, with no runtime
consumer under `psrl` or `examples`; `capacity.py` owns node admission now. The
resolver tests also referenced `resolve_sandbox_pool_limits`, a function that no
longer exists anywhere, so they could not pass even in isolation.

## RL workflow and overlap

```text
task spec ready
  +-- rollout image prepare ----+
  +-- TITO session create ------+--> acquire rollout capacity/container
  +-- grader image prepare -----------------------------+
                                    harness prepare     |
                                    rollout             |
                                    extract artifact    |
                                    release rollout     |
                                    acquire grader <----+
                                    grade in fresh container
                                    release grader
```

`SandboxManager.prepare(spec, backend=...)` warms reusable artifacts without
acquiring a container lease. Backends without a preparation implementation
inherit a no-op. Docker validates the image cache and pulls only when needed.
The generic harness overlaps rollout image warming with session creation and
grader image warming with rollout. The MiniSWE v1 loop also warms the grader
image while its synchronous runner executes.

This does not claim to overlap a command with the creation of its own container:
`harness.prepare` requires the container to exist. Preparation of one trajectory
can overlap work in other trajectories. No dataset-wide prefetch scheduler or
ready-container pool was added. Speculative warming reserves disk/network work,
not a second container's CPU/memory, avoiding rollout/grader capacity hold-and-wait.

Docker's image layers already provide reusable immutable data with separate
writable layers. Commit does not include mounted-volume contents and normally
pauses the container, so a commit before setup adds cost without caching setup.
See [Docker storage drivers](https://docs.docker.com/engine/storage/drivers/)
and [container commit](https://docs.docker.com/reference/cli/docker/container/commit/).
Full-state microVM snapshots keep their existing path; explicit Docker filesystem
snapshots remain available to callers that own deletion.

## Scale boundaries and remaining work

The implementation is suitable as a bounded node-local building block, not an
unqualified large-cluster scheduling service. Outstanding limits are explicit:

1. **Node-wide image admission:** the default two pulls are per backend. With W
   worker backends, the node can have up to 2W downloads. Connection pools are
   also per backend. Use node-local daemons and preloaded digest manifests; a
   shared image-distribution service is a separate scaling step.
2. **Crash ownership:** standalone backends without `PSRL_ACTOR_ID` have no owner
   heartbeat. A create whose server-side result is lost before receiving an ID
   relies on owner GC. SIGKILL recovery for explicit committed images is not
   implemented by the container collector. The default Docker workflow now
   avoids producing those per-trajectory images.
3. **GC deployment:** use the same absolute heartbeat mount path, TTL policy,
   and Docker endpoint within a namespace. Use separate heartbeat directories
   for distinct endpoints. Old collectors do not honor the new namespace label;
   stop them during upgrade, and explicitly handle older unlabeled containers.
   The detached collector is not an externally supervised node service and
   competing workers can still launch short-lived lock-losing collector processes.
4. **Memory/disk:** the exec limit bounds retained wire output, not total process
   memory; demultiplexing and decoding make additional bounded copies. File
   archives still materialize in memory and tar work remains synchronous.
   Disk headroom is a check, not a reservation or a quota. Container logs and
   accumulated image layers require daemon-level retention policy.
5. **Manager lifecycle:** shutdown still waits for in-flight acquisitions; a
   capacity request blocked indefinitely by another healthy owner can delay
   shutdown. Session expiry/exec termination also relies on callers releasing
   their manager lease to return accounting capacity. Follow-up work should
   define a bounded manager-wide drain and track connect/restore races as well.
6. **Snapshot API:** generic `branch(..., FILESYSTEM)` currently restores without
   passing the required Docker creation spec, and immediate temporary-image
   deletion needs an ownership contract for live child containers. This generic
   branch path is not used by the implemented grading workflow and is not
   claimed as validated Docker functionality.

CPU/memory limits must be configured together with the node admission envelope;
Docker itself imposes no such resource bounds by default. Configure log rotation
on the daemon (Docker recommends the rotating `local` driver for general use),
and export pool waits, pull latency/failures, disk headroom, GC failures, daemon
RSS, and container counts. See [Docker resource constraints](https://docs.docker.com/engine/containers/resource_constraints/)
and [logging configuration](https://docs.docker.com/engine/logging/configure/).
Anonymous-volume cleanup follows [Docker rm semantics](https://docs.docker.com/reference/cli/docker/container/rm/).

## Verification

Tests cover pull-stream errors, output overflow, final exec status, end-to-end
command deadlines, transport failure cleanup, pull sharing/cancellation,
concurrency bounds, shutdown draining, immutable snapshot references, namespace
filtering, lease I/O failure, and real harness-method scheduling with unrelated
Ray/Torch base classes replaced by test doubles.

The final targeted suite passed all 98 tests. Ruff lint and format checks pass
for the modified Python files, and both staged and unstaged `git diff --check`
checks pass. The wider sandbox, harness-preparation, harness-config and
integrity suites pass 138 tests. A disabled sandbox config is now unsupported
rather than a `None` manager: `SandboxManagerConfig.default_backend` is
required, and the test that still asserted the old `None` return was removed.

Reproduce the targeted suite with:

```bash
.venv/bin/python -m pytest -q \
  tests/sandbox/test_docker_backend.py tests/sandbox/test_docker_engine.py \
  tests/sandbox/test_docker_lifecycle.py tests/sandbox/test_docker_utils.py \
  tests/sandbox/test_docker_reliability.py tests/sandbox/test_manager.py \
  tests/sandbox/test_sync.py tests/sandbox/test_mini_swe_runner.py \
  tests/agent_loop/test_harness.py tests/agent_loop/test_harness_preparation.py
```

This machine has no Docker CLI/socket; live Docker and microVM tests remain
opt-in and were skipped. No container churn, SIGKILL/daemon-restart, cross-process
pull admission, or production RL throughput numbers are claimed. Before large
deployment, run the existing live conformance and benchmark scripts on the
target node, including cold/warm images and cancellation under saturation.
