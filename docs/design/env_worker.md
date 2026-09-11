# EnvWorker

The EnvWorker subsystem provides a managed pool of persistent sandbox containers for
agentic RL tasks that require a stateful execution environment. It is designed for tasks like
AIRS-Bench where the agent must run code across multiple turns and the environment state, such
as installed packages, intermediate files, and trained model checkpoints, must persist for the
duration of the episode.

---

## Motivation

Standard tool-use in PSRL invokes a stateless function per turn. That model works for tools
that look up facts, execute arithmetic, or call external APIs, but it fails for tasks where
the environment accumulates state. An ML training task, for example, expects to call a data
preprocessing script in turn 2 and read its output in turn 3. Stateless exec re-runs the
environment from scratch on every call and cannot satisfy this requirement.

The requirements that drive the design are:

1. **Persistent shell.** The container must run a single long-lived `bash -l` session so that
   the working directory, exported variables, and the active conda environment survive across
   agent turns.
2. **Placement separation.** On heterogeneous clusters some nodes are better suited for
   sandbox workloads than for GPU training. Sandboxes must be pinnable to specific nodes.
3. **Capacity accounting.** Each node has a finite number of concurrent sandboxes it can
   host. The coordinator must admit sandboxes within capacity and queue excess requests.
4. **GPU pass-through without the NVIDIA Container Toolkit.** Clusters that lack
   `nvidia-container-runtime` need a different path to expose GPUs to containers.
5. **Observation safety.** A training run inside a container can emit megabytes of log text.
   The worker must cap output before returning it to the agent.

---

## Component Triple

The subsystem is built from three components with separated concerns.

```
EnvWorkerManager (construction, placement)
    |
    +-- EnvWorkerCoordinator  (routing, capacity accounting)
    |       |
    |       +-- WorkerSlot × N  (per-worker bookkeeping)
    |
    +-- EnvWorker × N  (container lifecycle, shell I/O)
            |
            +-- ContainerShell × M  (one per live sandbox)
```

### EnvWorker (`worker.py`)

Owns the sandbox containers scheduled onto one placement unit. The class is plain Python with
no Ray dependency, so argument construction and shell mechanics stay unit-testable without a
running cluster. `EnvWorkerManager` wraps it with `ray.remote` at startup using a resource
request built from configuration.

Each `EnvWorker` tracks:

- The `CUDA_VISIBLE_DEVICES` range granted by Ray, from which it draws GPU indices for
  sandboxes. This makes it impossible for a sandbox to touch a GPU Ray did not assign.
- A `ContainerShell` per live sandbox, keyed by `sandbox_id`.
- A last-used timestamp per sandbox for the idle reaper.

Key operations: `create_sandbox`, `exec`, `read_file`, `write_file`, `destroy_sandbox`,
`reap_idle_sandboxes`.

### EnvWorkerCoordinator (`coordinator.py`)

Routes sandbox requests to workers and tracks their capacity. The coordinator is a Ray actor
with `max_concurrency` set from configuration to allow concurrent RPCs.

It holds a `WorkerSlot` per registered worker (populated by `EnvWorkerManager` at startup).
Each slot carries the live counts of occupied CPU and GPU slots. The routing logic in
`select_by_method` filters to slots with available capacity, then applies the configured
policy.

The `create_sandbox` method retries once on a different worker if the first attempt fails.
This covers the common case where the container image is not yet loaded on one node.

The coordinator is also the release target for `SandboxHandle.destroy`. After the worker
destroys the container, the handle calls `coordinator.release` to return the capacity
accounting, ensuring the coordinator never gets out of sync with the actual load.

### EnvWorkerManager (`manager.py`)

Constructs the pool at PSRL startup and owns the coordinator and worker Ray actors for one
training job. Its responsibilities are limited to startup and shutdown: it reads the
`psrl.env_worker` config, resolves which nodes receive workers via the placement policy, and
creates one coordinator actor and one or more worker actors per node.

---

## Placement Policies

The `psrl.env_worker.placement` key selects the node assignment policy. Switching between
policies requires only a config change. No code change is needed.

### `colocated`

Workers are created on every alive Ray node. The env node list is the full output of
`ray.nodes()` filtered to `Alive=True`.

Use this when the cluster has homogeneous nodes and sandbox CPU workloads can share resources
with train and rollout GPUs. It is simpler to operate.

### `dedicated`

Workers are created only on the nodes listed in `psrl.env_worker.dedicated_node_ips`. The
manager validates that every listed IP appears in the alive Ray node set and raises
`ValueError` if any are missing.

Use this when the cluster has nodes specialized for sandbox workloads (such as high-CPU, no
GPU), or when you want to fence sandbox I/O away from the training and rollout GPUs. The
AIRS-Bench recipe uses this policy: the env node `28.49.16.220` hosts only env workers, and
`total_nnodes=1` prevents PSRL from scheduling any other actor on it.

---

## Sandbox Handle Lifecycle

The caller (an agent loop or environment) never talks to the `EnvWorker` directly. It holds a
`SandboxHandle` that hides which node hosts the sandbox.

```
create_sandbox(spec)         coordinator selects a worker, increments used_cpu/used_gpu
                             worker starts the container, probes the shell
    returns SandboxHandle
        |
        exec(command)        worker writes command to stdin, reads until sentinel or deadline
        read_file(path)      worker runs "docker exec <name> cat <path>"
        write_file(path)     worker writes via base64 heredoc
        |
destroy()                    worker closes the shell, docker rm -f
                             handle calls coordinator.release to return capacity
```

`SandboxHandle.destroy` is idempotent: the `_destroyed` flag prevents double-release. Agent
loops should call it in a `finally` block to ensure capacity is always returned.

---

## Persistent Shell Protocol

A sandbox container is started with `docker run -i ... /bin/bash -l`. The `-i` flag keeps
stdin open, and `/bin/bash -l` starts a login shell. The shell never exits. Each command is
submitted by writing to its stdin and command completion is signaled by a sentinel line echoed
immediately after the command.

### Sentinel Design

After writing the command to stdin, the worker appends:

```bash
EXITSTATUS="$?"; sleep 0.01; echo ///PROCESS-DONE:$EXITSTATUS:PROCESS-DONE///
```

The short sleep before the echo keeps the sentinel on its own line even when the command's
own output is still being flushed. `parse_sentinel` in `shell.py` scans the accumulated
buffer for the regex pattern and returns the body and the exit code once the sentinel appears.

Why not stateless exec? Docker's `docker exec <name> bash -c "<cmd>"` starts a fresh shell for
each command. A fresh shell loses all state: the working directory resets, exported variables
vanish, and the activated conda environment must be re-sourced. MLGym's own environment
contract requires that the shell state accumulates across calls, so stateless exec is
incompatible with MLGym.

### Abandoned Command Drain

When a command times out, the abandoned command keeps running inside the shell and will
eventually emit its output and its sentinel. Without a drain, the stale sentinel would be
misread as the next command's exit code and the stale output would contaminate the next
observation.

`ContainerShell.drain` is called before every new command. It discards whatever is buffered
plus a non-blocking read sweep and logs the discarded character count at warning level.

---

## Observation Truncation

MLGym does not bound command output. A `pip install` or `python train.py` inside a sandbox
can emit enough text to saturate the model context in one turn.

`truncate_observation` in `shell.py` caps each observation to `max_observation_chars` per
side (head plus tail). The head carries the command echo and early errors, the tail carries
the final result. An elision notice shows the exact number of omitted characters. The total
output is at most `max_observation_chars * 2` characters, which the caller can rely on.

Truncation is applied by the `EnvWorker.exec` method before returning `ExecResult.stdout`.
The `max_observation_chars` value of 0 or less disables truncation entirely.

---

## GPU Passthrough

Some clusters lack `nvidia-container-runtime`, which is required for `docker run --gpus`. The
`build_gpu_argv` function in `sandbox.py` implements an alternative that works without the
toolkit.

The approach is:

1. For each assigned GPU index, pass `--device /dev/nvidiaN`.
2. For each NVIDIA control device (`/dev/nvidiactl`, `/dev/nvidia-uvm`,
   `/dev/nvidia-uvm-tools`), pass `--device` if the path exists on the host.
3. Bind-mount the driver libraries (`libcuda.so.1`, `libnvidia-ml.so.1`) from the host
   into the same path inside the container as read-only mounts.
4. Optionally bind-mount `nvidia-smi` from the host.

This was verified to work on H20 nodes with CUDA code running inside the container.

The `EnvWorker` derives the allowed GPU indices from `CUDA_VISIBLE_DEVICES`, which Ray sets
to exactly the GPUs it granted the worker actor. This partitions the physical GPUs safely:
a worker with indices 4 and 5 can only assign those indices to sandboxes and cannot touch
training GPUs 0 through 3.

---

## Configuration

All settings live under `psrl.env_worker` in the Hydra config tree. The defaults are in
`psrl/trainer/config/psrl/env_worker.yaml`.

| Key | Default | Purpose |
|---|---|---|
| `enable` | `False` | Build the pool at startup. Off by default. |
| `placement` | `colocated` | Node assignment policy. |
| `dedicated_node_ips` | `[]` | Node IPs for dedicated placement. |
| `workers_per_node` | `1` | Worker actors per node. |
| `cpu_slots_per_worker` | `8` | Concurrent sandboxes per worker. |
| `gpu_slots_per_worker` | `0` | GPU slots per worker. `0` for CPU-only sandboxes. |
| `sandbox_cpus` | `4.0` | Default `--cpus` for a sandbox. |
| `sandbox_memory` | `16g` | Default `--memory` for a sandbox. |
| `routing.method` | `least_loaded` | Routing policy. `least_loaded`, `round_robin`, or `random`. |
| `idle_sandbox_timeout_s` | `7200` | Seconds before an idle sandbox is reaped. |
| `exec_default_timeout_s` | `3600` | Default timeout per command. |
| `max_observation_chars` | `8000` | Per-side observation character cap. `0` disables. |
| `coordinator_max_concurrency` | `64` | Max concurrent RPCs to the coordinator. |

---

## Extension Points

The following seams exist for future additions without requiring changes to the existing
components.

**Locality-aware routing.** `select_by_method` receives the full `candidates` list and the
`SandboxSpec`. A future policy can consult `slot.node_ip` to prefer the node closest to the
agent worker issuing the request.

**Dynamic slot rescaling.** The coordinator's `WorkerSlot` counts are set at registration
time. A future mechanism could have workers report their current load and the coordinator
adjust `cpu_slots` based on observed utilization.

**Heterogeneous capacity.** The `SandboxSpec` already carries `gpus`, `cpus`, and `memory`.
A future routing policy can filter candidates by resource type beyond the current CPU and GPU
count checks.

**Per-sandbox network isolation.** `SandboxSpec.network` defaults to `host` for simplicity.
A future policy can set it to a named Docker network or `none` for stricter isolation.

---

## Relationship to Other PSRL Components

The `EnvWorkerManager` is constructed by the PSRL trainer before the agent loop workers
start. The `EnvWorkerCoordinator` handle is passed to agent loops via the config so that each
worker can call `coordinator.create_sandbox` to obtain a `SandboxHandle` for its episode.

EnvWorker is orthogonal to the Router, PS, and TransferQueue subsystems: it sits entirely
on the agent side of the data flow, between the agent loop and the environment, and produces
no training data itself. The training data path begins when the agent loop calls TITO via
the SessionRouter.

:::{seealso}
- {doc}`architecture`: full PSRL data and control flow
- {doc}`router_tito`: TITO session capture and SMG routing
- {doc}`staleness_control`: staleness buffer and Reserve/Occupy/Consume protocol
:::
