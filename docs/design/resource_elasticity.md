# Resource Elasticity

:::{admonition} Under Development
:class: warning
This feature is under active development. Documentation will continue to evolve as the implementation stabilizes.
:::

## Concept

Resource Elasticity lets PSRL dynamically reallocate GPU resources across its four major workload components, **training**, **generation**, **evaluation**, and **reward**: based on runtime demand.

Traditional asynchronous RL systems either statically partition GPUs (wasting capacity whenever a phase is idle) or rely on heavy external orchestration to swap workloads in and out. PSRL takes a lighter-weight route: GPU memory is transparently managed by TMS, so the same physical GPUs can host different workloads at different points in the training loop without any change to the workload code itself.

### Supported Components

| Component | Status | Elasticity granularity |
|-----------|--------|------------------------|
| Training | Supported | **Whole training pool**: all training workers pause/resume together (all-or-nothing) |
| Generation | Supported | **Per rollout instance**: each instance pauses/resumes independently |
| Evaluation | Supported | **Per validate instance**: each instance pauses/resumes independently |
| Reward | Work in progress | **Per reward worker** (planned, will follow the per-instance model) |

Training is "all-or-nothing" because mainstream parallelism backends (FSDP, Megatron) do not natively support scaling up/down at runtime: optimizer states are sharded with a fixed ZeRO-style partition that is bound to the world size at initialization, and re-sharding them on the fly would require a heavyweight repartition. The other components carry no such cross-worker state and can therefore scale at the finer per-instance granularity.

---

## TMS (torch_memory_saver)

PSRL uses [torch_memory_saver](https://github.com/fzyzcjy/torch_memory_saver) (TMS) as its underlying primitive for dynamic GPU memory management when multiple workloads are colocated on the same GPUs.

TMS operates in two phases:

1. **Pause**: when a workload is idle, its GPU memory is released (or offloaded to CPU).
2. **Resume**: when the workload becomes active again, its GPU memory is reclaimed (or reloaded from CPU).

This lifecycle is fully transparent to the workload: existing training and generation code does not need to be modified.

### Configuration

**Config**: `psrl.tms.*`

| Field | Type | Options | Description |
|-------|------|---------|-------------|
| `range` | str \| null | `null`, `train`, `all` | Scope of TMS-managed memory |
| `enable_cuda_graph` | bool | `true` / `false` | Also release captured CUDA graphs when pausing |
| `enable_nixl` | bool | `true` / `false` | Manage NIXL pinned transfer buffers under TMS |

```yaml
psrl:
  tms:
    range: all
    enable_cuda_graph: true
    enable_nixl: false
```

### Scope Options

| `range` | Behavior |
|---------|----------|
| `null` | TMS disabled. Training and generation must run on separate GPU allocations. |
| `train` | Only the training worker's memory is managed by TMS. While generation is active, training tensors are offloaded. |
| `all` | Both rollout and training workers' GPU memory is managed by TMS, enabling seamless colocated execution without OOM. |

:::{note}
The `range` interface currently covers **training, generation, and evaluation** workers. It will be extended to also cover the **reward worker** once reward-side elasticity lands, so that reward models can be colocated and TMS-managed on the same shared GPUs.
:::

When `range=all`, the execution flow becomes:

1. **Generation phase**: the rollout instance holds GPU memory, training tensors are offloaded to CPU.
2. **Training phase**: the training worker reclaims GPU memory, the rollout instance's model and KV cache are offloaded.
3. Transitions are driven by the training loop's consume/generate cycle.

:::{tip}
Use `range=all` for maximum GPU utilization in resource-constrained deployments. The offload/reload overhead is typically small compared with the compute saved by not dedicating separate GPUs to each phase.
:::

### CUDA Graph Management

When `enable_cuda_graph=true`, TMS additionally releases captured CUDA graphs during the pause phase. This matters because CUDA graphs pin GPU memory that ordinary tensor offloading cannot release, without it, a paused vLLM instance would still hold significant GPU memory via its captured attention kernels.

### NIXL Buffer Management

When `enable_nixl=true`, TMS also manages the pinned memory buffers used by NIXL for weight transfers. This lets the pinned memory be reclaimed when no transfer is active, reducing the steady-state memory footprint of the PS communication layer.
