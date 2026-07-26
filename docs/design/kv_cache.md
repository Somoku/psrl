# KV Cache Management

## Motivation

In multi-turn agentic RL, each trajectory may accumulate **10,000+ tokens** across multiple turns. Each turn involves:
1. Concatenating the full conversation history (prompt + all prior turns).
2. Sending the full sequence to the rollout instance for the next generation step.

Without KV cache reuse, each turn requires **full re-prefill** of all prior tokens, a massive waste of GPU compute that grows quadratically with conversation length. For a 10-turn trajectory with 1k tokens per turn, this means re-computing ~55k tokens of prefill across the trajectory, when only ~10k tokens of new computation are actually needed.

PSRL integrates with [LMCache](https://github.com/LMCache/LMCache) to solve this problem, enabling KV cache offloading, prefix reuse, and cross-instance transfer.

---

## LMCache Integration

**Config**: `psrl.lmcache.*`

LMCache operates as a secondary KV cache backend alongside vLLM's GPU-resident cache. When enabled:

1. **Offloading**: After prefill, KV blocks are copied to the offload backend (CPU memory by default). This frees GPU memory for new requests while preserving computed attention state.

2. **Hash-based chunk indexing**: KV blocks are indexed by **token content hashes** in fixed-size chunks (default: 256 tokens). This means matching is content-based, not position-based: if two requests share the same prefix tokens, they share KV cache regardless of when they were computed.

3. **Prefix retrieval**: On subsequent turns, the system checks the offload backend for matching prefix KV. Matching blocks are loaded back to GPU, and only the new (unmatched) tokens require fresh prefill computation.

### How It Works

```{mermaid}
sequenceDiagram
    participant AW as Agent Worker
    participant RI as Rollout Instance
    participant LMC as LMCache (CPU)

    Note over AW,LMC: Turn 1
    AW->>RI: generate(prompt, turn_1_tokens)
    RI->>RI: Full prefill (no cache)
    RI->>LMC: offload(KV blocks, hashes)
    RI-->>AW: response_1

    Note over AW,LMC: Turn 2
    AW->>RI: generate(prompt + response_1 + turn_2_tokens)
    RI->>LMC: lookup(prefix_hashes)
    LMC-->>RI: cached KV (prefix match)
    RI->>RI: Partial prefill (new tokens only)
    RI->>LMC: offload(new KV blocks)
    RI-->>AW: response_2
```

---

## Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable` | bool | `False` | Master switch for LMCache integration |
| `offload_size_gb` | float | `100.0` | Total CPU memory budget for KV offloading (divided across TP ranks) |
| `chunk_size` | int | `256` | Token chunk size for hash-based indexing |
| `save_decode_cache` | bool | `True` | Also cache KV from decode steps (helps multi-turn reuse) |
| `clear_on_weight_update` | bool | `False` | Invalidate the whole cache after model weight sync |
| `multi_version_kv` | bool | `True` | Tag entries with the model version so stale-weight KV is skipped instead of cleared |
| `enable_async_loading` | bool | `False` | Overlap KV retrieval with prefill computation (known bug, keep disabled) |
| `cache_policy` | str | `LRU` | Cache eviction: `LRU` or `FIFO` |
| `backend` | str | `cpu` | Storage backend: `cpu` (default), `disk`, `remote` (not yet implemented) |

```yaml
psrl:
  lmcache:
    enable: true
    offload_size_gb: 40.0
    chunk_size: 256
    save_decode_cache: true
    clear_on_weight_update: false
    multi_version_kv: true
    enable_async_loading: false
    cache_policy: LRU
    backend: cpu
```

### Key Considerations

:::{admonition} `clear_on_weight_update`
:class: important

When the rollout instance syncs to a new model version, all cached KV becomes **stale**: it was computed with the old weights. Serving subsequent turns from stale-weight KV causes accuracy degradation that compounds on top of the staleness bound itself, so stale entries must not be reused.

PSRL offers two mechanisms for this, and you should run exactly one of them:

- `multi_version_kv: true` (the shipped default) tags each cached entry with the model version that produced it, so a lookup at version *N* simply misses entries from earlier versions. Nothing is thrown away, which means KV from other still-valid versions survives the sync.
- `clear_on_weight_update: true` invalidates the entire cache on every sync. Correct, but coarse and expensive for multi-turn trajectories that span a version boundary.

P2P transfer forces the choice: `enable_p2p: true` requires `multi_version_kv: true` and `clear_on_weight_update: false`, because the P2P backend does not implement clear-on-weight-sync.
:::

:::{tip}
**Memory sizing**: A rule of thumb for `offload_size_gb` is:

$$
\text{offload\_size\_gb} \approx \frac{\text{num\_layers} \times \text{hidden\_dim} \times \text{max\_concurrent\_seqs} \times \text{avg\_seq\_len} \times 2 \times 2}{10^9}
$$

The factor of 2×2 accounts for key+value and fp16 storage. For a 7B model with 32 layers, 4096 hidden dim, 64 concurrent sequences at 4k average length: ~64 GB across all TP ranks.
:::

---

## P2P Cross-Instance Transfer

**Config**: `psrl.lmcache.enable_p2p`

When the Router moves a request to a different rollout instance (due to load balancing, migration, or sync-triggered re-routing), the accumulated KV cache for that request exists on the **source** instance. Without P2P transfer, the target instance must re-prefill from scratch.

### Architecture

- **LMCache Controller**: A shared process started by the Rollout Coordinator before
  vLLM initialization. It maintains worker registration and answers peer-lookup
  queries (`/query_worker_info`) so instances learn each other's transfer endpoints.
  It is **not** on the KV data path.
- **LMCache workers**: Source and destination workers perform the hot-path data move
  directly, avoiding a controller bottleneck.
- **Transport**: Transfer uses the NIXL library (same as PS weight transfer):
  - `nixl`: UCX transport: auto-selects shared memory (same node), IPC (same machine), or RDMA (cross-node).
  - `tcp`: Fallback TCP transport for environments without UCX/RDMA.

### Transfer Flow

A move is initiated on the **source** instance (triggered by the router/coordinator
when a request is re-routed). The source GenWorker calls `transfer_direct()`, which
sends a `MoveWorkerMsg` over a ZMQ REQ socket to its local LMCache worker, and the worker
pushes the KV blocks to the destination's endpoint (`new_position`) over NIXL/TCP and
replies with a `MoveWorkerRetMsg` carrying the number of tokens moved. The Controller
is consulted only beforehand, to resolve the destination's peer endpoint.

```{mermaid}
sequenceDiagram
    participant RC as Router / Coordinator
    participant Src as Source GenWorker
    participant SrcW as Source LMCache worker
    participant Dst as Dest LMCache worker

    RC->>Src: transfer_direct(tokens, src, dst)
    Src->>SrcW: MoveWorkerMsg (ZMQ REQ, new_position=dst endpoint)
    SrcW->>Dst: push KV blocks (NIXL / TCP)
    Dst-->>SrcW: blocks received
    SrcW-->>Src: MoveWorkerRetMsg(num_tokens)
    Src-->>RC: transfer result (bool)
```

### Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_p2p` | bool | `False` | Enable cross-instance KV transfer |
| `p2p_transfer_channel` | str | `nixl` | Transport backend: `nixl` or `tcp` |
| `controller_base_port` | int | `9000` | Base HTTP port for the LMCache Controller |
| `controller_pull_port` | int | `8300` | ZMQ registration/heartbeat port |
| `controller_reply_port` | int | `8400` | ZMQ task-dispatch port |

```yaml
psrl:
  lmcache:
    enable: true
    enable_p2p: true
    p2p_transfer_channel: nixl
    controller_base_port: 9000
```

### Integration with Routing

P2P KV transfer integrates with the `psrl.rollout_coordination.routing_strategy.kv_transfer` configuration (see {doc}`flexible_rollout`). The Router decides **whether** to transfer, LMCache handles **how** the data moves.

| `transfer_mode` | Behavior | Best For |
|-----------------|----------|----------|
| `async` | Start transfer, begin generation immediately (re-prefill if KV arrives late) | Latency-sensitive, short prefixes |
| `sync` | Wait for transfer, then begin generation (no re-prefill) | Long prefixes where re-prefill is expensive |
| `pin_sync` | Pin source KV + wait + unpin | Maximum reliability, highest overhead |

---

:::{admonition} Active Development
:class: warning
The P2P KV cache migration feature is under active development. API and configuration may change.
:::
