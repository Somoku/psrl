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
| `offload_size_gb` | float | `20.0` | Total CPU memory budget for KV offloading (divided across TP ranks) |
| `chunk_size` | int | `256` | Token chunk size for hash-based indexing |
| `save_decode_cache` | bool | `True` | Also cache KV from decode steps (helps multi-turn reuse) |
| `clear_on_weight_update` | bool | `True` | Invalidate stale KV after model weight sync |
| `enable_async_loading` | bool | `True` | Overlap KV retrieval with prefill computation |
| `eviction_policy` | str | `lru` | Cache eviction: `lru` (least recently used) or `fifo` |
| `backend` | str | `cpu` | Storage backend: `cpu` (default), `disk` |

```yaml
psrl:
  lmcache:
    enable: true
    offload_size_gb: 40.0
    chunk_size: 256
    save_decode_cache: true
    clear_on_weight_update: true
    enable_async_loading: true
    eviction_policy: lru
    backend: cpu
```

### Key Considerations

:::{admonition} `clear_on_weight_update`
:class: important

When the rollout instance syncs to a new model version, all cached KV becomes **stale**: it was computed with the old weights. Setting `clear_on_weight_update: true` (the default) invalidates the entire cache on sync, which is correct but expensive for multi-turn trajectories that span a version boundary.

Under any `staleness ≥ 1` setting, **keeping `clear_on_weight_update: true` is recommended**: continuing to serve subsequent turns from KV computed by stale weights causes accuracy degradation that compounds on top of the staleness bound itself. Set it to `false` only if you are willing to trade a measurable accuracy hit for the throughput win on long multi-turn trajectories.
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

- **LMCache Controller**: A shared coordinator process started by the Rollout Coordinator. It maintains a registry of which KV chunks exist on which instances and coordinates transfers.
- **Transport**: Transfer uses the NIXL library (same as PS weight transfer):
  - `nixl`: UCX transport: auto-selects shared memory (same node), IPC (same machine), or RDMA (cross-node).
  - `tcp`: Fallback TCP transport for environments without UCX/RDMA.

### Transfer Flow

```{mermaid}
sequenceDiagram
    participant Router
    participant Src as Source Instance
    participant Ctrl as LMCache Controller
    participant Dst as Destination Instance

    Router->>Ctrl: migrate_kv(request_id, src, dst)
    Ctrl->>Src: export_chunks(request_id)
    Src-->>Ctrl: chunk_metadata + handles
    Ctrl->>Dst: import_chunks(metadata, handles)
    Note over Src,Dst: RDMA transfer (data plane)
    Dst-->>Ctrl: import_complete
    Ctrl-->>Router: migration_done
    Router->>Dst: resume_generation(request_id)
```

### Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_p2p` | bool | `False` | Enable cross-instance KV transfer |
| `p2p_transport` | str | `nixl` | Transport backend: `nixl` or `tcp` |
| `controller_port` | int | `18080` | LMCache Controller listen port |

```yaml
psrl:
  lmcache:
    enable: true
    enable_p2p: true
    p2p_transport: nixl
    controller_port: 18080
```

### Integration with Routing

P2P KV transfer integrates with the `psrl.routing_strategy.kv_transfer` configuration (see {doc}`flexible_rollout`). The Router decides **whether** to transfer, LMCache handles **how** the data moves.

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
