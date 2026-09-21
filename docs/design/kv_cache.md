# KV Cache Management

## Motivation

In multi-turn agentic RL, a trajectory accumulates tens of thousands of tokens
across its turns. Each turn concatenates the full history and sends it to a
rollout instance for the next generation step.

Without KV reuse, every turn re-prefills all prior tokens. A ten turn trajectory
with roughly one thousand tokens per turn recomputes about 55k tokens of prefill
to produce about 10k tokens of new state.

PSRL integrates with [LMCache](https://github.com/LMCache/LMCache) to offload KV
cache, reuse prefixes across turns, and move prefixes between rollout instances.

## LMCache Integration

Config: `psrl.lmcache.*`

LMCache runs as a standalone multiprocess (MP) server, one per node. vLLM
workers are clients of that server, and the server owns all cache state. Three
tiers matter:

1. **L1** is the server's own memory pool, sized by `offload_size_gb`.
2. **L2** is optional extra storage registered per server, for example a local
   filesystem or an object store.
3. **Peers** are other instances' servers, reachable over a P2P transfer channel.

Chunks are indexed by **token content hash** in fixed size chunks (`chunk_size`,
default 256 tokens). Matching is content based, so two requests that share a
prefix share KV regardless of when either was computed.

```{mermaid}
sequenceDiagram
    participant AW as Agent Worker
    participant RI as Rollout Instance
    participant LMC as MP server (L1)

    Note over AW,LMC: Turn 1
    AW->>RI: generate(prompt, turn_1_tokens)
    RI->>RI: Full prefill (no cache)
    RI->>LMC: store(KV chunks, hashes)
    RI-->>AW: response_1

    Note over AW,LMC: Turn 2
    AW->>RI: generate(prompt + response_1 + turn_2_tokens)
    RI->>LMC: lookup(prefix hashes)
    LMC-->>RI: cached KV (prefix match)
    RI->>RI: Partial prefill (new tokens only)
    RI->>LMC: store(new KV chunks)
    RI-->>AW: response_2
```

## Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable` | bool | `False` | Master switch for LMCache integration |
| `offload_size_gb` | float | `100.0` | L1 capacity of one node's MP server |
| `chunk_size` | int | `256` | Token chunk size for hash based indexing |
| `hash_algorithm` | str | `blake3` | Chunk hash algorithm, `blake3` or `sha256` |
| `l1_align_bytes` | int | `4096` | L1 allocation alignment, raise to `65536` with P2P |
| `eviction_policy` | str | `LRU` | `LRU`, `IsolatedLRU`, or `noop` |
| `l2_adapters` | list | `[]` | L2 adapters as JSON objects, for example `{type: fs, path: /mnt/kv}` |
| `l2_store_policy` | str | `default` | L2 store policy name |
| `l2_prefetch_policy` | str | `default` | L2 prefetch policy name |
| `clear_on_weight_update` | bool | `False` | Clear the server cache after a weight sync |
| `multi_version_kv` | bool | `True` | Tag entries with the model version |
| `enable_p2p` | bool | `False` | Enable cross-instance KV transfer |
| `p2p_transfer_engine` | str | `nixl` | Transfer engine, `nixl` or `mooncake_te` |
| `coordinator_host` | str | `${psrl.ps_manager_ip}` | Host of the shared MP coordinator |
| `coordinator_port` | int | `9300` | Coordinator HTTP port |
| `coordinator_event_reporting` | bool | `False` | Stream cache events to the coordinator |
| `enable_kv_events` | bool | `False` | Publish the off-GPU tier to routing |
| `mq_timeout_s` | int | `300` | Connector request timeout |
| `gpu_pin_block_budget` | int | `0` | Max pinned GPU KV blocks, `0` disables the limit |

Ports for the MP server, its management HTTP API, the P2P transfer channel, and
the event publisher are allocated at launch. Do not set them by hand.

```yaml
psrl:
  lmcache:
    enable: true
    offload_size_gb: 40.0
    chunk_size: 256
    clear_on_weight_update: false
    multi_version_kv: true
    enable_p2p: true
    coordinator_host: ${psrl.ps_manager_ip}
```

### Version isolation

When the rollout instance syncs to new weights, all cached KV becomes stale. It
was computed with the old weights, and serving it silently degrades accuracy, so
stale entries must not be reused.

Run exactly one of these mechanisms:

- `multi_version_kv: true` (the shipped default) stamps each request with
  `lmcache.tag.model_version`, and that tag is part of the cache key. A lookup at
  version *N* misses entries from earlier versions while KV from other still
  valid versions survives.
- `clear_on_weight_update: true` clears the whole server cache on every sync.
  Correct but coarse, and it discards KV that is still valid.

P2P forces the choice. `enable_p2p: true` requires `multi_version_kv: true` and
`clear_on_weight_update: false`, because a peer's cache cannot be cleared
remotely on weight sync.

### Memory sizing

A rule of thumb for `offload_size_gb`:

$$
\text{offload\_size\_gb} \approx \frac{\text{num\_layers} \times \text{hidden\_dim} \times \text{max\_concurrent\_seqs} \times \text{avg\_seq\_len} \times 4}{10^9}
$$

The factor of four covers key plus value in fp16. For a 7B model with 32 layers,
4096 hidden dim, 64 concurrent sequences at 4k average length, this is about
64 GB of L1.

The cap is not committed up front. L1 grows lazily from `l1_init_size_gb` as it
fills, so a generous `offload_size_gb` costs nothing until the cache is actually
used. Set `l1_use_lazy: false` only to pre-commit the whole cap.

## Cross-instance transfer

Config: `psrl.lmcache.enable_p2p`

When the router moves a request to another rollout instance, the accumulated KV
for that request lives on the source. Without a transfer the destination
re-prefills from scratch.

MP transfer is a **pull**. The destination's MP server warms its own L1 from
whichever peer holds the prefix, so the source is never targeted and the request
is retried from cache rather than pushed.

```{mermaid}
sequenceDiagram
    participant RC as Router
    participant Src as Source instance
    participant DstS as Destination MP server
    participant SrcS as Source MP server

    RC->>Src: transfer_direct(tokens, src, dst)
    Src->>DstS: POST /cache/prefetches
    DstS->>SrcS: P2P lookup and lock
    SrcS-->>DstS: remote addresses
    DstS->>DstS: RDMA read into retained L1
    SrcS-->>DstS: unlock
    DstS-->>Src: found / total chunks
    Src-->>RC: transfer result
```

- The destination must acquire **every** chunk on **every node**. A partial
  result returns a failure so the destination re-prefills the remainder.
- `copy: false` deletes the prefix at the source after the destination confirms.
  That delete is best effort and non-atomic, and it is refused while the source
  prefix is pinned.
- Instances discover each other through the shared MP coordinator. No
  instance-to-instance configuration is required.

A replica can span nodes, and each node's MP server holds that node's share of
the KV. Every server registers its `replica_id` with the coordinator, so naming
either a replica or one of its nodes resolves to the same node set: the transfer
warms all of them, and `copy: false` deletes from all of them. `enable_p2p`
therefore works unchanged for a multi-node replica, whether or not
`mp_server_urls` splits its ranks across nodes.

### Routing modes

Transfer integrates with
`psrl.rollout_coordination.routing_strategy.kv_transfer` (see
{doc}`flexible_rollout`). The router decides whether to transfer, and LMCache
decides how.

| `transfer_mode` | Behavior | Best for |
|-----------------|----------|----------|
| `async` | Start transfer, begin generation immediately | Latency sensitive, short prefixes |
| `sync` | Wait for the transfer, then begin generation | Long prefixes where re-prefill is expensive |
| `pin_sync` | Deprecated alias for `sync` | Existing configurations only |

P2P transfer uses the NIXL transport, so `psrl.ps_mode` must be `nixl_cpu` or
`nixl_gpu` when `enable_p2p` is on.

## Event stream

Config: `psrl.lmcache.enable_kv_events`

With cache-aware routing, `lmcache_overlap_weight` scores the off-GPU tier. That
score is built from the cache events an MP server publishes. Enabling it implies
`coordinator_event_reporting` and turns on the server's event stream at
`/cache/events/stream`, which emits newline delimited JSON. The stream URL is
registered with the coordinator, so routers discover it from the instance
registry instead of being configured per instance.

The stream is advisory and lossy by design: a consumer that falls behind drops
batches instead of stalling the server's observability thread. `GET
/cache/events/stats` reports `dropped_total` alongside the subscriber count, so a
stalled consumer is visible instead of silently under-scoring the tier.

## Prefix pinning

Config: `psrl.lmcache.pin_policy`

A multi-turn trajectory's prefix has to survive the idle gap between two turns.
LRU cannot promise that: its horizon is how much unrelated traffic arrives, not
when the prefix comes back, and a burst of generation during a tool call is
exactly when a warm prefix is swept. Pinning replaces that unpredictable horizon
with the reuse interval itself.

A pin group is named `<model version>:<tail chunk hash>`, where the tail is the
last complete chunk of the request's token sequence. Identity comes from the
prefix, so no caller has to name its own trajectory, which the engine never
receives in auto trajectory mode. Two consequences follow:

- A longer prefix supersedes its own ancestor, because the previous turn's tail
  is now an interior chunk of the current prefix. The next turn therefore
  releases the previous group without either side tracking a group id, and the
  shared chunks stay pinned by the new group.
- Trajectories that share a system prompt but diverge have different tails, so
  they hold separate groups and share the common chunks through reference
  counts.

Three policies: `off`, `all`, and `tagged` (only requests carrying an
`lmcache.pin_group` request config). `off` is the default because pinning moves
capacity out of the general cache, so it is only a win when a prefix is being
evicted before its next turn. `all` is the production setting, chosen by
measuring turn-two hit rate and re-prefill volume.

The group table is bounded by design rather than by hope: an idle TTL releases
groups a client never released, a byte budget releases the least recently pinned
groups first, and a prefix larger than the whole budget stays unpinned instead
of overshooting the ceiling. Because every node's server derives the same group
from the same request, the mechanism is replica-wide without any cross-node
coordination.

## Gotchas

- **L1 is shared by every local KV rank.** `offload_size_gb` is one server's
  capacity, not per rank. Setting it to the per-rank value over-allocates by the
  tensor parallel size.
- **Multi-node instances must avoid pipeline parallelism.** The connector
  rejects multi-server plus PP greater than one. Keep
  `rollout_nnodes_per_instance: 1`, or use tensor parallelism only.
- **P2P alignment.** Below `l1_align_bytes: 65536`, RDMA reads work but are less
  efficient. Raise it when `enable_p2p` is on.
- **Adding tags changes L2 key encoding.** An L2 cache written before this
  version cannot be read back and must be invalidated.
- **`clear_on_weight_update` also drops pins.** A forced clear discards client
  pins, so re-pin after a weight sync if routing depends on them.
- **GPU and L1 pins are different mechanisms.** A GPU pin holds a prefix cache
  block open by raising its reference count, so vLLM refuses to reset the prefix
  cache while any pin is live. PSRL releases its own pins before a weight update
  and reports it if the reset still fails. L1 retention runs through pin groups
  instead, which the request path drives.
- **GPU pinning follows every KV cache group.** A hybrid model caches a
  different prefix per group, so the pin is limited to the prefix that all groups
  can serve, which is the prefix a later request can actually reuse.
- **L1 pins are applied where the store commits.** The store path pins the
  prefix in the same stream step that makes it visible, and the prefetch path
  pins what a transfer warms, so a prefix is never evictable in between. A pin
  no longer needs a retry loop against an in-flight store.
- **The pin budget is a hard ceiling.** A prefix larger than the whole budget
  stays unpinned and falls back to LRU. That is deliberate: overshooting would
  let one long prefix starve L1, and the warning names the bytes needed.
- **Pinning is off unless asked for.** It moves capacity out of the general
  cache, so it is only a win when prefixes are actually evicted before their
  next turn. Measure before enabling `pin_policy: all`.
- **`pin_sync` is retired.** A transfer's source prefix is protected by the read
  lock LMCache takes while resolving the objects, so the surrounding pin added
  nothing while costing two RPCs per transfer. `pin_sync` now behaves as `sync`.
- **Hybrid KV cache management is left to vLLM.** The MP connector declares
  support, so PSRL no longer forces it off. Models that require it, such as
  hybrid SSM models, now start as vLLM intends.
