# Reader-side FP8 quantization (Megatron direct supply)

Status: implementation note for `dev/fp8-chunk-quant`. The code is the authority; this
note records the intended protocol, the phase boundary, and the supported matrix.

## Goal

Keep the trainer in BF16 and the rollout engines in block FP8, but remove the
model-sized FP8 publish copy from the trainer GPU. The rollout reads BF16 live
parameters and quantizes into its own FP8 parameters. Readers may retry when a
write gate preempts them. The existing per-shard version stamps and torn detection
remain the correctness boundary.

This is the first phase of design 7.2. It implements reader-side quantization with
the existing per-shard sharding algebra. The canonical-chunk peer cache (cross-layout
sharing of one FP8 chunk) is phase 2.

## Why phase 1 is correct

- The unified sharding refactors every trainer and rollout client to the same finest
  global shard mesh. For a global shard, the trainer BF16 view and the rollout FP8
  view have the same logical shape. Only the dtype differs.
- The trainer keeps its live BF16 bytes. The rollout stages the source shard and
  quantizes it with the target's `VllmFp8BlockQuantizer` into the rollout's final FP8
  slot and `_scale_inv` view. No full BF16 or FP8 copy is added.
- The write gate is enabled for this path. The optimizer bumps the epoch before its
  first in-place write, preempts in-flight readers, and waits only for the bounded
  control acknowledgement. A reader whose pull spans the epoch change cannot commit;
  it retries the next version. Version equality and torn detection cover the rest.
- The trainer does not refresh publish copies in this mode. `publish_version` only
  stamps the BF16 shards. `_scale_inv` is derived by the reader.

## Configuration

```yaml
psrl:
  ps_mode: nixl_direct
  nixl:
    weight_quantization:
      scheme: fp8
    reader_quantization: true
    peer_diffusion: true           # exact shard groups share a quantized holder
    await_write_gate: auto        # resolves to true for this mode
```

Validation rules:

- `reader_quantization` requires `weight_quantization.scheme=fp8`.
- It requires `train_actor_rollout_ref.actor.strategy=megatron`. FSDP2 is rejected
  because the user does not require FSDP FP8 training and its flat local shards do
  not provide the tile geometry this path relies on.
- It requires direct supply (no PS data plane).
- Quantization tiles must be whole in the trainer shards. A configuration that cuts
  a 128x128 tile is rejected at registration, as in the current publisher.

### Network profile

Every node must source the same UCX profile before Ray starts. A split device list is
worse than a slow one: NIXL resolves a peer's UCX endpoints during the notification
handshake, and an address present on only one side is tried and times out before the
control message is delivered. The PSManager then raises `NIXL_ERR_NOT_FOUND`.
`scripts/start_nixl_cluster.sh` compares `UCX_NET_DEVICES` on all nodes and refuses to
start on a mismatch. On rail-isolated clusters, pass one list for all nodes through
`PSRL_UCX_NET_DEVICES`; do not configure the head by hand.

## Data flow

```text
trainer BF16 live shard
    -> NIXL READ into a bounded staging buffer (source dtype)
    -> VllmFp8BlockQuantizer.quantize
    -> rollout FP8 shard + _scale_inv
```

The pull engine:

1. Skips `_scale_inv` keys in the upstream plan. They are derived, not transferred.
2. Detects a weight key whose source meta dtype differs from the local FP8 dtype.
3. Stages the source shard and quantizes it after the read completes.
4. Marks the key complete only after the quantize/scatter, so a preemption cannot
   certify partially quantized bytes.
5. Records the source version for the torn/commit checks.

Peer diffusion is enabled for exact shard groups. The existing planner elects the
first reader of each (key, shard set) group as the designate. The designate reads BF16,
quantizes the weight, and writes its scale. A follower reads the designate's FP8 weight
and derived `_scale_inv` over NVLink. The scale key shares its weight key's wave. This
removes the repeated BF16 read and quantization for identical target shards. Different
target layouts form different groups and still quantize independently; the
layout-independent canonical-chunk cache is phase 2.

## Version and failure behavior

- A reader of version v reads BF16 source shards stamped v. If the optimizer starts
  a write, the writer closes the version and preempts the reader. The reader either
  commits at v if the epoch did not move, or abandons the local bytes and retries v+1.
- A scale is computed from the same source tile as its weight and written in the
  same scatter. There is no separate scale source version to mix.
- A failed or evicted source is rerouted or the reader reports `torn`/`unavailable`;
  the instance does not serve mixed bytes.

## Phase 2: peer diffusion / canonical chunks

Phase 1 shares quantized results only across readers with identical target shards. Phase 2 will
add the canonical-chunk relay described in design 7.2:

- Group readers by `(version, logical chunk, scale scheme)` instead of exact shard sets.
- Let the first reader quantize a layout-independent FP8 chunk into a bounded cache.
- Followers read the cached FP8 chunk and slice their target layout.
- The cache is bounded (K chunks), evicts, and is version/generation tagged.
- No designate process is added; the first holder is the existing peer planner's
  first reader for that chunk.

Phase 1 shares the designate's derived scale through the same peer wave as its
weight. A follower whose target shard differs cannot reuse it; that is the case the
canonical-chunk cache will cover. Version stamps on rollout holders are the existing
peer-diffusion semantics; the designate writes the scale before the wave is signalled.

## Supported matrix (phase 1)

| Dimension | Supported |
|---|---|
| Trainer | Megatron, BF16 parameters |
| Rollout | vLLM block FP8 (128x128, `_scale_inv`, fp32) |
| Supply | `ps_mode=nixl_direct` |
| Parallel layouts | Megatron TP/PP/EP vs vLLM TP/EP where every local shard holds whole quantization tiles |
| Model types | dense and grouped MoE keys that satisfy the tile rule |
| Colocate | allowed, because the reader still reads live BF16 and the write gate protects it |
| FSDP2 FP8 | rejected |

## Verification gates

1. CPU: per-shard BF16 -> FP8 pull with the memcpy transport, train/gen TP mismatch,
   two versions, and byte/scale equality against the publisher.
2. GPU: 8B Megatron TP2/TP4 -> vLLM TP2/TP4, two-round layout judge.
3. MoE: 30B-A3B EP/TP combinations that satisfy the tile rule.
4. Training: 10-step BF16/FP8 direct run with the write gate, compare reward,
   validation, sync time, and trainer HBM peak with the publisher baseline.
5. Scale test: 235B only if nodes and time permit; otherwise report the measured
   smaller-model result and the arithmetic extrapolation.

## Scale analysis

What phase 1 costs per rollout replica, per node, and across the whole cluster, in
terms the implementation already measures or bounds -- not a separate model.

**Per-reader compute.** A reader's quantize cost is `O(local shard elements)`: the
128x128 blockwise cast and the derived-scale write (`VllmFp8BlockQuantizer.quantize`)
touch only the bytes the reader already owns. It does not grow with model size for a
fixed shard size, so a reader's cost at a 30B MoE and at a 1T MoE is the same once TP/EP
hold the per-rank slice constant. This is what makes the design avoid a trainer-side
publish copy: the quantize work moves to wherever there is idle compute (the reader),
not a full-model pass on the trainer.

**Trainer egress per sync.** Without peer diffusion, egress is `O(rollout replicas on
a node) x (shard bytes)`: every same-node reader pulls the live BF16 shard upstream
independently. With peer diffusion, egress is `O(designate groups on a node) x (shard
bytes)` -- one upstream BF16 read per `(key, exact shard set)` group, however many
replicas share that group. At a fixed per-node replica count, growing the cluster to
10k GPUs multiplies the number of *nodes*, not the egress *per node*; each node's
trainer ranks still serve only their own node's designates. This is the same bound the
existing (non-reader-quantization) direct-supply path already measured and the
implementation reuses unchanged (`comm_plan._make_peer_groups`); reader-quantization
adds the quantize step onto the designate's copy without adding a second upstream read.

**Registration and handshake.** NIXL registration is a fixed ~30 ms per contiguous
region on this hardware (`nixl.yaml`'s `segment_registration` comment), not per
tensor -- `segment_registration=auto` registers one region per CUDA allocator segment,
so a model with 200 weights and a model with 2,000 register in the same number of
calls once segments, not tensors, are counted. Per-shard version stamps are checked at
0.06 ms for an 8B model (measured) and are O(1) per shard independent of model size,
dtype, or quantization, so this does not grow with scale either.

**Where heterogeneous layouts cost more.** Phase 1's peer groups are keyed by exact
shard set, so N different rollout TP/EP layouts sharing a node form N independent
designate groups for the same logical weight, each reading BF16 upstream and
quantizing once. At a handful of layouts per node this is a small constant factor; at
many simultaneous layouts (for example, elastic scaling mid-run) it degrades toward
the no-peer-diffusion bound above. The canonical-chunk cache in the next phase removes
this by sharing one layout-independent FP8 chunk across every layout on a node,
bringing egress back to one upstream BF16 read per node per logical chunk regardless
of how many target layouts read it.

**MoE.** A grouped expert tensor quantizes as `(1,) * (rank - 2) + block` -- each
expert's trailing 2-D slice tiles independently (`VllmFp8BlockQuantizer.tile_of`,
exercised end to end by
`test_reader_quantization_tiles_a_grouped_moe_shaped_key`). The per-reader cost is
still proportional to the reader's own shard, now summed over the experts it holds; a
TB-scale MoE model with a few thousand experts does not change this per-shard bound,
because canonical MoE grouping (`moe_grouping.py`) already collapses the
per-expert key explosion into one grouped key before any of this runs, keeping the
transport side's descriptor and registration count independent of expert count.

## Measured status at this snapshot

- CPU integration (`tests/nixl/test_pull_engine.py`, memcpy transport, real
  clients/planner): 5 reader-quantization tests pass. They cover train/gen TP
  combinations 1->2, 2->2, 4->2, and 2->4, a second version that replaces every
  FP8 byte and scale, and exact-shard peer diffusion (one upstream BF16 read per
  key/shard-set group, followers read the designate's FP8 weight and scale).
- CPU regression: 877 tests pass across the full `tests/unit` and CPU-marked
  `tests/nixl` suites (8 skipped, 668 deselected GPU/other tests).
- GPU FP8 quantizer: 46 tests pass, including bit and scale equality with vLLM's
  block quantizer and the full quantize/requantize round trip.
- GPU two-node weight transport (8B, Megatron TP4 x2 DP -> vLLM TP1 x4 instances,
  `reader_quantization=true`): the NIXL handshake, sharding exchange, and the
  first PS-to-rollout pull completed on real hardware once every node sourced the
  same UCX device list (see "Network profile" above). The handshake took 4.63 s.
  Each rollout replica pulled 16.38 GB upstream -- the live BF16 model, not a
  cast or compressed copy, confirming the trainer published no FP8 snapshot. The
  trainer ranks held 13.5-13.7 GiB each afterward, in the range of plain Megatron
  TP4 training memory with no separate FP8 publish allocation visible. The run
  did not reach a completed training step: rollout generation stalled
  indefinitely after the pull, reproduced across repeated clean runs. This is a
  dispatch fault in the serving layer the rollout gateway sits on, not in this
  transport: the gateway's own `/routing_loop/status` reports every backend
  `healthy`/`ready` with a full request queue (`queue_len` equal to every
  dispatched request) but `routing: false` and `running_tasks: 0` -- its
  requests are queued and never picked up for dispatch -- while the vLLM
  engine's own input queue stays empty throughout. The same stall reproduces
  with a direct request built to match PSRL's own payload, independent of
  `reader_quantization`. Several specific causes were checked against the
  gateway's source and ruled out: the per-worker generation-admission gate
  defaults open and never logs its "parking" path; the version/label-filtering
  selector it could use is not the one this deployment configures (`naive`,
  not `psrl`); and the migration-queue-starvation path in its dispatch loop
  requires a migration target that nothing in this deployment ever registers.
  The remaining gap is inside the gateway's own dispatch loop and needs
  Rust-level instrumentation this environment does not have tooling to add;
  it is tracked separately from this transport design, which the GPU numbers
  above already validate up through the completed pull.
- Peer diffusion for this topology (4 same-node rollout instances, each its own
  NIXL client, all wanting the identical unsharded tensor): a CPU-level
  reproduction of the exact shape (`train_tp=4, gen_tp=1, instances=4,
  reader_quantization=True`) elects a designate per key and gives every other
  instance a peer read, as intended (`test_reader_quantization_peer_diffusion_
  shares_training_reads_across_separate_instances`). The GPU run above did not
  show this: all four replicas read the full 16.38 GB upstream with zero peer
  bytes, reproduced across repeated clean runs. Since the planning algorithm is
  verified correct in isolation, the gap is in what the live multi-process
  deployment feeds it. Ray's log deduplication collapsed the per-replica
  `node_ip=...` diagnostic added this round across replicas sharing a template,
  which hid the one value (each replica's advertised `node_ip`, set once per
  process from `LOCAL_NODE_IP`/`get_local_ip()` in
  `psrl/utils/nixl/network_topology.py`, read inside vLLM's per-GPU spawned
  worker subprocess rather than the parent actor) most likely to explain it.
  The next real run should set `RAY_DEDUP_LOGS=0` to see all four values before
  changing the election logic.
- MoE: a grouped-expert-shaped key (3-D, expert axis unsharded, trailing two
  dims cut by TP) quantizes bit for bit and scale for scale through the real
  reader-quantization path (`quantize_into` -> `tile_of`) across train/gen TP
  1->2, 2->2, 4->2, and 2->4
  (`test_reader_quantization_tiles_a_grouped_moe_shaped_key`). This is a CPU
  reproduction of the tiling math, not a GPU run of an actual MoE checkpoint;
  the GPU verification gate for 30B-A3B EP/TP combinations has not been run.
- Canonical-chunk cache across different rollout layouts: not implemented. Exact
  shard groups are shared; different target layouts quantize independently.
