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

## Measured status at this snapshot

- CPU integration (`tests/nixl/test_pull_engine.py`, memcpy transport, real
  clients/planner): 5 reader-quantization tests pass. They cover train/gen TP
  combinations 1->2, 2->2, 4->2, and 2->4, a second version that replaces every
  FP8 byte and scale, and exact-shard peer diffusion (one upstream BF16 read per
  key/shard-set group, followers read the designate's FP8 weight and scale).
- CPU regression: 93 tests pass across the pull engine, publish-format config,
  write-gate config, and write-state suites.
- GPU FP8 quantizer: 46 tests pass, including bit and scale equality with vLLM's
  block quantizer and the full quantize/requantize round trip.
- Full GPU training end-to-end: not completed in this environment. Two-node and
  single-node attempts reached configuration validation, model construction, and
  vLLM engine startup. They then hit environment-level blockers: vLLM startup
  latency beyond the test window, a colocated validation KV-cache OOM, and a NIXL
  handshake timeout. No end-to-end latency or trainer-HBM number is claimed here.
- Canonical-chunk cache across different rollout layouts: not implemented. Exact
  shard groups are shared; different target layouts quantize independently.
