# KV Cache Transport Findings

Numbers behind the design in {doc}`kv_cache`. Each result lists the script that
reproduces it; run the scripts on a free GPU node with the pinned LMCache
v0.5.5 checkout and its patch applied.

## Test bed

- H20 8 x 95 GiB per node, driver 535.247.01 (CUDA 12.8).
- vLLM 0.30.1 (v0.30.0 checkout), LMCache v0.5.5 at commit `05a013b2`.
- Microbenchmarks: `Qwen2.5-0.5B-Instruct` / `Qwen2.5-1.5B-Instruct`.
- End-to-end: `Qwen3-8B`, GSM8K, 2 training steps, PS weight transport.

## Store and retrieve

`tests/lmcache/manual_store_retrieve.py` sends the same ~1.5k-token prompt
twice with vLLM's own prefix cache disabled, so the second prefill can only be
served by LMCache.

| Transfer path | Store | Retrieve | First prefill | Cached prefill | Speedup |
|---------------|-------|----------|---------------|----------------|---------|
| `lmcache_driven` (CUDA IPC) | 0.038 s | 0.002 s | 0.598 s | 0.196 s | 3.05x |
| `engine_driven` (worker copy) | 0.254 s | 0.149 s | 0.716 s | 0.394 s | 1.82x |

The zero-copy path is roughly 7x faster to store and 75x faster to retrieve.
It is also incompatible with PSRL's default vLLM sleep mode: the KV cache is
backed by the sleep-mode allocator, and legacy CUDA IPC handles cannot be
created for it (`cudaErrorInvalidValue` at `register_kv_caches`). The copy path
is therefore the shipped default; `lmcache_driven` stays available for
deployments that disable sleep mode.

With `LMCACHE_CHUNK=1024`, the copy path stored 1024 tokens in 0.205 s and
retrieved them in 0.082 s, with a cached prefill of 0.297 s. The per-chunk
overhead dominates at small chunk sizes, so long prefixes benefit from a
larger chunk; the default 256 keeps reuse granularity for short turns.

## P2P transfer

`tests/lmcache/manual_p2p.py` starts two MP servers with NIXL, stores a
1024-token prefix on A, warms B through `/cache/prefetches`, then generates
the same prompt on an engine attached to B.

- Warm prefetch: 4/4 chunks loaded into B's L1 in 203 ms (server-side);
  `transfer_direct` returned in 0.21 s.
- Retrieval on B: 1024 tokens in 0.108 s.
- The destination must have registered a KV layout first; a prefetch submitted
  before any destination engine starts is refused with 503.

The pin-group path exposed a bug in the v0.5.5 patch: the prefetch controller
read `request.pin_group`, which the in-flight request object did not define,
so every prefetch raised `AttributeError` after loading. The patch now carries
the pin fields on that object.

### Multi-node

`tests/lmcache/manual_p2p_multinode.py` runs on a Ray cluster with server A and
engine A on one node and server B and engine B on another. A 1024-token prefix
is stored on A, warmed into B, and fetched by a fresh engine B.

- Transfer: 0.211 s, 4/4 chunks loaded into B's L1.
- B retrieval: 1024 tokens in 0.103 s.
- B generation: 0.589 s with the migrated prefix against 2.386 s for a cold,
  different prompt of the same length: 4.05x.

The coordinator's instance metadata is what lets B resolve A's node from the
replica id; no node-to-node configuration is needed. The destination still
needs an engine registered before it accepts the prefetch, which the cold run
supplies.

## End-to-end

`examples/grpo_trainer/qwen3_8b_megatron.sh` with GSM8K, two steps, and
`cache_aware_v1`. Eight rollout and eight validation instances share the
rollout node, one GPU each. `manual_overhead.py` is the same shape at one
engine's scale: 64 unique prompts, about 900 tokens each, 64 output tokens.

| Run | Step 1 | Step 2 | Validation | Final val acc |
|-----|--------|--------|------------|---------------|
| No LMCache | 62.3 s | 247.5 s | 147.9 s | 0.3662 |
| Eager, 16 servers, 40 GiB L1, pin=all | 73.8 s | 300.1 s | 203.5 s | 0.3563 |
| Lazy, validation off, 8 GiB L1, pin=off | 75.5 s | 286.8 s | 163.0 s | 0.3662 |
| Auto-disabled (no reusable pool) | 71.6 s | 259.6 s | 152.6 s | 0.3533 |

The first eager run cost 18-20% per step. Three effects stacked:

1. **Stores compete with generation.** Eager mode submitted 62 stores for 64
   requests, 5.5 s of store time for 60k tokens. Even with the asynchronous
   engine-driven context, the GPU-to-CPU copies and the message queue consume
   CPU, PCIe, and copy-engine bandwidth while generation is running. The
   microbenchmark reproduces it without any cluster: eager is 18-23% slower,
   and the connector alone (lazy with a threshold the batch never reaches) is
   about 10% slower.
2. **Per-step allocation reporting.** LMCache's L0 report builds a token list
   for every new request on every scheduling step. PSRL does not consume it;
   turning it off (`report_block_allocations: false`, now the default) cut the
   lazy overhead roughly in half.
3. **Pools that cannot reuse anything.** Validation is half the requests in
   this run, and a generate-only training rollout has one request per
   trajectory with a new version every step. Both stored KV that the version
   tag made unreachable. Sixteen MP servers per node, each growing L1 to 40 GiB
   with `pin_policy=all`, amplified the contention.

The shipped policy therefore starts LMCache only for pools that can reuse KV:
rollout with a multi-turn agent loop. Validation, generate-only rollout,
reward, and teacher pools skip it unless `enable_validation` or
`enable_single_turn` is set. The last row is that policy: no MP server is
started, and the per-token generation time (`wait_for_gen / response tokens`)
matches the baseline within 1-3%; the remaining differences are response
length and seed variance.

When a deployment does enable stores, `lazy_offload: true` batches them behind
finished requests, and `report_block_allocations: false` removes the L0 work.
On the no-reuse microbenchmark that combination is 9% slower than no LMCache
instead of 18-23%. It is the right setting for a mixed workload where some
requests will be reused; a pool with no reuse at all should not start LMCache
in the first place.

On a workload with reuse, the win is large: the cached prefill in the table
above is 1.8x faster through the copy path and 3.05x through zero-copy. The
connector's ~10% per-request overhead is paid back by the first reused
1k+-token prefix.

## Weight transport regression

- `tests/nixl/test_nixl_e2e.py --config-name nixl_e2e_1node` (direct supply):
  both rounds passed, 399/399 keys on every client, 32.76 GB trainer egress,
  0.58 s pull, 723 descriptors, no fallbacks.
- `tests/nixl/test_nixl_e2e.py --config-name nixl_e2e_qwen35_4b_fp8`: FP8
  rollout check on Qwen3.5-4B. Initial pull 0.719 s, round-2 pull 0.201 s,
  24.48 GB trainer egress, 523 of 538 keys judged per client (the rest are the
  intentionally bf16 layers), 3869 descriptors, 520 MiB staged, no fallbacks.
- `tests/nixl/test_nixl_e2e.py --config-name nixl_e2e_colocate_qwen35_4b`:
  colocate on the same eight GPUs. Round-2 pull 0.010 s, 38.73 GB moved over
  CUDA IPC at 625-647 GB/s, 523/538 keys judged per client, 0 descriptors and
  0 fallbacks.
- `unit_tests/kv_cache` and `tests/gen_dplb`: 91 passed.
- `tests/nixl`: 1382 passed, 23 skipped. The remaining failures are stale
  source-text assertions and GPU/PCIe environment tests, all reproducible on
  the pre-merge commit; the registration fixture and the config-matrix
  default-resolution check were repaired so the suite reflects the current
  client and harness configs.

The LMCache merge leaves the NIXL client, transfer planner, and weight-sync
paths unchanged; the regression above is the guard on that claim.
