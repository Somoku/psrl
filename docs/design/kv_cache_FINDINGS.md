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

## End-to-end

`examples/grpo_trainer/qwen3_8b_megatron.sh` with GSM8K, two steps, and
`cache_aware_v1`, comparing LMCache off and on. Eight rollout and eight
validation instances share the rollout node, one GPU each.

| Metric | No LMCache | LMCache |
|--------|------------|---------|
| Initial validation accuracy | 0.3450 | 0.3434 |
| Final validation accuracy | 0.3662 | 0.3563 |
| Step 1 time | 62.3 s | 73.8 s |
| Step 2 time | 247.5 s | 300.1 s |
| Step 1 throughput | 70.6 tok/s | 59.2 tok/s |
| Step 2 throughput | 53.6 tok/s | 49.8 tok/s |

Both runs completed and every weight sync landed. The LMCache run stored
10,923 chunks and retrieved none: GSM8K prompts are unique per step, the 8
samples of one prompt are served by vLLM's own GPU prefix cache, and a weight
update changes the version tag, so earlier entries are correctly unreachable.
On this workload the copy path is pure overhead. It is the workload LMCache is
not for; the multichunk reuse case is the microbenchmark above and the
multi-turn agent loops described in the design.

## Weight transport regression

- `tests/nixl/test_nixl_e2e.py --config-name nixl_e2e_1node` (direct supply):
  both rounds passed, 399/399 keys on every client, 32.76 GB trainer egress,
  0.58 s pull, 723 descriptors, no fallbacks.
- `tests/nixl/test_nixl_e2e.py --config-name nixl_e2e_qwen35_4b_fp8`: FP8
  rollout check on Qwen3.5-4B. Initial pull 0.719 s, round-2 pull 0.201 s,
  24.48 GB trainer egress, 523 of 538 keys judged per client (the rest are the
  intentionally bf16 layers), 3869 descriptors, 520 MiB staged, no fallbacks.
- `unit_tests/kv_cache` and `tests/gen_dplb`: 91 passed.

The LMCache merge leaves the NIXL client, transfer planner, and weight-sync
paths unchanged; the regression above is the guard on that claim.
