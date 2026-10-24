# Quantized weight transport

This page describes how PSRL publishes quantized weights from the trainer and how the
rollout engines load them under direct supply. It covers the current implementation, the
generic plan that Phase 1 introduces, and the path to MXFP8, NVFP4 and arbitrary
compressed-tensors configurations.

The short version: the NIXL wire is already a generic `name + sharding -> bytes` channel.
What differs between quantization schemes is which auxiliary parameters exist, how they
are laid out, and whether computing them needs statistics from more than one rank. The
transport therefore needs a quantization plan and a publisher-side quantizer, not a new
transfer primitive.

## 1. Goals and non-goals

Goals:

- Keep direct supply free of full-weight materialization on every rank.
- Publish every tensor the rollout engine loads, including scale, zero-point and global
  scale tensors, with the same version and sharding metadata as the weight.
- Reuse the engine's own parameter geometry to decide what to quantize, so the trainer
  never maintains a second model-specific whitelist.
- Reuse upstream veRL and compressed-tensors quantization math instead of implementing
  MXFP8 or NVFP4 numerics in PSRL.
- Fail at configuration time, with the offending parameter and a concrete remedy, when a
  scheme or a shard layout cannot be expressed. Never fail after a rollout has started.

Non-goals for Phase 1:

- Implementing MXFP8, NVFP4 or arbitrary `config_file` support. Phase 1 provides the
  architecture and implements `bf16` and block FP8 on top of it.
- Replacing the trainer with a QAT trainer. NVFP4 ultimately needs activation scales
  observed during QAT, which is a separate capability.
- Supporting formats the target vLLM build cannot load on the current accelerator.

## 2. Two quantization mechanisms

The same word "quantization" describes two different flows in this stack.

| | Trainer-side QAT | Publisher-side transport quantization |
|---|---|---|
| Who quantizes | The trainer, on every weight sync, through compressed-tensors compressors | The NIXL publisher, when it prepares the published copy |
| What the trainer holds | BF16 master weights plus QAT observers | BF16 or FP8 master weights |
| What the engine is told | A compressed-tensors `quantization_config` | A serialized FP8 configuration that matches the publisher's bytes |
| Upstream API | `rollout.quantization` (`RolloutQuantizationConfig`) | PSRL-specific transport format |
| Purpose | Quantization-aware training plus low-precision rollout | Low-precision rollout without changing training precision |

PSRL direct supply uses the second flow. The trainer is the source of the bytes, so the
publisher must produce exactly the layout the engine's quant method expects. Mixing the
two flows means the engine advertises a format the trainer never produced, and the first
pull fails on a dtype or layout assertion.

The design keeps the two APIs explicit. `rollout.quantization` remains the upstream QAT
control. `psrl.nixl.weight_quantization` describes the wire format. The generation and
validation engines derive their load format from the wire format. A configuration that
enables both flows for the same engine is rejected.

## 3. Invariants

1. **The consumer declares the quantization structure.** Scale names, scale shapes and
   block geometry come from the rollout's live parameters and sharding. The trainer
   supplies bytes for those declarations rather than maintaining its own list.
2. **Auxiliary parameters are ordinary shards.** A scale or global scale travels with its
   own key, version and sharding. No side channel and no special-case transfer.
3. **One tile rule.** A trainer shard must contain whole quantization tiles. If a
   parallel split cuts a tile, configuration validation fails and names the parameter and
   the split.
4. **Publish is two-phase only when the scheme needs it.** Block-local schemes stay a
   single pass over the local shard. Global or per-tensor scales add one small reduction
   before the pass.
5. **The wire format and the engine format come from one plan.** Server, converter and
   publisher read the same plan, so a format change cannot reach only one side.
6. **Quantization is applied to a copy.** The optimizer continues to write BF16 or FP8
   master weights. The published copy is refreshed under the existing publish lock.

## 4. Architecture

### 4.1 QuantizationPlan

`psrl/utils/quantization/plan.py` resolves the wire format once, in
`validate_config`, and stores a serializable form in `psrl.nixl.publish_quantization`.
Every participant rebuilds the same plan from that field.

```python
@dataclass(frozen=True)
class QuantizationPlan:
    scheme: str                  # "none" | "fp8" | "mxfp8" | "nvfp4" | "compressed_tensors"
    block: tuple[int, int] | None
    scale_suffix: str | None     # "_scale_inv" for vLLM FP8, scheme-specific later
    scale_dtype: str | None
    weight_dtype: str | None
    preset: str | None           # compressed-tensors preset, future
    ignore: tuple[str, ...]      # modules kept high precision, future
    config_file: str | None      # arbitrary compressed-tensors config, future
```

`resolve_quantization_plan` performs four checks before any process starts:

- the scheme is known,
- the scheme is implemented in this phase,
- the block geometry is supported by the implemented publisher and engine path,
- the configured `block` and `preset` are consistent.

Reserved schemes fail with a message that names the phase that will implement them. An
unknown scheme fails as a configuration error.

### 4.2 PublishQuantizer

`psrl/utils/quantization/quantizer.py` defines the publisher interface.

```python
class PublishQuantizer(ABC):
    plan: QuantizationPlan

    def scale_key_of(self, weight_key: str) -> str | None: ...
    def base_of_scale_key(self, key: str) -> str | None: ...
    def scale_dtype(self) -> torch.dtype: ...
    def scale_shape(self, shard: torch.Tensor) -> tuple[int, ...]: ...
    def tile_of(self, shard: torch.Tensor) -> tuple[int, ...]: ...
    def validate_shard(self, key: str, shard: torch.Tensor) -> None: ...
    def quantize(self, shard: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
    def quantize_into(
        self,
        source: torch.Tensor,
        destination: torch.Tensor,
        scale: torch.Tensor,
    ) -> None: ...
```

Phase 1 has one implementation:

- `VllmFp8BlockQuantizer`: block FP8 with 128x128 tiles, `_scale_inv` suffix, float32
  scales. It uses the existing fused CUDA kernel and falls back to the portable
  blockwise implementation. Its bytes are identical to the pre-plan implementation.

Scheme `none` builds no quantizer, so the existing dtype cast is the entire publish path.

The interface is intentionally about one weight and one or more auxiliary tensors. MXFP8
and NVFP4 add implementations, not new transport paths.

### 4.3 Auxiliary parameter discovery

The vLLM converter already walks the live module tree and maps each fused or reshaped
parameter back to its canonical HF view. Phase 1 generalizes the auxiliary half of that
walk:

- the parameter iteration skips a parameter when it is the auxiliary tensor of a sibling
  weight (`is_auxiliary_parameter`),
- for each quantized weight, the converter looks up the module attribute declared by the
  plan (`fused_auxiliary`),
- the existing block-aware `scale_view` maps the fused scale tensor onto the canonical
  weight view and its sharding,
- the auxiliary parameter is emitted with the same sharding as its weight.

This keeps the current FP8 mapping unchanged while removing the compile-time assumption
that every auxiliary tensor is called `_scale_inv`.

### 4.4 Two-phase publish

Schemes split into two classes.

**Local schemes** (`FP8_BLOCK`, `MXFP8`, channel-wise): scale is a function of one tile.
A shard that contains whole tiles can be quantized independently. The plan validates
tile alignment once, and the publisher quantizes each local shard into its publish copy.

**Global schemes** (`TENSOR`, `TENSOR_GROUP`, NVFP4 weight global scale): one scale
depends on the whole tensor. The publisher runs a short collective before the local
quantization pass.

```
Phase A  local amax -> all-reduce over the tensor's shard group -> global amax
         optional cross-projection combine, for example gate and up sharing one scale
Phase B  quantize each local shard with the agreed global scale
         write weight bytes, local scales and global scales into publish copies
```

Phase A is issued before the publish version is stamped. Phase B runs under the publish
lock. A reader therefore sees either the previous complete version or the new complete
version, never a mixture.

For very large MoE models the reduction is over a small vector per projection, not over
weight bytes. The cost is bounded by the number of tensor groups, not by model size.

### 4.5 Engine integration

For the current native FP8 path the engine is created with vLLM's serialized block FP8
configuration. It uses Cutlass or another supported scaled matrix kernel. The publisher
writes weights and scales into the live parameter layout after the engine has run its
post-load hook, so the container geometry and the bytes agree.

For compressed-tensors schemes the engine is created from the config produced by
`build_compressed_tensors_config`. The worker extension already calls
`apply_vllm_quant_patches` when it detects a quantized model, which installs the refit
safe wrappers for compressed-tensors FP8 and MXFP8. The plan path will not force
DeepGEMM. Kernel choice is left to vLLM, with an explicit backend override available
when a cluster needs one.

Two engine details matter for correctness:

- the post-load hook must run exactly once per pulled layout. The current pull path runs
  it under `weights_already_processed` and falls back to a derived-weight refresh. Any
  new scheme must either declare support for pre-processed weights or publish checkpoint
  layout and re-run the hook.
- a second pull must produce the same runtime tensors as the first. This is a test, not
  an assumption.

### 4.6 Configuration

The transport format is a structured field:

```yaml
psrl:
  nixl:
    weight_quantization:
      scheme: fp8        # none | fp8 | mxfp8 | nvfp4 | compressed_tensors
      block: [128, 128]  # null selects the scheme default
      preset: null       # compressed-tensors preset, future
      ignore: []         # regexes kept high precision, future
      config_file: null  # arbitrary compressed-tensors config, future
```

`none` publishes the trainer dtype after the existing cast. `fp8` publishes 128x128
block FP8 with `_scale_inv` scales. The other names are recognized and rejected with a
precise message until their phase lands.

`validate_config` writes the resolved plan to `psrl.nixl.publish_quantization`.
`engine_train_worker` and the vLLM extension rebuild the plan from that field. The
generation and validation engines read the same plan, so the old rule that both rollout
roles must agree is now structural rather than a separate assertion.

The write gate follows the plan. A quantized publication writes into a private copy, so
the optimizer can keep running while readers consume the copy.

### 4.7 Failure semantics

Configuration validation rejects, with the parameter name and a remedy:

- an unknown scheme,
- a scheme reserved for a later phase,
- a block shape the engine does not support,
- a tensor whose shard cuts a tile, with the split that caused it,
- an auxiliary parameter whose shape does not tile its weight,
- mixed generation and validation formats.

Runtime still validates weights and scales before publishing. A malformed parameter is a
bug in the plan or the converter, so it fails the publish instead of writing bytes that
the engine would misread.

## 5. Scheme matrix and status

| Scheme | Wire weight | Wire auxiliary | Compute | Hardware | Status |
|---|---|---|---|---|---|
| `none` | trainer dtype | none | cast only | any | Implemented |
| `fp8` (vLLM serialized) | E4M3, 128x128 block | `_scale_inv`, fp32 | local | Ampere and newer | Implemented and e2e validated |
| `compressed_tensors` `FP8_BLOCK` | E4M3, 128x128 block | `weight_scale`, fp32 | local | Hopper with Cutlass, or DeepGEMM builds | Reserved, Phase 2 |
| `MXFP8` | E4M3, 32-element group | E8M0 scale, packed | local | Blackwell, or a vLLM supported emulation | Reserved, Phase 2 |
| `NVFP4` | E2M1, 16-element block | E4M3 local scale plus fp32 global scale | local plus global reduction | Blackwell | Reserved, Phase 3 |
| `compressed_tensors` `config_file` | scheme dependent | scheme dependent | local or global | scheme and kernel dependent | Reserved, Phase 4 |

Status meanings:

- Implemented: configuration, publisher, converter, engine and tests are in place.
- Reserved: the plan recognizes the name and rejects it with a message that points to the
  phase that adds it. No partial runtime path is left enabled.

Phase 1 status:

| Item | Status |
|---|---|
| `QuantizationPlan` resolution and capability errors | Implemented |
| `PublishQuantizer` interface with passthrough and vLLM FP8 block | Implemented |
| Plan-driven publisher with pooled auxiliary copies | Implemented |
| Plan-driven converter auxiliary discovery | Implemented |
| `psrl.nixl.weight_quantization` config and worker plumbing | Implemented |
| MXFP8, NVFP4 and compressed-tensors publishers | Phase 2 and later |
| Global-scale two-phase publish | Phase 3 |
| Arbitrary `config_file` | Phase 4 |

## 6. Performance analysis

### 6.1 Wire bytes

Let `E` be the number of weight elements and `B` the block size.

| Scheme | Weight bytes | Scale bytes | Relative to BF16 |
|---|---|---|---|
| `none` | `2E` | 0 | 1.0 |
| `fp8` | `E` | about `4E / 128^2` | about 0.5 |
| `MXFP8` | `E` | about `E / 32` | about 0.53 |
| `NVFP4` | `0.5E` | about `0.5E / 16` plus a per-tensor fp32 scale | about 0.53 |

The quantized path therefore halves the trainer egress and the rollout ingress for FP8
and MXFP8, and roughly halves them again for NVFP4. This is the reason to publish
quantized bytes rather than quantize in every rollout.

### 6.2 Quantization compute and memory

Blockwise quantization reads the source once and writes one quantized tile and one scale
per tile. The fused kernel keeps the source in cache and writes directly into the publish
copy, so the peak temporary is one source tensor. A non-fused fallback allocates one
quantized tensor per source tensor, which is the existing behavior.

Auxiliary scales are pooled into one allocation per device before NIXL registration. The
pool matters at MoE scale. One region per weight makes registration cost grow with the
number of weights, and registration happens on every wake. One region per device keeps
that cost constant.

Global schemes add one small all-reduce per tensor group. For a 1 TB MoE model the
reduced vector is on the order of one float per projection, so the collective cost is
negligible against the weight pass. It does add one synchronization point before the
local quantization pass, so the publisher must overlap it with the copy planning of
other tensors where possible.

### 6.3 Memory trade-off

The publisher holds one private copy of each published tensor. For BF16 the copy is the
weight itself. For FP8 the copy is half the size plus the scale pool. The master weights
stay in trainer dtype. Quantization therefore reduces publish memory for FP8 and MXFP8
and reduces it further for NVFP4.

The plan does not materialize full tensors on a rank. For block-local schemes the copy
is the same shape as the local shard. For global schemes the reduction exchanges
statistics, not tensors. An optional all-gather fallback for schemes that need the full
tensor is a deliberate escape hatch, not the default, because it breaks the scaling
property at TB scale.

### 6.4 Scaling to large clusters

- Publication is O(local shard) for block schemes. It does not grow with world size.
- Auxiliary registration is O(devices) regions, not O(weights).
- Scale reductions are O(projection groups) messages, not O(weights).
- Grouped experts batch their global scales into one reduction per projection and layer.
  A 128-expert layer therefore issues one reduction per projection, not one per expert.
- The version and sharding metadata are already O(keys) and are interned by shape and
  dtype, so the handshake does not serialize one copy per expert.
- Readers pull only the shards they hold. Quantized bytes reduce both the RDMA payload
  and the peer diffusion traffic by the same ratio.

## 7. Validation plan

Unit and CPU:

- plan resolution for every scheme name, including reserved and unknown names,
- byte parity between the quantizer and the pre-refactor FP8 path,
- `scale_view` mapping for fused, head-grouped and blocked views,
- converter auxiliary discovery with and without a plan,
- pooled scale allocation and sharding validation.

GPU single node:

- byte-for-byte comparison of published FP8 weights and scales before and after the
  refactor on the same checkpoint,
- two consecutive pulls of the same version, then a pull of a new version, with the
  engine output compared against a fresh engine,
- the existing FP8 rollout and score-centering run, comparing wire bytes, KL, head mass
  and step timing with the recorded baseline.

Multi-node:

- only when a change touches the transport itself. The Phase 1 refactor does not, so the
  multi-node acceptance is a smoke test on the validated node pair.

## 8. Phased plan

| Phase | Scope | Exit criterion |
|---|---|---|
| 1 | Plan and quantizer interface, passthrough and native FP8, plan-driven converter and publisher | Existing FP8 e2e is byte and metric identical, CPU and GPU suites pass |
| 2 | compressed-tensors `FP8_BLOCK` and `MXFP8`, kernel backend selection, refit tests | A supported GPU loads and refits both formats, parity against compressed-tensors reference |
| 3 | NVFP4, global weight scale reduction, QAT activation scales, ModelOpt refit patches | Blackwell e2e run with observed activation scales |
| 4 | arbitrary `config_file`, capability whitelist, optional all-gather fallback | Supported configs run, unsupported configs fail at validation |

## 9. Risks and open questions

- **Refit idempotence.** Compressed-tensors and ModelOpt methods may re-layout weights and
  scales on every post-load call. Upstream's patchers are the reference. The design
  requires a pull-twice test per scheme.
- **Runtime versus checkpoint layout.** The converter reads live parameters after the
  post-load hook. A scheme whose runtime layout is packed or swizzled must be published in
  that same layout, or the pull path must publish checkpoint layout and re-run the hook.
  This decision is per scheme and must be explicit in the plan.
- **Global scales.** NVFP4 quality depends on activation global scales observed during
  QAT. A BF16 RL trainer has no such observations. Phase 3 must either integrate QAT or
  state clearly that the format is transportable but quality is not validated.
- **Hardware.** MXFP8 and NVFP4 kernels are not available on every accelerator. Format
  and packing can be validated without the kernel. End to end validation needs a
  supported GPU.
- **Kernel backend defaults.** On the H20 environment, compressed-tensors `FP8_BLOCK`
  initializes with an explicit non-DeepGEMM backend such as `cutlass` or `marlin`. The
  automatic choice reaches DeepGEMM and fails while transforming block scales, because
  the DeepGEMM JIT requires a newer NVCC than the image ships. The native FP8 rollout
  path inherits the same upstream default, so the engine probes the NVCC release before
  accepting it and falls back to vLLM's own selection when the toolchain is older than
  12.9. Phase 2 still needs the same probe for compressed-tensors rather than inheriting
  upstream's DeepGEMM default.
- **Model layouts.** Fused QKV with grouped-query attention and head sizes smaller than
  the quantization tile cannot be expressed by a naive scale view. The converter must
  reject them at validation or map them with a dedicated rule.

## 10. Results

Phase 1 was validated on one node with eight H20 GPUs. The FP8 path ran the
score-centering configuration of `qwen3_8b_megatron_direct_fp8gen.sh` shape: Qwen3-8B,
Megatron TP2, two generation GPUs, one validation GPU, one training step. The same
configuration ran before and after the plan refactor.

| Measurement | Before the plan | After the plan |
|---|---|---|
| Wire payload per rollout rank | 4.72 GB | 4.72 GB |
| First pull, TP0 and TP1 | 0.058 s, 0.056 s | 0.061 s, 0.053 s |
| Peak training allocation | 73.0771 GB | 73.0774 GB |
| Sampler head mass | 0.99999905 | 0.99999914 |
| Trainer head mass | 0.99999908 | 0.99999921 |
| Actor update, per training token | 370 tok/s | 467 tok/s |

The two runs sampled different rollouts, so reward and KL differ. The transport and
memory invariants are identical: the same wire payload, the same pull latency within
noise, and the same peak allocation within 0.3 MB. The per-token actor rate is reported
for completeness, not as a comparison, because the two micro-batches had different token
counts.

CPU validation:

- 65 unit tests cover plan resolution, quantizer byte parity, scale geometry, auxiliary
  discovery and the publish-format configuration.
- 53 pull-engine simulations cover exact FP8 bytes through the plan-driven publisher for
  every layout, including peer diffusion and pooled scale copies.
- The quantizer's bytes and scales match the pre-plan `quantize_blockwise` path exactly
  on random, strided and sharded inputs.

GPU validation used the single-node layout above. A multi-node run is not required for
Phase 1 because the transport path and wire shape did not change, and the earlier
multi-node transport acceptance still applies.

After merging the sandbox and harness work, the same path was re-run as a colocated
one-step smoke on one eight-GPU H20 node: Qwen3-8B, block-FP8 publish, direct supply,
four prompts with two responses each. The step completed with `actor/pg_loss` -0.00265
and `actor/grad_norm` 0.400. The rollout and trainer distributions agreed at
`rollout_actor_probs_pearson_corr` 0.9878, `rollout_corr/kl` -0.00116 and
`rollout_corr/ppl_ratio` 0.9989, which is the same agreement the pre-merge runs showed.
The engine's DeepGEMM probe fell back to vLLM's own kernel selection because the host
NVCC is 12.8, and the rollout still ran.
