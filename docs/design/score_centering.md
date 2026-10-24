# Score centering

Score centering removes the bias that the training-inference mismatch leaves in the policy gradient.
The rollout sampler (vLLM) and the trainer (FSDP or Megatron) run the same weights through different
kernels, dtypes and batch shapes, so the distribution the agent drew from is not the distribution the
trainer scores. Score centering keeps a sparse record of the sampler's distribution and subtracts the
trainer's own expected score from the policy-gradient term, which removes the mismatch drift without
changing the objective in expectation. It follows arXiv 2609.20807 and is trainer-side only: the
sampler just reports its head.

This page explains what score centering adds on top of PPO, the two head models PSRL supports, how a
capture reaches the loss, and what it costs. The capture wire contract lives in
[sampler_distribution.md](sampler_distribution.md). A Chinese version of this page is in
[score_centering_zh.md](score_centering_zh.md).

## 1. What PPO gets wrong off-policy

PPO's clipped ratio `r = pi_theta / pi_old` assumes `pi_old` is the distribution that produced the
rollout. In an asynchronous RL loop that assumption fails twice.

- **Staleness.** `pi_old` is a few steps behind, which the ratio is designed to absorb.
- **Sampling mismatch.** Even at zero staleness, the log-probs the sampler recorded come from
  different numerics than the trainer's forward pass, so the behavior distribution `q` differs from
  the trainer's `p` by a small, systematic drift.

Score centering targets the second one. The drift does not average out on its own: under a constant
reward the policy still moves, because the sampled token's score is not zero-mean under the sampler.

When the mismatch is large, PSRL can use bypass mode, a REINFORCE objective with an importance weight
`w = f(p / q)`:

```
L = -E_{a~q} [ w(a) * log p(a) * A ]
```

This is unbiased for `p = q`, but its gradient still carries `E_{a~q}[w(a) * grad log p(a)]`, the drift
term. Score centering adds a correction that cancels it.

## 2. The score-centering term

Per response position, let `H` be a set of candidate tokens (the sampler's head, defined in section
3), `q_v` and `p_v` the sampler's and the trainer's probabilities at candidate `v`, and `A` the
advantage. Define:

```
head_mass_v = q_v * f(p_v / q_v)          # sampler mass reweighted by the IS rule
rho         = (1 - q_mass) / (1 - p_mass) # tail-mass ratio, floors at 1e-6
alpha       = rho * f(1 / rho)            # tail scale
residual_v  = head_mass_v - alpha * p_v
correction  = sum_{v in H} sg[residual_v] * log p_v
```

The training loss for the position becomes:

```
L = -A * w(a) * log p(a) + A * correction
```

Reading the gradients explains why this works. The correction's gradient is
`sum_v residual_v * grad log p_v`, with the residual held constant. Added to the main term's expected
gradient `-sum_v q_v * f(p_v/q_v) * grad log p_v * A`, the drift carried by the head cancels exactly,
and `alpha` projects the trainer's missing head mass onto its tail so the cancellation extends to the
whole distribution up to the accuracy of the tail model. Support mode needs no tail model at all,
because its head is complete: `p` sums to one over the head, so `alpha` is zero. In every case the
correction changes the gradient's variance, not its expectation.

Implementation notes that matter if you touch this code:

- The residual is computed under `torch.no_grad()` and only `log p_v` carries gradient. Removing that
  detach turns the correction into a different objective.
- A padding slot carries the sentinel id `-1` and `-inf` log-prob, so the head mask drops it before it
  can contribute.

## 3. Two head models

The head `H` is what the sampler recorded at rollout time. PSRL supports two models of it, selected by
the sampling parameters.

| | top-k (`top_p = 1`) | support (`top_p < 1`) |
|---|---|---|
| Head | the sampler's `k` highest-probability tokens | the sampler's complete nucleus |
| Trainer `p` on the head | full-vocabulary normalization | renormalized over the recorded support |
| Tail | modeled by `alpha`, with `rho` the tail-mass ratio | none, `alpha = 0` |
| Correction | `sum_H (q_v w_v - alpha p_v) log p_v` | `sum_H q_v w_v log p_v` |
| Capture width | `sampler_max_tokens` | `sampler_max_tokens` plus one sentinel slot |

**top-k** is the default when sampling is untruncated. The head is exact and the tail is modeled as the
trainer's tail rescaled to the sampler's tail mass, so the correction never has to see the full
vocabulary. `rho` and `alpha` carry the tail approximation.

**support** is the exact model for truncated sampling (`top_p < 1` or `top_k > 0`). The head is the
sampler's complete support, proven complete by a sentinel entry, so there is no tail and `alpha` is
zero. Here the trainer distribution is the support-conditional one: the trainer renormalizes its
`p` over the recorded support, and the sampled token's policy log-prob is shifted by
`sc_log_support_mass` before the REINFORCE term sees it, so the policy term and the correction share
one distribution and one gradient. The full-vocabulary log-prob is kept for the KL term. This is
what makes support mode exact rather than tail-approximated.

## 4. Importance-weight rules

`f` is chosen from the rollout-correction config and applies to both `head_mass` and `alpha`. The
work is done on probabilities or log-ratios, so an unbounded ratio is never exponentiated and the
`q_v = 0` case does not produce `0 * inf`.

| `rollout_is` / threshold | `f(r)` | Effect |
|---|---|---|
| `None` | `1` | no IS weight, `head_mass = q` |
| single float `c` (TIS) | `min(r, c)` | upper-clip, the head becomes `min(p, c q)` |
| `"low_high"` (IcePop) | `1` inside the band, `0` outside | zeroes ratios outside `[low, high]` |

The sampled token's own weight `w(a) = f(p(a)/q(a))` is applied by the REINFORCE term through
`rollout_is_weights`.

## 5. Workflow

```
rollout (vLLM)                       SMG gateway + TITO              PSRL agent loop
  sampler_top_logprobs width  --->   per-turn sampler_distribution   --->  SamplerDistribution
                                     (offsets, ids, log-probs,             (validate, trim)
                                      complete)                            |
                                                                           v
training (FSDP / Megatron)           veRL SC logits processor        sampler_adapter
  sc_correction  <--- centering  <---  p at head ids, q at head ids  <---  dense head or
  policy log-prob                     (chunked, vocab-parallel)            compact support
```

1. **Capture.** SessionRouter asks SMG for `sampler_top_logprobs` on chat, Responses, Anthropic
   Messages and `/generate`. Chat and Responses lower to the canonical chat training IR inside SMG.
   TITO stores one compact `SamplerDistribution` per turn and preserves it across partial rollouts.
2. **Assemble.** `psrl.utils.tito.training_data` walks the turns, aligns them by greedy match, trims
   boundary tokens, and concatenates the per-turn distributions. `psrl.utils.sampler_adapter` lays
   the head over the full prompt+response sequence at the prediction row `prompt_length - 1 + r`.
3. **Score the head.** The engine's logits processor computes the trainer's head log-probs and the
   centering term from the same logits, then the actor loss adds `A * sc_correction`.
4. **Log.** `sc_correction`, `sc_sampler_head_mass`, `sc_train_head_mass` and `sc_tail_ratio` are
   reported per token as `actor/sc_*`.

Minimal configuration (capture on the rollout, flag on the loss):

```yaml
algorithm:
  rollout_correction:
    score_centering: true
    bypass_mode: true          # required
    loss_type: reinforce       # required
    rollout_is: token          # None | token
    rollout_is_threshold: 2.0  # TIS float, or a "low_high" IcePop band
gen_actor_rollout_ref:
  rollout:
    sampler_distribution: auto # disabled | topk | support | auto
    sampler_max_tokens: 32
    sampler_compact_support: true
    logprobs_mode: processed_logprobs
    temperature: 1.0
    top_p: 1.0                 # topk needs 1.0, support allows < 1
```

`auto` resolves to `support` when `top_p < 1` or `top_k > 0`, and to `topk` otherwise. The flag is
mirrored onto the actor's static `policy_loss` before the worker selects its loss function, because
the worker picks `loss_fn` once in `init_model` and a driver-side mutation at train time would arrive
too late. Score centering requires materialized logits, rejects distillation, speculative sampling,
`mtp` training and dynamic context parallelism, and a validation session disables capture.

## 6. Performance

Score centering is not free. It needs the trainer's logits at arbitrary ids, so the fused
logits/loss kernels stay off and the full `(rows, vocab)` logits are materialized. On top of the
cross-entropy pass it adds the head normalizer and a backward recompute, about two more
`(rows, vocab)` passes. PSRL and the veRL patch keep that cost bounded and keep the communication and
host memory small.

| Optimization | Mechanism | Effect |
|---|---|---|
| Element-budget chunking | `score_centering_chunk_elems` on the actor engine config (64M elements, 256 MiB default) bounds the fp32 `(rows, vocab)` workspace. The row count per chunk is derived from the local vocabulary size | Memory stays flat as the vocabulary grows |
| Backward recompute | The normalizer primitive saves only the logits, the ids and two scalars per row and recomputes the softmax per chunk in backward | No `(rows, vocab)` softmax is stored for backward |
| Vocab-parallel reductions | Under tensor parallelism only the selected logits and the row normalizer cross ranks, never the full vocabulary | Rank memory scales with `vocab / TP` |
| Support path | Support mode normalizes over each row's recorded support, so it is `O(rows * k)` with a single reduction rather than a vocabulary-sized one | Support mode is cheap regardless of vocabulary |
| Compact transport | `sampler_compact_support` ships per-row lengths plus flat ids and log-probs instead of a dense `[T, k]` head, and the trainer reconstructs on device. It is the default and works under context parallelism | Much smaller rollout-to-trainer payload |
| Batched FSDP gather | All five per-token fields are gathered in one collective | One collective instead of five, per micro-batch |
| Host sync | The compact field reads min, max and sum of the row lengths in one device-to-host sync | One sync per field, not one per check |
| Validation gating | `sampler_strict_validation` gates the per-row duplicate, mass and agreement checks. Structural checks always run | Long responses skip the `O(rows)` Python pass in training |
| Vectorized TITO assembly | The per-turn token split and boundary match run in numpy | No Python loop over every response token |

Costs to budget for, not bugs:

- **Materialized logits.** At long sequence the peak is dominated by the `(rows, vocab)` logits plus
  their gradient. Use tensor or context parallelism to keep `rows` per rank manageable.
- **Megatron CE clone.** Megatron's fused vocabulary-parallel cross-entropy mutates its saved logits,
  and score centering reuses them in its own backward, so the engine clones the logits before the
  cross-entropy call. With entropy enabled that is a third `(rows, vocab)` copy at peak.
- **Per-chunk collectives.** The vocabulary-parallel path issues three all-reduces per row chunk, so
  the collective count scales as `rows * vocab / (chunk_elems * TP)`. Tensor parallelism and a larger
  budget both reduce it.
- **Sampler-side cost.** vLLM computes top-k log-probs per position to fill the capture width, and the
  gateway assembles one per-position head. The capture width is the main knob: raise
  `sampler_max_tokens` only as far as the sampling actually needs.

## 7. Code map

| Path | Role |
|---|---|
| `psrl/utils/sampler_distribution.py` | `SamplerCapture`, `SamplerDistribution`, capture-width definition and validation |
| `psrl/utils/sampler_adapter.py` | Dense and compact producers over the prompt+response sequence |
| `psrl/utils/tito/training_data.py` | Turn alignment, trimming, per-turn distribution assembly |
| `psrl/workers/gen/session_router.py` | Capture request contract per protocol |
| `psrl/workers/agent_loop/loops/` | Per-turn capture, TITO session handling, `/generate` decode |
| `third_party/verl/verl/trainer/ppo/score_centering.py` | Head log-prob primitives, centering term, logits processor, loss entry |
| `third_party/verl/verl/workers/utils/losses.py` | Adds `sc_correction` to the policy loss; the logits processor already renormalizes the sampled log-prob over the support |
| `third_party/verl/verl/workers/engine/{fsdp,megatron}/transformer_impl.py` | Engine wiring, chunk budget, CP split and Megatron CE clone |

The veRL changes ship as `patch/verl/dc709af.patch`, applied to the pinned veRL revision.

## Verification

The focused CPU suite runs from the PSRL root with the patched veRL on the path:

```bash
PYTHONPATH=third_party/verl python -m pytest -q \
  tests/utils/test_sampler_distribution.py \
  tests/utils/test_sampler_adapter.py \
  tests/tito/test_session_router.py \
  tests/tito/test_session_agent_loop.py \
  third_party/verl/tests/trainer/ppo/test_score_centering_on_cpu.py
```

This covers the capture contract, the two head models, the loss integration and the multi-turn
alignment. It does not replace accelerator runs: validate the FSDP/Ulysses and Megatron TP/CP/PP
combinations you actually train with, and measure peak training memory and capture bandwidth at the
chosen capture width.

## 8. Validation results

The CPU suites above pin the estimator. The accelerator path is checked end to end with GSM8K on a
single 8-GPU node: 4 training GPUs, one rollout instance and one validate instance. FSDP2 runs use
Qwen2.5-0.5B-Instruct, top-k 32 at `temperature=1.0, top_p=1.0`, and support captures at
`top_p=0.9`; responses are capped at 1024 tokens. Reward uses a test-only GSM8K scorer that accepts
both `####` and `\boxed{}` answers, because an instruct model often ignores the requested format and
would otherwise return zero reward for correct answers.

| Head | IS rule | `sc_support_aligned` | sampler head mass | trainer head mass | `sc_correction` |
|---|---|---|---|---|---|
| top-k 32 | none | 0 | 0.988 | 0.987 | -4.5e-4 |
| support | none | 1.000 | 1.000 | 0.976 | -8.2e-4 |
| support | IcePop 0.9-1.1 | 1.000 | 1.000 | 0.976 | +1.9e-2 |
| top-k 32 | TIS 2.0 | 0 | 0.994 | 0.994 | -5.4e-5 |

What the rows establish:

- Support rows report `sc_sampler_head_mass = 1`, `sc_sampler_tail_mass ≈ 0` and
  `sc_support_aligned = 1`, i.e. the captured nucleus is complete and the trainer renormalizes on it.
- The trainer and sampler head masses agree to ~1e-5 on top-k rows while the sampled log-probs agree
  to ~1e-3 (`rollout_corr/log_ppl_diff`), so the two engines see the same distribution to the
  precision the estimator needs. The correction stays near zero on-policy as it should.
- IcePop's band rejects 2.2% of token ratios (`rollout_is_oob_ratio`) while support alignment remains
  exact, and TIS reports `rollout_is_mean ≈ 1`, max ratio 1.19 and effective sample size 0.999.
- A support budget that is too small fails closed: with `sampler_max_tokens=128` the nucleus on
  GSM8K exceeds the budget and the agent loop raises `Sampler support exceeds the capture budget`.
  `512` covers this distribution; size the budget from the realized nucleus, not from the head mass.

The Megatron engine was run through the same path with tensor parallelism 2 on Qwen2.5-0.5B and
Qwen3-8B (single node, 2 training GPUs, 4 rollout GPUs). The vocab-parallel normalizer reports head
masses within 1e-4 of the sampler on the 0.5B model (0.99515 vs 0.99517) and within 1e-6 on the 8B
base model (0.9999987 vs 0.9999988, sampled log-prob difference 7.4e-4), the same agreement FSDP
shows. The THD split paths for heads, completion receipts and sampled labels are exercised on every
step; tensor parallelism 4 uses the same code path with a wider vocabulary shard. The same 8B TP2
run also completed with a block-FP8 rollout (`psrl.nixl.weight_quantization.scheme=fp8`): each rollout
rank pulled 4.72 GB, sampler and trainer head masses still agreed to 3e-7, and the expected
quantization gap appeared as KL 5.0e-3 with the CUTLASS kernel and 8.8e-3 with DeepGEMM.

### Training cost

The 8B Megatron TP2 runs above differ only in score centering. The optimizer-dominated peak is
unchanged (72.44 GB allocated, the same to 0.001 GB); the chunked fp32 normalizer workspace stays
under the 256 MiB element budget instead of growing with the vocabulary. Actor update time per
training token is 404 tok/s with centering against 417 tok/s without it (~3% more), and MFU drops
from 0.344 to 0.298. On FSDP2, 8B at top-k 32 peaks at 42.0 GB allocated (45.7 GB reserved) for a
~2.0k-token micro-batch and updates at 155 training tok/s: the processor's workspace is bounded on
top of the materialized logits, not a second vocabulary-sized buffer.
