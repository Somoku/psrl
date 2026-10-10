# Sampler distributions for score centering

This infrastructure captures the sampling distribution during rollout and exposes differentiable
sparse trainer log-probs, then feeds the score-centering correction into the actor loss through
veRL's `score_centering_ppo_loss`. Both the top-k head model (`top_p=1`) and the exact top-p support
model (`top_p<1`) are supported.

The algorithm, its two head models, the end-to-end workflow and the performance profile are described
in [score_centering.md](score_centering.md). This page covers the capture wire contract.

## Capture

Use a session agent that sends chat completions through SessionRouter and TITO.
Apply these overrides to its rollout configuration:

```yaml
gen_actor_rollout_ref:
  rollout:
    sampler_distribution: auto   # disabled | topk | support | auto
    sampler_max_tokens: 128      # topk head size, or the support budget
    sampler_compact_support: true
    sampler_strict_validation: false
    logprobs_mode: processed_logprobs
    temperature: 1.0
    top_p: 1.0                   # topk needs 1.0; support allows < 1
    top_k: -1
```

The vLLM engine receives a sufficient `max_logprobs` limit automatically. An
explicit smaller engine limit is rejected. Capture requires the matching SMG
source changes and a rebuilt gateway and gRPC servicer. A missing capture in a
training session is an error. Validation sessions disable capture.

For truncated sampling the compact capture can also hold a complete top-p support
(`sampler_distribution: support`, with `sampler_max_tokens` as the per-token budget and one extra
sentinel entry whose negative infinite log probability proves the finite support fits). Support mode
is trainable: the trainer renormalizes its log-probs over the recorded support, so the centering term
becomes the exact expected score with no tail model. Unused slots and masked environment/tool rows
carry the `HEAD_PADDING_ID` sentinel (`-1`, `-inf`) and are excluded by a head mask, so they never
contribute to the loss.

Capture requires positive temperature and non-speculative sampling. SessionRouter
buffers one non-streaming completion per turn. The chat, Responses and Anthropic Messages paths all
capture (Messages and Responses lower to the canonical chat training IR inside SMG). The native
`/generate` path returns the head in its own response metadata instead of TITO. Multiple turns,
tool/environment tokens, branching TITO trajectories and abort/continuation use the same row
alignment contract.

## Data contract

SMG accepts `sampler_top_logprobs` independently of the response's
`top_logprobs`. The former controls TITO capture. The latter controls alternatives
returned to the agent. PSRL requests zero response alternatives to avoid sending
the captured head twice.

Each TITO turn contains `sampler_capture_width` and, when captured,
`sampler_distribution`:

| Field | Wire representation | Meaning |
|---|---|---|
| `offsets` | Integer array | Entry boundaries for each generated token |
| `token_ids` | Base64 little-endian int32 | Flattened vocabulary IDs |
| `log_probs` | Base64 little-endian float32 | Processed sampler log probabilities |
| `complete` | Boolean array | A masked sentinel proved complete support |

IDs and probabilities have equal entry counts. Rows cannot contain duplicate IDs,
invalid probabilities or an inconsistent sampled-token probability. Complete
supports include the sampled token. Incomplete heads may omit it, so the sampled
token's own log probability remains a separate field.

The gateway preserves all rows across partial rollouts. A missing nonempty segment
invalidates the capture. An empty abort adds no rows. TITO tree copies share
immutable entry buffers. Only finite entries are serialized.

PSRL applies boundary trimming to tokens and distributions together. Tool and
environment tokens have empty rows. The batch carries the upstream dense
`rollout_topk_ids` / `rollout_topk_log_probs` pair over the full prompt+response
sequence: response token `r` sits at prediction row `prompt_length - 1 + r`, and
prompt, padding and masked environment/tool rows carry the sentinel head. Entry
buffers use int32/float32 and stay compact on the wire; the adapter expands them.
`sampler_distribution: auto` resolves to `support` when `top_p<1` or `top_k>0` and to
`topk` otherwise.

## Training interface

`psrl.utils.sampler_adapter` converts the compact rows into the training contract over the full
prompt+response sequence. Response token `r` sits at prediction row `prompt_length - 1 + r`; prompt,
padding and masked environment/tool rows carry the `HEAD_PADDING_ID` sentinel (`-1`, `-inf`) and are
excluded by a head mask. Three producers match the trainable paths:

- `to_dense_topk`: fixed-width top-k head, consumed with the full-vocabulary normalizer.
- `to_dense_support`: right-pads each row to `sampler_max_tokens + 1` for the complete support.
- `to_compact_support` (`rollout.sampler_compact_support`): ships per-row lengths plus the flat
  support ids and log-probs (instead of the dense `[T, K]` heads), and the trainer reconstructs both
  dense arrays on device. It is the default and works under context parallelism. It is a
  host/TransferQueue-memory win: GPU memory is unchanged, since the trainer still materializes the
  dense heads for the loss.

The engine consumes the head through veRL's `score_centering_ppo_loss` on FSDP, or the Megatron
`_lm_head_logits_processor` branch that normalizes with the tensor-parallel group. FSDP gathers the
DTensor vocabulary shard before the logits processor, so it needs no process group. Megatron scales
the logits by temperature in the engine and passes the tensor-parallel group plus the unpadded
vocabulary size, and clones the logits before its destructive cross-entropy backward. Both paths
require materialized logits; fused logits/loss paths are rejected. In support mode the processor
renormalizes the trainer over each row's recorded support (`_SupportHeadLogProbs`) instead of the
full vocabulary, so the centering term is the exact expected score with no tail model.

`rollout.sampler_strict_validation` gates the per-row checks that sort each row (duplicate ids,
probability mass, sampled-token agreement); structural checks always run, so normal training stays
cheap and CI-style runs can enable the full check. The logits processor emits `sc_correction`,
`sc_sampler_head_mass`, `sc_train_head_mass` and `sc_tail_ratio` per token, all logged as `actor/sc_*`
(`sc_tail_ratio` is the top-k tail-mass ratio; it is neutral in support mode, whose complete support
has no tail).

## Verification

Run the focused CPU suite from the PSRL root:

```bash
python -m pytest -q tests/utils/test_sampler_distribution.py tests/tito/test_training_data.py tests/tito/test_session_router.py tests/tito/test_session_agent_loop.py
```

The suite covers the compact layout, capture fail-closed checks, and the multi-turn row alignment.
It does not replace CUDA/NCCL tests or training runs.

Before enabling score centering, validate real FSDP/Ulysses and Megatron TP/CP/PP combinations,
dynamic batching, abort under weight synchronization and multi-turn tool trajectories on
accelerators. Measure capture bandwidth, TITO memory, peak training memory and throughput with the
intended capture width. The compact representation bounds storage by the captured entries; vLLM
still transfers its requested rectangular logprob buffer before the gateway compacts it.
