# Sampler distributions for score centering

This infrastructure captures the sampler distribution and exposes differentiable
sparse trainer log probabilities. It does not enable a score-centering loss.

## Capture

Use a session agent that sends chat completions through SessionRouter and TITO.
Apply these overrides to its rollout configuration:

```yaml
gen_actor_rollout_ref:
  rollout:
    sampler_distribution: topk
    sampler_max_tokens: 128
    logprobs_mode: processed_logprobs
    temperature: 1.0
    top_p: 1.0
    top_k: -1
```

The vLLM engine receives a sufficient `max_logprobs` limit automatically. An
explicit smaller engine limit is rejected. Capture requires the matching SMG
source changes and a rebuilt gateway and gRPC servicer. A missing capture in a
training session is an error. Validation sessions disable capture.

For truncated top-p, top-k or min-p sampling, use `sampler_distribution: support`.
Set `sampler_max_tokens` to the largest support to accept per token. The request
captures one additional entry. A masked entry with negative infinite log
probability proves that the finite support fits. A near-unit probability sum
does not prove completeness. Trajectories exceeding the budget are rejected.

Capture requires positive temperature and non-speculative sampling. SessionRouter
buffers one non-streaming completion per turn. Multiple turns, tool/environment
tokens, branching TITO trajectories and abort/continuation use the same row
alignment contract. The native `/generate`, Responses and Anthropic request
paths reject capture rather than returning incomplete training data.

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
environment tokens have empty rows. The batch fields are `sampler_offsets`,
`sampler_token_ids`, `sampler_log_probs` and `sampler_complete`. Entry buffers stay
compact and use int32/float32. Offsets cover the full sequence. Response row `r`
is placed at prediction position `prompt_length - 1 + r`. Prompt and padding
positions have no entries. Synthetic training samples also have empty buffers.

## Training interface

`psrl.utils.sparse_logprobs.sparse_log_probs` queries arbitrary vocabulary IDs.
It accepts a local vocabulary or a tensor-parallel process group. It uses FP32
normalization in bounded row chunks and recomputes softmax during backward.
Only row normalizers and sparse indices are retained alongside the input logits.
TP forward uses MAX/SUM reductions without gathering vocabulary shards.
Backward does not reduce replicated gradients a second time.

`LossWithLogitsProcessor` in `psrl.utils.sampler_training` attaches a callback to
an existing engine loss. The callback receives `logits`, `data`, `row_ids` and
`process_group`, and returns named per-token tensors. These tensors reach the
loss through the engine's existing model-output path. Use
`query_sampler_log_probs` to query the compact batch fields inside this callback.
Attaching a processor does not change the policy loss by itself.

FSDP assigns row IDs using the same packed padding and Ulysses partition as its
labels. Megatron sends row IDs through its model-forward preprocessing, including
THD/BSHD, zigzag/contiguous CP, dynamic CP, TP alignment and FP8 padding. Its
existing output postprocessing reconstructs per-token outputs. DP, PP and VPP
retain their existing microbatch ownership.

The callback requires materialized logits. Fused logits/loss paths are rejected.
Megatron MTP training and simultaneous top-k distillation require explicit
composition and are rejected. FSDP disables destructive logprob backward when a
callback is attached. Megatron isolates its destructive cross-entropy branch
with a logits clone. FSDP's existing DTensor vocabulary gather is unchanged.

## Verification

Run the focused CPU suite from the PSRL root:

```bash
python -m pytest -q tests/utils/test_sampler_distribution.py tests/utils/test_sparse_logprobs.py tests/utils/test_sampler_engine_hooks.py tests/tito/test_training_data.py tests/tito/test_session_router.py
```

The suite includes a two-process Gloo comparison with the full-vocabulary
reference. The engine hook tests exercise source methods without initializing
models. They do not replace CUDA/NCCL tests or training runs. The FSDP hook test
requires the nested tensor API used by the pinned veRL version.

Before enabling a score-centering loss, validate real FSDP/Ulysses and Megatron
TP/CP/PP combinations, dynamic batching, abort under weight synchronization and
multi-turn tool trajectories on accelerators. Measure capture bandwidth, TITO
memory, peak training memory and throughput with the intended support budget.
The sparse representation bounds storage by the captured entries. vLLM still
transfers its requested rectangular logprob buffer before the gateway compacts it.
