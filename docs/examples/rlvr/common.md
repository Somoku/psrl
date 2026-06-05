# Common Infrastructure

All RLVR recipes (PPO, GRPO, DAPO) share the same trainer wiring, cluster-layout
knobs, sequence-length settings, and monitoring metrics. This page documents the
algorithm-agnostic pieces in one place, so the individual algorithm pages can
focus on the bits that actually differ between recipes.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  PSRL RLVR Trainer                  │
│  (actor, vLLM rollout, reward scoring, TIS)         │
└──────────────────────┬──────────────────────────────┘
                       │
         ┌─────────────┴──────────────┐
         │                            │
         ▼                            ▼
┌─────────────────┐         ┌──────────────────┐
│  vLLM GenWorker │         │  FSDP / Megatron │
│  (rollout gen)  │         │  TrainWorker     │
└────────┬────────┘         └────────┬─────────┘
         │                           │
         └───── NIXL (GPU-direct) ───┘
```

The trainer orchestrates:
- **Rollout generation** on dedicated vLLM nodes
- **Reward scoring** via verifiable answer checking
- **Policy updates** with PPO-style clipped surrogate loss
- **Token Importance Sampling (TIS)** for off-policy correction under PSRL's
  decoupled train/gen architecture

For deeper background on the decoupled design, see {doc}`../../design/architecture`
and {doc}`../../design/staleness_control`.

---

## Cluster Layout Parameters

These shell variables show up in every RLVR launch script. Typical values are
shown below, but each recipe's launch script sets concrete values for its
target model and cluster size.

| Parameter | Typical | Description |
|-----------|---------|-------------|
| `NNODES` | `4` | Total nodes in the job |
| `GEN_NNODES` | `2` | Nodes dedicated to vLLM rollout |
| `TRAIN_NNODES` | `2` | Nodes dedicated to training |
| `GEN_TP` | `1` | Tensor parallelism for rollout |
| `TRAIN_SP` | `2` | Sequence parallelism during training |
| `TRAIN_FSDP` | `8` | FSDP shard group size (FSDP backend only) |

---

## Sequence Parameters

| Parameter | Typical | Description |
|-----------|---------|-------------|
| `max_prompt_length` | `2048` | Max tokens in the initial prompt |
| `max_response_length` | `10240` | Max tokens generated per response |
| `enable_overlong_buffer` | `True` | Penalize overly long responses |

---

## Monitoring

Track training progress in WandB:

| Metric | Description |
|--------|-------------|
| `critic/score/mean` | Mean raw task score on the training batch (strict answer correctness, primary progress indicator) |
| `critic/score/max` / `critic/score/min` | Best / worst raw score in the batch, useful for spotting reward collapse |
| `critic/rewards/mean` | Mean shaped reward (raw score + overlong / format penalties) fed into advantage estimation |
| `critic/advantages/mean` | Mean advantage used for the policy update (GAE for PPO, group-normalized for GRPO/DAPO) |
| `val-core/<data_source>/acc/mean@N` | Validation accuracy on each data source (evaluated every `test_freq` steps, `N` = `val_kwargs.n`) |
| `response_length/mean` | Mean generated response length, should stabilize as training converges |
| `actor/entropy` | Policy entropy, a slow decrease indicates healthy exploration → exploitation |
| `actor/grad_norm` | Gradient norm, sudden spikes often signal an unstable update |
