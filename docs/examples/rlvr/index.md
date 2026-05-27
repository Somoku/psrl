# RLVR: Reinforcement Learning from Verifiable Rewards

RLVR trains language models on tasks where correctness can be verified programmatically, math problems with numeric answers, code with test suites, or logic puzzles with deterministic solutions. This eliminates the need for learned reward models and enables stable, high-signal training.

---

## Supported Algorithms

Listed in order of increasing specialization, PPO is the foundational actor-critic
algorithm, GRPO removes the critic via per-prompt group normalization, and DAPO
extends GRPO with asymmetric clipping plus a dynamic sampling filter.

| Algorithm | Key Idea | Status |
|-----------|----------|--------|
| [PPO](ppo) | Proximal Policy Optimization with GAE and a learned critic | Coming soon |
| [GRPO](grpo) | Group Relative Policy Optimization, per-prompt advantage normalization, no critic | Coming soon |
| [DAPO](dapo) | GRPO variant with asymmetric clipping + dynamic sampling filter | Ready |

All algorithms support both FSDP and Megatron-LM training backends, with vLLM rollout and NIXL-based GPU-direct model transfer.

---

## Common Setup

Every RLVR example shares the same infrastructure. See {doc}`common` for the
full reference covering trainer architecture, cluster-layout parameters, and
monitoring metrics. At a glance:

1. **Data**: Parquet files with `prompt`, `data_source`, and `reward_model.ground_truth` fields.
2. **Reward**: A `compute_score(data_source, solution_str, ground_truth, extra_info)` function that returns `{"score": float, "acc": float}`.
3. **Training**: PSRL's decoupled trainer with configurable staleness, Token Importance Sampling (TIS), and dynamic batching.
4. **Monitoring**: WandB logging with `critic/score/mean` (raw task correctness), `critic/rewards/mean` (shaped reward), and `val-core/<data_source>/acc/mean@N` (validation accuracy) metrics.

```{toctree}
:maxdepth: 1
:hidden:

common
ppo
grpo
dapo
```
