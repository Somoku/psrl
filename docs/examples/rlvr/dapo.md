# DAPO: Dynamic Sampling Aligned Policy Optimization

DAPO (Dynamic sampling Aligned Policy Optimization) is the third entry in the
[PPO](ppo) → [GRPO](grpo) → DAPO progression: it keeps GRPO's per-prompt
group-relative advantages (no critic) and adds **asymmetric clipping** plus a
**dynamic sampling filter** for more stable RL training. Concretely, it groups rollouts
by prompt, normalizes advantages within each group, and uses different clip bounds for
the upper and lower probability ratio.

```{seealso}
This page covers DAPO-specific bits only. For the trainer architecture, cluster
layout, sequence parameters, and monitoring metrics shared by all RLVR recipes,
see {doc}`common`.
```

---

## DAPO vs. GRPO

DAPO inherits everything from [GRPO](grpo) and changes two things:

| Aspect | [GRPO](grpo) | DAPO |
|---|---|---|
| PPO clip bounds | Symmetric: `clip_ratio` | Asymmetric: `clip_ratio_low` < `clip_ratio_high` |
| Group filtering | All groups used | Dynamic sampling filter drops uninformative groups (all-correct / all-wrong) |

Both extensions aim to keep training on informative, exploratory updates: the
asymmetric upper clip lets the policy push more aggressively on under-probable but
high-reward tokens, while the dynamic sampling filter avoids wasting steps on prompts
where every rollout already shares the same outcome.

---

## Training Paths

| Path | Backend | Model | Cluster | Launch Script |
|------|---------|-------|---------|---------------|
| **FSDP 3B** | FSDP2 | Qwen2.5-3B-Instruct | 2 nodes × 8 GPU | `qwen2.5_3b_fsdp.sh` |
| **Megatron 8B** | Megatron | Qwen3-8B | 4+ nodes × 8 GPU | `qwen3_8b_megatron.sh` |
| **Megatron 30B** | Megatron | Qwen3-30B-A3B (MoE) | 8+ nodes × 8 GPU | `qwen3_30b_megatron.sh` |

:::{note}
The `advanced_*` launch scripts (e.g. `advanced_qwen2.5_7b_fsdp.sh`) additionally
turn on the full rollout-coordination stack: `throughput_optimal` routing with a cost
model, multi-priority queues, `status_based` sync, request migration, and proactive
filtering. They require extra tuning to run well. Start from the plain scripts above,
which use the safe defaults, before layering on advanced coordination.
:::

---

## Quick Launch

```bash
# FSDP path (default staleness=3)
bash examples/dapo_trainer/qwen2.5_3b_fsdp.sh

# With custom staleness
bash examples/dapo_trainer/qwen2.5_3b_fsdp.sh 2
```

---

## DAPO-specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `adv_estimator` | `grpo` | Group-normalized advantage estimation (shared with GRPO) |
| `clip_ratio_low` | `0.2` | Lower PPO clip bound (asymmetric) |
| `clip_ratio_high` | `0.28` | Upper PPO clip bound (asymmetric) |
| `filter_groups_metric` | `acc` | Dynamic sampling filter metric |
| `n_resp_per_prompt` | `8` | Rollouts per prompt, GRPO group size (shared with GRPO) |

---

```{seealso}
Full scripts and configuration details are in the [`examples/dapo_trainer/`](https://github.com/psrl-project/psrl/tree/main/examples/dapo_trainer) directory.
```
