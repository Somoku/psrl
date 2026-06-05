# Generative Reward Model

When the task is open-ended (creative writing, dialogue, instruction following) there
is no programmatic checker that can score a rollout. **Generative Reward Models
(GRMs)** fill this gap by having another language model produce the reward signal
itself, either by *judging* the response or by *teaching* the policy what a better
response would have looked like.

PSRL groups two complementary recipes under this umbrella:

| Recipe | One-line summary | Status |
|---|---|---|
| [LLM-as-a-Judge](llm_as_a_judge) | A frozen / fine-tuned LLM scores each rollout, the score is used as the RL reward | TBD |
| [On-Policy Distillation](on_policy_distillation) | A stronger teacher model emits token-level targets on the policy's own rollouts, the policy distills against those targets | TBD |

Both share the same plumbing as the RLVR recipes, vLLM rollout, NIXL weight transfer,
PSRL's `RewardManager` / `RewardLoop`, but swap the verifiable scoring function for an
LLM-derived signal.

```{admonition} Under Development
:class: warning

Both recipes are work-in-progress. The pages below describe the intended scope so you
can track what's coming, full launch scripts and configs are not yet published.
```

```{toctree}
:maxdepth: 1
:hidden:

llm_as_a_judge
on_policy_distillation
```
