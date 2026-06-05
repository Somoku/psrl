# LLM-as-a-Judge

```{admonition} Under Development
:class: warning

This recipe is under active development. The page below sketches the planned scope.
The launch script, judge prompts, and reference configs are not yet published.
```

---

## What it is

**LLM-as-a-Judge** turns a separate language model into the reward function. For every
rollout produced by the policy, a *judge* LLM reads the prompt together with the
response (and optionally a rubric, reference answer, or multiple candidates) and emits
a score, typically a scalar in `[0, 1]`, a discrete rating, or a pairwise preference.
That score is then used as the RL reward signal, so any algorithm in the
[RLVR](../rlvr/index) family (PPO / GRPO / DAPO) can train against it.

This unlocks RL on tasks where no programmatic checker exists: instruction following,
creative writing, dialogue, summarization, safety alignment, and so on.
