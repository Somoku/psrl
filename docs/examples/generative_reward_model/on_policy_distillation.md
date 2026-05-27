# On-Policy Distillation

```{admonition} Under Development
:class: warning

This recipe is under active development. The page below sketches the planned scope.
The launch script and reference configs are not yet published.
```

---

## What it is

**On-Policy Distillation** uses a stronger **teacher** LLM as the supervision signal
for the **student** policy, but, unlike classical offline distillation, the
trajectories being distilled are sampled from the *student's own* current policy. For
every rollout the student produces, the teacher provides token-level targets (full
distributions or log-probs), and the student is trained to match them via a KL-style
loss.

Because the student keeps generating its own rollouts, the training distribution stays
on-policy: the teacher tells the student "given *your* prefix, *here* is what a
stronger model would have said next", which avoids the compounding-error problem of
training purely on the teacher's offline trajectories.
