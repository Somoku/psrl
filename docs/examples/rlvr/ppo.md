# PPO: Proximal Policy Optimization

```{admonition} Coming Soon
:class: tip

This example is under active development. The PPO algorithm is fully implemented in PSRL's trainer but a standalone example recipe with documentation is forthcoming.
```

---

## Algorithm Overview

PPO (Proximal Policy Optimization) is the foundational policy-gradient algorithm in
PSRL. It trains an **actor** alongside a learned **critic** (value function) and
optimizes the policy with a clipped surrogate objective. Key ingredients:

- **Generalized Advantage Estimation (GAE)** with a learned value function
- **Clipped surrogate objective** to prevent large policy updates
- **KL divergence control** (fixed or adaptive) against a reference policy

The clipped surrogate loss for token $t$ is

$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[
\min\!\left(
  r_t(\theta)\,\hat{A}_t,\;
  \operatorname{clip}\!\bigl(r_t(\theta),\,1-\epsilon,\,1+\epsilon\bigr)\,\hat{A}_t
\right)
\right],
$$

where $r_t(\theta) = \pi_\theta(a_t\mid s_t) / \pi_{\theta_{\text{old}}}(a_t\mid s_t)$
is the importance-sampling ratio and $\hat{A}_t$ is the GAE advantage.

```{seealso}
For the trainer architecture, cluster layout, and monitoring metrics shared by
all RLVR recipes, see {doc}`common`.
```
