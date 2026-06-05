# Designs & Features

This section describes PSRL's core system innovations, the novel technical contributions that enable efficient asynchronous LLM post-training with fine-grained staleness control.

Each subsection covers one major design component, including motivation, architecture, configuration reference, and practical guidance.

---

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`workflow;1.5em` System Architecture
:link: architecture
:link-type: doc

Overall system design with decoupled training and generation, module responsibilities, and deployment topology.
:::

:::{grid-item-card} {octicon}`server;1.5em` Parameter Server
:link: parameter_server
:link-type: doc

CPU-based distributed weight storage with unified sharding, RDMA push/pull, and broadcast initialization.
:::

:::{grid-item-card} {octicon}`iterations;1.5em` Flexible Rollout
:link: flexible_rollout
:link-type: doc

Four complementary coordination techniques: partial rollout, redundant rollout, intelligent routing, and migration.
:::

:::{grid-item-card} {octicon}`clock;1.5em` Staleness Control
:link: staleness_control
:link-type: doc

Trajectory-level version binding with the Reserve/Occupy/Consume protocol for bounded off-policy training.
:::

:::{grid-item-card} {octicon}`database;1.5em` KV Cache Management
:link: kv_cache
:link-type: doc

LMCache integration for CPU offloading, prefix reuse, and cross-instance P2P KV transfer.
:::

:::{grid-item-card} {octicon}`pulse;1.5em` Resource Elasticity
:link: resource_elasticity
:link-type: doc

Dynamic GPU memory management via TMS for colocated training and generation workloads.
:::

::::

---

```{toctree}
:maxdepth: 2
:hidden:

architecture
parameter_server
flexible_rollout
staleness_control
kv_cache
resource_elasticity
```
