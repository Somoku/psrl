---
sd_hide_title: true
---

# PSRL Documentation

**PSRL** is an efficient asynchronous RL framework for LLM post-training.

Built on top of [veRL](https://github.com/volcengine/verl), PSRL features efficient RDMA weight synchronization via parameter servers, fine-grained staleness control, and flexible rollout coordination to achieve up to **2.68x throughput improvement** over existing systems.

---

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.5em` Quick Start
:link: tutorial/quickstart
:link-type: doc

Get PSRL running in minutes with a minimal DAPO training example.
:::

:::{grid-item-card} {octicon}`book;1.5em` Tutorial
:link: tutorial/index
:link-type: doc

Step-by-step guides for installation, configuration, and first training run.
:::

:::{grid-item-card} {octicon}`gear;1.5em` Architecture
:link: design/architecture
:link-type: doc

Understand the decoupled train/gen design and module interactions.
:::

:::{grid-item-card} {octicon}`graph;1.5em` Designs & Features
:link: design/index
:link-type: doc

Deep-dive into staleness control, flexible rollout, and parameter server.
:::

:::{grid-item-card} {octicon}`code;1.5em` Examples
:link: examples/index
:link-type: doc

Production-ready recipes for RLVR, Agentic RL, and more.
:::

:::{grid-item-card} {octicon}`versions;1.5em` Overview
:link: overview/index
:link-type: doc

Project introduction, key features, and performance highlights.
:::

::::

---

## Key Features

::::{grid} 2 2 3 3
:gutter: 2

:::{grid-item}
**Efficient RDMA-based Weight Transfer.**
Push/Pull weights to/from CPU-side Parameter Server and P2P RDMA transfers.
:::

:::{grid-item}
**Fine-grained Staleness Control.**
Trajectory-level version binding with Reserve/Occupy/Consume protocol ensures data freshness without sacrificing throughput.
:::

:::{grid-item}
**Flexible Rollout Coordination.**
Partial rollout, redundant rollout, intelligent routing, and load-balanced migration work together to minimize idle time.
:::

:::{grid-item}
**Easy-to-use Agentic RL Support.**
Native environment loops and SessionRouter/TITO support both integrated and black-box agents.
:::

:::{grid-item}
**Hierarchical KV Cache Management.**
SMG cache-aware routing, vLLM GPU prefix cache, LMCache offload, and P2P transfer reduce re-prefill.
:::

:::{grid-item}
**Multiple Backends and Algorithms Support.**
FSDP2 and Megatron training integrate with PPO, GRPO, DAPO, and generative reward workflows.
:::

::::

---

```{toctree}
:maxdepth: 2
:hidden:

overview/index
tutorial/index
examples/index
design/index
```
