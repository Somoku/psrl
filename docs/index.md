---
sd_hide_title: true
---

# PSRL Documentation

**PSRL** is a reinforcement learning (RL) framework for efficient large language model (LLM) post-training. It features decoupled training and generation, fine-grained staleness control, and flexible rollout coordination to achieve up to **2.68x throughput improvement** over existing systems.

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
**Decoupled architecture.**
Separate train and generation clusters communicate via Parameter Server for maximum hardware utilization.
:::

:::{grid-item}
**Fine-grained staleness control.**
Trajectory-level version binding with Reserve/Occupy/Consume protocol ensures data freshness without sacrificing throughput.
:::

:::{grid-item}
**Flexible rollout coordination.**
Partial rollout, redundant rollout, intelligent routing, and load-balanced migration work together to minimize idle time.
:::

:::{grid-item}
**Agentic RL support.**
Built-in multi-turn agent loops for tool-use training (ReTool, SWE-agent) with Docker-sandboxed environments.
:::

:::{grid-item}
**KV cache management.**
LMCache integration for CPU offloading and cross-instance P2P KV transfer to reduce re-prefill overhead.
:::

:::{grid-item}
**Multiple backends.**
FSDP2 and Megatron-LM training backends with PPO, GRPO, and DAPO algorithm support.
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
