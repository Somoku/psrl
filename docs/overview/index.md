# Overview

## What is PSRL?

**PSRL** is a reinforcement learning (RL) framework for efficient large language model (LLM) post-training. Its core idea is to decouple generation from training with a **Parameter Server (PS)** architecture, so rollout workers and training workers can progress asynchronously while keeping model-version staleness under explicit control.

Built on [veRL](https://github.com/volcengine/verl), PSRL focuses on the system bottlenecks that emerge in agentic, dynamic, and long-tailed RL workloads: uneven rollout latency, version-aware weight management, multi-turn KV cache reuse, and elastic resource allocation across rollout, reward, and training models.

## Key Capabilities

| Capability | Description |
|---|---|
| **Decoupled Train/Gen** | Lets training and generation workers run independently, with the Parameter Server coordinating weight publication and synchronization |
| **Fine-grained Staleness Control** | Binds model versions at trajectory level and uses a global consistency protocol (Reserve/Occupy/Consume) to keep staleness bounded |
| **Flexible Rollout Coordination** | Handles long-tailed rollout latency with partial rollout, redundant rollout, intelligent routing, and load-balanced migration |
| **KV Cache Management** | Improves multi-turn efficiency through LMCache-based CPU offloading and cross-instance P2P KV cache transfer |
| **Agentic RL** | Provides native support for multi-turn tool-use training (ReTool, mini-SWE-agent) in Docker-sandboxed environments |
| **Resource Elasticity** | Dynamically scales rollout, reward, and training models with TMS (torch_memory_saver) and Parameter Server integration |

## Supported Backends

| Component | Backend | Parallelism |
|---|---|---|
| **Rollout** | vLLM | vLLM serving |
| **Training** | FSDP2 | FSDP sharding + Ulysses SP |
| **Training** | Megatron-LM | TP + PP + CP + EP |
