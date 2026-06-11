# Overview

## What is PSRL?

**PSRL** is a reinforcement learning (RL) framework for efficient large language
model post-training. It decouples rollout, reward, and training while coordinating
them through a Parameter Server, an SMG-based rollout gateway, and TransferQueue.
Generation and training progress asynchronously while model-version staleness remains
explicitly bounded.

Built on [veRL](https://github.com/volcengine/verl), PSRL focuses on the system bottlenecks that emerge in agentic, dynamic, and long-tailed RL workloads: uneven rollout latency, version-aware weight management, multi-turn KV cache reuse, and elastic resource allocation across rollout, reward, and training models.

## Key Capabilities

| Capability | Description |
|---|---|
| **Efficient RDMA-based Weight Transfer** | Uses the Parameter Server and NIXL for local or UCX/RDMA model push/pull |
| **Fine-grained Staleness Control** | Binds model versions at trajectory level and uses a global consistency protocol (Reserve/Occupy/Consume) to keep staleness bounded |
| **Flexible Rollout Coordination** | Uses SMG routing, partial rollout, redundancy, migration, and PSRL-aware worker admission |
| **Easy-to-use Agentic RL Support** | Supports native environment loops plus SessionRouter/TITO integration for OpenAI-compatible black-box agents |
| **Hierarchical KV Cache Management** | Combines SMG cache-aware routing, vLLM GPU prefix KV, LMCache offload, and cross-instance transfer |
| **Multiple Backends and Algorithms Support** | Integrates FSDP2/Megatron training, vLLM rollout, and PPO/GRPO/DAPO-style algorithms |

## Supported Backends

| Component | Backend | Parallelism |
|---|---|---|
| **Rollout gateway** | SMG | HTTP/OpenAI ingress, PSRL-aware routing loop, gRPC proxy, TITO |
| **Rollout engine** | vLLM | DP + TP + PP serving through PSRL's gRPC integration |
| **Training** | FSDP2 | FSDP sharding + Ulysses SP |
| **Training** | Megatron-LM | TP + PP + CP + EP |

See {doc}`../design/architecture` for the three-plane architecture and
{doc}`../design/router_tito` for the SMG integration.
