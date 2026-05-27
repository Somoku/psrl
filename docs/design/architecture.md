# Overall System Architecture

PSRL decouples LLM post-training into two independent compute phases, **training** and **generation**: connected through a central Parameter Server. This architecture eliminates the synchronization barrier that limits throughput in tightly-coupled systems, enabling training and generation to progress concurrently at their own optimal pace.

```{figure} /_static/img/overview.svg
:alt: PSRL System Architecture
:width: 100%
:align: center

Overall PSRL system architecture showing decoupled training and generation.
```

---

## Module Overview

### Train Workers

Execute policy gradient updates using either FSDP2 or Megatron-LM as the distributed training backend. After each gradient step completes, train workers **push** the updated model weights to the Parameter Server via RDMA (cross-node) or shared memory (same node). Train workers operate independently of generation, they consume ready buffers from the staleness system and produce new model versions.

### Parameter Server (PS Manager + PS Workers)

The central weight storage and coordination hub. The **PS Manager** is a Ray actor that manages model versioning, tracks staleness buffers, and coordinates the Reserve/Occupy/Consume protocol. **PS Workers** are CPU-based processes distributed across nodes that store model weight shards. They are co-located with rollout instances to minimize pull latency.

:::{seealso}
{doc}`parameter_server` for detailed design of the PS architecture, sharding strategy, and synchronization modes.
:::

### Agent Manager

Dispatches prompts from the training dataset to Agent Workers. Collects completed trajectories, computes rewards, and fills staleness buffer entries (Reserve → Occupy). The Agent Manager maintains the global view of active and pending requests across all workers.

### Agent Workers

Each Agent Worker handles **one trajectory at a time**, running the agent loop to completion. Supports both single-turn generation (one prompt → one response) and multi-turn interaction (alternating LLM generation and tool execution). Agent Workers communicate with rollout instances through the Router.

### Rollout Coordinator

A centralized monitoring component that collects status from all rollout instances, queue depth, KV cache utilization, request throughput, and model version. Provides a **global view** used by the Router and sync/migration strategies to make informed decisions.

### Router (Rollout Router)

Dispatches generation requests from Agent Workers to rollout instances based on the configured routing strategy. Strategies range from simple (random, round-robin) to model-aware (throughput-optimal with cost model, KV-cache-aware). The Router is the central point where load balancing and locality decisions are made.

:::{seealso}
{doc}`flexible_rollout` for detailed routing strategies and their configuration.
:::

### Rollout Instances

vLLM engine pools that handle inference requests. Each instance manages a subset of generation requests and maintains its own KV cache. Rollout instances **pull** new model weights from co-located PS Workers when triggered by the sync strategy. Multiple instances can run different model versions concurrently during the transition period.

### Reward Service

Computes rewards for completed trajectories. Supports multiple reward sources:
- **Rule-based**: Deterministic scoring (format checking, exact match)
- **Model-based**: LLM-as-judge or process reward models
- **Execution-based**: SandboxFusion code execution with test verification

---

## Training Iteration Sequence

The following diagram shows the complete data flow for one training iteration:

```{mermaid}
sequenceDiagram
    participant AM as Agent Manager
    participant AW as Agent Worker
    participant Router
    participant RI as Rollout Instance
    participant PS as Parameter Server
    participant TW as Train Worker
    participant Reward

    AM->>AW: dispatch(prompt)
    AW->>Router: generate(tokens)
    Router->>RI: route(request)
    RI-->>AW: response
    AW->>Reward: compute_score(trajectory)
    Reward-->>AM: reward
    AM->>PS: occupy(buffer_entry)
    PS->>TW: buffer_ready signal
    TW->>PS: pull(weights) → train → push(new_weights)
    PS->>RI: sync_trigger(new_version)
    RI->>PS: pull(new_weights)
```

**Step-by-step:**

1. **Dispatch**: Agent Manager assigns a prompt (with a reserved buffer slot) to an available Agent Worker.
2. **Generation**: Agent Worker sends generation requests through the Router to a rollout instance.
3. **Multi-turn** (optional): For agentic tasks, the worker may execute tools and generate multiple turns.
4. **Reward**: Completed trajectory is scored by the Reward Service.
5. **Occupy**: The scored trajectory occupies its reserved slot in the staleness buffer.
6. **Training**: When a buffer is full (Ready state), the Train Worker consumes it for one gradient step.
7. **Weight Push**: Updated weights are pushed to PS via RDMA.
8. **Sync**: Rollout instances pull new weights when their sync strategy triggers.

:::{tip}
Steps 1-5 (generation) and steps 6-8 (training) run **concurrently**: this is the core advantage of PSRL's decoupled design. The staleness system (see {doc}`staleness_control`) ensures bounded off-policy error.
:::

---

## Deployment Topology

PSRL supports flexible node allocation between training and generation. The typical deployment separates GPU nodes by role:

### Example: 4-Node Setup (2 Gen + 2 Train)

| Node | Role | GPU Usage | Co-located Services |
|------|------|-----------|---------------------|
| Node 0 | Generation | 8× H100 for vLLM | Rollout Instance 0, PS Worker 0, Agent Workers |
| Node 1 | Generation | 8× H100 for vLLM | Rollout Instance 1, PS Worker 1, Agent Workers |
| Node 2 | Training | 8× H100 for FSDP/Megatron | Train Workers (ranks 0-7) |
| Node 3 | Training | 8× H100 for FSDP/Megatron | Train Workers (ranks 8-15) |

**Key placement decisions:**

- **PS Workers** are co-located with rollout instances (generation nodes) to minimize pull latency via local PCIe DMA.
- **Agent Workers** run on generation nodes as CPU processes: they orchestrate but do not consume GPU.
- **PS Manager** runs on the head node as a Ray actor (lightweight coordination).
- **Reward Service** can run on any node, GPU-based reward models get dedicated allocation.

### Configuration

Deployment topology is configured via `psrl.deployment.*`:

```yaml
psrl:
  deployment:
    train_nnodes: 2
    gen_nnodes: 2
    gen_tp: 8          # Tensor parallelism within each rollout instance
    num_instances: 2   # Number of rollout instances
```

:::{admonition} Scaling Guidance
:class: tip

- For **compute-bound** workloads (short generations, large models): allocate more nodes to training.
- For **generation-bound** workloads (long multi-turn trajectories, agentic RL): allocate more nodes to generation.
- The Parameter Server overhead is **constant** regardless of cluster size: it scales via distributed PS Workers, not by adding coordinator logic.
:::
