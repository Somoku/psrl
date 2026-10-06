# Designs & Features

This section describes the current PSRL architecture: SMG-based online request
routing, metadata-only sample flow through TransferQueue, asynchronous weight
transfer, and hierarchical KV cache management.

Each subsection covers one major design component, including motivation, architecture, configuration reference, and practical guidance.

---

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`workflow;1.5em` System Architecture
:link: architecture
:link-type: doc

Overall system design with decoupled training and generation, module responsibilities, and deployment topology.
:::

:::{grid-item-card} {octicon}`hubot;1.5em` Router, SessionRouter & TITO
:link: router_tito
:link-type: doc

SMG-based rollout routing, PSRL worker selection, session affinity, and exact
multi-turn training-data capture.
:::

:::{grid-item-card} {octicon}`stack;1.5em` TransferQueue
:link: transfer_queue
:link-type: doc

Asynchronous sample storage and metadata-only data exchange across rollout, reward,
and training.
:::

:::{grid-item-card} {octicon}`server;1.5em` Parameter Server
:link: parameter_server
:link-type: doc

CPU-based distributed weight storage with unified sharding, RDMA push/pull, and broadcast initialization.
:::

:::{grid-item-card} {octicon}`arrow-switch;1.5em` Weight Transport
:link: weight_transport
:link-type: doc

PS as a zero-byte version authority, trainer-direct reads with same-version peer
diffusion, redundancy-free planning, FP8/FP4 wire formats, and the Megatron/FSDP
backend matrix across colocate and disaggregate.
:::

:::{grid-item-card} {octicon}`checklist;1.5em` Weight Transport Review
:link: weight_transport_review
:link-type: doc

Self-review of the weight-transport design: feasibility tiers, the corrected
read/write conflict analysis, and the control-plane bottleneck it missed.
:::

:::{grid-item-card} {octicon}`graph;1.5em` Weight Transport Review 2
:link: weight_transport_review2
:link-type: doc

Second-round review: the metadata-plane blocker at MoE scale, FP4 extension limits,
and the round-count cost at 1TB.
:::

:::{grid-item-card} {octicon}`shield-check;1.5em` Weight Transport Review 3
:link: weight_transport_review3
:link-type: doc

Third-round review: the communication-stack analysis and the correctness gap under
all-rollout startup and elastic scaling.
:::

:::{grid-item-card} {octicon}`verified;1.5em` Weight Transport Review 4
:link: weight_transport_review4
:link-type: doc

Fourth-round review: verifies the earlier fixes landed, retires progressive
availability as unsupported by vLLM, and finds the colocate precision constraint.
:::

:::{grid-item-card} {octicon}`search;1.5em` Weight Transport Review 5
:link: weight_transport_review5
:link-type: doc

Fifth-round review: closes the NCCL gap as a non-issue, finds the epoch/version
interaction, and sizes what the existing e2e test can actually cover.
:::

:::{grid-item-card} {octicon}`tasklist;1.5em` Weight Transport Implementation
:link: weight_transport_implementation
:link-type: doc

The ordered work breakdown: six phases, every design item mapped to files, an
executable completion criterion per task, and the indivisible direct-supply block.
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

TMS-backed worker lifecycle and elastic inference resources, with current colocation
constraints documented explicitly.
:::

:::{grid-item-card} {octicon}`container;1.5em` Sandbox Lifecycle
:link: sandbox_lifecycle
:link-type: doc

Sandbox ownership, node capacity admission, and Docker failure semantics, with the
invariants that keep resource accounting aligned with running containers.
:::

:::{grid-item-card} {octicon}`tools;1.5em` Sandbox Refactor Plan
:link: sandbox_refactor
:link-type: doc

The thin control plane target, the phased migration order, multi node placement,
and the backend set the module runs on.
:::

:::{grid-item-card} {octicon}`checklist;1.5em` Sandbox Execution Plan
:link: sandbox_execution_plan
:link-type: doc

The ordered work breakdown for the refactor, with an exit criterion, a rollback
note, and a decision trace for every step.
:::

:::{grid-item-card} {octicon}`server;1.5em` Sandbox Service
:link: sandbox_service
:link-type: doc

The standalone service shape: a cluster control plane with a fleet monitor, one agent
per node owning lifecycle and reclamation, and a Python SDK as the only caller surface.
:::

::::

---

```{toctree}
:maxdepth: 2
:hidden:

architecture
router_tito
transfer_queue
parameter_server
weight_transport
weight_transport_review
weight_transport_review2
weight_transport_review3
weight_transport_review4
weight_transport_review5
weight_transport_implementation
flexible_rollout
staleness_control
kv_cache
resource_elasticity
sandbox_lifecycle
sandbox_architecture
sandbox_refactor
sandbox_execution_plan
sandbox_external_backends
sandbox_backend_integration
sandbox_agentenv
sandbox_opensandbox
sandbox_snapshot_store
sandbox_service
```
