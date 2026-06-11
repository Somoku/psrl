# TransferQueue Integration

TransferQueue is PSRL's asynchronous sample data plane. It stores the evolving fields
of each rollout sample and lets distributed components exchange lightweight
`KVBatchMeta` references instead of repeatedly serializing complete TensorDict
batches through Ray.

It is unrelated to SMG's online routing queue: SMG dispatches live inference
requests, while TransferQueue stores data used by rollout, reward, and training
operators.

## Data Model

Each sample is addressed by a key, normally its `uid`. When one request produces
multiple trajectories, PSRL uses a trajectory-specific key. Training and validation
data use separate `train` and `val` partitions.

Fields are filled incrementally:

| Producer | Representative fields |
|---|---|
| DataProcessor | initial prompt, raw chat, IDs, dataset metadata |
| AgentLoopWorker | prompt/response token IDs, response mask, rollout log-probabilities, trajectory metadata |
| Reward workers | reward-model or rule-based score fields |
| Actor/Critic/Ref workers | old/ref log-probabilities, entropy, values, advantages, returns, metrics |

Training stages pass a `KVBatchMeta(keys, tags, partition_id)` object. The
TransferQueue bridge resolves only the fields needed by that stage and writes
derived fields back under the same sample keys.

```{mermaid}
sequenceDiagram
    participant DP as DataProcessor
    participant AW as AgentLoopWorker
    participant ALM as AgentLoopManager
    participant TQ as TransferQueue
    participant TR as Trainer / Workers

    DP->>TQ: put prompt fields
    AW->>TQ: put completed trajectory fields
    AW->>ALM: notify completion metadata
    ALM-->>TR: ready KVBatchMeta(keys, tags, partition)
    TR->>TQ: get fields needed for logprob/value/reward
    TR->>TQ: put derived fields
    TR->>TR: update actor/critic
    TR->>TQ: clear consumed keys after the step
```

## Why PSRL Uses It

- Large trajectory tensors do not have to round-trip through the central trainer.
- Rollout, reward, and training can work on the same logical sample at different
  times and add columns independently.
- Metadata-only staleness buffers remain small even for long, multi-turn
  trajectories.
- Storage and transport can be changed without changing the algorithm-level batch
  contract.

## Backends

`SimpleStorage` is the default and provides distributed in-memory storage over ZMQ.
`MooncakeStore` is an experimental hierarchical KV backend that can use TCP or RDMA.

```yaml
transfer_queue:
  enable: true
  controller:
    sampler: SequentialSampler
    polling_mode: false
  backend:
    storage_backend: SimpleStorage
    SimpleStorage:
      total_storage_size: 100000
      num_data_storage_units: 8
```

Mooncake example:

```yaml
transfer_queue:
  backend:
    storage_backend: MooncakeStore
    MooncakeStore:
      auto_init: false
      metadata_server: head-node:50123
      master_server_address: head-node:50124
      local_hostname: ""
      protocol: rdma
      device_name: mlx5_0
```

`main_ppo.py` enables TransferQueue for the current PSRL training path and initializes
its controller before distributed workers connect. The `transfer_queue.enable` field
is therefore primarily an integration/runtime flag, not a switch back to the old
TensorDict-only data path.

## Operational Constraints

- Size `total_storage_size` for in-flight prompts, redundant trajectories,
  validation data, and at least one active training buffer.
- Use multiple SimpleStorage units for multi-node load distribution; the config
  recommends at least twice the number of nodes.
- Consumed keys are cleared only after the training step. Aborted request cleanup is
  coordinated with request-status tracking.
- `psrl.colocate=True` is not implemented with the current `KVBatchMeta` training
  flow.
- MooncakeStore is experimental. Validate metadata/master availability and RDMA
  device configuration before production use.

```{seealso}
- {doc}`architecture`: where TransferQueue sits in the complete system
- {doc}`staleness_control`: how ready entries become training buffers
- {doc}`../tutorial/configuration`: configuration reference
```
