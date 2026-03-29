# PS Broadcast Initialization Design

**Date:** 2026-03-29
**Status:** Approved
**Topic:** NIXL-based binary tree broadcast for PS storage worker initialization

---

## Background

Currently, every PS Storage Worker independently reads the full model checkpoint from disk during initialization. This means N disk reads for N PS workers, each worker loading the full parameter set. The goal is to support an optional mode where only rank-0 reads from disk, then distributes the loaded weights to all other PS workers via NIXL GPU-Direct writes, using a binary tree broadcast algorithm to avoid rank-0 becoming a bottleneck.

---

## Scope

- Add a config flag to enable/disable this mode (off by default, existing path unchanged).
- Extend the MetaServer/nixl_protocol so PS workers mutually hold each other's ClientInfo.
- Implement binary tree broadcast coordination in PSManager and PSStorageWorker.
- Handle both shared and non-shared train/gen buffer cases correctly.
- Preserve extensibility for future algorithms (ring, etc.) and non-full-replica sharding.

---

## Config

New fields under the `ps` config block in `psrl/trainer/config/psrl/psrl.yaml`:

```yaml
ps:
  broadcast_init:
    enabled: false          # Set true to use rank-0 broadcast initialization
    algorithm: binary_tree  # Broadcast algorithm; reserved for future: ring
```

Corresponding dataclasses (added to the existing psrl config schema):

```python
@dataclass
class BroadcastInitConfig:
    enabled: bool = False
    algorithm: str = "binary_tree"

@dataclass
class PSConfig:
    # ... existing fields ...
    broadcast_init: BroadcastInitConfig = field(default_factory=BroadcastInitConfig)
```

When `enabled=False`, the code takes the existing path with zero behavioral change.

---

## Architecture Overview

```
Existing path (broadcast_init.enabled = False):
  Each PSStorageWorker:
    preload_checkpoint_to_cpu() → write_checkpoint_to_registered_tensors()

New path (broadcast_init.enabled = True):
  rank 0 only:
    preload_checkpoint_to_cpu() → write_checkpoint_to_registered_tensors()

  PSManager coordinates binary tree broadcast:
    Round 0: rank 0 → rank 1, rank 2
    Round 1: rank 1 → rank 3, rank 4 | rank 2 → rank 5, rank 6
    ...  (ceil(log2(N)) rounds total)

  All workers (if not train_gen_model_share()):
    transfer_train_to_gen()
```

---

## Section 1: MetaServer Extension — PS-to-PS ClientInfo Distribution

**File:** `psrl/utils/nixl/server.py`
**Function:** `_get_relevant_client_names_for_agent`

This function determines which ClientInfos are broadcast to a given agent during `nixl_protocol` Phase 2b. Currently PS workers only receive ClientInfos for train/gen workers. The extension:

- If `broadcast_init.enabled` and the queried agent belongs to a PS worker, also include all other PS workers' client names in the returned list.
- PSManager passes the set of PS agent names to the MetaServer at bind time (already available via `bind_ps_worker_group`).
- No new coordination round is needed; this piggybacks on the existing Phase 2b info exchange.

After this change, every PS worker holds the GPU descriptors for all other PS workers' registered buffers, enabling direct NIXL writes between PS workers.

---

## Section 2: Binary Tree Broadcast Plan

**New file:** `psrl/workers/ps/broadcast.py`

Encapsulates broadcast topology logic, keeping PSManager and PSStorageWorker clean.

```python
class BroadcastPlan:
    """Abstract base for broadcast strategies."""
    def get_children(self, rank: int) -> list[int]: ...
    def get_parent(self, rank: int) -> int | None: ...
    def num_rounds(self) -> int: ...
    def senders_in_round(self, round_idx: int) -> list[int]: ...

class BinaryTreeBroadcastPlan(BroadcastPlan):
    """
    Static binary tree: parent(i) = (i-1)//2, children(i) = [2i+1, 2i+2].
    Built from rank indices alone; no external metadata required.
    """
    def __init__(self, world_size: int): ...

def build_broadcast_plan(world_size: int, algorithm: str) -> BroadcastPlan:
    if algorithm == "binary_tree":
        return BinaryTreeBroadcastPlan(world_size)
    raise ValueError(f"Unknown broadcast algorithm: {algorithm!r}")
```

The tree structure:
```
rank 0
├── rank 1
│   ├── rank 3
│   └── rank 4
└── rank 2
    ├── rank 5
    └── rank 6
```

`parent(i) = (i-1)//2`, `children(i) = [2i+1, 2i+2]` (clamped to world_size).
Depth = `ceil(log2(N))` rounds. For N=256, this is 8 rounds.

---

## Section 3: PSStorageWorker Changes

**File:** `psrl/workers/ps/ps_storage_worker.py`

### Initialization path change

```python
def preload_checkpoint_to_cpu(self):
    if not self._broadcast_init_enabled or self._rank == 0:
        # Existing logic: read safetensors shards from disk to CPU cache
        _do_preload_checkpoint()
    # else: non-root workers skip disk read entirely

def write_checkpoint_to_registered_tensors(self):
    if not self._broadcast_init_enabled or self._rank == 0:
        # Existing logic: copy CPU cache → GPU buffer
        _do_write_checkpoint()
    # else: non-root workers skip; they will receive data via broadcast

def broadcast_send_to_children(self, round_idx: int, plan: BroadcastPlan):
    """
    Send all keys to each child in the broadcast tree for this round.
    Called by PSManager via Ray remote after barrier for round_idx clears.
    Uses existing client_write path targeting ps_push_client of each child.
    """
    children = plan.get_children(self._rank)
    for child_rank in children:
        child_client_name = self._ps_client_name_for_rank(child_rank)
        for key in self._all_keys():
            self._nixl_client.client_write(
                target_agent=self._ps_agent_name_for_rank(child_rank),
                target_client=child_client_name,
                key=key,
                tag="ps_broadcast_init",  # fixed tag; unique per broadcast session
            )
        # wait for all transfers to complete before signalling PSManager

def do_transfer_train_to_gen_after_broadcast(self):
    """Called after broadcast completes if not train_gen_model_share()."""
    if not self._storage_plan.train_gen_model_share():
        self.transfer_train_to_gen(keys=self._all_keys())
```

### Key lookup: PS client names by rank

Each PS worker already has its own rank and knows the world size. The mapping from rank → (agent_name, client_name) for other PS workers is built from the ClientInfos received during nixl_protocol Phase 2b (available in `_all_client_infos`). A helper `_ps_client_name_for_rank(rank)` and `_ps_agent_name_for_rank(rank)` resolve this mapping using the sorted list of PS client names (ordered by rank, consistent across all workers).

---

## Section 4: PSManager Coordination

**File:** `psrl/workers/ps/ps_manager.py`

New method `_coordinate_broadcast_init()`, called after `nixl_protocol()` completes and rank 0 has called `preload_checkpoint_to_cpu()` + `write_checkpoint_to_registered_tensors()` (non-root workers skip both in broadcast_init mode):

```python
def _coordinate_broadcast_init(self, plan: BroadcastPlan):
    """
    Coordinate binary tree broadcast round by round.
    Each round: signal senders → wait for completion reports.
    """
    for round_idx in range(plan.num_rounds()):
        senders = plan.senders_in_round(round_idx)
        # Signal each sender to begin (Ray remote call)
        futures = [
            self._ps_workers[rank].broadcast_send_to_children.remote(round_idx, plan)
            for rank in senders
        ]
        # Barrier: wait for all senders to finish before next round
        ray.get(futures)

    # After broadcast: trigger transfer_train_to_gen on all workers if needed
    ray.get([w.do_transfer_train_to_gen_after_broadcast.remote()
             for w in self._ps_workers])
```

This replaces the per-worker `transfer_train_to_gen` call that would otherwise happen individually. The barrier between rounds ensures a worker only sends after it has received data from its parent.

---

## Section 5: Data Flow Summary

```
nixl_protocol() completes
  → All PS workers now hold each other's ClientInfo (GPU descriptors)

[broadcast_init.enabled = True]
  → rank 0: write_checkpoint_to_registered_tensors()  (train buffer populated)
  → PSManager: _coordinate_broadcast_init(plan)
      Round 0: rank 0 sends all keys to rank 1, rank 2 via client_write
               barrier: wait rank 0 done
      Round 1: rank 1 → rank 3, rank 4 | rank 2 → rank 5, rank 6 (parallel)
               barrier: wait rank 1, rank 2 done
      ... (ceil(log2(N)) rounds)
  → All workers: do_transfer_train_to_gen_after_broadcast()
      if train_gen_model_share(): no-op
      else: copy train buffer → gen buffer (existing transfer_train_to_gen)
  → PSManager: initialization complete

[broadcast_init.enabled = False]
  → Each worker: preload_checkpoint_to_cpu() → write_checkpoint_to_registered_tensors()
  → Each worker: transfer_train_to_gen() as before
  → PSManager: initialization complete (existing path, unchanged)
```

---

## Extensibility Notes

- **New algorithms**: Add a new subclass of `BroadcastPlan` in `psrl/workers/ps/broadcast.py` and register it in `build_broadcast_plan()`. No changes to PSManager or PSStorageWorker needed.
- **Non-full-replica sharding**: When a PS worker holds only a subset of keys, `broadcast_send_to_children` needs to filter keys to those held locally. The plan object can be extended with a `keys_for_sender(rank)` method. The rest of the coordination loop is unchanged.
- **Partial broadcast** (e.g., only broadcast a subset of layers): `_all_keys()` can be parameterized without touching the broadcast topology logic.

---

## Files Changed

| File | Change |
|------|--------|
| `psrl/trainer/config/psrl/psrl.yaml` | Add `ps.broadcast_init` config block |
| `psrl/trainer/config/` (dataclass schema) | Add `BroadcastInitConfig`, extend `PSConfig` |
| `psrl/utils/nixl/server.py` | Extend `_get_relevant_client_names_for_agent` to include PS-to-PS ClientInfos |
| `psrl/workers/ps/ps_storage_worker.py` | Conditionalize disk load on rank; add `broadcast_send_to_children`, `do_transfer_train_to_gen_after_broadcast` |
| `psrl/workers/ps/ps_manager.py` | Add `_coordinate_broadcast_init()`; call it when `broadcast_init.enabled` |
| `psrl/workers/ps/broadcast.py` | **New file**: `BroadcastPlan`, `BinaryTreeBroadcastPlan`, `build_broadcast_plan` |
