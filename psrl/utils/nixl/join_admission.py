"""Admission decision for a client joining an established NIXL mesh.

`NIXLMetaServer.make_unified_sharding` builds each key's finest shard mesh as the
LCM across all registered clients. A joiner whose own mesh does not divide that LCM
raises it, which invalidates every existing client's descriptors -- and if the new
LCM stops dividing some tensor's dimension, `NIXLSharding.get_local_sharded_tensors`
raises `ValueError` on *every* existing client, turning one scale-up into a
cluster-wide outage.

So admission is a three-way decision, not a boolean (design doc 3.3.1d):

======================  ==========================================================
`FAST_PATH`             joiner's mesh divides the current LCM -> LCM unchanged, every
                        existing descriptor stays valid, nothing to re-shard
`RESHARD`               LCM rises but still divides every tensor dimension -> legal,
                        but all clients must recompute descriptors and the plan must
                        be recompiled. Caller warns and proceeds.
`REJECT`                the raised LCM does not divide some tensor dimension -> the
                        sharding is inexpressible. Must be refused at admission.
======================  ==========================================================

Scale-up along the data-parallel dimension never changes `shard_mesh` (train-side
sharding carries no DP dimension), so routine elasticity always takes `FAST_PATH`;
the warning only appears when someone actually changes a parallelism degree.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum

from psrl.utils.nixl.nixl_spec import lcm

ShardMesh = OrderedDict[int, int] | dict[int, int]
TensorShapes = dict[str, tuple[int, ...]]


class JoinVerdict(Enum):
    """Outcome of an admission check. See module docstring for semantics."""

    FAST_PATH = "fast_path"
    RESHARD = "reshard"
    REJECT = "reject"


@dataclass
class JoinDecision:
    """A verdict plus everything the caller needs to log or act on it."""

    verdict: JoinVerdict
    current_finest: dict[int, int]
    new_finest: dict[int, int]
    #: (key, dim, dim_size, shard_count) for every tensor dimension the new LCM
    #: cannot divide. Non-empty exactly when ``verdict is REJECT``.
    blocking_dims: list[tuple[str, int, int, int]] = field(default_factory=list)
    #: Parallelism degrees that would divide the current LCM, offered as the
    #: actionable alternative when a join is rejected or triggers a re-shard.
    suggested_degrees: list[int] = field(default_factory=list)

    @property
    def requires_reshard(self) -> bool:
        return self.verdict is JoinVerdict.RESHARD

    @property
    def admitted(self) -> bool:
        return self.verdict is not JoinVerdict.REJECT

    def describe(self) -> str:
        """One-line human-readable summary for the admission log."""
        if self.verdict is JoinVerdict.FAST_PATH:
            return f"finest mesh unchanged at {dict(self.current_finest)}; existing descriptors stay valid"
        if self.verdict is JoinVerdict.RESHARD:
            ratio = _mesh_growth(self.current_finest, self.new_finest)
            return (
                f"finest mesh rises {dict(self.current_finest)} -> {dict(self.new_finest)} "
                f"(x{ratio}); every existing client must recompute descriptors and the "
                f"plan must be recompiled. Scale along the DP dimension, or pick a "
                f"parallelism degree dividing the current mesh "
                f"(e.g. {self.suggested_degrees}), to avoid this."
            )
        offenders = ", ".join(
            f"{key}.dim{dim}({size}) % {count} != 0" for key, dim, size, count in self.blocking_dims[:3]
        )
        more = "" if len(self.blocking_dims) <= 3 else f" (+{len(self.blocking_dims) - 3} more)"
        return (
            f"finest mesh would rise {dict(self.current_finest)} -> {dict(self.new_finest)}, "
            f"which cannot shard: {offenders}{more}. This is not a cost problem -- the "
            f"sharding is inexpressible and would make every existing client raise "
            f"ValueError at re-registration. Use a parallelism degree dividing the "
            f"current mesh (e.g. {self.suggested_degrees})."
        )


def _mesh_growth(current: dict[int, int], new: dict[int, int]) -> int:
    """Factor by which the total shard count grows. Used only for the log line."""
    cur_total = 1
    new_total = 1
    for dim in set(current) | set(new):
        cur_total *= current.get(dim, 1)
        new_total *= new.get(dim, 1)
    return new_total // cur_total if cur_total else 1


def merge_finest_mesh(current_finest: ShardMesh, new_mesh: ShardMesh) -> dict[int, int]:
    """LCM-merge a joiner's mesh into the established finest mesh, per dimension.

    Mirrors `NIXLSharding.find_finest_shard_mesh`, but for the incremental case
    where one new mesh meets an already-computed finest mesh.
    """
    merged: dict[int, int] = dict(current_finest)
    for dim, count in new_mesh.items():
        merged[dim] = lcm(merged.get(dim, 1), count)
    return merged


def suggest_divisor_degrees(current_finest: ShardMesh, requested: int, limit: int = 4) -> list[int]:
    """Parallelism degrees nearest ``requested`` that keep the finest mesh unchanged.

    A degree is safe exactly when it divides the current LCM on every dimension it
    touches. We only need the per-dimension maximum, since a divisor of that also
    divides the LCM. Ties break toward the larger degree -- when 2 and 4 are equally
    distant from 3, suggesting 4 keeps more parallelism.
    """
    if not current_finest:
        return []
    ceiling = max(current_finest.values())
    divisors = [d for d in range(1, ceiling + 1) if ceiling % d == 0]
    divisors.sort(key=lambda d: (abs(d - requested), -d))
    return divisors[:limit]


def _reshard_blocks(dim_size: int, old_count: int, new_count: int) -> bool:
    """Would refactoring one dimension from ``old_count`` to ``new_count`` shards raise?

    This mirrors what actually happens at re-registration, which is subtler than
    "does the global dimension divide by the new count":

    `refactor_based_on_finer_shard_mesh` keeps each client's *local* slice and splits
    it further. A client holding 1 of ``old_count`` shards owns ``dim_size //
    old_count`` rows, and must now split them into ``new_count // old_count`` pieces.
    `get_local_sharded_tensors` then checks divisibility against that **local** count,
    not the global one (`nixl_spec.py` `_local_shard_mesh` is derived from
    `shard_indices`, not from `shard_mesh`).

    The global condition disagrees with the runtime in both directions -- e.g.
    dim=1536, 8 -> 12 succeeds globally (1536 % 12 == 0) but raises in reality
    (192 local rows do not split into 12/8), while dim=100, 8 -> 16 fails globally
    yet never gets that far because 100 % 8 != 0 already. Only the local form agrees
    on every case; see ``TestAgreementWithShardingRuntime`` in the tests, which
    compares the two exhaustively against the real runtime.
    """
    if old_count and dim_size % old_count != 0:
        # The established sharding is already inexpressible; not this join's fault,
        # but re-sharding will still raise, so report it.
        return True
    if old_count == 0 or new_count % old_count != 0:
        # refactor_based_on_finer_shard_mesh requires the new mesh to be a multiple.
        return True
    growth = new_count // old_count
    return (dim_size // old_count) % growth != 0


def classify_join(
    current_finest: ShardMesh,
    new_mesh: ShardMesh,
    tensor_shapes: TensorShapes,
) -> JoinDecision:
    """Decide whether a client with ``new_mesh`` may join.

    Args:
        current_finest: the established finest shard mesh, ``{dim: shard_count}``.
            Empty means no client has registered yet, so anything is admissible.
        new_mesh: the joining client's own ``shard_mesh``.
        tensor_shapes: ``{key: shape}`` over all registered tensors. Needed because
            expressibility is a property of the *tensors*, not of the meshes.

    Returns:
        A `JoinDecision`. Callers must treat `REJECT` as fatal for the joiner while
        leaving the existing mesh untouched and serving.
    """
    current = dict(current_finest)
    merged = merge_finest_mesh(current, new_mesh)

    if merged == current:
        return JoinDecision(JoinVerdict.FAST_PATH, current, merged)

    if not current:
        # First client to register: there are no existing descriptors to invalidate
        # and no re-shard to perform, so its own mesh simply becomes the baseline.
        # Whether that mesh can shard its tensors is `get_sharding_for_param`'s
        # concern at declaration time (item A5), not an admission question.
        return JoinDecision(JoinVerdict.FAST_PATH, current, merged)

    # The LCM rose. Expressibility is now the question, and it must be checked
    # against every registered tensor -- not just the joiner's -- because
    # re-sharding re-runs the divisibility check on all of them.
    blocking: list[tuple[str, int, int, int]] = []
    for key, shape in tensor_shapes.items():
        for dim, new_count in merged.items():
            if dim >= len(shape):
                continue
            old_count = current.get(dim, 1)
            if new_count != old_count and _reshard_blocks(shape[dim], old_count, new_count):
                blocking.append((key, dim, shape[dim], new_count))

    requested = max(new_mesh.values()) if new_mesh else 1
    suggestions = suggest_divisor_degrees(current, requested)

    if blocking:
        return JoinDecision(JoinVerdict.REJECT, current, merged, blocking, suggestions)
    return JoinDecision(JoinVerdict.RESHARD, current, merged, [], suggestions)
