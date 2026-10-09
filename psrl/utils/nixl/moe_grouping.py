"""Canonical MoE expert grouping (implementation item C1).

PSRL's canonical namespace is the HF-expanded form, so every expert's every projection
is its own key: ``model.layers.3.mlp.experts.57.gate_proj.weight``. Measured on the
checkpoints at hand, that is 18,867 keys for Qwen3-30B-A3B (97.7% of them expert keys)
and 70,272 for a Kimi-K2-class model. Both the handshake payload and the per-key
formatting cost scale with that count, which makes it the blocking item at MoE scale
rather than a tuning opportunity.

Grouping replaces the per-expert keys of one (layer, projection) with a single key whose
expert axis becomes a sharding dimension. `NIXLSharding.shard_mesh` is already an
``OrderedDict[dim, count]``, so expressing it costs nothing new:

    model.layers.3.mlp.experts.{0..127}.gate_proj.weight    128 keys
    -> model.layers.3.mlp.experts.gate_proj.weight          1 key, shard_mesh {0: 128}

This is also closer to what the inference engine wants: vLLM's fused loader expects the
3-D ``w13_weight``, and the converter currently splits it apart only to have the
transport carry the pieces separately.

The two sides are not symmetric, and that asymmetry is what `GroupedShards` exists for:

* **vLLM** holds ``w13_weight`` as one fused 3-D tensor, so the expert axis is a real
  tensor dimension and each projection is a *view* of it.
* **Megatron/TE** registers every expert as its own ``torch.empty`` (TE's
  ``grouped_linear.py`` constructs ``weight{i}`` per GEMM), so the experts of one layer
  share no storage at all and no view can span them.

A grouped key is therefore one tensor on one side and N tensors on the other. Stacking
Megatron's experts to force a single tensor would copy the whole MoE every sync, which
is the cost the grouping exists to avoid, so the transport takes the shards as they lie.

Two consequences the design calls out, both handled by the chunk/commit-group split
(item B1) rather than here:

* commit groups get coarser -- a group is now the whole fused tensor
* a single key gets very large: Kimi-K2's per-layer ``w13_weight`` is ~11GB at fp8, so
  chunking must be able to split one key, which `plan_chunks` does
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass

import torch

from psrl.utils.nixl.nixl_spec import NIXLSharding

#: ``<prefix>.experts.<id>.<projection>``. The prefix carries the layer, so grouping
#: never merges across layers.
_EXPERT_KEY = re.compile(r"^(?P<prefix>.+)\.experts\.(?P<expert>\d+)\.(?P<projection>.+)$")


@dataclass(frozen=True)
class ExpertGroup:
    """One (prefix, projection) pair and the expert keys it subsumes."""

    grouped_key: str
    prefix: str
    projection: str
    #: Source keys ordered by expert id, which fixes the shard index of each expert.
    member_keys: tuple[str, ...]
    expert_ids: tuple[int, ...]

    @property
    def num_experts(self) -> int:
        return len(self.expert_ids)

    def shard_mesh(self, inner_mesh: OrderedDict[int, int] | None = None) -> OrderedDict[int, int]:
        """Shard mesh for the grouped key: expert axis first, then any inner sharding.

        ``inner_mesh`` is the per-expert tensor's existing mesh -- ETP sharding within an
        expert, for instance. The expert axis is prepended as dimension 0, so a grouped
        key under ETP != 1 ends up two-dimensional, which is the case design 3.5.5 flags
        for verification against `find_finest_shard_mesh`.
        """
        mesh: OrderedDict[int, int] = OrderedDict([(0, self.num_experts)])
        if inner_mesh:
            for dim, count in inner_mesh.items():
                mesh[dim + 1] = count
        return mesh

    def shard_index_of(self, expert_id: int) -> int:
        """Position of ``expert_id`` along the expert axis."""
        return self.expert_ids.index(expert_id)


@dataclass(frozen=True)
class GroupedShards:
    """A key whose local shards live in separate storages.

    The transport registers each shard where it already is. The alternative -- stacking
    the experts into one tensor so the key has a single storage -- would copy the entire
    MoE on every sync, which is larger than the metadata cost grouping removes.

    `shards` is ordered to match `shard_indices` positionally; that pairing is the whole
    contract, since the shard index decides which expert the bytes belong to.
    """

    shards: tuple[torch.Tensor, ...]
    shard_indices: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if len(self.shards) != len(self.shard_indices):
            raise ValueError(
                f"GroupedShards has {len(self.shards)} tensors but "
                f"{len(self.shard_indices)} shard indices; the two are paired positionally."
            )

    @property
    def dtype(self) -> torch.dtype:
        return self.shards[0].dtype

    @property
    def device(self) -> torch.device:
        return self.shards[0].device

    @property
    def shard_shape(self) -> torch.Size:
        return self.shards[0].shape


@dataclass(frozen=True)
class ExpertGrouping:
    """Result of grouping a state dict: the grouped view plus how to address it.

    `source_map` is what lets a loader that still speaks per-expert keys -- a checkpoint
    reader, for instance -- write into the grouped form without reconstructing the
    grouping rule itself.
    """

    state_dict: dict[str, object]
    sharding_dict: dict[str, NIXLSharding]
    #: original per-expert key -> (grouped key, that expert's shard index)
    source_map: dict[str, tuple[str, tuple[int, ...]]]
    groups: tuple[ExpertGroup, ...]


def parse_expert_key(key: str) -> tuple[str, int, str] | None:
    """Split an expert key into (prefix, expert_id, projection), or None if not one."""
    match = _EXPERT_KEY.match(key)
    if match is None:
        return None
    return match["prefix"], int(match["expert"]), match["projection"]


def group_expert_keys(keys: list[str]) -> tuple[list[ExpertGroup], list[str]]:
    """Group expert keys; return the groups and the untouched non-expert keys.

    Expert ids are sorted numerically rather than lexically, because the sort order
    *is* the shard index: a lexical sort would place expert 10 before expert 2 and
    silently permute the expert dimension.
    """
    buckets: dict[tuple[str, str], list[tuple[int, str]]] = {}
    passthrough: list[str] = []

    for key in keys:
        parsed = parse_expert_key(key)
        if parsed is None:
            passthrough.append(key)
            continue
        prefix, expert_id, projection = parsed
        buckets.setdefault((prefix, projection), []).append((expert_id, key))

    groups = []
    for (prefix, projection), members in buckets.items():
        members.sort(key=lambda item: item[0])
        expert_ids = tuple(expert_id for expert_id, _key in members)
        if len(set(expert_ids)) != len(expert_ids):
            raise ValueError(
                f"{prefix}.experts.*.{projection} has duplicate expert ids: {expert_ids}"
            )
        groups.append(
            ExpertGroup(
                grouped_key=f"{prefix}.experts.{projection}",
                prefix=prefix,
                projection=projection,
                member_keys=tuple(key for _expert_id, key in members),
                expert_ids=expert_ids,
            )
        )

    groups.sort(key=lambda group: group.grouped_key)
    return groups, passthrough


def grouped_key_count(keys: list[str]) -> int:
    """Number of keys after grouping -- the quantity the metadata plane scales with."""
    groups, passthrough = group_expert_keys(keys)
    return len(groups) + len(passthrough)


def grouping_report(keys: list[str]) -> dict[str, int | float]:
    """Summary for the handshake log, so the reduction is visible rather than assumed."""
    groups, passthrough = group_expert_keys(keys)
    after = len(groups) + len(passthrough)
    expert_keys = sum(group.num_experts for group in groups)
    return {
        "keys_before": len(keys),
        "keys_after": after,
        "expert_keys": expert_keys,
        "expert_groups": len(groups),
        "reduction": len(keys) / after if after else 1.0,
    }


def _inner_mesh_of(sharding: NIXLSharding) -> OrderedDict[int, int]:
    return OrderedDict(sharding.shard_mesh)


def local_shards_of(entry: object, sharding: NIXLSharding) -> list[torch.Tensor]:
    """Local shards of a state-dict entry, whether it is one tensor or many.

    Registration slices a plain tensor with `get_local_sharded_tensors`, but a grouped
    expert key has no single tensor to slice -- its shards are separate storages. Both
    produce a list positionally aligned with ``sharding.shard_indices``, so callers that
    only need the shards can stay unaware of which case they have.
    """
    if isinstance(entry, GroupedShards):
        if len(entry.shards) != len(sharding.shard_indices):
            raise ValueError(
                f"grouped entry carries {len(entry.shards)} shards but its sharding "
                f"declares {len(sharding.shard_indices)}; the two are paired positionally."
            )
        return list(entry.shards)
    return sharding.get_local_sharded_tensors(entry)


def entry_is_empty(entry: object) -> bool:
    """Whether an entry holds no transferable bytes.

    A grouped key is empty only when every one of its shards is: under expert
    parallelism a rank can legitimately hold a subset, and treating the whole group as
    empty because one expert is would drop the rest silently.
    """
    if isinstance(entry, GroupedShards):
        return all(
            shard.device.type == "meta" or shard.untyped_storage().nbytes() == 0
            for shard in entry.shards
        )
    return entry.device.type == "meta" or entry.untyped_storage().nbytes() == 0


def group_expert_tensors(
    state_dict: dict[str, torch.Tensor],
    sharding_dict: dict[str, NIXLSharding],
    num_experts: int,
) -> ExpertGrouping:
    """Collapse per-expert keys into grouped keys with the expert axis as dimension 0.

    ``num_experts`` is the *global* expert count and must be supplied rather than
    inferred from the keys present. Under expert parallelism a rank holds only its own
    slice, so inferring would give each rank a different mesh for the same key, and the
    server's LCM reconciliation would then invent a finest mesh no one can satisfy.

    Every member of a group must share one inner sharding. They are the same projection
    of the same layer, differing only in expert, so a disagreement means the caller
    assembled the group wrongly -- raising beats silently taking the first one, which
    would mis-address every expert after it.
    """
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")

    groups, passthrough = group_expert_keys(list(state_dict))

    grouped_state: dict[str, object] = {key: state_dict[key] for key in passthrough}
    grouped_sharding: dict[str, NIXLSharding] = {
        key: sharding_dict[key] for key in passthrough if key in sharding_dict
    }
    source_map: dict[str, tuple[str, tuple[int, ...]]] = {}

    for group in groups:
        if group.expert_ids[-1] >= num_experts:
            raise ValueError(
                f"{group.grouped_key} holds expert id {group.expert_ids[-1]}, which is "
                f"outside a model of {num_experts} experts."
            )

        inner_meshes = {key: _inner_mesh_of(sharding_dict[key]) for key in group.member_keys}
        reference = inner_meshes[group.member_keys[0]]
        for key, mesh in inner_meshes.items():
            if mesh != reference:
                raise ValueError(
                    f"{group.grouped_key}: expert key {key} has inner shard mesh {dict(mesh)} "
                    f"but {group.member_keys[0]} has {dict(reference)}; experts of one "
                    "projection must be sharded identically."
                )

        # The expert axis carries the *global* count, while the inner dims keep whatever
        # the per-expert tensor already declared. Dimension numbers shift by one because
        # the expert axis is prepended.
        #
        # An inner dim of count 1 is dropped: it says the tensor is unsharded along it,
        # which is already the default reading of an absent dim. Keeping it would append
        # a degenerate axis to every grouped key -- harmless to the lcm, but it makes a
        # dense-inner key look two-dimensional and forces a matching pad into every shard
        # index.
        kept_dims = [dim for dim, count in reference.items() if count > 1]
        mesh: OrderedDict[int, int] = OrderedDict([(0, num_experts)])
        for position, dim in enumerate(kept_dims):
            mesh[position + 1] = reference[dim]

        reference_dims = list(reference.keys())
        keep_positions = [reference_dims.index(dim) for dim in kept_dims]

        # `keep_positions` is bound as a default rather than captured: it is redefined on
        # every group, and a late-binding closure would silently index later groups with
        # an earlier group's layout.
        def _full_index(
            expert_id: int,
            inner_index: tuple[int, ...],
            positions: tuple[int, ...] = tuple(keep_positions),
        ) -> tuple[int, ...]:
            return (expert_id, *(inner_index[position] for position in positions))

        shards: list[torch.Tensor] = []
        indices: list[tuple[int, ...]] = []
        for expert_id, member_key in zip(group.expert_ids, group.member_keys):
            inner_indices = sharding_dict[member_key].shard_indices
            member_shards = sharding_dict[member_key].get_local_sharded_tensors(state_dict[member_key])
            for inner_index, shard in zip(inner_indices, member_shards):
                shards.append(shard)
                indices.append(_full_index(expert_id, inner_index))
            # A member key maps to its first shard: that identifies the expert, which is
            # all a per-expert loader needs to find its slot.
            source_map[member_key] = (group.grouped_key, _full_index(expert_id, inner_indices[0]))

        order = sorted(range(len(indices)), key=lambda i: indices[i])
        grouped_state[group.grouped_key] = GroupedShards(
            shards=tuple(shards[i] for i in order),
            shard_indices=tuple(indices[i] for i in order),
        )
        grouped_sharding[group.grouped_key] = NIXLSharding(
            shard_mesh=mesh,
            shard_indices=[indices[i] for i in order],
        )

    return ExpertGrouping(
        state_dict=grouped_state,
        sharding_dict=grouped_sharding,
        source_map=source_map,
        groups=tuple(groups),
    )
