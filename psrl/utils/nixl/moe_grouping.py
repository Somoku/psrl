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

    def shard_mesh(self, inner_mesh: "OrderedDict[int, int] | None" = None) -> "OrderedDict[int, int]":
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
