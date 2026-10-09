"""Chunk plan attached to the communication plan (implementation item B1).

`test_chunking.py` proves `plan_chunks` and `GroupCompletionTracker` in isolation. This
file covers the wiring: that the planner derives a chunk plan from what each pull client
is actually scheduled to read, using that client's own shard sizes.

The property B1 exists to establish is that **chunk and commit group are independent**
(design 3.6.1). A chunk is a byte budget and may split one key; a commit group is what
the engine loader requires atomically and may span several chunks. Collapsing them into
one unit makes either the pipeline or version visibility wrong, so the tests below check
the two can be varied independently rather than merely that both exist.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from psrl.utils.nixl.comm_plan import CommunicationPlanner, NIXLClientInfo, NIXLClientType
from psrl.utils.nixl.nixl_spec import NIXLSharding, NIXLShardMetaInfo, NIXLTensorInfo

pytestmark = pytest.mark.cpu_test

MB = 1024 * 1024


def _tensor_info(shard_count: int, shard_indices: list[int], shape) -> NIXLTensorInfo:
    sharding = NIXLSharding(
        shard_mesh=OrderedDict([(0, shard_count)]),
        shard_indices=[(i,) for i in sorted(shard_indices)],
    )
    meta = NIXLShardMetaInfo(
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        shape=torch.Size(shape),
        stride=(shape[1], 1),
        is_contiguous=True,
    )
    n = len(shard_indices)
    return NIXLTensorInfo(
        desc_bytes_list=[None] * n,
        temp_desc_bytes_list=[None] * n,
        sharding=sharding,
        shard_meta_infos=[meta] * n,
    )


def _clients(keys_to_shape: dict[str, tuple[int, int]], shard_count: int = 2):
    """One rollout puller and one PS, both holding every key."""
    shards = list(range(shard_count))
    infos = {k: _tensor_info(shard_count, shards, shape) for k, shape in keys_to_shape.items()}
    return {
        "ps0": NIXLClientInfo(
            name="ps0",
            node_ip="10.0.0.1",
            node_gpu_id=0,
            type=NIXLClientType.PS_FOR_PULL,
            tensor_infos=infos,
            meta=b"",
        ),
        "pull0": NIXLClientInfo(
            name="pull0",
            node_ip="10.0.0.2",
            node_gpu_id=0,
            type=NIXLClientType.PULL_SIDE,
            tensor_infos=infos,
            meta=b"",
        ),
    }


def _plan(keys_to_shape, chunk_bytes):
    planner = CommunicationPlanner(chunk_bytes=chunk_bytes)
    return planner.make_comm_plan(_clients(keys_to_shape))


class TestChunkPlanIsProduced:
    def test_pull_client_receives_a_chunk_plan(self):
        plan = _plan({"w": (1024, 1024)}, chunk_bytes=64 * MB)
        chunks, groups = plan.get_pull_chunks("pull0")
        assert chunks and groups

    def test_push_side_has_no_chunk_plan(self):
        """Only the pull side runs a chunk pipeline; planning for the push side would be
        dead metadata that still has to be serialised and shipped."""
        plan = _plan({"w": (1024, 1024)}, chunk_bytes=64 * MB)
        assert plan.get_pull_chunks("ps0") == ([], [])

    def test_unknown_client_degrades_to_no_chunking(self):
        """An absent entry means "no chunk plan", not an error: callers fall back to the
        per-key path that predates B1."""
        plan = _plan({"w": (1024, 1024)}, chunk_bytes=64 * MB)
        assert plan.get_pull_chunks("nobody") == ([], [])

    def test_one_commit_group_per_key(self):
        plan = _plan({"a": (512, 512), "b": (512, 512), "c": (512, 512)}, chunk_bytes=64 * MB)
        _chunks, groups = plan.get_pull_chunks("pull0")
        assert {g.name for g in groups} == {"a", "b", "c"}


class TestChunkAndGroupAreIndependent:
    """The core B1 claim: the two concepts vary independently."""

    def test_small_budget_splits_one_key_across_chunks(self):
        """A key larger than the budget must span several chunks -- an 11GB MoE fused
        weight is the real case, and it cannot be made to fit."""
        plan = _plan({"big": (4096, 4096)}, chunk_bytes=4 * MB)
        chunks, groups = plan.get_pull_chunks("pull0")

        assert len(chunks) > 1, "a key above the budget must occupy several chunks"
        assert len(groups) == 1, "but it is still one atomic commit group"
        assert len(groups[0].chunk_indices) > 1

    def test_large_budget_packs_many_keys_into_one_chunk(self):
        """The other direction: several groups share a chunk."""
        keys = {f"k{i}": (256, 256) for i in range(8)}
        plan = _plan(keys, chunk_bytes=512 * MB)
        chunks, groups = plan.get_pull_chunks("pull0")

        assert len(chunks) == 1
        assert len(groups) == 8
        assert all(g.chunk_indices == (0,) for g in groups)

    def test_group_membership_is_invariant_to_chunk_size(self):
        """Changing the byte budget must not change what a group contains.

        This is the separation judged directly: if group membership moved with the
        chunk size, the two concepts would still be entangled.
        """
        keys = {"a": (1024, 1024), "b": (512, 512)}
        memberships = []
        for chunk_bytes in (1 * MB, 4 * MB, 64 * MB, 512 * MB):
            _chunks, groups = _plan(keys, chunk_bytes).get_pull_chunks("pull0")
            memberships.append({g.name: g.shards for g in groups})

        assert all(m == memberships[0] for m in memberships), (
            "commit group membership changed with the chunk budget; the two concepts "
            "are still entangled"
        )

    def test_chunk_count_grows_as_the_budget_shrinks(self):
        keys = {"a": (2048, 2048)}
        counts = [len(_plan(keys, b).get_pull_chunks("pull0")[0]) for b in (64 * MB, 8 * MB, 1 * MB)]
        assert counts == sorted(counts), f"chunk count did not grow monotonically: {counts}"


class TestChunkPlanCoversEverythingExactlyOnce:
    def test_every_scheduled_shard_appears_in_some_chunk(self):
        """A shard missing from the chunk plan would simply never be transferred."""
        keys = {"a": (1024, 1024), "b": (512, 512), "c": (256, 256)}
        plan = _plan(keys, chunk_bytes=4 * MB)
        chunks, groups = plan.get_pull_chunks("pull0")

        in_chunks = {piece.shard for chunk in chunks for piece in chunk.pieces}
        in_groups = {shard for g in groups for shard in g.shards}
        assert in_groups <= in_chunks, f"shards missing from chunks: {in_groups - in_chunks}"

    def test_chunk_bytes_sum_to_the_scheduled_bytes(self):
        """Nothing is dropped or double-counted when a key is split."""
        shape = (1024, 1024)
        plan = _plan({"a": shape}, chunk_bytes=4 * MB)
        chunks, _groups = plan.get_pull_chunks("pull0")

        # 2 shards of a (1024, 1024) bf16 tensor, each the full declared shape.
        expected = 2 * shape[0] * shape[1] * 2
        assert sum(c.nbytes for c in chunks) == expected

    def test_no_chunk_exceeds_the_budget(self):
        budget = 4 * MB
        plan = _plan({"a": (2048, 2048), "b": (1024, 1024)}, chunk_bytes=budget)
        chunks, _groups = plan.get_pull_chunks("pull0")
        assert all(c.nbytes <= budget for c in chunks)


class TestPlanSurvivesSerialization:
    def test_chunk_plan_round_trips(self):
        """The plan is pickled to each client, so the chunk plan has to survive it."""
        from psrl.utils.nixl.comm_plan import NIXLCommPlan

        plan = _plan({"a": (1024, 1024)}, chunk_bytes=4 * MB)
        restored = NIXLCommPlan.deserialize(plan.serialize())

        before_chunks, before_groups = plan.get_pull_chunks("pull0")
        after_chunks, after_groups = restored.get_pull_chunks("pull0")
        assert len(after_chunks) == len(before_chunks)
        assert [g.name for g in after_groups] == [g.name for g in before_groups]
        assert [g.chunk_indices for g in after_groups] == [g.chunk_indices for g in before_groups]
