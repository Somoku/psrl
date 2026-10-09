"""Staggered chunk start offsets (implementation item D5).

The point of staggering is measurable: how many distinct chunks the first wave holds at
any instant, and therefore whether a released queuer can find a peer without waiting for
anyone to finish a full round. These tests measure that rather than checking that
offsets differ, because differing offsets that still collide would pass the weaker check.
"""

from __future__ import annotations

import pytest
from psrl.utils.nixl.chunk_offset import (
    distinct_chunks_held,
    first_wave_overlap,
    group_start_chunks,
    plan_read_orders,
    read_order,
    should_stagger,
    start_chunk_for_instance,
)

pytestmark = pytest.mark.cpu_test


class TestDeterminism:
    """The index predicts holders without communicating, so this cannot be random."""

    def test_the_same_instance_always_gets_the_same_start(self):
        first = start_chunk_for_instance(3, num_chunks=32, num_instances=8)
        second = start_chunk_for_instance(3, num_chunks=32, num_instances=8)
        assert first == second

    def test_starts_are_spread_across_the_ring(self):
        starts = [start_chunk_for_instance(i, 32, 8) for i in range(8)]
        assert starts == [0, 4, 8, 12, 16, 20, 24, 28]

    def test_starts_stay_inside_the_ring(self):
        for index in range(20):
            start = start_chunk_for_instance(index, num_chunks=8, num_instances=20)
            assert 0 <= start < 8

    def test_non_positive_sizes_are_rejected(self):
        with pytest.raises(ValueError, match="num_chunks must be positive"):
            start_chunk_for_instance(0, 0, 4)
        with pytest.raises(ValueError, match="num_instances must be positive"):
            start_chunk_for_instance(0, 4, 0)


class TestReadOrder:
    def test_the_ring_covers_every_chunk_exactly_once(self):
        order = read_order(start_chunk=5, num_chunks=8)
        assert sorted(order) == list(range(8))

    def test_the_ring_wraps(self):
        assert read_order(start_chunk=6, num_chunks=8) == [6, 7, 0, 1, 2, 3, 4, 5]

    def test_starting_at_zero_is_plain_order(self):
        assert read_order(0, 4) == [0, 1, 2, 3]


class TestGroupAlignment:
    """A group split across the ring wrap makes its visibility check wrap-aware."""

    def test_group_starts_are_the_minimum_chunk_of_each_group(self):
        groups = [(0, 1, 2), (3,), (4, 5)]
        assert group_start_chunks(groups) == [0, 3, 4]

    def test_empty_groups_are_ignored(self):
        assert group_start_chunks([(0, 1), (), (4,)]) == [0, 4]

    def test_offsets_snap_to_a_group_boundary(self):
        groups = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15)]
        starts = group_start_chunks(groups)

        for index in range(8):
            start = start_chunk_for_instance(index, 16, 8, starts)
            assert start in starts, f"instance {index} starts mid-group at {start}"

    def test_no_group_is_split_by_the_ring_start(self):
        """The constraint, checked directly: each group is contiguous in read order."""
        groups = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15)]
        starts = group_start_chunks(groups)

        for index in range(8):
            order = read_order(start_chunk_for_instance(index, 16, 8, starts), 16)
            positions = {chunk: pos for pos, chunk in enumerate(order)}
            for group in groups:
                group_positions = sorted(positions[chunk] for chunk in group)
                assert group_positions == list(
                    range(group_positions[0], group_positions[0] + len(group))
                ), f"group {group} is not contiguous for instance {index}"

    def test_snapping_keeps_instance_order_monotonic(self):
        """The index relies on adjacent instances not swapping places."""
        groups = [(i, i + 1) for i in range(0, 16, 2)]
        starts = group_start_chunks(groups)
        computed = [start_chunk_for_instance(i, 16, 8, starts) for i in range(8)]

        assert computed == sorted(computed)


class TestStaggeringHelps:
    """The measurable claim."""

    def test_unstaggered_instances_all_hold_the_same_chunk(self):
        plain = [list(range(16)) for _ in range(4)]
        assert first_wave_overlap(plain, wave=1) == 4
        assert distinct_chunks_held(plain, wave=1) == 1

    def test_staggered_instances_hold_disjoint_chunks(self):
        orders = plan_read_orders(num_chunks=16, num_instances=4, trainer_fanout=2)
        assert first_wave_overlap(orders, wave=1) == 1
        assert distinct_chunks_held(orders, wave=1) == 4

    def test_a_released_queuer_finds_more_peers_sooner(self):
        """After two reads each, staggering covers 8 chunks rather than 2."""
        staggered = plan_read_orders(num_chunks=16, num_instances=4, trainer_fanout=2)
        plain = [list(range(16)) for _ in range(4)]

        assert distinct_chunks_held(staggered, wave=2) == 8
        assert distinct_chunks_held(plain, wave=2) == 2

    def test_the_advantage_holds_across_the_whole_first_wave(self):
        staggered = plan_read_orders(num_chunks=32, num_instances=8, trainer_fanout=4)
        plain = [list(range(32)) for _ in range(8)]

        for wave in range(1, 5):
            assert distinct_chunks_held(staggered, wave) > distinct_chunks_held(plain, wave)


class TestOnlyWhenItHelps:
    def test_staggering_is_skipped_when_everyone_reads_the_trainer(self):
        """R <= W means no queuing, so scattering costs locality for nothing."""
        assert not should_stagger(num_instances=4, trainer_fanout=4)
        assert not should_stagger(num_instances=2, trainer_fanout=4)
        assert should_stagger(num_instances=8, trainer_fanout=4)

    def test_plan_returns_plain_order_below_the_threshold(self):
        orders = plan_read_orders(num_chunks=8, num_instances=4, trainer_fanout=4)
        assert all(order == list(range(8)) for order in orders)

    def test_plan_covers_every_chunk_for_every_instance(self):
        orders = plan_read_orders(num_chunks=12, num_instances=5, trainer_fanout=2)
        for order in orders:
            assert sorted(order) == list(range(12))
