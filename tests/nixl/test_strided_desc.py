"""Strided layout analysis for non-contiguous shards (implementation item A2).

`describe_shard` decides whether a shard can skip the contiguous staging copy. Both
answers are costly to get wrong: a false negative keeps a staging buffer and a serial
slot wait that the design is trying to remove, while a false positive would transfer
the wrong bytes -- a strided descriptor over a view that is not uniformly spaced reads
whatever sits in the gaps.

So the central test is equivalence: for every shape and sharding the converters
produce, the strided description must cover exactly the bytes a contiguous copy would,
and `to_nx3_rows` must agree with `to_nx5` element for element.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from psrl.utils.nixl.nixl_spec import NIXLSharding
from psrl.utils.nixl.strided_desc import StridedLayout, can_avoid_staging, describe_shard

pytestmark = pytest.mark.cpu_test


def _shard(tensor: torch.Tensor, dim: int, count: int, index: int) -> torch.Tensor:
    sharding = NIXLSharding(
        shard_mesh=OrderedDict([(dim, count)]),
        shard_indices=[(index,)],
    )
    full = NIXLSharding(
        shard_mesh=OrderedDict([(dim, count)]),
        shard_indices=[(i,) for i in range(count)],
    )
    return full.get_local_sharded_tensors(tensor)[index]


class TestContiguousShards:
    """Dim-0 sharding yields contiguous shards; one plain descriptor suffices."""

    def test_dim0_shard_is_single_block(self):
        tensor = torch.arange(8 * 6, dtype=torch.int32).reshape(8, 6)
        shard = _shard(tensor, dim=0, count=2, index=0)
        assert shard.is_contiguous()

        layout = describe_shard(shard)
        assert layout is not None
        assert layout.row_count == 1
        assert layout.is_contiguous
        assert layout.total_bytes == shard.numel() * shard.element_size()

    def test_whole_tensor_is_single_block(self):
        tensor = torch.zeros(16, 32, dtype=torch.bfloat16)
        layout = describe_shard(tensor)
        assert layout is not None
        assert layout.row_count == 1
        assert layout.total_bytes == tensor.numel() * 2


class TestStridedShards:
    """Dim-1 sharding yields the uniform row gap a strided descriptor expresses."""

    @pytest.mark.parametrize("count", [2, 3, 4])
    def test_dim1_shard_is_strided(self, count):
        rows, cols = 8, 12
        tensor = torch.arange(rows * cols, dtype=torch.int32).reshape(rows, cols)
        shard = _shard(tensor, dim=1, count=count, index=0)
        assert not shard.is_contiguous()

        layout = describe_shard(shard)
        assert layout is not None
        assert layout.row_count == rows
        assert layout.row_bytes == (cols // count) * 4
        assert layout.stride_bytes == cols * 4
        assert not layout.is_contiguous

    def test_each_shard_points_at_its_own_offset(self):
        """A shard's address must be its own, or every shard transfers shard 0."""
        tensor = torch.arange(8 * 12, dtype=torch.int32).reshape(8, 12)
        addresses = []
        for index in range(3):
            layout = describe_shard(_shard(tensor, dim=1, count=3, index=index))
            assert layout is not None
            addresses.append(layout.address)
        assert len(set(addresses)) == 3, "shards share an address"
        assert addresses == sorted(addresses)

    def test_strided_rows_cover_exactly_the_shard_bytes(self):
        """The equivalence that justifies deleting the staging copy."""
        rows, cols, count = 8, 12, 3
        tensor = torch.arange(rows * cols, dtype=torch.int32).reshape(rows, cols)
        shard = _shard(tensor, dim=1, count=count, index=1)
        layout = describe_shard(shard)
        assert layout is not None

        # Reconstruct from the descriptor rows and compare against a contiguous copy.
        base = tensor.data_ptr()
        element_size = tensor.element_size()
        flat = tensor.reshape(-1)
        rebuilt = []
        for address, length, _device in layout.to_nx3_rows():
            start = (address - base) // element_size
            rebuilt.append(flat[start : start + length // element_size])
        assert torch.equal(torch.cat(rebuilt), shard.contiguous().reshape(-1))

    def test_nx5_and_nx3_describe_the_same_bytes(self):
        tensor = torch.arange(6 * 10, dtype=torch.int32).reshape(6, 10)
        layout = describe_shard(_shard(tensor, dim=1, count=2, index=0))
        assert layout is not None

        address, row_bytes, device, stride, count = layout.to_nx5()
        rows = layout.to_nx3_rows()
        assert len(rows) == count
        assert rows[0] == (address, row_bytes, device)
        assert rows[-1] == (address + (count - 1) * stride, row_bytes, device)


class TestUnsupportedLayouts:
    """False positives would transfer gap bytes, so these must report None."""

    def test_two_gapped_levels_are_rejected(self):
        """A gap at two levels cannot be one strided row.

        Note which 3-D slices do *not* qualify as two-level: slicing the middle
        dimension of a row-major tensor leaves the trailing dimensions packed, so the
        view collapses to equal blocks at a constant stride and is expressible (see
        `test_three_dim_middle_slice_is_expressible`). Two levels requires a gap that
        survives that collapse -- slicing the last dimension *and* an outer one.
        """
        tensor = torch.arange(4 * 6 * 8, dtype=torch.int32).reshape(4, 6, 8)
        view = tensor[:, :3, :4]
        assert not view.is_contiguous()
        assert describe_shard(view) is None

    def test_three_dim_middle_slice_is_expressible(self):
        """Verified byte-exact: the trailing dims are packed, so blocks are uniform."""
        tensor = torch.arange(4 * 6 * 8, dtype=torch.int32).reshape(4, 6, 8)
        view = tensor[:, :3, :]
        layout = describe_shard(view)
        assert layout is not None
        assert layout.row_count == 4
        assert layout.row_bytes == 3 * 8 * 4

        base, element_size = tensor.data_ptr(), tensor.element_size()
        flat = tensor.reshape(-1)
        rebuilt = torch.cat(
            [
                flat[(address - base) // element_size :][: length // element_size]
                for address, length, _device in layout.to_nx3_rows()
            ]
        )
        assert torch.equal(rebuilt, view.contiguous().reshape(-1))

    def test_broadcast_view_is_rejected(self):
        """A zero stride aliases the same bytes; transferring it would duplicate them."""
        tensor = torch.zeros(1, 8, dtype=torch.int32).expand(4, 8)
        assert describe_shard(tensor) is None

    def test_flip_is_a_copy_not_a_negative_stride_view(self):
        """Recording why there is no negative-stride test tensor here.

        `torch.flip` materialises a contiguous copy rather than a reversed view, and
        `torch.as_strided` refuses negative strides outright ("Negative strides are not
        supported at the moment"). So a negative stride is unreachable from PyTorch, and
        the guard in `describe_shard` exists for descriptors arriving from elsewhere --
        it cannot be exercised through a tensor.
        """
        tensor = torch.arange(16, dtype=torch.int32).reshape(4, 4)
        flipped = torch.flip(tensor, dims=[0])
        assert flipped.is_contiguous()
        assert all(stride > 0 for stride in flipped.stride())

        with pytest.raises(RuntimeError, match="Negative strides"):
            torch.as_strided(tensor, size=(4, 4), stride=(-4, 1), storage_offset=12)

    def test_zero_and_negative_strides_are_rejected_by_the_predicate(self):
        """Exercise the guard directly, since no PyTorch tensor can carry it."""

        class FakeView:
            """Just the attributes `describe_shard` reads."""

            def __init__(self, sizes, strides):
                self._sizes, self._strides = sizes, strides
                self.shape = torch.Size(sizes)
                self.device = torch.device("cpu")

            def numel(self):
                count = 1
                for size in self._sizes:
                    count *= size
                return count

            def element_size(self):
                return 4

            def stride(self):
                return tuple(self._strides)

            def data_ptr(self):
                return 0x1000

        assert describe_shard(FakeView([4, 4], [-4, 1])) is None
        assert describe_shard(FakeView([4, 4], [0, 1])) is None

    def test_meta_tensor_is_rejected(self):
        """Meta tensors have no address to describe."""
        assert describe_shard(torch.zeros(4, 4, device="meta")) is None

    def test_empty_tensor_is_rejected(self):
        assert describe_shard(torch.zeros(0, dtype=torch.int32)) is None


class TestAgreementWithContiguityCheck:
    """`can_avoid_staging` must be consistent with what the shards actually look like."""

    @pytest.mark.parametrize("shape", [(8, 12), (16, 32), (4096, 128)])
    @pytest.mark.parametrize("dim,count", [(0, 2), (0, 4), (1, 2), (1, 4)])
    def test_every_converter_shape_is_expressible(self, shape, dim, count):
        """Both sharding dims over realistic shapes must avoid staging entirely.

        This is the claim that makes A2 worth doing: if a common case still needed
        staging, the buffers and the slot wait would have to stay.
        """
        if shape[dim] % count != 0:
            pytest.skip("uneven")
        tensor = torch.zeros(shape, dtype=torch.bfloat16)
        shard = _shard(tensor, dim=dim, count=count, index=0)
        assert can_avoid_staging(shard), (
            f"shape={shape} dim={dim} count={count} would still need a staging copy"
        )

    @pytest.mark.parametrize("dim,count", [(0, 2), (1, 2), (1, 4)])
    def test_total_bytes_matches_the_shard(self, dim, count):
        tensor = torch.zeros(8, 16, dtype=torch.float32)
        shard = _shard(tensor, dim=dim, count=count, index=0)
        layout = describe_shard(shard)
        assert layout is not None
        assert layout.total_bytes == shard.numel() * shard.element_size()


class TestMoEExpertShapes:
    """Three-dimensional expert tensors, which MoE grouping introduces."""

    def test_expert_dim_shard_is_contiguous(self):
        """Splitting experts (dim 0) keeps each shard packed."""
        experts = torch.zeros(8, 256, 512, dtype=torch.bfloat16)
        shard = _shard(experts, dim=0, count=4, index=0)
        layout = describe_shard(shard)
        assert layout is not None
        assert layout.row_count == 1, "expert-dim shards should need no striding"

    def test_projection_dim_shard_is_strided_per_expert(self):
        """Splitting the projection of a 3-D expert tensor gives one block per expert.

        Each expert contributes a packed `(proj_slice, hidden)` block, and the blocks sit
        a whole expert apart -- so it is one strided row with `row_count == num_experts`,
        not an unsupported two-level gap.
        """
        num_experts, projection, hidden = 8, 256, 512
        experts = torch.zeros(num_experts, projection, hidden, dtype=torch.bfloat16)
        shard = _shard(experts, dim=1, count=2, index=0)

        layout = describe_shard(shard)
        assert layout is not None
        assert layout.row_count == num_experts
        assert layout.row_bytes == (projection // 2) * hidden * 2
        assert layout.stride_bytes == projection * hidden * 2
        assert layout.total_bytes == shard.numel() * shard.element_size()

    def test_single_expert_projection_shard_is_one_block(self):
        """With one expert the outer dimension is degenerate: a single block."""
        experts = torch.zeros(1, 256, 512, dtype=torch.bfloat16)
        shard = _shard(experts, dim=1, count=2, index=0)
        layout = describe_shard(shard)
        assert layout is not None
        assert layout.row_count == 1
        assert layout.total_bytes == shard.numel() * shard.element_size()


class TestLayoutArithmetic:
    def test_contiguous_flag_detects_a_gapless_stride(self):
        layout = StridedLayout(address=0x1000, row_bytes=256, device_id=0, stride_bytes=256, row_count=4)
        assert layout.is_contiguous
        assert layout.total_bytes == 1024

    def test_nx3_expansion_is_evenly_spaced(self):
        layout = StridedLayout(address=0x1000, row_bytes=64, device_id=0, stride_bytes=256, row_count=3)
        assert layout.to_nx3_rows() == [
            (0x1000, 64, 0),
            (0x1100, 64, 0),
            (0x1200, 64, 0),
        ]
