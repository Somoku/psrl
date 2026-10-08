"""Strided descriptors for non-contiguous shards (implementation item A2).

Sharding a tensor along any dimension but the first produces local shards that are
*views* with a row gap: shard 0 of an 8x6 tensor split on dim 1 is 8 rows of 3 elements
with a stride of 6. NIXL cannot transfer such a view directly, so the current code
copies each one into a contiguous staging buffer, registers that, and transfers it --
which costs a D2D copy per shard, extra GPU memory for the staging slots, and a serial
wait when the slots run out (`_pinned_slot_running_read_xfer`).

A uniform row gap is exactly what an Nx5 strided descriptor expresses:
``(address, row_bytes, device, stride_bytes, row_count)``. One such row replaces the
whole staging round trip. Verified byte-equivalent to the plain descriptor list,
including that it leaves the gaps untouched, in
`tests/nixl/test_descriptor_correctness_gpu.py`.

Two limits, both deliberate:

* **Only 2-D-style uniformity.** A view with gaps at more than one level (three or more
  dimensions where the middle one is sliced) needs several strided rows or several plain
  ones. `describe_shard` reports that case as unsupported rather than guessing, and the
  caller keeps staging for it.
* **`nixlXferDList` is Nx3-only in every NIXL build.** Strided descriptors exist only
  via `prep_xfer_dlist`, so they are usable from the prepped path and nowhere else.
  Treating that asymmetry as a missing feature is a trap worth naming.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StridedLayout:
    """A shard expressible as ``row_count`` equal blocks at a constant byte stride."""

    address: int
    row_bytes: int
    device_id: int
    stride_bytes: int
    row_count: int

    @property
    def total_bytes(self) -> int:
        return self.row_bytes * self.row_count

    @property
    def span_bytes(self) -> int:
        """Bytes from the first address to the end of the last row, gaps included.

        Distinct from `total_bytes`, which counts only the payload. The span is what a
        containment check against a registered region needs: the gaps sit inside the
        region too, and the final row must end before the region does.
        """
        if self.row_count == 0:
            return 0
        return (self.row_count - 1) * self.stride_bytes + self.row_bytes

    @property
    def is_contiguous(self) -> bool:
        """A stride equal to the block size means the view has no gaps after all."""
        return self.stride_bytes == self.row_bytes

    def to_nx5(self) -> tuple[int, int, int, int, int]:
        """The tuple `prep_xfer_dlist` consumes as one row of an Nx5 array."""
        return (self.address, self.row_bytes, self.device_id, self.stride_bytes, self.row_count)

    def to_nx3_rows(self) -> list[tuple[int, int, int]]:
        """The equivalent plain descriptors, for builds or paths without strided support."""
        return [
            (self.address + i * self.stride_bytes, self.row_bytes, self.device_id)
            for i in range(self.row_count)
        ]


def describe_shard(tensor: torch.Tensor) -> StridedLayout | None:
    """Describe ``tensor`` as a strided layout, or None if it needs staging.

    Returns None rather than raising: an inexpressible view is a normal case that falls
    back to the staging path, not an error.

    The test is that the view is a sequence of equal, evenly spaced blocks. Walking the
    dimensions from the fastest-varying end, the trailing dimensions must be packed (so
    each block is itself contiguous), and at most one outer dimension may then carry a
    gap. Anything else -- two gapped levels, or a negative/zero stride from a flip or
    broadcast -- is reported as unsupported.
    """
    if tensor.numel() == 0 or tensor.device.type == "meta":
        return None

    element_size = tensor.element_size()
    sizes = list(tensor.shape)
    strides = list(tensor.stride())

    # Broadcast (stride 0) and reversed (negative stride) views are not transferable as
    # a simple strided block, and a 0-stride view would alias the same bytes repeatedly.
    if any(stride <= 0 for size, stride in zip(sizes, strides) if size > 1):
        return None

    # Collapse trailing packed dimensions into one block.
    block_elements = 1
    index = len(sizes) - 1
    while index >= 0 and strides[index] == block_elements:
        block_elements *= sizes[index]
        index -= 1

    if index < 0:
        # Fully packed: contiguous, so a single plain descriptor covers it.
        return StridedLayout(
            address=tensor.data_ptr(),
            row_bytes=block_elements * element_size,
            device_id=tensor.device.index or 0,
            stride_bytes=block_elements * element_size,
            row_count=1,
        )

    # One gapped dimension is expressible; everything outside it must be degenerate.
    row_count = sizes[index]
    stride_bytes = strides[index] * element_size
    for outer in range(index):
        if sizes[outer] != 1:
            return None

    return StridedLayout(
        address=tensor.data_ptr(),
        row_bytes=block_elements * element_size,
        device_id=tensor.device.index or 0,
        stride_bytes=stride_bytes,
        row_count=row_count,
    )


def can_avoid_staging(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` can be transferred without a contiguous staging copy."""
    return describe_shard(tensor) is not None
