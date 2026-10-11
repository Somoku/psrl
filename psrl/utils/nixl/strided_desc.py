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
        """The equivalent plain descriptors, for builds or paths without strided support.

        One descriptor per row, and that count is the dominant cost of a transfer:
        `postXferReq` charges ~3.5-4us *per descriptor*, so a shard with 4096 rows pays
        ~8.6ms against ~4us for the same bytes in one descriptor (swept with
        `bench_cross_node_bw.py --tensors` at a fixed 2 GiB payload: 0.026ms at one
        descriptor, 16.942ms at 4096, bandwidth 354 -> 114 GB/s).

        There is no shortcut to take here: a gapless view never reaches this method,
        because `describe_shard` already collapses one to a single row. Every layout that
        gets here has real gaps, so the rows cannot be merged without moving bytes the
        caller did not ask for. Reducing the count means changing the *shard's layout*,
        not its description.
        """
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


#: Microseconds `postXferReq` charges per transfer descriptor.
#:
#: Measured by sweeping descriptor count at a fixed 2 GiB payload
#: (`tests/nixl/bench_cross_node_bw.py --tensors`): 0.026ms at 1 descriptor against
#: 16.942ms at 4096, i.e. a flat ~3.5-4us each, with bandwidth falling 354 -> 114 GB/s.
#: 2.1 is the lower figure observed in the 5-node pull itself (399 reads x 4096
#: descriptors in 3.426s), so estimates built on it are conservative rather than
#: flattering.
#:
#: Used only for reporting. Nothing branches on the absolute value -- `STRIDED_ROW_BUDGET`
#: is the decision -- so a drift in this constant misstates a log line rather than
#: changing behaviour.
STRIDED_US_PER_DESCRIPTOR = 2.1

#: Rows above which registering a shard in place costs more than staging it.
#:
#: `postXferReq` charges per descriptor, and a strided shard needs one per row
#: (`to_nx3_rows`). Staging needs exactly one descriptor plus a local device copy. So the
#: comparison is `row_count * per_desc` against `one_desc + copy_bytes / local_bw`, and
#: with the measured ~2.1-4us per descriptor the crossover is low: at 4096 rows, in-place
#: costs ~8.6ms against ~4us + ~5ms/8GB of copy. Measured on the 5-node run, the strided
#: path spent 3.43s where staging would have spent single-digit milliseconds.
#:
#: 64 rather than a computed crossover: the copy bandwidth is not known at registration
#: time, the per-descriptor cost varies ~3.5-4us with count, and the function is steep
#: either side of the threshold -- so a round number inside the flat region beats a
#: precise-looking number derived from constants that move. Below 64 the in-place path
#: keeps its advantage (no staging buffer, no copy-back, no slot to wait for); above it
#: the descriptor count dominates everything else.
STRIDED_ROW_BUDGET = 64


def strided_is_cheaper_than_staging(
    layout: StridedLayout | None, *, must_be_readable: bool = False
) -> bool:
    """Whether to register ``layout`` in place rather than stage it contiguously.

    Returns False for a layout that is None (nothing to register in place) and for one
    whose row count exceeds `STRIDED_ROW_BUDGET`.

    This is the decision that `is_strided_expressible` alone cannot make: a shard can be
    perfectly expressible as a strided range and still be the wrong thing to transfer
    that way, because expressibility says nothing about how many descriptors the
    expression costs.

    ``must_be_readable`` overrides the budget, and it is not an optimisation knob -- it is
    a correctness requirement. A staged shard has **no permanent remote descriptor**: the
    staging buffer is a transfer-local destination, so `client_read` refuses a source
    whose `desc_bytes_list` entry is None ("Remote descriptor must be contiguous for
    client read"). Any shard that another client may *read from* -- a peer-diffusion
    designate, or any client under direct supply -- therefore has to keep its in-place
    registration however many descriptors that costs. Staging is only available to a
    shard that is purely a destination.
    """
    if layout is None:
        return False
    if must_be_readable:
        return True
    return layout.row_count <= STRIDED_ROW_BUDGET
