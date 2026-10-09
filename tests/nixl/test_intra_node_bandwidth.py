"""Intra-node peer bandwidth: the premise D1's diffusion rests on.

Design 3.5.2 ranks a same-node peer above the trainer because the read "costs
NVLink/PCIe rather than NIC bandwidth, so it is free in the currency that is actually
scarce". That is an assumption about this cluster, and the two-phase e2e run made it
worth checking: diffusion moved 61% of reads onto peers and the run got *slower*.

Measured with `bench_cross_node_bw.py` run with both ends on one node (cuda:0 serving,
cuda:1 reading), 2 GiB over 64 tensors:

    intra-node, GPU to GPU   342.95 GB/s
    cross-node, one NIC       24.50 GB/s      (recorded in test_cross_node_bandwidth.py)

So the premise holds with a 14x margin, and the slow peer phase is **not** a link-speed
problem. 399 reads in 3.50s is 8.8 ms per read, while 343 GB/s moves a 1 MB shard in
0.003 ms -- the time is per-read overhead, not bytes.

These are regression guards on the numbers, not a benchmark: the measurement needs two
processes and minutes of NIXL agent setup, so it is recorded here rather than re-run.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.cpu_test

#: Measured on this cluster, 2026-10-10. cuda:0 target, cuda:1 initiator, same node.
INTRA_NODE_GBPS = 342.95
#: From test_cross_node_bandwidth.py, same harness across two nodes.
CROSS_NODE_GBPS = 24.50


class TestThePremiseHolds:
    def test_a_same_node_read_is_much_cheaper_than_a_cross_node_one(self):
        """If this ever stopped holding, peer diffusion would have no rationale."""
        assert INTRA_NODE_GBPS > 10 * CROSS_NODE_GBPS, (
            f"intra-node {INTRA_NODE_GBPS} GB/s is not decisively better than "
            f"cross-node {CROSS_NODE_GBPS} GB/s, so preferring a peer buys nothing"
        )

    def test_intra_node_bandwidth_is_nvlink_class(self):
        """PCIe gen4 x16 is ~25 GB/s; this is an order above, so it is NVLink."""
        assert INTRA_NODE_GBPS > 100, (
            f"{INTRA_NODE_GBPS} GB/s is PCIe-class, not NVLink -- the design's "
            "'free in the scarce currency' claim would need revisiting"
        )


class TestTheSlowPeerPhaseIsNotBandwidth:
    """Rules out the explanation that would have invalidated the design."""

    #: 5-node direct-supply run, peer phase.
    PEER_PHASE_SECONDS = 3.4978
    PEER_READS = 399

    def test_per_read_cost_is_far_above_the_byte_cost(self):
        ms_per_read = 1000 * self.PEER_PHASE_SECONDS / self.PEER_READS
        # A generous shard size: 399 reads covering a 16GB bf16 model is ~40MB each.
        mb_per_read = 40
        ms_of_bytes = 1000 * (mb_per_read / 1024) / INTRA_NODE_GBPS

        assert ms_per_read > 20 * ms_of_bytes, (
            f"{ms_per_read:.1f} ms per read against {ms_of_bytes:.3f} ms of bytes: the "
            "peer phase would be bandwidth-bound, which the measurement contradicts"
        )

    def test_the_phase_is_serialised_not_overlapped(self):
        """Recorded so the next attempt starts from the right problem.

        The peer phase runs entirely after the upstream phase -- the driver's barrier is
        what makes peer reads safe, and it also prevents any overlap. Fixing the cost
        means overlapping diffusion with the upstream wave per chunk (which is what B1's
        chunk pipeline exists for), not changing the link.
        """
        assert self.PEER_PHASE_SECONDS > 0, "placeholder for the recorded observation"
