"""Measured cross-node bandwidth vs the cost model's assumption (items 1-4, 10).

`test_cost_model.py` derives `BW_NIC = 170 GB/s` as
`8 NICs x 200 Gbps x 0.85 / 8`, i.e. the **aggregate** of all eight NICs on the node.
Every bandwidth-judged verification item (trainer egress == M, per-node ingress == M_v,
AGST sublinearity, chunk-size economics) is evaluated against that constant, so what one
NIXL agent can actually reach decides whether those judgements mean anything.

Measured with `bench_cross_node_bw.py` between two nodes of this cluster
(29.162.247.148 <-> 28.59.83.232), NIXL READ over the cluster's RoCE config:

| tensors | payload | steady-state |
|---|---|---|
| 64  | 4 GiB | 24.50 GB/s |
| 512 | 8 GiB | 24.51 GB/s |

The number does not move with descriptor count or payload size, which is what makes it a
ceiling rather than a sample: **24.5 GB/s = 196.1 Gbps = 98% of one 200 Gbps NIC**. One
NIXL agent drives one NIC, not the eight the cost model aggregates -- a 6.9x gap.

That is not a defect. It is the reason the design's own planner matters: a single
connection cannot saturate the node, so reaching anything near `BW_NIC` requires the
multi-source, multi-path structure of §3.5 (W concurrent readers, peer diffusion,
per-node ingress accounting). Which in turn means **`BW_NIC = 170 GB/s` is a
node-aggregate budget and must never be used to predict one transfer's duration.**

These tests pin the relationship so that a future change to either the constant or the
transport has to confront it. They do not re-run the benchmark: it needs two nodes and
a coordinated target process, so the measured values are recorded here as constants and
`bench_cross_node_bw.py` is the reproducer.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.cpu_test

# --- measured, 2026-10-08, 29.162.247.148 <-> 28.59.83.232 -------------------
#: Steady-state NIXL READ bandwidth for a single agent pair, GB/s. Identical at
#: (64 tensors, 4 GiB) and (512 tensors, 8 GiB).
MEASURED_SINGLE_AGENT_GBS = 24.5
#: First repetition is slower (connection warm-up); reported for honesty, not asserted.
MEASURED_FIRST_REP_GBS = 15.9
NIC_LINE_GBPS = 200.0
NIC_COUNT = 8
#: What test_cost_model.py assumes, as a node aggregate.
COST_MODEL_BW_NIC_GBS = 170.0


def test_single_agent_saturates_exactly_one_nic():
    """The measured figure must be ~one NIC's line rate, not a fraction of it.

    If this drops materially the transport has regressed to a slower path -- the TCP
    fallback that `scripts/nixl_env.sh` exists to prevent lands around 0.24 GB/s, two
    orders of magnitude below.
    """
    one_nic_gbs = NIC_LINE_GBPS / 8
    efficiency = MEASURED_SINGLE_AGENT_GBS / one_nic_gbs
    assert 0.90 <= efficiency <= 1.0, (
        f"measured {MEASURED_SINGLE_AGENT_GBS} GB/s is {efficiency:.0%} of one "
        f"{NIC_LINE_GBPS} Gbps NIC; expected 90-100%"
    )


def test_single_agent_does_not_reach_the_node_aggregate():
    """One agent must NOT be assumed to deliver BW_NIC. This is the load-bearing claim.

    Asserting the gap rather than hoping nobody conflates the two: if a future change
    makes one agent reach 170 GB/s, this test fails and the cost model's interpretation
    should be revisited, which is the correct outcome either way.
    """
    assert MEASURED_SINGLE_AGENT_GBS < COST_MODEL_BW_NIC_GBS / 2, (
        f"one agent measured {MEASURED_SINGLE_AGENT_GBS} GB/s against a "
        f"{COST_MODEL_BW_NIC_GBS} GB/s node budget; the two are no longer distinct and "
        "the cost model's single-transfer predictions need rechecking"
    )


def test_cost_model_constant_is_the_eight_nic_aggregate():
    """Pin where 170 comes from, so the 6.9x gap is attributable rather than mysterious."""
    aggregate = NIC_COUNT * NIC_LINE_GBPS * 0.85 / 8
    assert abs(aggregate - COST_MODEL_BW_NIC_GBS) < 1.0, (
        f"the documented derivation gives {aggregate} GB/s, not {COST_MODEL_BW_NIC_GBS}"
    )


def test_reaching_the_aggregate_needs_multiple_concurrent_readers():
    """How many single-NIC streams the node budget implies -- the design's W.

    §3.5.2.1 sizes W as `min(NIC_count, nodes_per_instance)`. This checks the bandwidth
    arithmetic agrees: the aggregate is only reachable with roughly NIC_COUNT concurrent
    streams, which is precisely why the planner has to spread sources.
    """
    streams_needed = COST_MODEL_BW_NIC_GBS / MEASURED_SINGLE_AGENT_GBS
    assert 6.0 <= streams_needed <= NIC_COUNT + 1, (
        f"node budget implies {streams_needed:.1f} concurrent single-NIC streams, which "
        f"is inconsistent with {NIC_COUNT} NICs"
    )


def test_warm_up_penalty_is_bounded():
    """The first transfer is slower; the design's chunk pipeline amortises it.

    Worth pinning because a *growing* warm-up cost would undermine the per-round
    accounting in §3.6.5 -- the model treats per-chunk cost as constant across rounds.
    """
    penalty = 1.0 - MEASURED_FIRST_REP_GBS / MEASURED_SINGLE_AGENT_GBS
    assert penalty < 0.5, f"first-rep penalty {penalty:.0%} is larger than expected"


def test_measured_bandwidth_is_far_above_the_tcp_fallback():
    """Guards the specific regression `scripts/nixl_env.sh` was written to prevent.

    A misconfigured UCX_NET_DEVICES completes every functional test while delivering
    ~0.24 GB/s. The gap to RDMA is ~100x, so any accidental fallback is unmistakable
    in a bandwidth number -- provided someone is looking at one.
    """
    tcp_fallback_gbs = 4.0 * 1024**3 / 17.3 / 1e9  # 4 GiB in 17.3 s, measured
    assert MEASURED_SINGLE_AGENT_GBS > 20 * tcp_fallback_gbs, (
        f"RDMA {MEASURED_SINGLE_AGENT_GBS} GB/s vs TCP {tcp_fallback_gbs:.2f} GB/s: "
        "the margin that makes a silent fallback detectable has narrowed"
    )
