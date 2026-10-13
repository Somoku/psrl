#!/usr/bin/env python
"""Measure cross-node NIXL READ bandwidth (verification items 1-4, 10 input).

The cost model in `test_cost_model.py` assumes `BW_NIC = 170 GB/s`, derived as
8 NICs x 200 Gbps x 0.85 / 8. That is an *aggregate* figure across all eight NICs; a
single NIXL agent reading over one connection cannot reach it. Before any item that
judges on bandwidth (trainer egress == M, per-node ingress == M_v, AGST sublinearity)
can mean anything, the achievable per-agent and aggregate numbers have to be measured
rather than assumed -- otherwise a result is being compared against a ceiling this
fabric may not offer.

`ib_write_bw` cannot be used here: its TCP setup channel binds to the RoCE netdev
address (bond2..bond9), which is on a per-node subnet and not reachable from the other
side. NIXL sidesteps that by doing metadata exchange over bond1 and the data path over
RoCE, so the measurement has to go through NIXL to happen at all.

Run as two processes:

    # on the target node
    python bench_cross_node_bw.py --mode target --ip <target-bond1-ip>
    # on the initiator node
    python bench_cross_node_bw.py --mode initiator --ip <target-bond1-ip>

Source `scripts/nixl_env.sh` on both sides first -- without the cluster's full
UCX_NET_DEVICES list the transfer silently falls back to TCP and reports roughly 1/24th
of the real figure (see that script's comment).
"""

from __future__ import annotations

import argparse
import time

import torch
from nixl._api import nixl_agent, nixl_agent_config

GIB = 1024**3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ip", required=True, help="target's bond1 IP (metadata channel)")
    p.add_argument("--port", type=int, default=29990)
    p.add_argument("--mode", choices=["target", "initiator"], default="initiator")
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument(
        "--total-gib",
        type=float,
        default=4.0,
        help="payload per repetition; must match on both sides",
    )
    p.add_argument(
        "--tensors",
        type=int,
        default=64,
        help="how many tensors the payload is split across. Few large tensors measure "
        "the link; many small ones measure per-descriptor cost. Must match both sides.",
    )
    p.add_argument("--reps", type=int, default=3, help="repetitions; the best is reported")
    p.add_argument(
        "--start-at",
        type=float,
        default=None,
        help="epoch seconds to begin the first repetition. Lets several initiators on one "
        "node start together, so their reps overlap and the sum measures the node's "
        "aggregate rather than one NIC at a time.",
    )
    p.add_argument("--serve-s", type=float, default=120.0, help="target: how long to stay up")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_default_device(f"cuda:{args.cuda}")

    # The target listens on a fixed port; the initiator takes an ephemeral one.
    listen_port = args.port if args.mode == "target" else 0
    agent = nixl_agent(args.mode, nixl_agent_config(True, True, listen_port))

    elems = int(args.total_gib * GIB) // 2 // args.tensors  # bfloat16
    fill = 0.0 if args.mode == "target" else 1.0
    tensors = [torch.full((elems,), fill, dtype=torch.bfloat16) for _ in range(args.tensors)]
    payload_bytes = sum(t.numel() * t.element_size() for t in tensors)

    reg = agent.register_memory(tensors)
    assert reg, "register_memory failed"

    if args.mode == "target":
        # `send_notif` raises NIXL_ERR_NOT_FOUND unless the peer is already known, so
        # the target must block on check_remote_metadata first -- the initiator pushes
        # its metadata via send_local_metadata below. Ordering taken from
        # test_send_recv.py, which is the reference for this handshake.
        target_descs = reg.trim()
        desc_blob = agent.get_serialized_descs(target_descs)
        print(f"target: serving {payload_bytes / GIB:.2f} GiB, waiting for peer", flush=True)
        while not agent.check_remote_metadata("initiator"):
            time.sleep(0.1)
        agent.send_notif("initiator", desc_blob)
        print("target: descriptors published, waiting for reads", flush=True)
        # A READ is one-sided: the target sees no completion, so it just stays alive
        # long enough for every repetition to finish.
        time.sleep(args.serve_s)
        print("target: done", flush=True)
        return

    # --- initiator ---
    agent.fetch_remote_metadata("target", args.ip, args.port)
    agent.send_local_metadata(args.ip, args.port)

    notifs = agent.get_new_notifs()
    waited = 0.0
    while not notifs and waited < 120:
        time.sleep(0.2)
        waited += 0.2
        notifs = agent.get_new_notifs()
    assert notifs, "no descriptor notification from target within 120s"
    remote_descs = agent.deserialize_descs(notifs["target"][0])

    while not agent.check_remote_metadata("target"):
        time.sleep(0.1)
    local_descs = reg.trim()

    best = None
    best_post_ms = None
    if args.start_at is not None:
        delay = args.start_at - time.time()
        assert delay > 0, f"--start-at is {-delay:.1f}s in the past because agent setup overran it."
        time.sleep(delay)
    for rep in range(args.reps):
        handle = agent.initialize_xfer("READ", local_descs, remote_descs, "target", b"bw")
        assert handle, "initialize_xfer failed"
        start = time.perf_counter()
        state = agent.transfer(handle)
        # Post time is reported separately from completion time, because the two answer
        # different questions. The 5-node pull spends nearly all of its peer-read time
        # inside `transfer()` itself, which is consistent with two different cost models
        # -- a fixed per-submission price, or a per-descriptor one -- and only sweeping
        # `--tensors` at fixed payload distinguishes them. If post time grows with the
        # tensor count, the cost is per descriptor and merging submissions cannot help.
        post_ms = (time.perf_counter() - start) * 1e3
        assert state != "ERR", "transfer submission failed"
        while True:
            state = agent.check_xfer_state(handle)
            assert state != "ERR", "transfer reported ERR"
            if state == "DONE":
                break
        elapsed = time.perf_counter() - start
        agent.release_xfer_handle(handle)

        gbps = payload_bytes / elapsed / 1e9
        print(
            f"  rep {rep}: start={start + time.time() - time.perf_counter():.4f} "
            f"{elapsed:.4f} s -> {gbps:.2f} GB/s "
            f"(post {post_ms:.3f} ms, {post_ms * 1e3 / args.tensors:.2f} us/desc)",
            flush=True,
        )
        best = gbps if best is None else max(best, gbps)
        best_post_ms = post_ms if best_post_ms is None else min(best_post_ms, post_ms)

    print(
        f"RESULT tensors={args.tensors} payload={payload_bytes / GIB:.2f}GiB "
        f"best={best:.2f} GB/s post_ms={best_post_ms:.3f} "
        f"post_us_per_desc={best_post_ms * 1e3 / args.tensors:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
