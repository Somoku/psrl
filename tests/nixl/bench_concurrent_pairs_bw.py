"""Concurrent N-pair cross-node NIXL READ bandwidth sweep (W12).

Each pair is identified by a GPU index, so all N target agents and N initiator
agents can run simultaneously without NIXL name collisions.

    # target node (runs N agents, one per GPU)
    source scripts/nixl_env.sh
    python tests/nixl/bench_concurrent_pairs_bw.py \
        --role target --ip <target-bond1-ip> --pairs 8

    # initiator node (runs N agents, one per GPU, synchronized start)
    source scripts/nixl_env.sh
    python tests/nixl/bench_concurrent_pairs_bw.py \
        --role initiator --ip <target-bond1-ip> --pairs 8

The initiator launches all N agents as threads, waits until every agent has
completed its handshake, then fires them all at the same wall-clock time using
a threading.Barrier.  This gives true simultaneous READ starts so the window
overlap is tight and aggregate throughput is comparable across pair counts.

Prints one RESULT line per pair, then a SUMMARY line with the aggregate.
"""

from __future__ import annotations

import argparse
import json
import threading
import time

import numpy as np
import torch
from nixl._api import nixl_agent, nixl_agent_config

GIB = 1024 ** 3
BASE_PORT = 31950


def agent_name(role: str, idx: int) -> str:
    return f"{role}-{idx}"


def target_worker(idx: int, args) -> None:
    torch.cuda.set_device(idx)
    port = BASE_PORT + idx
    name = agent_name("tgt", idx)
    peer = agent_name("ini", idx)
    agent = nixl_agent(name, nixl_agent_config(True, True, port))
    elems = int(args.total_gib * GIB) // 2
    buf = torch.zeros(elems, dtype=torch.bfloat16, device=f"cuda:{idx}")
    reg = agent.register_memory(buf)
    assert reg, f"target {idx}: register_memory failed"
    descs = reg.trim()
    desc_blob = agent.get_serialized_descs(descs)
    print(f"[tgt-{idx}] agent up, waiting for initiator", flush=True)
    deadline = time.time() + 300
    while not agent.check_remote_metadata(peer):
        if time.time() > deadline:
            print(f"[tgt-{idx}] timeout waiting for {peer}", flush=True)
            return
        time.sleep(0.1)
    agent.send_notif(peer, desc_blob)
    # READ is one-sided; stay alive until hold_s
    time.sleep(args.hold_s)
    print(f"[tgt-{idx}] done", flush=True)


def initiator_worker(
    idx: int,
    args,
    barrier: threading.Barrier,
    results: list,
) -> None:
    torch.cuda.set_device(idx)
    port = BASE_PORT + idx
    name = agent_name("ini", idx)
    peer = agent_name("tgt", idx)
    agent = nixl_agent(name, nixl_agent_config(True, True, 0))
    elems = int(args.total_gib * GIB) // 2
    local = torch.ones(elems, dtype=torch.bfloat16, device=f"cuda:{idx}")
    reg = agent.register_memory(local)
    assert reg, f"initiator {idx}: register_memory failed"

    agent.fetch_remote_metadata(peer, args.ip, port)
    agent.send_local_metadata(args.ip, port)

    # Wait for target to send its descriptor blob
    deadline = time.time() + 120
    notifs = {}
    while not notifs and time.time() < deadline:
        notifs = agent.get_new_notifs()
        if not notifs:
            time.sleep(0.1)
    assert notifs, f"initiator {idx}: no descriptor notification within 120s"

    remote_descs = agent.deserialize_descs(notifs[peer][0])
    while not agent.check_remote_metadata(peer):
        time.sleep(0.05)
    local_descs = reg.trim()
    payload_bytes = elems * 2  # bfloat16

    print(f"[ini-{idx}] handshake done, waiting at barrier", flush=True)
    barrier.wait()  # all initiators start at the same moment

    best_gbps = 0.0
    best_post_ms = float("inf")
    for rep in range(args.reps):
        handle = agent.initialize_xfer("READ", local_descs, remote_descs, peer, b"bw")
        assert handle, f"initiator {idx}: initialize_xfer failed"
        t0 = time.perf_counter()
        state = agent.transfer(handle)
        post_ms = (time.perf_counter() - t0) * 1e3
        assert state != "ERR", f"initiator {idx}: transfer ERR"
        while True:
            state = agent.check_xfer_state(handle)
            assert state != "ERR"
            if state == "DONE":
                break
        elapsed = time.perf_counter() - t0
        agent.release_xfer_handle(handle)
        gbps = payload_bytes / elapsed / 1e9
        if gbps > best_gbps:
            best_gbps = gbps
            best_post_ms = post_ms
        print(
            f"[ini-{idx}] rep {rep}: {elapsed:.4f}s -> {gbps:.2f} GB/s "
            f"(post {post_ms:.3f}ms)",
            flush=True,
        )

    row = {"idx": idx, "GBps": round(best_gbps, 2), "post_ms": round(best_post_ms, 3)}
    results[idx] = row
    print(f"RESULT pair={idx} " + json.dumps(row), flush=True)


def run_targets(args) -> None:
    threads = [
        threading.Thread(target=target_worker, args=(i, args), daemon=True)
        for i in range(args.pairs)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def run_initiators(args) -> None:
    barrier = threading.Barrier(args.pairs)
    results: list = [None] * args.pairs
    threads = [
        threading.Thread(
            target=initiator_worker, args=(i, args, barrier, results), daemon=False
        )
        for i in range(args.pairs)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    valid = [r for r in results if r is not None]
    if not valid:
        print("SUMMARY no results", flush=True)
        return

    per_pair = [r["GBps"] for r in valid]
    agg = sum(per_pair)
    print(
        f"SUMMARY pairs={len(valid)} "
        f"per_pair_min={min(per_pair):.2f} per_pair_max={max(per_pair):.2f} "
        f"aggregate={agg:.1f} GB/s",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["target", "initiator"], required=True)
    ap.add_argument("--ip", required=True, help="target node's metadata IP")
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--total-gib", type=float, default=4.0)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--hold-s", type=float, default=300.0)
    args = ap.parse_args()

    if args.role == "target":
        run_targets(args)
    else:
        run_initiators(args)


if __name__ == "__main__":
    main()
