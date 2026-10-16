"""One-pair cross-node NIXL READ bandwidth as a function of segment size and segments per post.

W12 (FINDINGS §1c) attributes the 235B shortfall to per-expert segments being too small to
fill a link. That claim fixes what the remedy is, so it is measured here instead of argued:
a fixed byte budget is read as N equal segments in one prepped handle, for N spanning
1 segment of the whole buffer down to segments of a few KB, and the same with several
handles in flight at once.

    source scripts/nixl_env.sh
    # target node (serves GPU --gpu)
    python tests/nixl/bench_segment_size.py --role target --ip <target-ip> --gpu 0
    # initiator node
    python tests/nixl/bench_segment_size.py --role initiator --ip <target-ip> --gpu 0

The target's agent listens on --port; the initiator fetches its metadata, then reads.
Every read is verified against the target's pattern.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from nixl._api import nixl_agent, nixl_agent_config

MIB = 1 << 20


def pattern(nbytes: int, device: str) -> torch.Tensor:
    idx = torch.arange(nbytes, dtype=torch.int64, device=device)
    return ((idx * 2654435761) >> 7 & 0xFF).to(torch.uint8)


def rows(base: int, seg: int, count: int, stride: int, dev: int) -> np.ndarray:
    starts = base + np.arange(count, dtype=np.int64) * stride
    return np.stack([starts, np.full(count, seg), np.full(count, dev)], axis=1).astype(np.uint64)


def target(args) -> None:
    torch.cuda.set_device(args.gpu)
    agent = nixl_agent("seg-target", nixl_agent_config(True, True, args.port))
    buf = pattern(args.buf_mb * MIB, "cuda")
    agent.register_memory(buf)
    print(f"agent up base={buf.data_ptr()} bytes={buf.numel()}", flush=True)
    # The initiator pushes its metadata; only then can the target address it.
    while not agent.check_remote_metadata("seg-init"):
        time.sleep(0.05)
    agent.send_notif("seg-init", json.dumps({"base": buf.data_ptr(), "dev": args.gpu}).encode())
    deadline = time.time() + args.hold_s
    while time.time() < deadline:
        if any(b"done" in msgs for msgs in agent.get_new_notifs().values()):
            print("initiator done", flush=True)
            return
        time.sleep(0.05)


def initiator(args) -> None:
    torch.cuda.set_device(args.gpu)
    agent = nixl_agent("seg-init", nixl_agent_config(True, True, 0))
    nbytes = args.buf_mb * MIB
    local = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    agent.register_memory(local)
    agent.fetch_remote_metadata("seg-target", args.ip, args.port)
    agent.send_local_metadata(args.ip, args.port)
    info = None
    deadline = time.time() + 600
    while info is None and time.time() < deadline:
        msgs = agent.get_new_notifs().get("seg-target", [])
        if msgs:
            info = json.loads(msgs[0].decode())
        time.sleep(0.05)
    while not agent.check_remote_metadata("seg-target"):
        time.sleep(0.05)
    rbase, rdev = info["base"], info["dev"]
    expected = pattern(nbytes, "cuda")
    ldev = args.gpu

    results = []
    for seg_kb in args.seg_kb:
        seg = seg_kb * 1024
        for inflight in args.inflight:
            # Split the budget into `inflight` equal handles of `count` segments each.
            per_handle = nbytes // inflight
            if seg > per_handle:
                continue  # a segment larger than one handle's share would run past the buffer
            count = per_handle // seg
            moved = count * seg * inflight
            handles = []
            for h in range(inflight):
                off = h * per_handle
                lrows = rows(local.data_ptr() + off, seg, count, seg, ldev)
                rrows = rows(rbase + off, seg, count, seg, rdev)
                ld = agent.prep_xfer_dlist("NIXL_INIT_AGENT", lrows, mem_type="VRAM")
                rd = agent.prep_xfer_dlist("seg-target", rrows, mem_type="VRAM")
                idx = np.arange(count, dtype=np.int32)
                handles.append(agent.make_prepped_xfer("READ", ld, idx, rd, idx, b"seg"))
            best = float("inf")
            for _rep in range(args.reps):
                local.zero_()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for h in handles:
                    assert agent.transfer(h) != "ERR"
                pending = list(handles)
                while pending:
                    pending = [h for h in pending if agent.check_xfer_state(h) != "DONE"]
                elapsed = time.perf_counter() - t0
                best = min(best, elapsed)
            torch.cuda.synchronize()
            ok = all(
                torch.equal(local[h * per_handle : h * per_handle + count * seg], expected[h * per_handle : h * per_handle + count * seg])
                for h in range(inflight)
            )
            gbps = moved / best / 1e9
            row = {"seg_kb": seg_kb, "inflight": inflight, "descs": count * inflight, "GBps": round(gbps, 1), "ok": ok}
            results.append(row)
            print("ROW " + json.dumps(row), flush=True)
            for h in handles:
                agent.release_xfer_handle(h)
    agent.send_notif("seg-target", b"done")
    print("RESULT " + json.dumps(results), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["target", "initiator"], required=True)
    ap.add_argument("--ip", required=True)
    ap.add_argument("--port", type=int, default=31555)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--buf-mb", type=int, default=2048)
    ap.add_argument("--seg-kb", type=int, nargs="+", default=[2097152, 65536, 4096, 1024, 256, 64, 16, 4])
    ap.add_argument("--inflight", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--hold-s", type=int, default=900)
    args = ap.parse_args()
    target(args) if args.role == "target" else initiator(args)


if __name__ == "__main__":
    main()
