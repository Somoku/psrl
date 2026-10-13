#!/usr/bin/env python
"""Read a gen client's weights at cell vs whole-tensor granularity, real layout (D1 / S5).

`_make_peer_diffusion_plan` routes a follower to a same-node designate *per finest-mesh
cell*. Under train TP=4 / gen TP=2 the finest mesh on a row-parallel key is {1: 4}, so
the gen tensor `[out, in/2]` is cut into two `[out, in/4]` column cells -- each a strided
view with `out` rows -- and a peer read of it pays one descriptor per row. But follower
and designate hold the *same* gen TP rank, so the cells they exchange tile the same
contiguous parameter on both sides: one descriptor moves the same bytes.

Two processes (target serves, initiator reads), so the transport is the one production
uses between two agents -- intra-node or cross-node depending on where they run:

* ``cells``: one Nx3 descriptor per row of every strided cell and one per contiguous
  cell -- what `client_read` posts today for a peer read;
* ``whole``: one descriptor per key, covering the gen-local tensor;
* ``flat``:  one descriptor for the whole client (upper bound for the link).

Correctness is byte equality against a position-dependent pattern.

    source scripts/nixl_env.sh
    # same node uses two GPUs, cross node runs the target on the other host
    python tests/nixl/bench_peer_whole_tensor.py --role target    --ip <target-ip> --gpu 0 &
    python tests/nixl/bench_peer_whole_tensor.py --role initiator --ip <target-ip> --gpu 1
"""

from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np
import torch
from nixl._api import nixl_agent, nixl_agent_config

from tests.nixl.bench_direct_supply_geometry import LLAMA_70B, QWEN3_8B, model_keys

MODES = ("cells", "whole", "flat")


def gen_local_layout(model, gen_tp: int, finest_tp: int):
    """Per key: (local shape, cells as (start_elem, rows, row_elems, stride_elems)).

    Mirrors `get_local_sharded_tensors` on the gen-local tensor after the refactor to
    the finest mesh: a dim-0 split gives contiguous cells, a dim-1 split strided ones.
    """
    out = {}
    cells_per_local = finest_tp // gen_tp
    for key, (shape, dim) in model_keys(model).items():
        if dim is None:
            n = int(np.prod(shape))
            out[key] = (shape, [(0, 1, n, n)])
            continue
        local = list(shape)
        local[dim] //= gen_tp
        rows, cols = local
        if dim == 0:
            per = rows // cells_per_local
            cells = [(c * per * cols, 1, per * cols, per * cols) for c in range(cells_per_local)]
        else:
            per = cols // cells_per_local
            cells = [(c * per, rows, per, cols) for c in range(cells_per_local)]
        out[key] = (tuple(local), cells)
    return out


#: Elements per step when writing or checking the pattern. A single `arange` over an 8 GB
#: client is 4e9 int64 elements, a 65 GB transient that OOMs a shared GPU.
_PATTERN_STEP = 1 << 28


def _pattern(start: int, stop: int, device) -> torch.Tensor:
    return (torch.arange(start, stop, device=device, dtype=torch.int64) % 32749).to(torch.int16)


def fill_pattern(buf: torch.Tensor) -> None:
    view = buf.view(torch.int16)
    for i in range(0, view.numel(), _PATTERN_STEP):
        j = min(i + _PATTERN_STEP, view.numel())
        view[i:j].copy_(_pattern(i, j, buf.device))


def matches_pattern(buf: torch.Tensor) -> bool:
    view = buf.view(torch.int16)
    for i in range(0, view.numel(), _PATTERN_STEP):
        j = min(i + _PATTERN_STEP, view.numel())
        if not torch.equal(view[i:j], _pattern(i, j, buf.device)):
            return False
    return True


def build_descs(base_ptr: int, total: int, offsets: dict, layout: dict, gpu: int, mode: str) -> np.ndarray:
    if mode == "flat":
        return np.array([(base_ptr, total * 2, gpu)], dtype=np.uint64)
    rows = []
    for key, (shape, cells) in layout.items():
        off = offsets[key]
        if mode == "whole":
            rows.append((base_ptr + off * 2, int(np.prod(shape)) * 2, gpu))
            continue
        for start, nrows, row_elems, stride in cells:
            base = base_ptr + (off + start) * 2
            for r in range(nrows):
                rows.append((base + r * stride * 2, row_elems * 2, gpu))
    return np.array(rows, dtype=np.uint64)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["target", "initiator"], required=True)
    ap.add_argument("--ip", required=True, help="target's metadata IP")
    ap.add_argument("--port", type=int, default=29991)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model", choices=["qwen3-8b", "70b"], default="qwen3-8b")
    ap.add_argument("--layers", type=int, default=None, help="truncate layers to fit memory")
    ap.add_argument("--gen-tp", type=int, default=2)
    ap.add_argument("--finest-tp", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--hold-s", type=int, default=900)
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument(
        "--start-at", type=float, default=0.0, help="unix time to start the timed loop, so parallel initiators overlap"
    )
    ap.add_argument(
        "--barrier-dir", default=None, help="shared directory where initiators wait until --barrier-n are ready"
    )
    ap.add_argument("--barrier-n", type=int, default=1)
    ap.add_argument("--port-file", default=None, help="target writes its listen port here")
    return ap.parse_args()


def main():
    args = parse()
    model = QWEN3_8B if args.model == "qwen3-8b" else LLAMA_70B
    if args.layers:
        model = type(model)(**{**model.__dict__, "layers": args.layers})
    layout = gen_local_layout(model, args.gen_tp, args.finest_tp)
    offsets, total = {}, 0
    for key, (shape, _) in layout.items():
        offsets[key] = total
        total += int(np.prod(shape))

    torch.cuda.set_device(args.gpu)
    buf = torch.zeros(total, dtype=torch.bfloat16, device=f"cuda:{args.gpu}")
    if args.role == "target":
        fill_pattern(buf)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    if args.role == "target":
        # Busy nodes hold random ports, so probe upward and publish the bound port on the shared filesystem.
        for port in range(args.port, args.port + 200):
            try:
                agent = nixl_agent(args.role, nixl_agent_config(True, True, port))
                break
            except RuntimeError:
                continue
        args.port = port
        if args.port_file:
            with open(args.port_file + ".tmp", "w") as f:
                f.write(str(port))
            import os

            os.replace(args.port_file + ".tmp", args.port_file)
    else:
        if args.port_file:
            import os

            while not os.path.exists(args.port_file):
                time.sleep(0.2)
            args.port = int(open(args.port_file).read())
        agent = nixl_agent(args.role, nixl_agent_config(True, True, 0))
    print(
        f"{args.role}: agent up in {time.perf_counter() - t0:.0f}s, "
        f"{model.name} {len(layout)} keys {total * 2 / 1e9:.2f} GB",
        flush=True,
    )
    reg = agent.register_memory([buf])

    if args.role == "target":
        while not agent.check_remote_metadata("initiator"):
            time.sleep(0.1)
        blobs = {
            m: agent.get_serialized_descs(
                agent.get_xfer_descs(build_descs(buf.data_ptr(), total, offsets, layout, args.gpu, m), "VRAM")
            )
            for m in MODES
        }
        agent.send_notif("initiator", pickle.dumps(blobs))
        print("target: descriptors published", flush=True)
        time.sleep(args.hold_s)
        agent.deregister_memory(reg)
        return

    agent.fetch_remote_metadata("target", args.ip, args.port)
    agent.send_local_metadata(args.ip, args.port)
    notifs = {}
    deadline = time.time() + 1200
    while "target" not in notifs and time.time() < deadline:
        notifs = agent.get_new_notifs()
        time.sleep(0.2)
    blobs = pickle.loads(notifs["target"][0])
    while not agent.check_remote_metadata("target"):
        time.sleep(0.1)

    if args.barrier_dir:
        import os

        os.makedirs(args.barrier_dir, exist_ok=True)
        open(os.path.join(args.barrier_dir, f"ready_{os.getpid()}_{args.gpu}"), "w").close()
        while len(os.listdir(args.barrier_dir)) < args.barrier_n:
            time.sleep(0.05)
    if args.start_at:
        time.sleep(max(0.0, args.start_at - time.time()))
    results = {}
    for mode in args.modes:
        remote = agent.deserialize_descs(blobs[mode])
        ld = build_descs(buf.data_ptr(), total, offsets, layout, args.gpu, mode)
        local = agent.get_xfer_descs(ld, "VRAM")
        best_total = best_post = None
        window_start = time.time()
        for rep in range(args.reps):
            last = rep == args.reps - 1
            # Zero and verify only on the last rep: between-rep work would let parallel
            # initiators take turns, and the aggregate would measure the gaps.
            if last:
                buf.zero_()
                torch.cuda.synchronize()
            h = agent.initialize_xfer("READ", local, remote, "target", mode.encode())
            start = time.perf_counter()
            st = agent.transfer(h)
            post = time.perf_counter() - start
            assert st != "ERR"
            while (st := agent.check_xfer_state(h)) != "DONE":
                assert st != "ERR"
            elapsed = time.perf_counter() - start
            agent.release_xfer_handle(h)
            if last:
                window_end = time.time()
                torch.cuda.synchronize()
                assert matches_pattern(buf), f"{mode}: bytes differ"
            best_total = elapsed if best_total is None else min(best_total, elapsed)
            best_post = post if best_post is None else min(best_post, post)
        results[mode] = {
            "descriptors": int(len(ld)),
            "post_ms": round(best_post * 1e3, 3),
            "total_ms": round(best_total * 1e3, 3),
            "GB_per_s": round(total * 2 / best_total / 1e9, 1),
            "bytes_equal": True,
            # Wall-clock window of all reps, so parallel initiators can be aggregated as
            # total bytes / (latest end - earliest start) rather than by summing bests.
            "window": [window_start, window_end],
            "bytes_moved": total * 2 * args.reps,
        }
        print(mode, results[mode], flush=True)
    print(
        "RESULT "
        + json.dumps(
            {
                "model": model.name,
                "GB": round(total * 2 / 1e9, 2),
                "gen_tp": args.gen_tp,
                "finest_tp": args.finest_tp,
                **results,
            }
        ),
        flush=True,
    )
    agent.deregister_memory(reg)


if __name__ == "__main__":
    main()
