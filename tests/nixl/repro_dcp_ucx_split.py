"""Direct supply's real layout: checkpointing train ranks serve, separate reader processes read.

`repro_dcp_ucx.py` puts the reader thread inside the process that runs the collective. In
production the reader is a vLLM process that never checkpoints; the train rank only serves
(passive target, progress thread on) while it runs `dist_checkpointing.save`. This script
runs the two sides as separate torchrun jobs with separate process groups, rendezvousing
through a shared directory, so the train side's collective traffic is the only variable.

    # train job: serves its buffer, runs --stage on its own process group
    torchrun --nproc_per_node=4 repro_dcp_ucx_split.py --role train --dir D --stage dcp
    # reader job (same node on other GPUs, or another node): reads train rank r, verifies
    torchrun --nproc_per_node=4 repro_dcp_ucx_split.py --role read  --dir D

Reader r reads train rank r. Every read is verified byte for byte.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import threading
import time

import torch
import torch.distributed as dist

from nixl._api import nixl_agent, nixl_agent_config

sys.path.insert(0, os.path.dirname(__file__))
from repro_dcp_ucx import Reader, pattern, stage_dcp, stage_objgather  # noqa: E402


def wait_for(path: str, timeout: float) -> None:
    deadline = time.time() + timeout
    while not os.path.exists(path):
        if time.time() > deadline:
            raise TimeoutError(path)
        time.sleep(0.2)


def train(args, rank: int) -> int:
    name = f"split-train-{rank}"
    agent = nixl_agent(name, nixl_agent_config(enable_prog_thread=True))
    nbytes = args.buf_mb << 20
    source = pattern(rank, nbytes)
    agent.register_memory(source)
    blob = (name, agent.get_agent_metadata(), agent.get_serialized_descs(agent.get_xfer_descs(source, mem_type="VRAM")))
    with open(os.path.join(args.dir, f"train_{rank}.tmp"), "wb") as f:
        pickle.dump(blob, f)
    os.replace(os.path.join(args.dir, f"train_{rank}.tmp"), os.path.join(args.dir, f"train_{rank}.pkl"))
    wait_for(os.path.join(args.dir, f"reader_ready_{rank}"), args.wait_s)
    dist.barrier()
    time.sleep(2.0)  # readers settle before the stage starts
    open(os.path.join(args.dir, f"stage_start_{rank}"), "w").close()
    t0 = time.time()
    err = ""
    iters = 0
    try:
        if args.stage == "objgather":
            iters = stage_objgather(args.seconds, args.payload_mb)
        elif args.stage in ("dcp", "dcp_async"):
            iters = stage_dcp(args.seconds, args.ckpt_dir, args.stage == "dcp_async", args.ckpt_mb, args.persistent)
        else:
            time.sleep(args.seconds)
    except Exception as exc:  # noqa: BLE001
        err = repr(exc)
    print(f"[train {rank}] stage={args.stage} iters={iters} stage_s={time.time() - t0:.1f} error={err!r}", flush=True)
    dist.barrier()
    open(os.path.join(args.dir, f"stage_done_{rank}"), "w").close()
    # Keep serving until the reader has finished its post-stage reads.
    wait_for(os.path.join(args.dir, f"reader_done_{rank}"), args.wait_s)
    return 0 if not err else 1


def read(args, rank: int) -> int:
    agent = nixl_agent(f"split-read-{rank}", nixl_agent_config(enable_prog_thread=True))
    nbytes = args.buf_mb << 20
    recv = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    agent.register_memory(recv)
    wait_for(os.path.join(args.dir, f"train_{rank}.pkl"), args.wait_s)
    with open(os.path.join(args.dir, f"train_{rank}.pkl"), "rb") as f:
        peer_name, peer_meta, peer_descs = pickle.load(f)
    agent.add_remote_agent(peer_meta)
    remote = agent.deserialize_descs(peer_descs)
    reader = Reader(agent, peer_name, recv, remote, pattern(rank, nbytes))
    thread = threading.Thread(target=reader.run, args=(args.read_timeout_s,), daemon=True)
    thread.start()
    time.sleep(1.0)
    open(os.path.join(args.dir, f"reader_ready_{rank}"), "w").close()
    wait_for(os.path.join(args.dir, f"stage_start_{rank}"), args.wait_s)
    before = reader.reads
    start = time.time()
    wait_for(os.path.join(args.dir, f"stage_done_{rank}"), args.wait_s + args.seconds * 4)
    during = reader.reads - before
    stage_s = time.time() - start
    time.sleep(3.0)
    after = reader.reads - before - during
    reader.stop.set()
    thread.join(timeout=args.read_timeout_s + 5)
    open(os.path.join(args.dir, f"reader_done_{rank}"), "w").close()
    gbps = during * nbytes / max(stage_s, 1e-9) / 1e9
    print(
        f"[read {rank}] reads_during={during} ({gbps:.1f} GB/s) reads_after={after} errors={reader.errors} "
        f"timeouts={reader.timeouts} mismatches={reader.mismatches} max_read_s={reader.max_read_s:.3f} "
        f"first_error={reader.first_error!r}",
        flush=True,
    )
    for line in reader.mismatch_log:
        print(f"[read {rank}] MISMATCH {line} (stage began t={start - reader.t0:.2f}s)", flush=True)
    ok = reader.errors == 0 and reader.timeouts == 0 and reader.mismatches == 0 and after > 0
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["train", "read"], required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--stage", choices=["idle", "objgather", "dcp", "dcp_async"], default="idle")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--buf-mb", type=int, default=256)
    ap.add_argument("--payload-mb", type=int, default=64)
    ap.add_argument("--ckpt-dir", default="/tmp/repro_dcp_split_ckpt")
    ap.add_argument("--ckpt-mb", type=int, default=512)
    ap.add_argument("--read-timeout-s", type=float, default=60.0)
    ap.add_argument("--wait-s", type=float, default=900.0)
    ap.add_argument("--persistent", action="store_true")
    args = ap.parse_args()

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    os.makedirs(args.dir, exist_ok=True)
    code = train(args, rank) if args.role == "train" else read(args, rank)
    flags = torch.tensor([code], device="cuda")
    dist.all_reduce(flags)
    if rank == 0:
        print(f"RESULT role={args.role} stage={args.stage} {'PASS' if flags.item() == 0 else 'FAIL'} ({flags.item()} rank(s) failed)", flush=True)
    dist.destroy_process_group()
    return int(flags.item() != 0)


if __name__ == "__main__":
    sys.exit(main())
