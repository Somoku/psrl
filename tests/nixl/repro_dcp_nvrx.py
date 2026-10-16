"""DCP async save with nvrx strategy vs NIXL direct supply (same-process reader thread).

Reproduces the stack closest to production: one process runs both sides (trainer
serving via NIXL progress thread + reader thread pulling via NIXL read), while the
trainer's main thread runs `dist_checkpointing.async_save(..., async_strategy="nvrx")`.

Usage:
    torchrun --nproc_per_node=2 repro_dcp_nvrx.py --stage nvrx --seconds 30
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import torch
import torch.distributed as dist
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.strategies import torch as mcore_torch
from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelSaveStrategyWrapper
from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncCallsQueue

from nixl._api import nixl_agent, nixl_agent_config

sys.path.insert(0, os.path.dirname(__file__))
from repro_dcp_ucx import Reader, pattern  # noqa: E402


def stage_dcp_nvrx(seconds: float, ckpt_dir: str, ckpt_mb: int) -> int:
    """Run DCP async save with nvrx strategy in a loop.

    nvrx's AsyncCallsQueue(persistent=True) starts a worker via multiprocessing.spawn.
    spawn re-imports the calling script as __main__, so all setup here must be inside
    a function (never at module level) to avoid recursion and double NIXL init.
    """
    rank = dist.get_rank()
    world = dist.get_world_size()
    os.makedirs(ckpt_dir, exist_ok=True)
    data = torch.arange(ckpt_mb << 17, dtype=torch.float32, device="cuda")
    # rank_offsets: (axis, offset_along_axis, num_shards_along_axis)
    sharded = ShardedTensor.from_rank_offsets(
        "layer.weight",
        data,
        (0, rank, world),
    )
    strategy = FullyParallelSaveStrategyWrapper(
        mcore_torch.TorchDistSaveShardedStrategy("torch_dist", 1),
        dist.GroupMember.WORLD,
        True,
    )
    # Import here so the module-level import does NOT trigger on spawn re-import
    from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncCallsQueue as _Q
    async_queue = _Q(persistent=True)
    async_queue.warmup_persistent_caller(rank)

    start, iters = time.time(), 0
    while time.time() - start < seconds:
        step_dir = os.path.join(ckpt_dir, f"step_{iters}")
        async_queue.maybe_finalize_async_calls(blocking=True)
        dist.barrier()
        async_request = strategy.async_save({"model": sharded}, step_dir, async_strategy="nvrx")
        async_queue.schedule_async_request(async_request)
        dist.barrier()
        iters += 1
    async_queue.maybe_finalize_async_calls(blocking=True)
    async_queue.close()
    return iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["idle", "nvrx"], default="idle")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--buf-mb", type=int, default=256)
    ap.add_argument("--ckpt-dir", default="/tmp/repro_dcp_nvrx_ckpt")
    ap.add_argument("--ckpt-mb", type=int, default=512)
    ap.add_argument("--read-timeout-s", type=float, default=60.0)
    args = ap.parse_args()

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()

    # Each rank serves its buffer via NIXL
    name = f"nvrx-train-{rank}"
    agent = nixl_agent(name, nixl_agent_config(enable_prog_thread=True))
    nbytes = args.buf_mb << 20
    source = pattern(rank, nbytes)
    agent.register_memory(source)
    meta, descs = agent.get_agent_metadata(), agent.get_serialized_descs(
        agent.get_xfer_descs(source, mem_type="VRAM")
    )

    # Gather peer metadata
    obj = [{"name": name, "meta": meta, "descs": descs}]
    peers = [None] * world
    dist.all_gather_object(peers, obj[0])
    for p in peers:
        if p["name"] != name:
            agent.add_remote_agent(p["meta"])

    # Start reader threads: rank r reads from rank (r+1) % world
    peer_idx = (rank + 1) % world
    peer = peers[peer_idx]
    recv = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    agent.register_memory(recv)
    remote = agent.deserialize_descs(peer["descs"])
    reader = Reader(agent, peer["name"], recv, remote, pattern(peer_idx, nbytes))
    thread = threading.Thread(target=reader.run, args=(args.read_timeout_s,), daemon=True)
    thread.start()

    dist.barrier()
    time.sleep(1.0)
    print(f"[rank {rank}] stage={args.stage} starting for {args.seconds}s", flush=True)

    # Run the stage
    t0 = time.time()
    err, iters = "", 0
    try:
        if args.stage == "nvrx":
            iters = stage_dcp_nvrx(args.seconds, args.ckpt_dir, args.ckpt_mb)
        else:
            time.sleep(args.seconds)
    except Exception as exc:  # noqa: BLE001
        err = repr(exc)
    stage_s = time.time() - t0

    dist.barrier()
    time.sleep(2.0)
    reader.stop.set()
    thread.join(timeout=args.read_timeout_s + 5)

    gbps = reader.reads * nbytes / max(stage_s, 1e-9) / 1e9
    print(
        f"[rank {rank}] stage={args.stage} iters={iters} stage_s={stage_s:.1f} reads={reader.reads} "
        f"({gbps:.1f} GB/s) errors={reader.errors} timeouts={reader.timeouts} "
        f"mismatches={reader.mismatches} error={err!r}",
        flush=True,
    )

    ok = reader.errors == 0 and reader.timeouts == 0 and reader.mismatches == 0 and not err
    flags = torch.tensor([0 if ok else 1], device="cuda")
    dist.all_reduce(flags)
    if rank == 0:
        result = "PASS" if flags.item() == 0 else "FAIL"
        print(f"RESULT stage={args.stage} {result} ({flags.item()} rank(s) failed)", flush=True)
    dist.destroy_process_group()
    return int(flags.item() != 0)


if __name__ == "__main__":
    sys.exit(main())
