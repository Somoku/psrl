"""Does a torch.distributed collective break a NIXL progress thread running in the same process?

The claim being tested (engine_train_worker.py, config.py, megatron_manager.py): Megatron
`dist_checkpointing` calls `all_gather_object`, which "corrupts the UCX endpoint addresses"
the NIXL progress thread reads, so direct supply (which needs that thread) cannot save with DCP.

Each torchrun rank is a "train rank": it owns a NIXL agent with the progress thread on and a
registered GPU buffer filled with a rank-specific pattern. A separate "reader" rank-pair
relationship is used: rank r reads rank (r+1) % world over NIXL in a background thread, in a
tight loop, verifying every byte. While that runs, the main thread drives the stage selected
by --stage:

  idle      no collective (baseline: the read loop alone)
  objgather torch.distributed.all_gather_object on large pickles, repeatedly
  dcp       megatron dist_checkpointing.save of a sharded state dict (sync)
  dcp_async the same with async_sharded_save=True (forks a writer process)

Outcome per rank: reads completed, read errors, byte mismatches, collective errors.
Any process crash (e.g. a UCX assertion) shows up as a non-zero torchrun exit.

Metadata between agents goes over torch.distributed (all_gather_object) once, before the
read loop starts, so the stage under test is the only collective traffic during the loop.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist

from nixl._api import nixl_agent, nixl_agent_config


def log(msg: str) -> None:
    print(f"[rank {dist.get_rank()}] {msg}", flush=True)


def pattern(rank: int, nbytes: int) -> torch.Tensor:
    base = torch.arange(nbytes, dtype=torch.int64, device="cuda")
    return ((base * 131 + rank * 7919) % 251).to(torch.uint8)


class Reader:
    """Read `peer`'s registered buffer over NIXL in a loop and verify it."""

    def __init__(self, agent, peer_name, local_buf, remote_descs, expected):
        self.agent = agent
        self.peer_name = peer_name
        self.local_buf = local_buf
        self.expected = expected
        local_descs = agent.get_xfer_descs(local_buf, mem_type="VRAM")
        self.local_descs = local_descs
        self.remote_descs = remote_descs
        self.reads = 0
        self.errors = 0
        self.mismatches = 0
        self.timeouts = 0
        self.first_error = ""
        self.stop = threading.Event()
        self.max_read_s = 0.0
        self.mismatch_log: list[str] = []
        self.t0 = time.time()

    def run(self, read_timeout_s: float) -> None:
        # The current CUDA device is per thread and a new thread starts on device 0. A bare
        # `torch.cuda.synchronize()` here synced the wrong card on every rank but 0, so the
        # `zero_()` below could still be queued behind a busy collective when the NIC wrote
        # the data, and then ran afterwards and erased it.
        device = self.local_buf.device
        torch.cuda.set_device(device)
        while not self.stop.is_set():
            self.local_buf.zero_()
            torch.cuda.synchronize(device)
            try:
                handle = self.agent.initialize_xfer("READ", self.local_descs, self.remote_descs, self.peer_name)
                state = self.agent.transfer(handle)
                start = time.time()
                while state not in ("DONE", "ERR"):
                    if time.time() - start > read_timeout_s:
                        state = "TIMEOUT"
                        break
                    # Yield the GIL: a bare spin starves the main thread's collective, and a
                    # starved collective looks exactly like the hang this script is probing.
                    time.sleep(0)
                    state = self.agent.check_xfer_state(handle)
                elapsed = time.time() - start
                self.max_read_s = max(self.max_read_s, elapsed)
                if state == "TIMEOUT":
                    self.timeouts += 1
                    self.first_error = self.first_error or f"read stuck in PROC for {read_timeout_s}s"
                    return
                if state == "ERR":
                    self.errors += 1
                    self.first_error = self.first_error or "transfer state ERR"
                    continue
                self.agent.release_xfer_handle(handle)
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                self.errors += 1
                self.first_error = self.first_error or repr(exc)
                time.sleep(0.01)
                continue
            torch.cuda.synchronize()
            if not torch.equal(self.local_buf, self.expected):
                self.mismatches += 1
                diff = self.local_buf != self.expected
                ndiff = int(diff.sum())
                idx = diff.nonzero()
                first, last = int(idx[0]), int(idx[-1])
                all_zero = bool((self.local_buf[diff] == 0).all())
                # Re-check after a second sync: a late write landing would make it match now.
                torch.cuda.synchronize()
                late = int((self.local_buf != self.expected).sum())
                if len(self.mismatch_log) < 8:
                    self.mismatch_log.append(
                        f"t={time.time() - self.t0:.2f}s diff={ndiff}B span=[{first},{last}] "
                        f"zeros={all_zero} after_resync={late}B"
                    )
            self.reads += 1


def stage_objgather(seconds: float, payload_mb: int) -> int:
    payload = os.urandom(payload_mb << 20)
    out = [None] * dist.get_world_size()
    n = 0
    deadline = time.time() + seconds
    while time.time() < deadline:
        dist.all_gather_object(out, {"rank": dist.get_rank(), "blob": payload})
        n += 1
        if n % 10 == 0:
            log(f"objgather iter {n}")
    return n


def stage_dcp(seconds: float, ckpt_dir: str, async_save: bool, mb_per_rank: int, persistent: bool = False) -> int:
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.mapping import ShardedTensor
    from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelSaveStrategyWrapper
    from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy

    rank, world = dist.get_rank(), dist.get_world_size()
    rows = (mb_per_rank << 20) // (4 * 1024)
    local = torch.randn(rows, 1024, device="cuda")
    n = 0
    deadline = time.time() + seconds
    while time.time() < deadline:
        sd = {
            f"w{i}": ShardedTensor.from_rank_offsets(f"w{i}", local, (0, rank, world)) for i in range(4)
        }
        path = os.path.join(ckpt_dir, f"iter{n}")
        if rank == 0:
            os.makedirs(path, exist_ok=True)
        dist.barrier()
        strategy = FullyParallelSaveStrategyWrapper(
            TorchDistSaveShardedStrategy(backend="torch_dist", version=1), dist.group.WORLD
        )
        # "mcore" is the fork-based async writer the original report suspected; the default
        # ("nvrx") needs nvidia-resiliency-ext, which this environment does not have.
        req = dist_checkpointing.save(
            sd, path, sharded_strategy=strategy, async_sharded_save=async_save, async_strategy="mcore"
        )
        if async_save:
            from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue

            # persistent=True keeps one *spawned* writer process alive across saves, so nothing
            # forks the parent; persistent=False forks one child per save.
            queue = AsyncCallsQueue(persistent=persistent)
            queue.schedule_async_request(req)
            queue.maybe_finalize_async_calls(blocking=True, no_dist=False)
            queue.close()
        dist.barrier()
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["idle", "objgather", "dcp", "dcp_async"], required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--buf-mb", type=int, default=256)
    ap.add_argument("--payload-mb", type=int, default=64)
    ap.add_argument("--ckpt-dir", default="/tmp/repro_dcp_ucx")
    ap.add_argument("--ckpt-mb", type=int, default=512)
    ap.add_argument("--read-timeout-s", type=float, default=60.0)
    ap.add_argument("--backend", default="nccl")
    ap.add_argument("--persistent", action="store_true", help="dcp_async via the spawn-based persistent caller")
    ap.add_argument("--hog-gb", type=float, default=0.0, help="GPU memory pinned down before NIXL starts")
    args = ap.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(args.backend)
    rank, world = dist.get_rank(), dist.get_world_size()

    # The original report says the failure appeared "under high memory pressure".
    hog = None
    if args.hog_gb > 0:
        hog = torch.empty(int(args.hog_gb * (1 << 30)), dtype=torch.uint8, device="cuda")
        hog.fill_(1)
        log(f"holding {args.hog_gb:.0f} GB; free={torch.cuda.mem_get_info()[0] / (1 << 30):.1f} GB")

    name = f"repro-agent-{rank}"
    agent = nixl_agent(name, nixl_agent_config(enable_prog_thread=True))
    nbytes = args.buf_mb << 20
    source = pattern(rank, nbytes)
    agent.register_memory(source)
    recv = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    agent.register_memory(recv)

    # Exchange metadata and the source descriptors once, before the loop.
    meta = agent.get_agent_metadata()
    src_descs = agent.get_serialized_descs(agent.get_xfer_descs(source, mem_type="VRAM"))
    gathered = [None] * world
    dist.all_gather_object(gathered, (name, meta, src_descs))
    peer = (rank + 1) % world
    peer_name, peer_meta, peer_descs = gathered[peer]
    agent.add_remote_agent(peer_meta)
    remote_descs = agent.deserialize_descs(peer_descs)
    expected = pattern(peer, nbytes)

    reader = Reader(agent, peer_name, recv, remote_descs, expected)
    dist.barrier()
    thread = threading.Thread(target=reader.run, args=(args.read_timeout_s,), daemon=True)
    thread.start()
    time.sleep(2.0)
    reads_before = reader.reads

    def heartbeat():
        while not reader.stop.is_set():
            log(f"hb reads={reader.reads} errors={reader.errors} timeouts={reader.timeouts}")
            reader.stop.wait(10.0)

    threading.Thread(target=heartbeat, daemon=True).start()
    log(f"reader running, {reads_before} reads in warmup")
    collective_error = ""
    iterations = 0
    t0 = time.time()
    try:
        if args.stage == "idle":
            time.sleep(args.seconds)
        elif args.stage == "objgather":
            iterations = stage_objgather(args.seconds, args.payload_mb)
        else:
            iterations = stage_dcp(args.seconds, args.ckpt_dir, args.stage == "dcp_async", args.ckpt_mb, args.persistent)
    except Exception:  # noqa: BLE001
        collective_error = traceback.format_exc().splitlines()[-1]
    stage_s = time.time() - t0
    reads_during = reader.reads - reads_before

    time.sleep(3.0)  # reads after the stage: does the agent still work?
    reads_after = reader.reads - reads_before - reads_during
    reader.stop.set()
    thread.join(timeout=args.read_timeout_s + 5)

    gbps = reads_during * nbytes / max(stage_s, 1e-9) / 1e9
    log(
        f"stage={args.stage} iters={iterations} stage_s={stage_s:.1f} "
        f"reads_during={reads_during} ({gbps:.1f} GB/s) reads_after={reads_after} "
        f"errors={reader.errors} timeouts={reader.timeouts} mismatches={reader.mismatches} "
        f"max_read_s={reader.max_read_s:.3f} collective_error={collective_error!r} "
        f"first_error={reader.first_error!r}"
    )
    log(f"stage started at t={t0 - reader.t0:.2f}s, ended at t={t0 + stage_s - reader.t0:.2f}s")
    for line in reader.mismatch_log:
        log(f"MISMATCH {line}")
    ok = (
        not collective_error
        and reader.errors == 0
        and reader.timeouts == 0
        and reader.mismatches == 0
        and reads_after > 0
    )
    flags = torch.tensor([0 if ok else 1], device="cuda")
    dist.all_reduce(flags)
    if rank == 0:
        print(f"RESULT stage={args.stage} {'PASS' if flags.item() == 0 else 'FAIL'} ({flags.item()} rank(s) failed)", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0 if flags.item() == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
