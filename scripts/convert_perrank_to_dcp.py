#!/usr/bin/env python
"""Convert per-rank checkpoint format back to DCP (dist_checkpointing) format.

Our training save uses per-rank torch.save() to avoid UCX heap corruption.
This script converts those checkpoints back to verl's standard DCP format.

Must be run with torchrun using the SAME TP/PP/CP/EP config as training,
but in an environment WITHOUT NIXL/UCX (so DCP save works without conflict).

Usage:
    # Same TP=4, PP=1 as training, 16 GPUs across 2 nodes:
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=$RANK \
        --master_addr=$MASTER --master_port=$PORT \
        scripts/convert_perrank_to_dcp.py \
        --input_dir /tmp/lhy/global_step_1/actor/dist_ckpt \
        --output_dir /path/to/dcp_output/global_step_1/actor/dist_ckpt \
        --tp_size 4 --pp_size 1 --cp_size 1 --ep_size 1

    # world_size must exactly match the value in parallel_config.json.
    # 'per_rank_torch_save' format only — 'per_rank_plain_tensors' (new format)
    # lacks ShardedBase metadata and cannot be converted to DCP.
"""

import argparse
import json
import os

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description="Convert per-rank .pt checkpoint to DCP format")
    parser.add_argument("--input_dir", required=True, help="Directory containing rank_*.pt files")
    parser.add_argument("--output_dir", required=True, help="Output directory for DCP format checkpoint")
    parser.add_argument("--tp_size", type=int, default=4, help="Tensor model parallel size (must match training)")
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline model parallel size (must match training)")
    parser.add_argument("--cp_size", type=int, default=1, help="Context parallel size (must match training)")
    parser.add_argument("--ep_size", type=int, default=1, help="Expert model parallel size (must match training)")
    args = parser.parse_args()

    # Initialize torch.distributed
    if "WORLD_SIZE" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # Initialize Megatron parallel state
    from megatron.core import parallel_state as mpu
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=args.cp_size,
        expert_model_parallel_size=args.ep_size,
    )
    model_parallel_cuda_manual_seed(0)

    # Validate checkpoint metadata
    metadata_path = os.path.join(args.input_dir, "parallel_config.json")
    assert os.path.exists(metadata_path), (
        f"Metadata file not found: {metadata_path!r}.  "
        f"Is {args.input_dir!r} a valid per-rank checkpoint directory?"
    )
    with open(metadata_path) as f:
        metadata = json.load(f)

    fmt = metadata.get("format")
    assert fmt == "per_rank_torch_save", (
        f"Unsupported checkpoint format {fmt!r}.  "
        f"Only 'per_rank_torch_save' checkpoints contain ShardedBase metadata needed "
        f"to reconstruct DCP sharding.  'per_rank_plain_tensors' checkpoints (saved by "
        f"the current megatron_saver.py) have plain tensors with no sharding metadata "
        f"and cannot be converted to DCP with this script."
    )

    saved_ws = metadata.get("world_size")
    assert saved_ws == world_size, (
        f"Checkpoint world_size={saved_ws} != current world_size={world_size}.  "
        f"Run with exactly {saved_ws} processes (torchrun --nproc_per_node=... --nnodes=...)."
    )

    if rank == 0:
        print(f"Checkpoint metadata: {metadata}")

    # Verify rank file exists
    rank_path = os.path.join(args.input_dir, f"rank_{rank}.pt")
    assert os.path.exists(rank_path), (
        f"[Rank {rank}] Per-rank file not found: {rank_path!r}"
    )

    # Load per-rank checkpoint
    print(f"[Rank {rank}] Loading {rank_path}...")
    state_dict = torch.load(rank_path, map_location="cpu", weights_only=False)
    assert isinstance(state_dict, dict), (
        f"[Rank {rank}] Expected state_dict to be a dict, got {type(state_dict)}"
    )
    print(f"[Rank {rank}] Loaded. Keys: {list(state_dict.keys())}")

    # Re-save using DCP (no NIXL in this process → no UCX conflict)
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.serialization import get_default_save_sharded_strategy

    os.makedirs(args.output_dir, exist_ok=True)

    from megatron.core.dist_checkpointing.strategies.fully_parallel import (
        FullyParallelSaveStrategyWrapper,
    )

    save_strategy = FullyParallelSaveStrategyWrapper(
        get_default_save_sharded_strategy("torch_dist"),
        parallelization_group=None,  # defaults to WORLD — all ranks coordinate shard assignment
        do_cache_distribution=False,  # one-shot conversion, no need to cache
    )

    # === Fix: prevent NCCL corruption from forked DCP writer processes ================
    # Root cause (confirmed by nccl_experiment.py tests 1–4):
    #   TorchDistSaveShardedStrategy.async_save() returns an AsyncRequest whose async_fn
    #   is FileSystemWriterAsync.write_preloaded_data_multiproc() (filesystem_async.py).
    #   execute_sync() calls it synchronously in the main process.  That function forks N
    #   worker child processes via mp.get_context("fork") — one per write bucket.  The
    #   forked children inherit the parent's ProcessGroupNCCL objects (live NCCL comms).
    #   When each child finishes and exits normally, Python's __del__ finalizers call
    #   ncclCommAbort / ncclCommDestroy on the inherited (shared) communicator handles,
    #   corrupting the parent's NCCL state.  The dist.barrier() at execute_sync:97 is then
    #   enqueued on a broken communicator, its CUDA kernel never fires, and NCCL reports
    #   "the scheduled collective, for some reason, didn't run" → 600 s timeout → SIGABRT.
    #
    # Attempted fix (os._exit on write_preloaded_data): FAILED — same SeqNum=3 timeout.
    #   os._exit alone is insufficient; CUDA device cleanup at child process exit still
    #   corrupts NCCL IPC state even when Python __del__ finalizers are bypassed.
    #
    # Correct fix: replace write_preloaded_data_multiproc with a sequential no-fork version
    #   that calls write_preloaded_data directly in the main process — one bucket at a time.
    #   No child process is ever spawned, so NCCL communicator handles are never inherited
    #   and corruption is impossible.  stdlib queue.SimpleQueue / queue.Queue are drop-in
    #   replacements for mp.SimpleQueue / mp.JoinableQueue: write_preloaded_data calls
    #   results_queue.put(), count_queue.get(), and count_queue.task_done() — all present
    #   on the stdlib types.  The dict written to global_results_queue is identical in
    #   format to what the original multiproc version produced.
    import queue as _stdlib_queue

    from megatron.core.dist_checkpointing.strategies import filesystem_async as _fsa

    _orig_write_preloaded_data = _fsa.FileSystemWriterAsync.write_preloaded_data

    def _write_preloaded_data_multiproc_nofork(
        transform_list, use_msc, rank, write_buckets, global_results_queue
    ):
        """Sequential, no-fork replacement for write_preloaded_data_multiproc.

        Calls write_preloaded_data() directly in the main process for each bucket.
        No subprocesses are spawned, so NCCL state is never corrupted.
        """
        write_results_or_exc = {}
        for i, write_bucket in enumerate(write_buckets):
            # Per-call queues: write_preloaded_data puts result into results_q and
            # calls count_q.get() + count_q.task_done() before returning.
            results_q = _stdlib_queue.SimpleQueue()
            count_q = _stdlib_queue.Queue()
            count_q.put(i)
            kwargs = {
                "local_proc_idx": i,
                "write_bucket": write_bucket,
                "results_queue": results_q,
                "count_queue": count_q,
                "use_fsync": True,
            }
            if use_msc:
                kwargs["use_msc"] = use_msc
            try:
                _orig_write_preloaded_data(transform_list, **kwargs)
            except Exception as e:
                write_results_or_exc = RuntimeError(f"[nofork] Bucket {i} failed: {e}")
                break
            local_proc_idx, local_results_or_exc = results_q.get()
            if isinstance(local_results_or_exc, Exception):
                write_results_or_exc = local_results_or_exc
                break
            assert isinstance(local_results_or_exc, list), type(local_results_or_exc)
            write_results_or_exc[local_proc_idx] = local_results_or_exc
        global_results_queue.put(write_results_or_exc)

    _fsa.FileSystemWriterAsync.write_preloaded_data_multiproc = staticmethod(
        _write_preloaded_data_multiproc_nofork
    )
    # ==================================================================================

    print(f"[Rank {rank}] Saving to DCP format: {args.output_dir}")
    dist_checkpointing.save(
        state_dict,
        args.output_dir,
        sharded_strategy=save_strategy,
        async_sharded_save=False,
    )

    dist.barrier()
    if rank == 0:
        print(f"Conversion complete: {args.input_dir} → {args.output_dir}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
