import logging
import os

import torch
import torch.distributed as dist

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


def _byte_to_gb(n: int):
    return round(n / (1024**3), 2)


def available_memory():
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    return {
        "gpu": str(device),
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(torch.cuda.memory_allocated(device)),
        "reserved_GB": _byte_to_gb(torch.cuda.memory_reserved(device)),
    }


def print_memory(msg, rank_diff=True):
    memory_info = available_memory()
    # Need to print for all ranks, different rank can have different behaviors
    if rank_diff and dist.is_initialized():
        rank = dist.get_rank()
        psrl_logger.info(f"[Rank {rank}] Memory-Usage {msg}: {memory_info}")
    else:
        psrl_logger.info(f"Memory-Usage {msg}: {memory_info}")
    return memory_info
