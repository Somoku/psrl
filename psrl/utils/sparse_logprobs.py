"""
Compute differentiable sparse log probabilities over local or sharded vocabularies.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


class _SparseLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, row_ids, token_ids, group, vocab_size, chunk_size):
        ctx.empty = row_ids.numel() == 0
        if ctx.empty:
            ctx.save_for_backward(logits)
            dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
            return logits.new_empty(0, dtype=dtype)
        distributed = group is not None
        rank = dist.get_rank(group) if distributed else 0
        rows, local_vocab = logits.shape
        start = rank * local_vocab
        valid = max(0, min(local_vocab, vocab_size - start))
        dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        normalizers = torch.empty(rows, device=logits.device, dtype=dtype)
        for begin in range(0, rows, chunk_size):
            end = min(begin + chunk_size, rows)
            block = logits[begin:end].to(dtype, copy=True)
            block[:, valid:] = -torch.inf
            maximum = block.amax(-1)
            if distributed:
                dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
            block.sub_(maximum[:, None]).exp_()
            denominator = block.sum(-1)
            if distributed:
                dist.all_reduce(denominator, group=group)
            normalizers[begin:end] = maximum + denominator.log()

        owned = (token_ids >= start) & (token_ids < start + valid)
        local_ids = (token_ids - start).clamp(0, local_vocab - 1).long()
        selected = logits[row_ids, local_ids].to(dtype)
        selected = torch.where(owned, selected, 0)
        if distributed:
            dist.all_reduce(selected, group=group)
        selected.sub_(normalizers[row_ids])
        ctx.save_for_backward(logits, row_ids, local_ids, owned, normalizers)
        ctx.valid = valid
        ctx.chunk_size = chunk_size
        return selected

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.empty:
            return torch.zeros_like(ctx.saved_tensors[0]), None, None, None, None, None
        logits, row_ids, local_ids, owned, normalizers = ctx.saved_tensors
        rows = logits.shape[0]
        dtype = normalizers.dtype
        weights = grad_output.to(dtype)
        row_weights = torch.zeros(rows, device=logits.device, dtype=dtype)
        row_weights.scatter_add_(0, row_ids, weights)
        grad_logits = torch.empty_like(logits)
        for begin in range(0, rows, ctx.chunk_size):
            end = min(begin + ctx.chunk_size, rows)
            block = logits[begin:end].to(dtype, copy=True)
            block[:, ctx.valid :] = -torch.inf
            block.sub_(normalizers[begin:end, None]).exp_()
            block.mul_(-row_weights[begin:end, None])
            grad_logits[begin:end] = block.to(logits.dtype)
        # Outputs are replicated across TP ranks. Reducing this gradient would multiply it by TP size.
        grad_logits.index_put_((row_ids, local_ids), torch.where(owned, weights, 0).to(logits.dtype), accumulate=True)
        return grad_logits, None, None, None, None, None


def sparse_log_probs(
    logits: torch.Tensor,
    row_ids: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    process_group=None,
    vocab_size: int | None = None,
    chunk_size: int = 256,
) -> torch.Tensor:
    """
    Gather arbitrary token log probabilities without a full softmax activation.

    Args:
        logits (torch.Tensor): Local contiguous vocabulary partition of shape `[N, V_local]`.
        row_ids (torch.Tensor): Local prediction row for each sparse entry, as int64.
        token_ids (torch.Tensor): Global vocabulary ID for each entry.
        process_group: Tensor parallel group, or None for a local vocabulary.
        vocab_size (int | None): Logical vocabulary size excluding padded vocabulary entries.
        chunk_size (int): Maximum rows of temporary FP32 softmax storage.

    Returns:
        torch.Tensor: One log probability per entry. TP outputs are replicated.
    """
    if logits.ndim != 2 or row_ids.ndim != 1 or token_ids.shape != row_ids.shape:
        raise ValueError("Sparse log probabilities require [N,V] logits and aligned entry vectors.")
    if chunk_size < 1 or logits.shape[1] < 1:
        raise ValueError("Chunk size and local vocabulary size must be positive.")
    if row_ids.dtype != torch.int64 or token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Sparse row IDs must be int64 and token IDs must be int32 or int64.")
    world_size = dist.get_world_size(process_group) if process_group is not None else 1
    total_vocab = world_size * logits.shape[1]
    vocab_size = total_vocab if vocab_size is None else vocab_size
    if not 0 < vocab_size <= total_vocab:
        raise ValueError(f"Invalid logical vocabulary size {vocab_size!r}.")
    torch._assert_async(((row_ids >= 0) & (row_ids < logits.shape[0])).all(), "Invalid sparse row ID.")
    torch._assert_async(((token_ids >= 0) & (token_ids < vocab_size)).all(), "Invalid sparse token ID.")
    return _SparseLogProbs.apply(logits, row_ids, token_ids, process_group, vocab_size, chunk_size)


def select_sparse_rows(
    offsets: torch.Tensor,
    selected_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Map arbitrary CP or SP prediction rows to sparse entries without densifying.

    Returns local row IDs and source entry indices. `selected_rows` must use the
    backend's actual token permutation, including its padding and zigzag order.
    """
    counts = offsets[selected_rows + 1] - offsets[selected_rows]
    rows = torch.repeat_interleave(torch.arange(len(selected_rows), device=offsets.device), counts)
    starts = torch.repeat_interleave(offsets[selected_rows], counts)
    local_starts = torch.repeat_interleave(counts.cumsum(0) - counts, counts)
    entries = starts + torch.arange(len(rows), device=offsets.device) - local_starts
    return rows, entries
