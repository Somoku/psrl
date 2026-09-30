"""
Expose sparse sampler queries to engine logits processors independently of the RL loss.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from psrl.utils.sparse_logprobs import select_sparse_rows, sparse_log_probs


def query_sampler_log_probs(
    logits: torch.Tensor,
    data,
    row_ids: torch.Tensor,
    *,
    process_group=None,
    vocab_size: int | None = None,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return local sparse row IDs, trainer log probabilities and sampler log probabilities.

    `row_ids` maps each local logit row into the unpadded global microbatch.
    Negative IDs identify padding. The engine owns CP and SP permutations.
    """
    sample_offsets = data["sampler_offsets"]
    if sample_offsets.is_nested:
        counts = torch.cat([offsets.diff() for offsets in sample_offsets.unbind()])
        ids = data["sampler_token_ids"].values()
        log_probs = data["sampler_log_probs"].values()
    else:
        counts = sample_offsets.diff().reshape(-1)
        ids = data["sampler_token_ids"].reshape(-1)
        log_probs = data["sampler_log_probs"].reshape(-1)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    # A dedicated empty row absorbs every padding position, including CP alignment padding.
    offsets = torch.cat((offsets, offsets[-1:]))
    selected = torch.where(row_ids.reshape(-1) >= 0, row_ids.reshape(-1), len(counts))
    rows, entries = select_sparse_rows(offsets, selected)
    trainer_log_probs = sparse_log_probs(
        logits.reshape(-1, logits.shape[-1]),
        rows,
        ids[entries],
        process_group=process_group,
        vocab_size=vocab_size,
        chunk_size=chunk_size,
    )
    return rows, trainer_log_probs, log_probs[entries]


@dataclass
class LossWithLogitsProcessor:
    """
    Attach a differentiable per-token processor to an existing engine loss.
    """

    loss: Callable
    logits_processor: Callable

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)
