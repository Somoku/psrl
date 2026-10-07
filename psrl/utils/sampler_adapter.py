"""Adapt the compact sampler capture to the training contract.

The capture layer (SMG + TITO + :class:`SamplerDistribution`) stores one compact ragged row per
response token: assistant rows hold the sampler head, environment/tool rows are empty. The trainer
consumes a dense, full-sequence ``rollout_topk_ids`` / ``rollout_topk_log_probs`` pair (upstream
score centering, ``top_p=1``). This module is the single conversion point so the capture format does
not leak into the loss.

Three producers match the three trainable paths: :func:`to_dense_topk` for the top-k head model,
:func:`to_dense_support` for exact top-p support, and :func:`to_compact_support` for the compact
support transport (per-row lengths plus flat ids/log-probs).
"""

from __future__ import annotations

import numpy as np
import torch

from psrl.utils.sampler_distribution import SamplerDistribution


def _fixed_width(distribution: SamplerDistribution, response_mask: list[int]) -> tuple[np.ndarray, int]:
    """Return the active-row boolean mask and the common head width, rejecting a ragged capture."""
    rows = len(distribution)
    if rows != len(response_mask):
        raise ValueError(f"sampler distribution has {rows} rows but response_mask has {len(response_mask)}.")
    counts = np.diff(distribution.offsets)
    active = np.asarray(response_mask, dtype=bool)
    if np.any(counts[~active] != 0):
        raise ValueError("environment/tool rows must have an empty sampler head.")
    active_counts = counts[active]
    if active_counts.size == 0:
        return active, 0
    width = int(active_counts[0])
    if width < 1 or np.any(active_counts != width):
        raise ValueError(f"topk capture must be fixed width, got widths {np.unique(active_counts).tolist()}.")
    return active, width


def _validate_support(distribution: SamplerDistribution, response_mask: list[int], width: int) -> np.ndarray:
    """Validate a support capture and return the per-row entry counts.

    Checks the row count matches the response mask, no support exceeds ``width``, and masked
    environment/tool rows are empty. Shared by the dense and compact support adapters (the compact
    one only needs the counts, so it avoids materializing the padded ``[rows, width]`` array).
    """
    rows = len(distribution)
    if rows != len(response_mask):
        raise ValueError(f"sampler distribution has {rows} rows but response_mask has {len(response_mask)}.")
    if width < 1:
        raise ValueError("support capture width must be positive.")
    counts = np.diff(distribution.offsets)
    if counts.size and int(counts.max()) > width:
        raise ValueError(f"sampler support width {int(counts.max())} exceeds the capture budget {width}.")
    active = np.asarray(response_mask, dtype=bool)
    if np.any(counts[~active] != 0):
        raise ValueError("environment/tool rows must have an empty sampler support.")
    return counts


def _pad_support(flat: np.ndarray, offsets: np.ndarray, counts: np.ndarray, width: int, *, fill, dtype) -> np.ndarray:
    """Scatter a flat per-row support buffer into a ``[rows, width]`` array, right-padded with ``fill``.

    The scatter is vectorized (``np.repeat`` over the per-row counts), so the row count, which can be
    tens of thousands, costs no Python loop.
    """
    rows = len(counts)
    out = np.full((rows, width), fill, dtype=dtype)
    total = int(counts.sum())
    if total:
        row_idx = np.repeat(np.arange(rows, dtype=np.int32), counts)
        col_idx = np.arange(total, dtype=np.int32) - np.repeat(offsets[:-1].astype(np.int32), counts)
        out[row_idx, col_idx] = flat
    return out


def _support_rows(
    distribution: SamplerDistribution, response_mask: list[int], width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a support capture and right-pad each row's log-probs to ``[rows, width]``.

    Returns ``(counts, padded_log_probs)``. The padded array carries ``-inf`` (the padding sentinel)
    in unused columns and on masked rows.
    """
    counts = _validate_support(distribution, response_mask, width)
    log_probs = _pad_support(
        distribution.log_probs, distribution.offsets, counts, width, fill=-np.inf, dtype=np.float32
    )
    return counts, log_probs


def to_dense_topk(
    distribution: SamplerDistribution,
    response_mask: list[int],
    *,
    prompt_length: int,
) -> dict[str, torch.Tensor]:
    """Lay the captured head over the full prompt+response sequence as upstream ``rollout_topk_*``.

    Response token ``r`` is predicted by logits row ``prompt_length - 1 + r``, prompt, padding and
    masked environment/tool rows get the sentinel dummy head so they contribute nothing once the loss
    masks them out.
    """
    from verl.trainer.ppo.score_centering import dummy_rollout_topk

    rows = len(distribution)
    active, width = _fixed_width(distribution, response_mask)
    if width == 0:
        # No trainable token carried a head. Upstream still expects a well-shaped dummy head.
        raise ValueError("topk capture has no active row.")
    ids_active = torch.as_tensor(distribution.token_ids.reshape(-1, width), dtype=torch.int32)
    log_probs_active = torch.as_tensor(distribution.log_probs.reshape(-1, width), dtype=torch.float32)
    if ids_active.shape[0] != int(active.sum()):
        raise ValueError(
            f"sampler head covers {ids_active.shape[0]} rows but {int(active.sum())} response rows are active."
        )
    # Fill the full sequence with the sentinel head and scatter the active heads straight into their
    # prediction rows (`prompt_length - 1 + r`), skipping a response-sized intermediate + copy.
    seq_ids, seq_log_probs = dummy_rollout_topk(prompt_length + rows, width)
    dest = torch.as_tensor(np.flatnonzero(active), dtype=torch.long) + (prompt_length - 1)
    seq_ids[dest] = ids_active
    seq_log_probs[dest] = log_probs_active
    return {"rollout_topk_ids": seq_ids, "rollout_topk_log_probs": seq_log_probs}


def to_dense_support(
    distribution: SamplerDistribution,
    response_mask: list[int],
    *,
    width: int,
    prompt_length: int,
) -> dict[str, torch.Tensor]:
    """Lay the compact support rows over the sequence as a dense, right-padded head.

    Every row keeps its recorded support ids and probabilities in the leading columns. Unused slots
    and masked environment/tool rows carry the padding sentinel, so the head mask excludes them. The
    trainer renormalizes over each row's ids, which makes the correction the exact expected score on
    the support, so ``width`` only has to cover the largest recorded support.
    """
    from verl.trainer.ppo.score_centering import HEAD_PADDING_ID, pad_rollout_topk

    counts, log_probs = _support_rows(distribution, response_mask, width)
    ids = _pad_support(
        distribution.token_ids, distribution.offsets, counts, width, fill=HEAD_PADDING_ID, dtype=np.int32
    )
    seq_ids, seq_log_probs = pad_rollout_topk(
        torch.from_numpy(ids), torch.from_numpy(log_probs), prompt_length=prompt_length
    )
    return {"rollout_topk_ids": seq_ids, "rollout_topk_log_probs": seq_log_probs}


def to_compact_support(
    distribution: SamplerDistribution,
    response_mask: list[int],
    *,
    width: int,
    prompt_length: int,
) -> dict[str, torch.Tensor]:
    """Compact support transport: per-row lengths plus the flat support ids and log-probs.

    Ships the concatenated support ids and log-probs (``Σ entries``) instead of the dense
    ``rollout_topk_ids``/``rollout_topk_log_probs`` ``[T, K]`` arrays, so both the id and the log-prob
    transport shrink to the realized support rather than the budget width. The trainer reconstructs
    both dense arrays from ``rollout_topk_lengths`` (see ``score_centering_logits_processor``). This is
    a host/TransferQueue-memory win. The trainer still materializes the dense ``[T, K]`` arrays on the
    GPU, so it does not lower GPU memory.
    """
    rows = len(distribution)
    counts = _validate_support(distribution, response_mask, width)
    seq_lengths = np.zeros(prompt_length + rows, dtype=np.int32)
    seq_lengths[prompt_length - 1 : prompt_length - 1 + rows] = counts
    return {
        "rollout_topk_lengths": torch.from_numpy(seq_lengths),
        "rollout_support_ids": torch.from_numpy(distribution.token_ids.astype(np.int32)),
        "rollout_support_log_probs": torch.from_numpy(distribution.log_probs.astype(np.float32)),
    }
