"""Build the per-token reward tensor the trainer reads back as `rm_scores`."""

from collections.abc import Sequence

import torch


def rm_scores_from_lengths(lengths: Sequence[int], scores: Sequence[float]) -> list[torch.Tensor]:
    """
    Build one `rm_scores` row per trajectory, each as long as that trajectory's response.

    A trajectory's reward is a single outcome scalar, so the token tensor is a representation
    of it rather than its own quantity. It has to be built from the trajectory's own response
    length. A length taken from elsewhere, such as the last trajectory of the session, gives a
    shorter trajectory a tensor that disagrees with its `response_mask`. Because a batch pads
    each field to that field's own maximum, the two widths then diverge downstream.

    Args:
        lengths (Sequence[int]): Per-trajectory response length in tokens.
        scores (Sequence[float]): Outcome reward per trajectory, aligned with `lengths`.

    Returns:
        list[torch.Tensor]: One 1-D float tensor per trajectory, holding the score at its last
            token. A trajectory that responded with no tokens gets an empty row.

    Raises:
        ValueError: If `lengths` and `scores` have different sizes.
    """
    if len(lengths) != len(scores):
        raise ValueError(f"Got {len(lengths)} response lengths but {len(scores)} reward scores.")
    rows = []
    for length, score in zip(lengths, scores, strict=True):
        row = torch.zeros(int(length), dtype=torch.float32)
        if row.numel():
            row[-1] = float(score)
        rows.append(row)
    return rows
