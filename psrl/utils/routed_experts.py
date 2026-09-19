"""Canonical routed-expert tensor validation and alignment."""

from typing import Any

import numpy as np
import torch

_WIRE_DTYPES = (np.dtype(np.uint8), np.dtype(np.uint16))
_MAX_EXPERT_ID = np.iinfo(np.int16).max


def validate_routed_experts_array(routed_experts: Any) -> np.ndarray:
    """
    Validate a compact routed-expert array from an inference boundary.

    Args:
        routed_experts (Any): Array-like payload with shape ``(num_tokens, num_layers, top_k)``.

    Returns:
        np.ndarray: The validated array, unchanged.

    Raises:
        ValueError: If the array is not 3D, is empty, uses a non-wire dtype, or holds an
            expert id that exceeds veRL's signed int16 replay representation.
    """
    array = np.asarray(routed_experts)
    if array.ndim != 3:
        raise ValueError(f"routed_experts must be 3D, got shape {array.shape!r}.")
    if any(dimension <= 0 for dimension in array.shape):
        raise ValueError(f"routed_experts dimensions must be positive, got shape {array.shape!r}.")
    if array.dtype not in _WIRE_DTYPES:
        raise ValueError(f"routed_experts must use uint8 or uint16 on the wire, got dtype {array.dtype!r}.")
    if array.size and int(array.max()) > _MAX_EXPERT_ID:
        raise ValueError(
            "routed_experts holds an expert id that veRL's signed int16 replay tensor cannot represent, "
            f"got max={int(array.max())} and limit={_MAX_EXPERT_ID}."
        )
    if not array.flags.c_contiguous:
        raise ValueError(f"routed_experts must be C-contiguous, got strides {array.strides!r}.")
    return array


def canonicalize_routed_experts(
    routed_experts: Any | None,
    prompt_length: int,
    response_length: int,
) -> torch.Tensor | None:
    """
    Build the full prompt-plus-response int16 tensor used by router replay.

    The compact inference payload omits the final sampled token because that token is never
    forwarded through the model. The canonical representation restores that one row as zero
    filler. Missing, overlapping, or extra positions are rejected.

    Args:
        routed_experts (Any | None): Compact payload with shape ``(num_rows, num_layers, top_k)``.
            Uint8 or uint16 arrays arrive from the SMG wire, integer tensors from an in-process
            model output. None disables routed experts.
        prompt_length (int): Number of prompt tokens in the canonical sequence.
        response_length (int): Number of response tokens in the canonical sequence.

    Returns:
        torch.Tensor | None: An int16 tensor of shape ``(prompt_length + response_length, num_layers,
            top_k)``, or None when no payload was supplied.

    Raises:
        ValueError: If the payload dtype or expert ids are unsupported, or the row count does not
            cover every forwarded sequence position.
    """
    if routed_experts is None:
        return None
    if prompt_length < 0 or response_length <= 0:
        raise ValueError(
            "Canonical routed_experts needs a non-negative prompt length and a positive response length, "
            f"got prompt_length={prompt_length} and response_length={response_length}."
        )

    sequence_length = prompt_length + response_length
    if isinstance(routed_experts, torch.Tensor):
        tensor = routed_experts.detach().cpu()
        if tensor.ndim != 3:
            raise ValueError(f"routed_experts must be 3D, got shape {tuple(tensor.shape)!r}.")
        if tensor.shape[1] <= 0 or tensor.shape[2] <= 0:
            raise ValueError(f"routed_experts dimensions must be positive, got shape {tuple(tensor.shape)!r}.")
        if tensor.dtype not in (torch.uint8, torch.int16, torch.int32, torch.int64):
            raise ValueError(f"routed_experts must contain integer expert ids, got dtype {tensor.dtype!r}.")
        if tensor.numel():
            min_expert_id, max_expert_id = torch.aminmax(tensor)
            if min_expert_id.item() < 0 or max_expert_id.item() > _MAX_EXPERT_ID:
                raise ValueError(
                    "routed_experts expert ids must fit veRL's non-negative int16 replay representation, "
                    f"got range [{min_expert_id.item()}, {max_expert_id.item()}]."
                )
        rows = tensor.shape[0]
        compact = tensor
    else:
        array = validate_routed_experts_array(routed_experts)
        rows = array.shape[0]
        if not array.flags.writeable:
            array = array.copy()
        compact = torch.from_numpy(array)

    if rows not in (sequence_length - 1, sequence_length):
        raise ValueError(
            "routed_experts must cover every forwarded prompt and response position, "
            f"got {rows} rows for sequence length {sequence_length}."
        )

    canonical = torch.zeros(
        (sequence_length, compact.shape[1], compact.shape[2]),
        dtype=torch.int16,
    )
    canonical[:-1].copy_(compact[: sequence_length - 1])
    return canonical
