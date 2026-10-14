"""
Utility helpers for working with TransferQueue's KVBatchMeta.
"""

import logging
from collections import Counter
from collections.abc import Mapping
from enum import Enum
from typing import Any

import torch
import transfer_queue as tq
from transfer_queue import KVBatchMeta

psrl_logger = logging.getLogger(__file__)

# Fields indexed by a position within the trajectory's response. They describe the same
# tokens from different angles, so a row's length is one fact they must all agree on.
RESPONSE_ALIGNED_FIELDS = (
    "response_mask",
    "loss_mask",
    "rm_scores",
    "rollout_log_probs",
    "old_log_probs",
    "entropy",
    "ref_log_prob",
    "values",
)


class PayloadState(Enum):
    """Lifecycle state for payloads stored in TransferQueue."""

    PENDING_GROUP = "pending_group"
    OCCUPIED = "occupied"
    READY = "ready"
    DROPPED = "dropped"
    CONSUMED = "consumed"

    @property
    def is_terminal(self) -> bool:
        """Whether payload ownership has ended and its TQ storage may be cleared."""
        return self in {PayloadState.DROPPED, PayloadState.CONSUMED}


def request_payload_keys(request_id: int, n_trajectory: int) -> list[str]:
    """Reconstruct the TQ payload keys for one request."""
    if n_trajectory < 1:
        raise ValueError(f"n_trajectory must be positive, got {n_trajectory}.")
    if n_trajectory == 1:
        return [str(request_id)]
    return [f"{request_id}_{i}" for i in range(n_trajectory)]


def response_lengths(field: torch.Tensor | list | tuple | None) -> torch.Tensor | None:
    """
    Return each row's token count for a response-aligned field, or None when unreadable.

    A response-aligned field is stored ragged, so a row's length is the gap between its
    offsets. A field that arrives as an already padded rectangle cannot report a per-row
    length. Its width is a batch maximum, and a worker output padded to `responses` is
    legitimately wider than the mask, so such a field returns None instead of a guess.

    Args:
        field (torch.Tensor | list | tuple | None): A batch field read from the transfer queue.

    Returns:
        torch.Tensor | None: Per-row lengths, or None when the field cannot supply them.
    """
    if field is None:
        return None
    if hasattr(field, "offsets"):
        return field.offsets().diff()
    if isinstance(field, torch.Tensor):
        if field.dim() == 1:
            return torch.tensor([field.numel()])
        return None
    if isinstance(field, (list, tuple)):
        return torch.tensor([int(element.numel()) for element in field])
    return None


def _uniform_width(field: Any) -> int | None:
    """Return a field's width when it arrives as one padded rectangle, else None."""
    if isinstance(field, torch.Tensor) and not hasattr(field, "offsets") and field.dim() == 2:
        return int(field.shape[1])
    return None


def assert_response_aligned_lengths(
    fields: Mapping[str, Any],
    *,
    reference: str = "responses",
    stage: str,
) -> torch.Tensor | None:
    """
    Fail unless every response-aligned field has the same per-row length as `reference`.

    A batch pads each field on its own (`TensorDict.to_padded_tensor` pads a field to that
    field's own maximum), so a length that one producer gets wrong stays invisible until it
    changes a tensor's width far downstream. Comparing the whole set here makes such a
    producer fail at the boundary that crossed it, naming the field.

    A field that shares one width across the batch cannot report per-row lengths, so only its
    width is checked against the longest response. A worker output padded to `responses` is
    legitimately wider than the mask and passes.

    Args:
        fields (Mapping[str, Any]): Batch fields by name, as read from the transfer queue.
            A field that is absent is skipped.
        reference (str): The field carrying the authoritative response length.
        stage (str): The pipeline stage, for the log line and the error message.

    Returns:
        torch.Tensor | None: The reference's per-row lengths, or None when it is unreadable.

    Raises:
        AssertionError: If the reference is unreadable, a field has a different row count,
            a padded field is narrower than the longest response, or a field disagrees with
            the reference on per-row length.
    """
    reference_lengths = response_lengths(fields.get(reference))
    if reference_lengths is None:
        raise AssertionError(
            f"[length-contract] stage={stage}: reference field {reference!r} is missing or its per-row "
            f"length cannot be read, so the contract cannot be checked. Fields: {sorted(fields)}."
        )
    if reference_lengths.numel() == 0:
        return reference_lengths
    longest = int(reference_lengths.max())

    offenders: dict[str, list[tuple[int, int, int]]] = {}
    truncated: dict[str, tuple[int, int]] = {}
    for name, field in fields.items():
        if name == reference:
            continue
        lengths = response_lengths(field)
        if lengths is None:
            width = _uniform_width(field)
            if width is not None and width < longest:
                truncated[name] = (width, longest)
            continue
        if lengths.numel() != reference_lengths.numel():
            raise AssertionError(
                f"[length-contract] stage={stage}: field {name!r} has {lengths.numel()} rows but "
                f"{reference!r} has {reference_lengths.numel()}."
            )
        divergent = (lengths != reference_lengths).nonzero().flatten().tolist()
        if divergent:
            offenders[name] = [(int(row), int(reference_lengths[row]), int(lengths[row])) for row in divergent[:5]]

    psrl_logger.debug(
        f"[length-contract] stage={stage}: rows={reference_lengths.numel()} reference={reference} "
        f"min={int(reference_lengths.min())} max={longest} offenders={sorted(offenders)} "
        f"truncated={sorted(truncated)}."
    )
    if truncated:
        raise AssertionError(
            f"[length-contract] stage={stage}: {len(truncated)} field(s) are narrower than the longest "
            f"response, each value being (field width, longest {reference}): {truncated}. Such a field was "
            "built from a length other than the response's own, so at least one row's reward or mask "
            "cannot be aligned."
        )
    if offenders:
        raise AssertionError(
            f"[length-contract] stage={stage}: {len(offenders)} field(s) disagree with {reference!r} on "
            f"per-row length, each value being (row, {reference}, field): {offenders}. Every response-aligned "
            "field is indexed by position within the response, so they must share one length per row."
        )
    return reference_lengths


def _require_terminal_state(state: PayloadState) -> None:
    if not state.is_terminal:
        raise ValueError(f"Cannot clear payload in non-terminal state {state.value!r}.")


def _existing_payload_keys(
    keys: list[str] | str,
    partition_id: str,
    partitions: dict[str, dict[str, Any]],
) -> list[str]:
    """Return only keys currently registered in the target partition."""
    requested_keys = [keys] if isinstance(keys, str) else keys
    partition = partitions.get(partition_id, {})
    return [key for key in requested_keys if key in partition]


def clear_payload(
    keys: list[str] | str,
    partition_id: str,
    state: PayloadState,
) -> None:
    """Synchronously clear any present payload after entering a terminal state."""
    _require_terminal_state(state)
    existing_keys = _existing_payload_keys(keys, partition_id, tq.kv_list(partition_id=partition_id))
    if existing_keys:
        tq.kv_clear(keys=existing_keys, partition_id=partition_id)


async def async_clear_payload(
    keys: list[str] | str,
    partition_id: str,
    state: PayloadState,
) -> None:
    """Asynchronously clear any present payload after entering a terminal state."""
    _require_terminal_state(state)
    partitions = await tq.async_kv_list(partition_id=partition_id)
    existing_keys = _existing_payload_keys(keys, partition_id, partitions)
    if existing_keys:
        await tq.async_kv_clear(keys=existing_keys, partition_id=partition_id)


def validate_ready_payload(
    keys: list[str],
    partition_id: str,
    partitions: dict[str, dict[str, dict[str, Any]]],
) -> None:
    """Validate that every key in a prospective READY batch is committed."""
    duplicate_keys = sorted(key for key, count in Counter(keys).items() if count > 1)
    partition = partitions.get(partition_id, {})
    missing_keys = [key for key in keys if key not in partition]
    non_success_keys = [
        (key, partition[key].get("status"))
        for key in keys
        if key in partition and partition[key].get("status") != "success"
    ]
    if duplicate_keys or missing_keys or non_success_keys:
        raise RuntimeError(
            "Cannot publish READY payload: "
            f"partition_id={partition_id!r}, total={len(keys)}, "
            f"duplicate={len(duplicate_keys)}, missing={len(missing_keys)}, "
            f"non_success={len(non_success_keys)}, "
            f"duplicate_keys[:20]={duplicate_keys[:20]!r}, "
            f"missing_keys[:20]={missing_keys[:20]!r}, "
            f"non_success_keys[:20]={non_success_keys[:20]!r}."
        )


def kv_batch_meta_update_tags(batch: KVBatchMeta, key: str, value) -> KVBatchMeta:
    """Return a new KVBatchMeta with ``key`` set to ``value`` in every tag.

    If ``value`` is a list it must have the same length as ``batch``, and
    each element is assigned to the corresponding tag. Otherwise the scalar
    ``value`` is broadcast to all tags.

    Args:
        batch: Source ``KVBatchMeta``.
        key: Tag field name to set / overwrite.
        value: Scalar (broadcast) or list (per-sample) new value.

    Returns:
        A new ``KVBatchMeta`` instance with updated tags.
    """
    n = len(batch)
    if isinstance(value, list):
        if len(value) != n:
            raise ValueError(
                f"kv_batch_meta_update_tags: value list length {len(value)} does not match batch size {n}."
            )
        new_tags = [{**tag, key: value[i]} for i, tag in enumerate(batch.tags)]
    else:
        new_tags = [{**tag, key: value} for tag in batch.tags]

    return KVBatchMeta(
        keys=batch.keys,
        tags=new_tags,
        partition_id=batch.partition_id,
        fields=batch.fields,
        extra_info=batch.extra_info,
    )
