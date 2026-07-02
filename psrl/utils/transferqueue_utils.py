"""
Utility helpers for working with TransferQueue's KVBatchMeta.
"""

from transfer_queue import KVBatchMeta


def kv_batch_meta_update_tags(batch: KVBatchMeta, key: str, value) -> KVBatchMeta:
    """Return a new KVBatchMeta with ``key`` set to ``value`` in every tag.

    If ``value`` is a list it must have the same length as ``batch``, and
    each element is assigned to the corresponding tag.  Otherwise the scalar
    ``value`` is broadcast to all tags.

    Args:
        batch: Source ``KVBatchMeta``.
        key:   Tag field name to set / overwrite.
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
