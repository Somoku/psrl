import logging
import os

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

_clamp_warned = False


def apply_block_table_clamp_patch() -> None:
    """Clamp the v2 runner's block-table appends to the row capacity.

    The v2 model runner keeps its own, append-only view of a request's block
    IDs: `BlockTables.append_block_ids` accumulates into `num_blocks` and writes
    at the accumulated offset. The scheduler can shrink a request's block list
    without signalling a replace -- a sync KV load failure under the recompute
    policy truncates `num_computed_tokens` and frees the orphaned tail without
    routing through `resumed_from_preemption` -- so repeated rollbacks push the
    accumulated offset past the row width. v2 catches that with an explicit
    `RuntimeError`; the v1 runner has the equivalent guard as a source patch
    (see `patch/vllm/v0.29.0.patch`, `gpu_model_runner.py`).

    This patch pre-truncates the incoming block IDs to the remaining row
    capacity so both runners drop the same dead tail instead of failing the
    engine. The surplus is always a dead tail: positions beyond
    `max_model_len` are never addressed. v2's own `RuntimeError` stays in place
    as a backstop for any other cause of overflow.

    The patch is applied unconditionally but only takes effect on the v2 runner,
    which is the only user of `BlockTables`.
    """
    try:
        from vllm.v1.worker.gpu.block_table import BlockTables
    except ImportError:
        # The v2 runner's block table is only importable when its package tree is
        # available; nothing to patch otherwise.
        psrl_logger.debug("vllm.v1.worker.gpu.block_table is unavailable; skipping the block-table clamp patch.")
        return

    _patch_append_block_ids(BlockTables)


def _patch_append_block_ids(target: type) -> None:
    _SENTINEL = "_psrl_block_table_clamp_patched"
    if getattr(target, _SENTINEL, False):
        psrl_logger.debug("BlockTables.append_block_ids already patched, skipping.")
        return

    _original = target.append_block_ids

    def append_block_ids(self, req_index, new_block_ids, overwrite):
        global _clamp_warned

        clamped = []
        clamped_groups = 0
        for group in range(self.num_kv_cache_groups):
            block_ids = new_block_ids[group]
            # `len(...)` rather than truthiness: the incoming container may be a
            # numpy array, for which `not x` is ambiguous.
            if block_ids is None or len(block_ids) == 0:
                clamped.append(block_ids)
                continue

            # `num_blocks` and the row width are both counted in kernel blocks,
            # while the incoming IDs are KV-manager blocks that expand to
            # `blocks_per_kv_block` kernel blocks each.
            blocks_per_kv_block = self.blocks_per_kv_block[group]
            row_capacity = self.block_tables[group].gpu.shape[1]
            start = 0 if overwrite else int(self.num_blocks.np[group, req_index])
            remaining_kv_blocks = max(row_capacity - start, 0) // blocks_per_kv_block

            if len(block_ids) > remaining_kv_blocks:
                block_ids = block_ids[:remaining_kv_blocks]
                clamped_groups += 1
            clamped.append(block_ids)

        if clamped_groups and not _clamp_warned:
            _clamp_warned = True
            psrl_logger.warning(
                "Clamped block-table append for request %s in %d group(s): the "
                "runner's block IDs had drifted past the row capacity. The "
                "truncated tail is never addressed. This indicates a scheduler/"
                "runner desync after a KV load failure; report it upstream.",
                req_index,
                clamped_groups,
            )

        return _original(self, req_index, tuple(clamped), overwrite)

    target.append_block_ids = append_block_ids
    setattr(target, _SENTINEL, True)

    psrl_logger.info("Patched BlockTables.append_block_ids to clamp appends to the row capacity.")
