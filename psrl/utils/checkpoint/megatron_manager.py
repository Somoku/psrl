"""A Megatron checkpoint manager that writes per-rank files instead of ``torch_dist``.

Direct supply (``ps_mode=nixl_direct``) needs the NIXL progress thread running on the
train ranks, because rollouts RDMA-read them while they compute and that path waits for
``DONE``. Megatron's distributed checkpointing cannot coexist with it: its
``all_gather_object`` corrupts the UCX endpoint addresses the thread reads, which is why
``checkpoint.use_dcp_save`` turns the thread off.

So a direct-supply run that also saves checkpoints needs a saver that does not use
``dist_checkpointing``. ``psrl.utils.checkpoint.megatron_saver`` is that saver -- it
writes one ``torch.save`` file per rank -- and this class is what routes verl's manager
through it.

This used to be a patch applied inside the vendored verl checkout: that copy's
``MegatronCheckpointManager`` took a ``use_per_rank_checkpoint`` flag and called
``psrl.utils.checkpoint`` from inside verl. Upstream verl has neither, and re-applying the
patch would put it back on the list of things that silently rot across a verl bump.
Subclassing keeps the behaviour in psrl and depends on only two seams:
``_save_dist_checkpoint``, and the three ``load_dist_checkpointing`` calls that
``load_checkpoint`` makes.

**The two formats are not interchangeable.** A checkpoint written per-rank can only be
read back by the same world size and parallelism layout, where a ``torch_dist`` one can be
resharded. That is a real cost of using direct supply with saving enabled, not an
implementation detail, and it is why the mode is chosen by configuration rather than
inferred.
"""

from __future__ import annotations

import logging
from typing import Any

from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager

from psrl.utils.checkpoint.megatron_saver import (
    load_megatron_checkpoint,
    save_megatron_checkpoint,
)

logger = logging.getLogger(__name__)


class PSRL_PerRankMegatronCheckpointManager(MegatronCheckpointManager):
    """Writes and reads per-rank checkpoints, bypassing Megatron dist_checkpointing.

    Drop-in for verl's manager: it overrides only how bytes reach disk, leaving the
    directory layout, the state-dict composition and the retention policy alone.
    """

    def _save_dist_checkpoint(self, dist_checkpoint_path: str, state_dict: dict) -> None:
        """Write ``state_dict`` as one file per rank.

        Returns ``None`` rather than an async save request: the per-rank saver is
        synchronous, and reporting a request that does not exist would make the caller
        wait on it.
        """
        save_megatron_checkpoint(
            sharded_state_dict=state_dict,
            ckpt_path=dist_checkpoint_path,
            async_save=False,
            metadata=self._build_sharded_state_dict_metadata(),
        )
        logger.info("Saved per-rank Megatron checkpoint to %s", dist_checkpoint_path)
        return None

    def load_checkpoint(self, *args: Any, **kwargs: Any):
        """Run verl's load with `load_dist_checkpointing` redirected to the per-rank reader.

        Patching the module attribute for the duration of the call, rather than
        reproducing `load_checkpoint`: that method is long, it reads model, optimizer and
        extra state through three separate calls, and copying it would mean re-copying it
        at the next verl bump -- the exact failure this class exists to avoid.
        """
        import verl.utils.checkpoint.megatron_checkpoint_manager as verl_manager

        original = verl_manager.load_dist_checkpointing
        verl_manager.load_dist_checkpointing = _load_per_rank
        try:
            return super().load_checkpoint(*args, **kwargs)
        finally:
            verl_manager.load_dist_checkpointing = original


def _load_per_rank(sharded_state_dict, ckpt_dir, **kwargs):
    """Adapter matching `load_dist_checkpointing`'s keyword shape."""
    return load_megatron_checkpoint(
        sharded_state_dict=sharded_state_dict,
        ckpt_dir=ckpt_dir,
    )
