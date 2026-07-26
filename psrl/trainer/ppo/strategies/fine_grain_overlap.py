"""FineGrainOverlapStrategy: overlap per-sample stages with ongoing rollout.

recompute scope: per-sample stages (old_log_prob/ref/values/reward) run
  per chunk as chunks arrive; advantage + updates run on the full batch.
  Math is IDENTICAL to FullBatchStepStrategy for all advantage estimators.

pre_step + mini_batch scope: each chunk IS one mini-batch; advantage and
  one optimizer step run per chunk.  Exact for GRPO (group-local normalization);
  approximate for GAE/REINFORCE++/GDPO (masked_whiten scope changes).
  PS weight push is deferred until the last chunk so ``maybe_delete_buffer``
  does not tear down the current buffer mid-step.

``run_step`` switches to trainer mode immediately, then pulls chunks from the
manager as they become available, so per-sample GPU work on chunk N overlaps
with rollout generating chunk N+1.

TODO (future): pre_step + micro_batch scope — true cross-chunk gradient
  accumulation where optimizer.step() fires at mini-batch boundaries.
  Requires: (1) new actor_grad_zero/actor_accumulate_grad/actor_optimizer_step
  RPCs on engine_train_worker; (2) batch_num_tokens_override in veRL's FSDP
  forward_backward_batch to share the full-mini-batch loss denominator across
  chunks; (3) ppo_epochs == 1 constraint.  Currently guarded by ValueError.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import ray
from transfer_queue.metadata import KVBatchMeta
from verl.utils.debug import marked_timer

from psrl.trainer.ppo.strategies.base import StepStrategy, psrl_logger
from psrl.utils.config import resolve_fine_grain_chunk_size
from psrl.utils.logger import EventType, log_dual_events

if TYPE_CHECKING:
    from psrl.trainer.ppo.ray_trainer import PSRL_RayPPOTrainer


class FineGrainOverlapStrategy(StepStrategy):
    """Overlap training stages with rollout using chunk-level pipelining.

    ``run_step`` switches to trainer mode immediately (no blocking on the
    full batch), then iterates over chunks yielded by the manager, running
    per_sample stages on each chunk concurrently with the rollout generating
    the next chunk.

    overlap_scope:
      recompute  — only per_sample stages overlap; advantage+update on full batch.
      pre_step   — advantage+update also per chunk (mini_batch granularity only;
                   micro_batch is a future phase).
    """

    def __init__(self, trainer: PSRL_RayPPOTrainer, cfg) -> None:
        """Initialize and register the chunk size with the manager.

        Args:
            trainer: The ``PSRL_RayPPOTrainer`` instance.
            cfg: The ``psrl.fine_grain_overlap`` config sub-tree (OmegaConf node).
        """
        super().__init__(trainer)
        t = trainer
        dp_size = t._get_dp_size(t.actor_wg, "actor")
        self.effective_granularity, self.chunk_groups = resolve_fine_grain_chunk_size(t.config, dp_size)
        self.overlap_scope = str(cfg.get("overlap_scope", "recompute"))

        if self.overlap_scope == "pre_step" and self.effective_granularity != "mini_batch":
            raise ValueError(
                f"pre_step scope requires mini_batch granularity (micro_batch pre_step is Phase 4, "
                f"not yet implemented); got effective_granularity={self.effective_granularity!r}. "
                "Use overlap_scope=recompute with micro_batch, or reduce multiplier so chunk "
                "clamps to mini_batch."
            )

        ray.get(t.agent_loop_manager.set_chunk_size.remote(self.chunk_groups))
        psrl_logger.info(
            "FineGrainOverlapStrategy initialized: granularity=%r chunk_groups=%d overlap_scope=%r.",
            self.effective_granularity,
            self.chunk_groups,
            self.overlap_scope,
        )

    def run_step(self, buffer_id: int, metrics: dict, timing_raw: dict):
        """Pipeline per_sample stages over chunks, then run full-batch updates.

        For ``recompute`` scope: per_sample stages run per chunk; advantage
        and optimizer updates run once on the concatenated full batch.

        For ``pre_step + mini_batch`` scope: advantage and one optimizer step
        also run per chunk immediately after per_sample stages.

        Args:
            buffer_id: Training buffer ID (``global_steps - 1``).
            metrics: Accumulator dict updated in-place by phase methods.
            timing_raw: Timing dict for ``marked_timer`` instrumentation.

        Returns:
            KVBatchMeta: The concatenated full batch after all phases complete.
        """
        t = self.trainer

        # Switch to trainer mode before any GPU work. Unlike FullBatchStepStrategy,
        # we do not block on the full batch here — chunks arrive as rollout progresses.
        t.switch_to_trainer_mode()

        chunks: list[KVBatchMeta] = []
        chunk_idx = 0

        while True:
            # Use warning so these show under default PSRL_LOGGING_LEVEL=WARN.
            psrl_logger.warning(
                "FineGrainOverlap: waiting for buffer=%d chunk=%d",
                buffer_id,
                chunk_idx,
            )
            chunk_meta, is_last = ray.get(t.agent_loop_manager.wait_for_training_chunk.remote(buffer_id, chunk_idx))
            psrl_logger.warning(
                "FineGrainOverlap: got buffer=%d chunk=%d size=%d is_last=%s; sampling replay buffer",
                buffer_id,
                chunk_idx,
                len(chunk_meta),
                is_last,
            )

            t.replay_buffer.sample(chunk_meta.keys, chunk_meta.partition_id)
            psrl_logger.warning(
                "FineGrainOverlap: replay sample ready for buffer=%d chunk=%d",
                buffer_id,
                chunk_idx,
            )

            if t.config.trainer.balance_batch:
                chunk_meta = t._balance_batch(chunk_meta, metrics=metrics)

            chunk_meta.extra_info["temperature"] = t.config.gen_actor_rollout_ref.rollout.temperature
            chunk_meta.extra_info["global_steps"] = t.global_steps

            # --- per_sample stages (overlap with rollout generating the next chunk) ---
            with marked_timer("old_log_prob", timing_raw, color="orange"):
                chunk_meta = t._compute_old_log_prob(chunk_meta, metrics=metrics)

            if t.use_reference_policy:
                with marked_timer("ref", timing_raw, color="olive"):
                    chunk_meta = t._compute_ref_log_prob(chunk_meta, metrics=metrics)

            if t.use_critic:
                with marked_timer("values", timing_raw, color="cyan"):
                    chunk_meta = t._compute_values(chunk_meta, metrics=metrics)

            if t.config.reward.launch_reward_fn_async:
                with marked_timer("async_reward_get", timing_raw, color="yellow"):
                    chunk_meta = ray.get(t.reward_manager.wait_for_reward_of_requests.remote(chunk_meta))
            else:
                chunk_meta = ray.get(t.reward_manager.normalize_reward.remote(chunk_meta))

            # --- pre_step + mini_batch: per-chunk advantage + optimizer step ---
            if self.overlap_scope == "pre_step" and self.effective_granularity == "mini_batch":
                with marked_timer("adv", timing_raw, color="brown"):
                    with log_dual_events(
                        "Compute advantage",
                        psrl_logger,
                        event_type=EventType.OTHER,
                    ):
                        chunk_meta = t._compute_advantage(chunk_meta, metrics=metrics)

                if t.use_critic:
                    with marked_timer("update_critic", timing_raw, color="pink"):
                        with log_dual_events(
                            "Update critic",
                            psrl_logger,
                            event_type=EventType.TRAIN,
                        ):
                            chunk_meta = t._update_critic(chunk_meta, metrics=metrics)

                if t.config.trainer.critic_warmup <= t.global_steps:
                    # Push only on the last chunk. Intermediate pushes advance the
                    # PS version and trigger maybe_delete_buffer(version-1), which
                    # tears down the current training buffer before remaining
                    # chunks are consumed (deadlock / silent stall).
                    with marked_timer("update_actor", timing_raw, color="red"):
                        with log_dual_events(
                            f"Update actor (chunk {chunk_idx}, push={is_last})",
                            psrl_logger,
                            event_type=EventType.TRAIN,
                        ):
                            chunk_meta = t._update_actor(chunk_meta, metrics=metrics, push_model=is_last)
                            # Ephemeral per-chunk control flag; must not survive into
                            # KVBatchMeta.concat (False on intermediate, True on last).
                            chunk_meta.extra_info.pop("push_model", None)

            chunks.append(chunk_meta)
            chunk_idx += 1
            if is_last:
                break

        full_batch = KVBatchMeta.concat(chunks)

        # --- recompute scope: advantage + updates run on the full batch ---
        if self.overlap_scope == "recompute":
            with marked_timer("adv", timing_raw, color="brown"):
                with log_dual_events(
                    "Compute advantage",
                    psrl_logger,
                    event_type=EventType.OTHER,
                ):
                    full_batch = t._compute_advantage(full_batch, metrics=metrics)

            if t.use_critic:
                with marked_timer("update_critic", timing_raw, color="pink"):
                    with log_dual_events(
                        "Update critic",
                        psrl_logger,
                        event_type=EventType.TRAIN,
                    ):
                        full_batch = t._update_critic(full_batch, metrics=metrics)

            if t.config.trainer.critic_warmup <= t.global_steps:
                with marked_timer("update_actor", timing_raw, color="red"):
                    with log_dual_events(
                        "Update actor",
                        psrl_logger,
                        event_type=EventType.TRAIN,
                    ):
                        full_batch = t._update_actor(full_batch, metrics=metrics)

        self._run_ckpt_and_validate(
            full_batch,
            metrics,
            timing_raw,
            actor_updated=(t.config.trainer.critic_warmup <= t.global_steps),
        )

        return full_batch
