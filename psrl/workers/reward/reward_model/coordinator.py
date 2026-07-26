import asyncio
import logging
import os

from psrl.workers.gen.rollout_coordination import RolloutCoordinator

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class RewardModelCoordinator(RolloutCoordinator):
    """
    Coordinator for reward-model replicas.

    Reuses from RolloutCoordinator:
    - ``_command_handler_loop()``    — ABORT / SLEEP / WAKE_UP command dispatch
    - ``_process_status_queue()``    — ZMQ engine-stats collection (for elastic RM)
    - ``add_worker()``               — replica registration
    - ``get_status_sink_endpoint()`` — ZMQ bind address for servers to push stats

    Overrides (no-op or reward-specific):
    - ``start_busy_loop()``          — skips rollout-router sync and PS sync tasks
    - ``sync_model()``               — reward model weights never update
    - ``update_model_version()``     — no version tracking needed
    - ``_greedy_sync_and_migrate_loop()``       — no sync loop
    - ``_status_based_sync_and_migrate_loop()`` — no sync loop
    - ``_sync_status_to_router()``   — no router to sync to
    """

    def __init__(
        self,
        config,
        rm_config,
        rollout_gateway_url: str,
    ) -> None:
        # ps_manager=None: no PS handle needed for reward models.
        super().__init__(
            config=config,
            ps_manager=None,
            rollout_gateway_url=rollout_gateway_url,
        )
        self.rm_config = rm_config
        self.reward_model_name = rm_config.reward_model_name
        psrl_logger.info("RewardModelCoordinator initialized for model_name=%s", self.reward_model_name)

    # ── Reward-specific start_busy_loop ────────────────────────────────────

    async def start_busy_loop(self) -> None:
        """
        Start background tasks for reward model coordination.

        Unlike RolloutCoordinator.start_busy_loop():
        - Does NOT start _sync_status_to_router (no router handle)
        - Does NOT start sync/migration loops (weights are static)
        - Does NOT read sync_and_mig_strategy config (not applicable for reward models)
        """
        if self.command_handler_task is not None and not self.command_handler_task.done():
            return

        self.running_loop = asyncio.get_running_loop()
        self.command_handler_task = self.running_loop.create_task(self._command_handler_loop())
        self.command_handler_task.add_done_callback(lambda f: f.result())

        if self.config.psrl.status_collection.enable:
            self.process_status_queue_task = self.running_loop.create_task(self._process_status_queue())
            self.process_status_queue_task.add_done_callback(lambda f: f.result())

        psrl_logger.info("RewardModelCoordinator busy loop started for model_name=%s", self.reward_model_name)

    # ── No-op PS-sync and router-sync overrides ────────────────────────────

    async def sync_model(self, *args, **kwargs) -> None:
        """Reward model weights are static; weight sync is a no-op."""

    async def update_model_version(self, model_version: int, *args, **kwargs) -> None:
        """No model version tracking for static reward models."""

    async def _greedy_sync_and_migrate_loop(self) -> None:
        """No sync/migration loop for static reward models."""

    async def _status_based_sync_and_migrate_loop(self) -> None:
        """No sync/migration loop for static reward models."""

    async def _sync_status_to_router(self) -> None:
        """No router to sync engine status to for reward models."""

    # ── Elastic RM: sleep/wake hooks ───────────────────────────────────────

    def _get_sleep_level(self) -> int:
        """Reward model sleep level=1: releases KV cache but retains model weights in GPU memory."""
        return 1

    async def _do_sleep_instance(self, replica_id: str) -> None:
        """Reward model sleep: calls server.sleep() (non-nixl path; no NIXL deregistration needed)."""
        await self.server_handles[replica_id].sleep.remote(
            level=self._get_sleep_level(),
        )

    async def _do_wake_up_instance(self, replica_id: str) -> None:
        """Reward model wake_up: calls server.wake_up() (non-nixl path)."""
        await self.server_handles[replica_id].wake_up.remote()

    # ── Elastic RM: gateway backlog monitoring ─────────────────────────────

    async def get_router_backlog_size(self) -> int:
        """
        Return the total number of in-flight requests across all smg gateway workers.

        Uses GET /workers → sums ``load`` field per worker (``load`` is the
        active request counter maintained by smg for round-robin scheduling).
        Falls back to 0 on any error so elastic RM degrades gracefully.
        """
        if not self.rollout_gateway_url:
            return 0
        try:
            data = await self._gateway_get_json("/workers")
            workers = data.get("workers", [])
            return sum(int(w.get("load", 0)) for w in workers)
        except Exception:
            psrl_logger.debug("RewardModelCoordinator.get_router_backlog_size: gateway query failed, returning 0")
            return 0
