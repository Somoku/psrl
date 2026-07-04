"""StatusMixin: engine-status queue processing and router sync loops."""

from __future__ import annotations

import asyncio
import logging
import os

from psrl.workers.gen.smg_adapter import (
    ROUTING_LOOP_STATUS_PATH,
    WORKERS_STATS_PATH,
    WORKERS_UPDATE_STATS_PATH,
    build_worker_stats_update,
)
from psrl.workers.gen.utils import RolloutInstanceId
from psrl.workers.gen.stats_collector import EngineStats

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

STATUS_QUEUE_POLL_TIMEOUT_SECS = 0.2


class StatusMixin:
    """Provides status-collection and router-sync loops for RolloutCoordinator.

    Expects ``self`` to carry: ``status_queue``, ``replica_idx_to_replica_id``,
    ``instance_to_engine_status``, ``stop_process_status_queue``,
    ``stop_sync_status_to_router``, ``stop_stats_recorder``,
    ``rollout_router``, ``use_rust_gateway``, ``config``,
    ``_stats_recorder``, and the ``_gateway_post_json`` helper.
    """

    async def _process_status_queue(self):
        psrl_logger.info("Starting to process ZMQ status stream")
        while not self.stop_process_status_queue:
            # Naive implementation version
            """
            recv_stats = await self.status_queue.get_async(block=True, timeout=None)
            replica_id = self.replica_idx_to_replica_id.get(recv_stats.replica_idx)
            if replica_id is None:
                raise RuntimeError(f"Received engine stats with unknown replica_idx {recv_stats.replica_idx}")
            instance_id = (replica_id, recv_stats.data_parallel_rank)
            self.instance_to_engine_status[instance_id] = recv_stats
            psrl_logger.debug(
                f"Updated engine status for instance "
                f"{instance_id}: {self.instance_to_engine_status[instance_id]} "
            )
            """

            # Coalesce implementation version
            try:
                recv_stats = await self.status_queue.get_async(
                    block=True,
                    timeout=STATUS_QUEUE_POLL_TIMEOUT_SECS,
                )
            except TimeoutError:
                # Periodically wake up so shutdown flag can be observed even when no messages arrive.
                continue
            latest_by_instance: dict[RolloutInstanceId, EngineStats] = {}

            # Put first blocking message
            replica_id = self.replica_idx_to_replica_id.get(recv_stats.replica_idx)
            if replica_id is None:
                raise RuntimeError(f"Received engine stats with unknown replica_idx {recv_stats.replica_idx}")
            instance_id = (replica_id, recv_stats.data_parallel_rank)
            latest_by_instance[instance_id] = recv_stats

            # Drain all currently queued messages and coalesce by instance
            while True:
                next_stats = await self.status_queue.get_async(block=False)
                if next_stats is None:
                    break
                replica_id = self.replica_idx_to_replica_id.get(next_stats.replica_idx)
                if replica_id is None:
                    raise RuntimeError(f"Received engine stats with unknown replica_idx {next_stats.replica_idx}")
                instance_id = (replica_id, next_stats.data_parallel_rank)
                latest_by_instance[instance_id] = next_stats

            # Apply only the newest snapshot per instance
            self.instance_to_engine_status.update(latest_by_instance)
        psrl_logger.info("Stopped processing ZMQ status stream.")

    def get_instance_engine_status_snapshot(self) -> dict[RolloutInstanceId, dict]:
        """Return a lightweight snapshot map for elastic scaling decisions."""
        snapshot = {}
        for instance_id, engine_status in self.instance_to_engine_status.items():
            snapshot[instance_id] = engine_status.snapshot
        return snapshot

    async def _sync_status_to_router(self):
        """Broadcast the engine status to the router."""
        assert self.rollout_router is not None, "Rollout router is not set in RolloutCoordinator"

        while not self.stop_sync_status_to_router:
            # Broadcast the engine status to the router every coordinator sync interval
            await asyncio.sleep(self.config.psrl.status_collection.coordinator_sync_interval_in_ms / 1000)
            if self.use_rust_gateway:
                if not self.instance_to_engine_status:
                    continue
                updates = []
                for instance_id, engine_status in self.instance_to_engine_status.items():
                    replica_id, dp_rank = instance_id
                    updates.append(build_worker_stats_update(replica_id, dp_rank, engine_status.snapshot))
                await self._gateway_post_json(WORKERS_UPDATE_STATS_PATH, payload=updates)
            else:
                await self.rollout_router.update_instance_status.remote(self.instance_to_engine_status)
        psrl_logger.info("Stopped syncing engine status to router.")

    async def _stats_recorder_loop(self):
        """Periodically snapshot per-replica stats to JSONL files."""
        psrl_logger.info("Starting stats recorder loop")
        interval = self.config.psrl.status_collection.stats_recorder.interval_in_s
        while not self.stop_stats_recorder:
            await asyncio.sleep(interval)
            if not self.stop_stats_recorder:
                self._stats_recorder.record(self.instance_to_engine_status)
                routing_loop_status = await self._gateway_get_json(ROUTING_LOOP_STATUS_PATH)
                workers_stats = await self._gateway_get_json(WORKERS_STATS_PATH)
                self._stats_recorder.record_smg_routing_status(routing_loop_status, workers_stats)
        psrl_logger.info("Stopped stats recorder loop.")
