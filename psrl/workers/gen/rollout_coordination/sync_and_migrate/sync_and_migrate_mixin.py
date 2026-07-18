"""SyncAndMigrateMixin: shared sync/migrate helpers for RolloutCoordinator."""

from __future__ import annotations

import asyncio
import logging
import os

from psrl.utils.logger import EventType, log_dual_events
from psrl.utils.server.command import Command, CommandType
from psrl.workers.gen.smg_adapter import (
    WORKERS_UPDATE_WEIGHT_VERSION_PATH,
    build_pause_resume_payload,
    build_weight_version_updates,
)
from psrl.workers.gen.utils import RolloutInstanceId

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class SyncAndMigrateMixin:
    """Sync/migrate helpers shared by greedy and status-based loops.

    Expects ``self`` to carry: ``instance_ids``, ``server_handles``,
    ``instance_to_model_version``, ``instance_to_latest_stale_model_version``,
    ``instance_to_version_after_sync``, ``model_sync_tasks``, ``replica_sync_tasks``,
    ``ps_model_version``, ``ps_manager``,
    ``config``, ``command_queue``, and the ``_gateway_post/get_json``,
    ``_set_routing_loop_running``, ``exec_command``, ``set_rollout_instance_model_version``,
    ``_broadcast_kv_current_version`` methods.
    """

    def _expand_replica_instance_ids(self, instance_ids: list[RolloutInstanceId]) -> list[RolloutInstanceId]:
        replica_ids = {instance_id[0] for instance_id in instance_ids}
        return sorted(instance_id for instance_id in self.instance_ids if instance_id[0] in replica_ids)

    async def _publish_weight_version_updates(self, updates: list[dict]) -> None:
        result = await self._gateway_post_json(WORKERS_UPDATE_WEIGHT_VERSION_PATH, payload=updates)
        if int(result.get("rejected", 0)) != 0 or int(result.get("updated", 0)) != len(updates):
            raise RuntimeError(f"SMG gateway rejected replica weight-version update: {result}.")

    async def _finish_gateway_sync(
        self,
        replica_ids: list[str],
        instance_ids: list[RolloutInstanceId],
        advertised_version: int,
        pull_futures: list,
    ) -> None:
        try:
            actual_versions = await asyncio.gather(*pull_futures)
            actual_updates = []
            for replica_id, actual_version in zip(replica_ids, actual_versions):
                replica_instance_ids = [instance_id for instance_id in instance_ids if instance_id[0] == replica_id]
                for instance_id in replica_instance_ids:
                    self.set_rollout_instance_model_version(instance_id, actual_version)
                    self.instance_to_version_after_sync[instance_id] = actual_version
                if actual_version != advertised_version:
                    actual_updates.extend(build_weight_version_updates(replica_instance_ids, actual_version))
            if actual_updates:
                await self._publish_weight_version_updates(actual_updates)
            await asyncio.gather(
                *[self.server_handles[replica_id].resume_after_sync.remote() for replica_id in replica_ids]
            )
        except Exception:
            psrl_logger.exception(f"Model sync failed for replicas {replica_ids}; keeping them unavailable")
            await asyncio.gather(
                *[self.server_handles[replica_id].fail_sync.remote() for replica_id in replica_ids],
                return_exceptions=True,
            )
            await self._gateway_post_json("/workers/pause", payload=build_pause_resume_payload(instance_ids))
            raise

    async def _quarantine_failed_replicas(
        self,
        replica_ids: list[str],
        instance_ids: list[RolloutInstanceId],
    ) -> None:
        await asyncio.gather(
            *[self.server_handles[replica_id].fail_sync.remote() for replica_id in replica_ids],
            return_exceptions=True,
        )
        await self._gateway_post_json("/workers/pause", payload=build_pause_resume_payload(instance_ids))
        await self._set_routing_loop_running(True)

    async def _wait_for_replica_syncs(self, replica_ids: list[str]) -> None:
        pending = {
            self.replica_sync_tasks[replica_id] for replica_id in replica_ids if replica_id in self.replica_sync_tasks
        }
        if pending:
            await asyncio.gather(*pending)

    def _track_model_sync_task(self, task: asyncio.Task, replica_ids: list[str]) -> None:
        self.model_sync_tasks.add(task)
        for replica_id in replica_ids:
            self.replica_sync_tasks[replica_id] = task

        def on_done(completed: asyncio.Task) -> None:
            self.model_sync_tasks.discard(completed)
            for replica_id in replica_ids:
                if self.replica_sync_tasks.get(replica_id) is completed:
                    del self.replica_sync_tasks[replica_id]
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(on_done)

    async def _fetch_filtered_request_meta(self, version_tag: int) -> list[tuple[int, bool]]:
        requests = await self._gateway_get_json("/routing_loop/filter", params={"version_tag": version_tag})
        filtered_request_meta: list[tuple[int, bool]] = []
        for req in requests:
            request_id = int(req.get("request_id"))
            if request_id is None:
                continue
            is_validate = bool(req.get("is_validate", False))
            filtered_request_meta.append((request_id, is_validate))
        return filtered_request_meta

    async def sync_with_ps(
        self,
        instance_ids: list[RolloutInstanceId],
        wait_model_sync: bool = False,
        wait_interrupted_partial_requests_loop_back: bool = True,
    ):
        """
        Synchronize with PS for the given instance IDs.
        """
        # Add batching SYNC command to the command queue to interrupt the instance
        # This will stop the instance, pull the model weights from PS, and resume generation.
        # But this will not block the current loop.
        # NOTE(lhy): we don't need to update the instance version here because the version
        # is updated in the `sync_with_ps` method of the GenWorker
        # when calling `pull_model` or `pull_model_async` from the GenWorker, the ps manager
        # will update the instance version.
        # However, we need to update the latest stale model version here to avoid stale stats
        # being handled after the synchronization.
        with log_dual_events(
            f"Synchronize rollout instances {instance_ids} with PS "
            f"(model pull is {'non-blocking' if not wait_model_sync else 'blocking'} "
            f"for the coordinator)",
            psrl_logger,
            level=logging.INFO,
            event_type=EventType.OTHER,
        ):
            instance_ids = self._expand_replica_instance_ids(instance_ids)
            replica_ids = sorted({instance_id[0] for instance_id in instance_ids})
            sleeping = await asyncio.gather(
                *[self.server_handles[replica_id].is_sleeping.remote() for replica_id in replica_ids]
            )
            replica_ids = [replica_id for replica_id, is_sleeping in zip(replica_ids, sleeping) if not is_sleeping]
            instance_ids = [instance_id for instance_id in instance_ids if instance_id[0] in replica_ids]
            if not replica_ids:
                return
            await self._wait_for_replica_syncs(replica_ids)
            for instance_id in instance_ids:
                self.instance_to_latest_stale_model_version[instance_id] = self.instance_to_model_version.get(
                    instance_id, 0
                )

            await self._set_routing_loop_running(False)
            try:
                await asyncio.gather(
                    *[self.server_handles[replica_id].pause_for_sync.remote() for replica_id in replica_ids]
                )
                if (
                    wait_interrupted_partial_requests_loop_back
                    and self.config.psrl.rollout_coordination.partial_rollout.enable
                ):
                    await self._wait_interrupted_partial_requests_loop_back(instance_ids)

                if self.config.psrl.lmcache.multi_version_kv:
                    await self._broadcast_kv_current_version(self.ps_model_version)

                pull_futures = [
                    self.server_handles[replica_id].pull_model_for_sync.remote(self.ps_model_version)
                    for replica_id in replica_ids
                ]
                updates = build_weight_version_updates(instance_ids, self.ps_model_version)
                await self._publish_weight_version_updates(updates)
                for instance_id in instance_ids:
                    self.instance_to_version_after_sync[instance_id] = self.ps_model_version
                await self._set_routing_loop_running(True)
                psrl_logger.info(
                    f"Published version {self.ps_model_version} and resumed routing for replicas {replica_ids!r}."
                )
            except Exception:
                await self._quarantine_failed_replicas(replica_ids, instance_ids)
                raise

            completion = self._finish_gateway_sync(
                replica_ids,
                instance_ids,
                self.ps_model_version,
                pull_futures,
            )
            if wait_model_sync:
                await completion
            else:
                self._track_model_sync_task(asyncio.create_task(completion), replica_ids)

    async def check_no_activate_tasks(self, instance_id: RolloutInstanceId) -> bool:
        """
        Check whether the instance has no active tasks.
        """
        replica_id, data_parallel_rank = instance_id
        active_task_num = await self.server_handles[replica_id].get_active_task_num.remote(data_parallel_rank)
        return active_task_num == 0

    async def check_should_sync(self, instance_id: RolloutInstanceId) -> bool:
        """
        Check whether to synchronize with PS for the instance.
        """
        assert self.config.psrl.rollout_coordination.sync_and_mig_strategy.method == "status_based", (
            "Partial rollout is only supported for status-based sync strategy"
        )
        assert self.config.psrl.status_collection.enable, (
            "Partial rollout is only supported when status collection is enabled"
        )
        assert self.config.psrl.rollout_coordination.partial_rollout.enable, "Partial rollout is not enabled"

        return await self._check_should_sync(instance_id)

    async def _check_should_sync(self, instance_id: RolloutInstanceId) -> bool:
        instance_status = self.instance_to_engine_status[instance_id]
        if instance_status.get_waiting_queue_size() > 0:
            return False

        # 1. Check if there are any requests version satisfies the condition before synchronization
        current_instance_version = self.instance_to_version_after_sync[instance_id]
        if await self.ps_manager.check_aborted_model_versions.remote(current_instance_version):
            filtered_request_ids = []
        else:
            filtered_request_meta = await self._fetch_filtered_request_meta(current_instance_version)
            filtered_request_ids = [request_meta[0] for request_meta in filtered_request_meta]

        # 2. Check if there are any requests
        # that can be RESERVED for the instance but no need to reserve new entry
        # before synchronization
        if len(filtered_request_ids) > 0:
            is_aborted = await self.ps_manager.check_aborted_requests.remote(filtered_request_ids, remove=False)
            filtered_request_ids = [
                request_id for i, request_id in enumerate(filtered_request_ids) if not is_aborted[i]
            ]
            can_reserve_without_new_reserve_entry = await self.ps_manager.can_reserve_request.remote(
                filtered_request_ids,
                [current_instance_version],
                without_new_reserve_entry=True,
            )
            filtered_request_ids = [
                request_id
                for i, request_id in enumerate(filtered_request_ids)
                if can_reserve_without_new_reserve_entry[i] == [True]
            ]

        # If there are requests that can still be routed to the instance
        # before synchronization without new reserve entry
        # we will not attempt to synchronize with PS
        if (
            len(filtered_request_ids) > 0
            and self.config.psrl.rollout_coordination.sync_and_mig_strategy.sync.check_req_before_sync
        ):
            return False

        # 3. Check indicator to determine whether to synchronize with PS
        if self.config.psrl.rollout_coordination.sync_and_mig_strategy.sync.indicator == "request_num":
            # Check whether request num is above threshold
            request_num = instance_status.get_waiting_and_running_queue_size()
            psrl_logger.debug(
                f"Instance {instance_id} (version {self.instance_to_version_after_sync[instance_id]}) "
                f"request_num: {request_num}, "
                f"threshold: {self.config.psrl.rollout_coordination.sync_and_mig_strategy.sync.threshold}"
            )
            if request_num > self.config.psrl.rollout_coordination.sync_and_mig_strategy.sync.threshold:
                return False
        elif self.config.psrl.rollout_coordination.sync_and_mig_strategy.sync.indicator == "throughput":
            # Check whether throughput is above threshold
            throughput = self.instance_to_engine_status[instance_id].get_generation_throughput()
            psrl_logger.debug(
                f"Instance {instance_id} (version {self.instance_to_version_after_sync[instance_id]}) "
                f"throughput: {throughput}, "
                f"threshold: {self.config.psrl.rollout_coordination.sync_and_mig_strategy.sync.threshold}"
            )
            if throughput > self.config.psrl.rollout_coordination.sync_and_mig_strategy.sync.threshold:
                return False
        elif self.config.psrl.rollout_coordination.sync_and_mig_strategy.sync.indicator == "kv_cache":
            # Check whether KV Cache is above threshold
            kv_cache_utilization = instance_status.get_kv_cache_utilization()
            psrl_logger.debug(
                f"Instance {instance_id} (version {self.instance_to_version_after_sync[instance_id]}) "
                f"kv_cache_utilization: {kv_cache_utilization}, "
                f"threshold: {self.config.psrl.rollout_coordination.sync_and_mig_strategy.sync.threshold}"
            )
            if kv_cache_utilization > self.config.psrl.rollout_coordination.sync_and_mig_strategy.sync.threshold:
                return False
        else:
            raise ValueError(
                f"Unknown sync indicator: {self.config.psrl.rollout_coordination.sync_and_mig_strategy.sync.indicator}"
            )
        return True

    async def check_and_migrate(self, wait_interrupted_partial_requests_loop_back: bool = True):
        """
        Check if any instance is starving and do migration if necessary.
        """
        assert self.config.psrl.rollout_coordination.sync_and_mig_strategy.mig.enable, (
            "Rollout migration is not enabled"
        )
        assert self.config.psrl.status_collection.enable, (
            "Rollout migration is only supported when status collection is enabled"
        )

        migrate_instance_ids = await self._check_should_migrate()

        if migrate_instance_ids:
            with log_dual_events(
                f"Migrating instances {migrate_instance_ids}",
                psrl_logger,
                level=logging.INFO,
                event_type=EventType.OTHER,
            ):
                await self._set_routing_loop_running(False)
                psrl_logger.info("Paused routing for migration")
                await self.exec_command(
                    Command(
                        type=CommandType.ABORT,
                        instance_ids=migrate_instance_ids,
                    ),
                    blocking=True,
                )
                if wait_interrupted_partial_requests_loop_back:
                    await self._wait_interrupted_partial_requests_loop_back(migrate_instance_ids)
                    psrl_logger.info(
                        f"All interrupted requests on the migrated instances "
                        f"{migrate_instance_ids} have been looped back"
                    )
                await self._set_routing_loop_running(True)
                psrl_logger.info("Resumed routing after migration")

    async def _check_should_migrate(self) -> list[RolloutInstanceId]:
        filtered_instance_ids = []
        instance_to_status = self.instance_to_engine_status
        for instance_id in self.instance_ids:
            if instance_to_status[instance_id].get_waiting_queue_size() != 0:
                continue
            instance_version = self.instance_to_version_after_sync[instance_id]
            if await self.ps_manager.check_aborted_model_versions.remote(instance_version):
                continue
            filtered_request_meta = await self._fetch_filtered_request_meta(instance_version)
            filtered_request_ids = [meta[0] for meta in filtered_request_meta]
            if len(filtered_request_ids) > 0:
                is_aborted = await self.ps_manager.check_aborted_requests.remote(filtered_request_ids, remove=False)
                filtered_request_ids = [
                    request_id for i, request_id in enumerate(filtered_request_ids) if not is_aborted[i]
                ]
                is_validate_list = [request_meta[1] for request_meta in filtered_request_meta]
                can_reserve = await self.ps_manager.can_reserve_request.remote(
                    filtered_request_ids,
                    [instance_version],
                    without_new_reserve_entry=False,
                    is_validate=is_validate_list,
                )
                filtered_request_ids = [
                    request_id for i, request_id in enumerate(filtered_request_ids) if can_reserve[i] == [True]
                ]
            if len(filtered_request_ids) == 0:
                filtered_instance_ids.append(instance_id)

        candidate_migrate_instance_ids = []  # (instance_id, ratio)
        for starved_instance_id in filtered_instance_ids:
            for instance_id in self.instance_ids:
                if instance_id == starved_instance_id:
                    continue
                if (
                    self.instance_to_version_after_sync[instance_id]
                    > self.instance_to_version_after_sync[starved_instance_id]
                ):
                    continue

                if self.config.psrl.rollout_coordination.sync_and_mig_strategy.mig.indicator == "request_num":
                    request_num = instance_to_status[instance_id].get_waiting_and_running_queue_size()
                    starved_request_num = instance_to_status[starved_instance_id].get_waiting_and_running_queue_size()
                    if starved_request_num == 0:
                        ratio = float("inf") if request_num > 0 else 1
                    else:
                        ratio = request_num / starved_request_num
                elif self.config.psrl.rollout_coordination.sync_and_mig_strategy.mig.indicator == "throughput":
                    throughput = instance_to_status[instance_id].get_generation_throughput()
                    starved_throughput = instance_to_status[starved_instance_id].get_generation_throughput()
                    if starved_throughput == 0:
                        ratio = float("inf") if throughput > 0 else 1
                    else:
                        ratio = throughput / starved_throughput
                elif self.config.psrl.rollout_coordination.sync_and_mig_strategy.mig.indicator == "kv_cache":
                    kv_cache_utilization = instance_to_status[instance_id].get_kv_cache_utilization()
                    starved_kv_cache_utilization = instance_to_status[starved_instance_id].get_kv_cache_utilization()
                    if starved_kv_cache_utilization == 0:
                        ratio = float("inf") if kv_cache_utilization > 0 else 1
                    else:
                        ratio = kv_cache_utilization / starved_kv_cache_utilization
                else:
                    raise ValueError(
                        f"Unknown migrate indicator: "
                        f"{self.config.psrl.rollout_coordination.sync_and_mig_strategy.mig.indicator}"
                    )

                if ratio > self.config.psrl.rollout_coordination.sync_and_mig_strategy.mig.threshold:
                    # psrl_logger.info(
                    #     f"Instance {instance_id} (version {self.instance_to_version_after_sync[instance_id]}) "
                    #     f"has a ratio of {ratio} for migrating to instance {starved_instance_id} "
                    #     f"(version {self.instance_to_version_after_sync[starved_instance_id]})"
                    # )
                    candidate_migrate_instance_ids.append((instance_id, ratio))

        # We choose the instance with the highest ratio to migrate
        # TODO(lhy): support multiple instances to migrate and finer-grained migration strategy
        # Currently, we only support one instance to migrate,
        # and all the requests on the instance will be interrupted and looped back to the router.
        if len(candidate_migrate_instance_ids) > 0:
            candidate_migrate_instance_ids.sort(key=lambda x: x[1], reverse=True)
            migrate_instance_id = candidate_migrate_instance_ids[0][0]
            if self.config.psrl.rollout_coordination.sync_and_mig_strategy.mig.stop_indicator == "request_num":
                request_num = instance_to_status[migrate_instance_id].get_waiting_and_running_queue_size()
                if request_num < self.config.psrl.rollout_coordination.sync_and_mig_strategy.mig.stop_threshold:
                    return []
            elif self.config.psrl.rollout_coordination.sync_and_mig_strategy.mig.stop_indicator == "throughput":
                throughput = instance_to_status[migrate_instance_id].get_generation_throughput()
                if throughput < self.config.psrl.rollout_coordination.sync_and_mig_strategy.mig.stop_threshold:
                    return []
            elif self.config.psrl.rollout_coordination.sync_and_mig_strategy.mig.stop_indicator == "kv_cache":
                kv_cache_utilization = instance_to_status[migrate_instance_id].get_kv_cache_utilization()
                if (
                    kv_cache_utilization
                    < self.config.psrl.rollout_coordination.sync_and_mig_strategy.mig.stop_threshold
                ):
                    return []
            else:
                raise ValueError(
                    f"Unknown stop indicator: "
                    f"{self.config.psrl.rollout_coordination.sync_and_mig_strategy.mig.stop_indicator}"
                )
            return [migrate_instance_id]
        return []

    async def _wait_interrupted_partial_requests_loop_back(self, instance_ids: list[RolloutInstanceId]):
        futures = [
            self.server_handles[replica_id].wait_for_requests_to_drain.remote()
            for replica_id in sorted({instance_id[0] for instance_id in instance_ids})
        ]
        await asyncio.gather(*futures)
