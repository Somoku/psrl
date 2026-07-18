"""CommandHandlerMixin: periodic background loop for ABORT/SYNC/SLEEP/WAKE_UP commands."""

from __future__ import annotations

import asyncio
import logging
import os

import numpy as np

from psrl.utils.server.command import Command, CommandType

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class CommandHandlerMixin:
    """Provides ``_command_handler_loop`` for RolloutCoordinator.

    Expects ``self`` to carry: ``command_queue``, ``stop_command_handler``,
    ``server_handles`` and the
    ``_set_routing_loop_running`` / ``_do_sleep_instance`` / ``_do_wake_up_instance``
    methods provided by ``CoordinatorBase`` / ``RolloutCoordinator``.
    """

    async def _command_handler_loop(self):
        """
        Background loop for processing commands from the command queue.

        This method continuously processes different types of commands:
        - ABORT: Interrupt specific requests on instances
        - SYNC: Interrupt instance, pull new model weights, and resume generation

        The loop runs until stop_command_handler is set to True.
        """
        while not self.stop_command_handler:
            # Command processing
            if not self.command_queue.empty():
                command = self.command_queue.get_nowait()

                assert isinstance(command, Command), f"Expected Command type, got {type(command)}"

                # Unpack command attributes
                command_type = command.type
                command_id = command.get_kwargs()["id"]
                command_args = command.get_args()
                psrl_logger.debug(
                    f"Receive command: type = {command_type}, kwargs = {command.get_kwargs()}, args = {command_args}"
                )

                result = None
                # Process the command based on its type
                if command_type == CommandType.ABORT:
                    instance_to_uids = command_args.get("instance_to_uids", None)
                    instance_ids = command_args.get("instance_ids", None)
                    if instance_to_uids is None and instance_ids is None:
                        raise ValueError("ABORT command must contain 'instance_to_uids' or 'instance_ids' in args.")

                    psrl_logger.info(
                        f"Received ABORT command with instance_to_uids: "
                        f"{instance_to_uids} and instance_ids: {instance_ids}"
                    )
                    futures = []

                    if instance_to_uids is not None:
                        replica_to_uids = {}
                        for instance_id, uids in instance_to_uids.items():
                            if not uids:
                                continue
                            replica_id = instance_id[0]
                            if not isinstance(uids, (list, set)):
                                uids = [uids]
                            abort_requests = set(uids)  # Ensure uniqueness
                            replica_to_uids.setdefault(replica_id, set()).update(abort_requests)
                        for replica_id, abort_requests in replica_to_uids.items():
                            abort_requests = [str(req_id) for req_id in abort_requests]
                            futures.append(self.server_handles[replica_id].abort_requests.remote(abort_requests))
                    if instance_ids is not None:
                        for instance_id in instance_ids:
                            # TODO(linsh): merge dp ranks of the same replica to reduce RPC calls
                            replica_id, data_parallel_rank = instance_id
                            futures.append(self.server_handles[replica_id].abort_all_requests.remote())

                    if not futures:
                        interrupted_request_num = 0
                    else:
                        interrupted_request_status = await asyncio.gather(*futures)
                        interrupted_request_nums = [
                            status if isinstance(status, int) else status.get("aborted_count", 0)
                            for status in interrupted_request_status
                        ]
                        interrupted_request_num = np.sum(interrupted_request_nums)

                    result = interrupted_request_num
                    psrl_logger.info(f"Received ABORT command, interrupted {interrupted_request_num} requests")
                    # Post process the command result
                    self._complete_command(command_id, result)

                elif command_type == CommandType.SYNC:
                    # Interrupt the instance, pull the model weights from PS and resume generation.
                    instance_ids = command_args.get("instance_ids", None)
                    curr_ps_model_version = command_args.get("curr_ps_model_version", None)
                    wait_model_sync = command_args.get("wait_model_sync", False)
                    if not isinstance(instance_ids, list):
                        instance_ids = [instance_ids]
                    if instance_ids is None or curr_ps_model_version is None:
                        raise ValueError(
                            "SYNC command must contain 'instance_ids' and 'curr_ps_model_version' in args."
                        )
                    psrl_logger.info(
                        f"Received SYNC command for instances {instance_ids} "
                        f"with PS model version {curr_ps_model_version}"
                    )

                    # Skip instances that are sleeping
                    is_sleeping = await asyncio.gather(
                        *[self.server_handles[instance_id[0]].is_sleeping.remote() for instance_id in instance_ids]
                    )
                    sync_instance_ids = [
                        instance_id for instance_id, sleeping in zip(instance_ids, is_sleeping) if not sleeping
                    ]
                    sleeping_instance_ids = [
                        instance_id for instance_id, sleeping in zip(instance_ids, is_sleeping) if sleeping
                    ]
                    if sleeping_instance_ids:
                        psrl_logger.info(f"Skipping SYNC for sleeping instances: {sleeping_instance_ids}")

                    if not sync_instance_ids:
                        self._complete_command(command_id, None)
                    else:
                        # Sync with PS (interrupt, pull model, and resume generation)
                        sync_futures = []

                        for instance_id in instance_ids:
                            # NOTE(linsh): ignore dp rank for now
                            replica_id, _ = instance_id
                            sync_future = self.server_handles[replica_id].sync_with_ps.remote(
                                curr_ps_model_version,
                                pause_generation=True,
                            )
                            sync_futures.append(sync_future)

                        # Post process the command result
                        if wait_model_sync:
                            await asyncio.gather(*sync_futures)
                            self._complete_command(command_id, None)
                        else:
                            # NOTE(linsh): sometimes it's not necessary for the caller to wait for pulling from PS
                            self._complete_command(command_id, None)
                            await asyncio.gather(*sync_futures)  # Wait for the sync to complete
                elif command_type == CommandType.SLEEP:
                    instance_ids = command_args.get("instance_ids", None)
                    if not isinstance(instance_ids, list):
                        instance_ids = [instance_ids]
                    if instance_ids is None:
                        raise ValueError("SLEEP command must contain 'instance_ids' in args.")

                    # Pause routing to prevent new requests from being dispatched
                    # to the instances that are going to sleep
                    await self._set_routing_loop_running(False)

                    sleep_futures = []
                    for instance_id in instance_ids:
                        # NOTE(linsh): ignore dp rank for now
                        replica_id, _ = instance_id
                        sleep_futures.append(self._do_sleep_instance(replica_id))
                    await asyncio.gather(*sleep_futures)
                    self._complete_command(command_id, None)
                elif command_type == CommandType.WAKE_UP:
                    instance_ids = command_args.get("instance_ids", None)
                    if not isinstance(instance_ids, list):
                        instance_ids = [instance_ids]
                    if instance_ids is None:
                        raise ValueError("WAKE_UP command must contain 'instance_ids' in args.")
                    psrl_logger.info(f"Received WAKE_UP command for instances {instance_ids}")

                    wake_up_futures = []
                    for instance_id in instance_ids:
                        # NOTE(linsh): ignore dp rank for now
                        replica_id, _ = instance_id
                        wake_up_futures.append(self._do_wake_up_instance(replica_id))
                    await asyncio.gather(*wake_up_futures)

                    # Resume routing after the instances have woken up
                    await self._set_routing_loop_running(True)

                    self._complete_command(command_id, None)
                else:
                    raise ValueError(f"Unknown command type: {command_type}")

            await asyncio.sleep(0)
        psrl_logger.info("Background command handler of rollout coordinator has finished.")
