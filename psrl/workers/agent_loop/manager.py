import asyncio
import logging
import os
from collections import Counter

import ray
import transfer_queue as tq
from omegaconf import DictConfig
from tensordict import TensorDict
from transfer_queue import KVBatchMeta
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig

from psrl.utils.logger import (
    DualOutputHandler,
    EventType,
    log_dual_events,
    log_single_event,
)
from psrl.utils.ray import AsyncBusyPollingRayLock
from psrl.utils.transferqueue_utils import kv_batch_meta_update_tags
from psrl.workers.gen_dplb.utils import RolloutInstanceId
from psrl.workers.ps.request_status_tracker import PSRL_RequestStatus
from psrl.workers.ps.staleness_controller import EntryInfo

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class PSRL_AgentLoopManager:
    def __init__(
        self,
        config: DictConfig,
        data_queue_size: int,
        agent_loop_workers: list[ray.actor.ActorHandle],
        ps_manager_handle: ray.actor.ActorHandle,
        group_post_process_fn=None,
        buffer_post_process_fn=None,
    ):
        """Initialize agent loop manager.
        Agent loop manager that manages a group of agent loop workers.
        Handles data distribution, versioning, and coordination between workers.

        Args:
            config (DictConfig): Configuration containing training and rollout settings.
            data_queue_size (int): Size of the data queue.
            agent_loop_workers (list[ray.actor.ActorHandle]): List of agent loop worker instances.
            ps_manager_handle (ray.actor.ActorHandle): Handle to the parameter server manager.
            group_post_process_fn (Optional[callable]): Optional function to post-process
                grouped entry data before occupying the buffer
            buffer_post_process_fn (Optional[callable]): Optional function to post-process
                ready buffer data
        """
        self.config = config
        model_config = config.gen_actor_rollout_ref.model
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config)
        self.tokenizer = self.model_config.tokenizer
        self.processor = self.model_config.processor

        # TransferQueue bootstrap.
        tq.init()

        self.staleness = self.config.psrl.staleness
        self.group_post_process_fn = group_post_process_fn
        self.buffer_post_process_fn = buffer_post_process_fn
        if self.config.psrl.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n
        self.val_rollout_n = self.config.train_actor_rollout_ref.rollout.val_kwargs.n

        if self.config.psrl.redundant_rollout.enable:
            self.entries_per_buffer = self.config.psrl.redundant_rollout.redundant_global_batch_size
            self.ready_entries_per_buffer = self.config.psrl.redundant_rollout.alg_global_batch_size
        else:
            self.entries_per_buffer = self.config.psrl.staleness_buffer_entries
            self.ready_entries_per_buffer = self.config.psrl.staleness_buffer_entries

        self.train_data_queue: asyncio.Queue = asyncio.Queue(maxsize=data_queue_size)
        self.val_data_queue: asyncio.Queue = asyncio.Queue(maxsize=data_queue_size)
        self.agent_loop_workers = agent_loop_workers
        self.ps_manager_handle = ps_manager_handle

        self._request_counter = 0
        self._dispatch_idx = 0
        self._val_buffer_id = 0
        self.running_loop: asyncio.AbstractEventLoop | None = None
        self.train_dispatch_task: asyncio.Task | None = None
        self.val_dispatch_task: asyncio.Task | None = None
        self.stop_train_dispatch_task = False
        self.stop_val_dispatch_task = False

        self.curr_ps_version_tag = 0

        # Accumulated EntryInfo buffers (train path).
        self.train_data_buffers: dict[int, KVBatchMeta] = {}  # metadata of READY buffer in ps manager
        self.train_accumulated_buffers: dict[
            int, dict[int, list[EntryInfo]]
        ] = {}  # Maps buffer_id to dict of model_version to READY entry_info list
        self.train_accumulated_buffer_size: dict[int, int] = {}  # Maps buffer id to current accumulated size

        # Accumulated EntryInfo buffers (val path).
        self.val_data_buffers: dict[int, KVBatchMeta] = {}  # metadata of READY buffer in ps manager
        self.val_accumulated_buffers: dict[
            int, dict[int, list[EntryInfo]]
        ] = {}  # Maps buffer_id to dict of model_version to READY entry_info list
        self.val_accumulated_buffer_size: dict[int, int] = {}  # Maps buffer id to current accumulated size
        self.val_buffer_size: int | None = None  # Set by main trainer when starting validation

        # Set of buffer ids that have been logged as ready, to avoid duplicate logging
        self.logged_ready_train_buffer_ids: set[int] = set()
        self.logged_ready_val_buffer_ids: set[int] = set()

        # Waiting lists for training batches
        self._train_buffer_waiters: dict[
            int, list[asyncio.Future]
        ] = {}  # Maps buffer IDs to a set of futures waiting for that buffer
        self._val_buffer_waiters: dict[
            int, list[asyncio.Future]
        ] = {}  # Maps buffer IDs to a set of futures waiting for that buffer

        # Track finished child requests for Group Sampling
        self.rollout_request_tracker: dict[
            str | int, list[EntryInfo]
        ] = {}  # Maps parent request ids to "occupied" child entries

        # Build logger
        self.log_prefix = "AgentLoopManager"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))

    # AGENT(VERL): `generate_sequences`, `_run_agent_loop` are moved to agent loop workers.
    # The manager only handles data distribution and coordination.

    def set_val_buffer_size(self, val_buffer_size: int):
        """Set the validation buffer size."""
        self.val_buffer_size = val_buffer_size

    async def start_busy_loop(self):
        """Start the busy loop for continuous data processing from the queue."""
        if (
            self.train_dispatch_task is not None
            and not self.train_dispatch_task.done()
            or self.val_dispatch_task is not None
            and not self.val_dispatch_task.done()
        ):
            return

        # Start the busy loop of agent loop workers.
        await asyncio.gather(
            *[worker.start_busy_loop.remote() for worker in self.agent_loop_workers]
        )

        # Start the background task to process data
        self.running_loop = asyncio.get_running_loop()
        self.train_dispatch_task = self.running_loop.create_task(self._train_dispatch_data())
        self.train_dispatch_task.add_done_callback(lambda f: f.result())
        self.val_dispatch_task = self.running_loop.create_task(self._val_dispatch_data())
        self.val_dispatch_task.add_done_callback(lambda f: f.result())

    async def stop_busy_loop(self):
        """Stop the busy loop and wait for all tasks to complete."""
        if (
            (not self.train_dispatch_task or self.train_dispatch_task.done())
            and (not self.val_dispatch_task or self.val_dispatch_task.done())
        ):
            return

        self.stop_train_dispatch_task = True
        self.stop_val_dispatch_task = True
        await asyncio.gather(self.train_dispatch_task, self.val_dispatch_task)

        await asyncio.gather(
            *[worker.stop_busy_loop.remote() for worker in self.agent_loop_workers]
        )

    async def put_data(self, batch_meta: KVBatchMeta, is_validate: bool = False):
        """Put objectref of data into the manager's data queue."""
        queue = self.val_data_queue if is_validate else self.train_data_queue
        await queue.put(batch_meta)

    async def _train_dispatch_data(self):
        """Main dispatch loop that processes data from the queue and routes to workers."""
        while not self.stop_train_dispatch_task:
            if not self.train_data_queue.empty():
                data: KVBatchMeta | None = self.train_data_queue.get_nowait()
            else:
                await asyncio.sleep(0)
                continue

            # Receive END signal to stop processing data queue
            if data is None:
                psrl_logger.info("Received END signal, stopping agent loop manager train dispatch task.")
                self.stop_train_dispatch_task = True
                continue

            batch_size = len(data)

            # Wait for version update in ps
            # NOTE(lhy): we restrict the extra dispatched data to be no more than (staleness + 1) * buffer_size
            expected_ps_version = self._get_expected_ps_version()
            if expected_ps_version > self.curr_ps_version_tag:
                psrl_logger.debug(f"Waiting for ps model version: {expected_ps_version}")
                # Busy polling until the PS worker has the needed model version
                while (
                    await self.ps_manager_handle.get_ps_model_version.remote(debug_info="agent_loop_manager")
                ) < expected_ps_version:
                    await asyncio.sleep(0.1)
                self.curr_ps_version_tag = expected_ps_version
                psrl_logger.info(f"ps model version updated to {self.curr_ps_version_tag}, continue to dispatch")

            # Initialize the version tag to -1 for all requests
            data = kv_batch_meta_update_tags(data, "version_tag", -1)

            # Dispatch data to agent loop workers
            await self._inner_dispatch_data(data, is_validate=False)
            # Increment counter after dispatch so _get_expected_ps_version reflects the number
            # of requests that have actually been sent out.
            self._request_counter += batch_size
            await asyncio.sleep(0)  # Yield control to the event loop

    async def _val_dispatch_data(self):
        """Main dispatch loop that processes data from the queue and routes to workers."""
        while not self.stop_val_dispatch_task:
            if not self.val_data_queue.empty():
                data: KVBatchMeta | None = self.val_data_queue.get_nowait()
            else:
                await asyncio.sleep(0)
                continue

            # Receive END signal to stop processing data queue
            if data is None:
                psrl_logger.info("Received END signal, stopping agent loop manager validation dispatch task.")
                self.stop_val_dispatch_task = True
                continue

            batch_size = len(data)
            # Validation samples all share the current PS version.
            data = kv_batch_meta_update_tags(data, "version_tag", [self.curr_ps_version_tag] * batch_size)

            # Dispatch data to agent loop workers
            await self._inner_dispatch_data(data, is_validate=True)
            await asyncio.sleep(0)  # Yield control to the event loop

        psrl_logger.info("Agent loop manager validation dispatch task stopped.")

    async def _retry_data(self, data: KVBatchMeta | None = None):
        """Notify the agent loop manager to retry processing some data."""
        if not (self.running_loop and not self.stop_train_dispatch_task):
            psrl_logger.warning("Busy loop of the agent loop manager has stopped, the retry operation will be skipped")
            return

        # If data is None, the new data from the data queue will be used.
        if data is None:
            if self.train_data_queue.empty():
                return
            data = await self.train_data_queue.get()
            if data is None:
                raise ValueError("Data queue should not contain None when retrying requests.")

        batch_size = len(data)
        data = kv_batch_meta_update_tags(data, "version_tag", -1)
        psrl_logger.info(f"Retry {batch_size} requests")
        await self._inner_dispatch_data(data, is_validate=False)

    def _get_expected_ps_version(self):
        """
        Get the expected PS version tag based on the current staleness and request counter.
        """
        if self.config.psrl.redundant_rollout.enable:
            buffer_size = self.config.psrl.redundant_rollout.redundant_global_batch_size * self.rollout_n
        else:
            buffer_size = self.config.psrl.staleness_buffer_entries * self.rollout_n

        expected_ps_version = max(self._request_counter - self.staleness * buffer_size, 0) // buffer_size
        return expected_ps_version

    async def _inner_dispatch_data(self, data: KVBatchMeta, is_validate: bool = False):
        """Update request status to RUNNING in PSManager, then fan out to workers."""
        # Rows are ordered as contiguous groups of `rollout_n` children per parent.
        uids = [int(key) for key in data.keys]
        versions = [tag["version_tag"] for tag in data.tags]

        # Update request status from PENDING to RUNNING
        update_status_success = await self.ps_manager_handle.update_request_status.remote(
            uids,
            PSRL_RequestStatus.RUNNING,
            model_version=versions,
            is_validate=is_validate,
        )
        if not update_status_success:
            return

        dispatch_plan = self.get_dispatch_plan(data, is_validate=is_validate)
        for worker_index, batch in dispatch_plan.items():
            if not batch:
                continue

            # Dispatch data to the corresponding worker
            requests = batch.chunk(len(batch))
            tasks = [
                self.agent_loop_workers[worker_index].add_agent_program.remote(request)
                for request in requests
            ]
            await asyncio.gather(*tasks)

    def get_dispatch_plan(
        self, data: KVBatchMeta, is_validate: bool = False
    ) -> dict[int, KVBatchMeta]:
        """Round-robin dispatch plan keyed by worker index, co-locating siblings.

        Children sharing a ``parent_id`` (group sampling) land on the same worker.
        """
        keys_by_worker: dict[int, list[str]] = {}
        prompt_to_worker: dict[int, int] = {}
        rollout_n = self.val_rollout_n if is_validate else self.rollout_n
        if rollout_n > 1:
            prompt_ids = [tag["parent_id"] for tag in data.tags]
        else:
            prompt_ids = [int(key) for key in data.keys]

        # Round-robin dispatching
        for i, prompt_id in enumerate(prompt_ids):
            if prompt_id in prompt_to_worker:
                worker_index = prompt_to_worker[prompt_id]
            else:
                worker_index = (self._dispatch_idx + prompt_id) % len(self.agent_loop_workers)
                prompt_to_worker[prompt_id] = worker_index
            keys_by_worker.setdefault(worker_index, []).append(data.keys[i])

        return {
            worker_index: data.select_keys(keys) if keys else None
            for worker_index, keys in keys_by_worker.items()
        }

    async def occupy_requests(
        self,
        request_id: int,
        prompt_id: int,
        rollout_instance_id: RolloutInstanceId | tuple | list,
        version_tag: int,
        is_validate: bool = False,
    ):
        """Flat-arg RPC invoked by rollout workers once a request finishes.

        The rollout worker has already written the per-sample TensorDict to
        TQ under ``str(request_id)``. This method just appends the
        corresponding ``EntryInfo`` into the manager's trackers and triggers
        PSManager occupation, running group/buffer post-processing on
        KVBatchMeta slices (never on tensor payload).
        """
        async with AsyncBusyPollingRayLock(self.ps_manager_handle):
            rollout_n = self.val_rollout_n if is_validate else self.rollout_n
            alg_rollout_n = self.val_rollout_n if is_validate else self.alg_rollout_n

            ready_buffer_ids: set[int] = set() # Buffer IDs that are READY after occupation
            occupy_futures: list = []
            abort_request_ids: list[int] = [] # Used to abort requests in the data pool
            prompt_to_occupy_requests: dict[int, list[EntryInfo]] = {}

            # 1. Judge whether to abort requests and occupy requests in the PS worker
            if rollout_n > 1:
                entry_info = EntryInfo(
                    rollout_instance_id=rollout_instance_id,
                    request_idx=request_id % rollout_n,
                    prompt_id=prompt_id,
                    model_version=version_tag,
                    is_validate=is_validate,
                )
                self.rollout_request_tracker.setdefault(prompt_id, []).append(entry_info)
                psrl_logger.debug(
                    f"Store data for prompt {prompt_id} with info {entry_info}, "
                    f"request num: {len(self.rollout_request_tracker[prompt_id])}"
                )

                if len(self.rollout_request_tracker[prompt_id]) >= alg_rollout_n:
                    psrl_logger.debug(
                        f"Reached/Required: "
                        f"({len(self.rollout_request_tracker[prompt_id])}/{alg_rollout_n}) "
                        f"samples for prompt {prompt_id}"
                    )
                    entry_infos = self.rollout_request_tracker.pop(prompt_id)
                    psrl_logger.debug(
                        f"Popped entry_infos from rollout_request_tracker for prompt_id {prompt_id}, "
                        f"entry count: {len(entry_infos)}"
                    )

                    all_child_idxs = set(range(rollout_n))
                    stored_child_idxs = set()
                    for entry_info in entry_infos:
                        assert isinstance(entry_info.request_idx, int), (
                            f"entry_info.request_idx should be int, but got {type(entry_info.request_idx)}"
                        )
                        request_idx: int = entry_info.request_idx
                        stored_child_idxs.add(request_idx)
                    abort_child_idxs = all_child_idxs - stored_child_idxs
                    abort_child_ids = [prompt_id * rollout_n + idx for idx in abort_child_idxs]
                    psrl_logger.debug(
                        f"Stored child IDs: "
                        f"{[prompt_id * rollout_n + idx for idx in stored_child_idxs]}, "
                        f"Abort child IDs: {abort_child_ids}"
                    )

                    # Notify the request status manager to abort the child requests
                    if abort_child_ids:
                        assert not is_validate, "Abort child requests should not happen in validation"
                        psrl_logger.info(f"Aborting child requests {abort_child_ids} for sample {prompt_id}.")
                        with log_dual_events(
                            f"Abort {len(abort_child_ids)} requests",
                            psrl_logger,
                            level=logging.INFO,
                            event_type=EventType.OTHER,
                        ):
                            await self.ps_manager_handle.abort_requests.remote(
                                list(abort_child_ids), blocking=False
                            )

                    # Abort the extra finished entries beyond alg_rollout_n
                    abort_request_ids.extend(
                        [
                            prompt_id * rollout_n + entry_info.request_idx
                            for entry_info in entry_infos[alg_rollout_n:]
                        ]
                    )

                    alg_entry_infos = entry_infos[:alg_rollout_n]
                    add_data = True
                    # Perform group post-processing for training data only
                    if self.group_post_process_fn:
                        add_data = await self._group_post_process(alg_entry_infos)

                    if not add_data:
                        # Retry immediately and no occupation
                        # NOTE(linsh): data has been cleared in `_group_post_process`
                        psrl_logger.info(
                            f"Post-processing function returned empty data for "
                            f"prompt {prompt_id}. Retrying immediately."
                        )
                        # Clear the reserved entries for the group entry
                        await self.ps_manager_handle.clear_reserved_entries.remote(prompt_id, is_validate)
                        # Notify agent loop manager to retry new requests
                        await self._retry_data()
                    else:
                        prompt_to_occupy_requests[prompt_id] = alg_entry_infos
                        request_ids = [
                            prompt_id * rollout_n + entry_info.request_idx for entry_info in alg_entry_infos
                        ]
                        occupy_futures.append(
                            self.ps_manager_handle.occupy_rollout_instance_request.remote(
                                prompt_id=prompt_id,
                                request_ids=request_ids,
                                is_validate=is_validate,
                            )
                        )
            else:
                # Without group sampling (e.g., PPO)
                # Group post processing is not used and every data will be added
                occupy_futures.append(
                    self.ps_manager_handle.occupy_rollout_instance_request.remote(
                        prompt_id=request_id,
                        is_validate=is_validate,
                    )
                )

            # 2. Occupy requests in the PS worker
            if not occupy_futures:
                return
            with log_dual_events(
                "Occupy requests",
                psrl_logger,
                level=logging.DEBUG,
                event_type=EventType.OTHER,
            ):
                results = await asyncio.gather(*occupy_futures)

            # 3. Handle the occupied results to accumulate data
            for result in results:
                buffer_id, occupy_num, prompt_entry_info = result
                # If occupy failed due to READY status, the requests must be aborted already
                # Just continue
                if buffer_id is None:
                    continue

                psrl_logger.debug(
                    f"Successfully occupied prompt {prompt_entry_info} into "
                    f"buffer {buffer_id} with occupy_num {occupy_num}."
                )

                # Accumulate data
                accumulated_buffers = (
                    self.val_accumulated_buffers if is_validate else self.train_accumulated_buffers
                )
                accumulated_buffer_size = (
                    self.val_accumulated_buffer_size if is_validate else self.train_accumulated_buffer_size
                )
                expected_buffer_size = (
                    self.val_buffer_size if is_validate else self.ready_entries_per_buffer
                )

                if buffer_id not in accumulated_buffers:
                    accumulated_buffers[buffer_id] = {}
                    accumulated_buffer_size[buffer_id] = 0
                model_version = prompt_entry_info.get_entry_version()
                accumulated_buffers[buffer_id].setdefault(model_version, []).append(prompt_entry_info)
                accumulated_buffer_size[buffer_id] += 1
                psrl_logger.info(
                    f"Accumulated buffer {buffer_id} size: "
                    f"{accumulated_buffer_size[buffer_id]}/{expected_buffer_size}"
                )

                # Check if the buffer is the earliest waiting buffer
                # If so, handle the waiting buffer using the abort and truncate strategy
                if not is_validate and self._train_buffer_waiters:
                    min_waiter_buffer_id = min(self._train_buffer_waiters.keys())
                    if min_waiter_buffer_id == buffer_id:
                        await self.handle_waiting_buffer(buffer_id)

                # Check for READY buffers
                if (
                    accumulated_buffer_size[buffer_id] == expected_buffer_size
                    and buffer_id not in ready_buffer_ids
                ):
                    psrl_logger.info(f"Add buffer {buffer_id} to ready_buffer_ids")
                    ready_buffer_ids.add(buffer_id)

            # 4. Release TQ state for aborted entries (beyond alg_rollout_n).
            if abort_request_ids:
                await tq.async_kv_clear(
                    keys=[str(request_id) for request_id in abort_request_ids],
                    partition_id="val" if is_validate else "train",
                )

            # 5. Process READY buffers
            for buffer_id in sorted(list(ready_buffer_ids)):
                # Collect all prompt entry infos for the buffer
                accumulated_buffers = (
                    self.val_accumulated_buffers if is_validate else self.train_accumulated_buffers
                )
                accumulated_buffer_size = (
                    self.val_accumulated_buffer_size if is_validate else self.train_accumulated_buffer_size
                )

                prompt_entry_infos: list[EntryInfo] = []
                for model_version in sorted(list(accumulated_buffers[buffer_id].keys())):
                    prompt_entry_infos.extend(accumulated_buffers[buffer_id][model_version])

                if is_validate:
                    prompt_entry_infos.sort(key=lambda ei: ei.prompt_id)

                batch = self.entry_infos_to_kv_batch_meta(prompt_entry_infos, is_validate)
                # Apply buffer post-processing if exists and add to data_buffers
                add_buffer = self.maybe_add_buffer(buffer_id, batch, is_validate)
                if add_buffer:
                    psrl_logger.info(f"Buffer {buffer_id} is READY with {len(batch)} entries.")
                    await self.handle_ready_buffer(buffer_id, is_validate)
                    accumulated_buffers.pop(buffer_id)
                    accumulated_buffer_size.pop(buffer_id)

    def maybe_add_buffer(self, buffer_id: int, batch: KVBatchMeta, is_validate: bool = False) -> bool:
        """
        Apply buffer post-processing function if defined and add the buffer to data_buffers.

        Args:
            buffer_id (int): The ID of the buffer to be added.
            batch (KVBatchMeta): The data buffer to be potentially post-processed and added.
            is_validate (bool): Whether the buffer is for validation.
        Returns:
            bool: whether the buffer was added to data_buffers.
        """
        if is_validate:
            self.val_data_buffers[buffer_id] = batch
            psrl_logger.debug(f"Buffer {buffer_id} is added to val_data_buffers without post-processing.")
            return True

        add_buffer = True
        if self.buffer_post_process_fn:
            add_buffer, batch = self._buffer_post_process(buffer_id, batch)

        if add_buffer:
            self.train_data_buffers[buffer_id] = batch
            psrl_logger.debug(f"Buffer {buffer_id} is added to train_data_buffers after post-processing.")
        return add_buffer

    def entry_infos_to_kv_batch_meta(
        self, entry_infos: list[EntryInfo], is_validate: bool
    ) -> KVBatchMeta:
        """Build a ``KVBatchMeta`` for the given EntryInfos."""
        rollout_n = self.val_rollout_n if is_validate else self.rollout_n
        partition = "val" if is_validate else "train"
        keys: list[str] = []
        tags: list[dict] = []
        for entry_info in entry_infos:
            request_idxs = entry_info.request_idx if isinstance(entry_info.request_idx, list) else [entry_info.request_idx]
            model_versions = (
                entry_info.model_version if isinstance(entry_info.model_version, list) else [entry_info.model_version]
            )
            for j, request_idx in enumerate(request_idxs):
                keys.append(str(entry_info.prompt_id * rollout_n + request_idx))
                tags.append(
                    {
                        "uid": entry_info.prompt_id * rollout_n + request_idx,
                        "parent_id": entry_info.prompt_id,
                        "version_tag": (
                            model_versions[j] if j < len(model_versions) else model_versions[-1]
                        ),
                        "rollout_instance_id": entry_info.rollout_instance_id,
                    }
                )
        return KVBatchMeta(keys=keys, tags=tags, partition_id=partition)

    async def _group_post_process(self, entry_infos: list[EntryInfo]) -> bool:
        """Apply post-processing function to a group of entry infos.

        This method retrieves data from the data pool for each entry, applies
        the group post-processing function, and stores the processed data back.

        Args:
            entry_infos (List[EntryInfo]): List of entry info objects to process

        Returns:
            bool: whether the group data is reserved
        """
        assert self.group_post_process_fn is not None, "Group post-processing function is not set."
        assert all(not entry_info.is_validate for entry_info in entry_infos), (
            "Group post-processing should not be applied to validation data."
        )

        keys = [
            str(entry_info.prompt_id * self.rollout_n + entry_info.request_idx) for entry_info in entry_infos
        ]
        # TODO(linsh): optimize by only fetching necessary columns for post-processing instead of the full TD.
        meta = KVBatchMeta(
            keys=keys,
            tags=[{} for _ in keys],
            partition_id="train",
            fields=None,
        )
        data = await tq.async_kv_batch_get_by_meta(meta)

        processed_data = self.group_post_process_fn(data)
        if processed_data is None:
            await tq.async_kv_clear(keys=keys, partition_id="train")
            return False

        # Mutation path: re-upsert the processed TensorDict under the same keys.
        await tq.async_kv_batch_put(keys=keys, partition_id="train", fields=processed_data)
        return True

    def _buffer_post_process(self, buffer_id: int, batch_meta: KVBatchMeta) -> tuple[bool, KVBatchMeta | None]:
        """Apply post-processing function to a full buffer of data.

        This method applies the buffer post-processing function to the data
        in the buffer and stores the processed data if valid.
        Note that the buffer post process only targets training data.

        Args:
            buffer_id (int): The ID of the buffer to process
            buffer_data (KVBatchMeta): The data in the buffer to be processed
        Returns:
            Tuple[bool, Optional[KVBatchMeta]]: A tuple where the first element indicates
            whether to add the buffer to data_buffers, and the second element is the
            processed KVBatchMeta or None if not added.
        """
        assert self.buffer_post_process_fn is not None, "Buffer post-processing function is not set."

        original_keys = list(batch_meta.keys)
        # TODO(linsh): optimize by only fetching necessary columns for post-processing instead of the full TD.
        data = tq.kv_batch_get_by_meta(batch_meta)
        processed_data = self.buffer_post_process_fn(data)

        original_size = len(batch_meta)
        processed_size = (
            0 if processed_data is None else len(processed_data)
        )

        if processed_data is not None and processed_size == original_size:
            # Just write mutations and keep the original meta.
            tq.kv_batch_put(
                keys=original_keys, partition_id=batch_meta.partition_id, fields=processed_data
            )
            return True, batch_meta

        # Clear entries from accumulated_data_buffer
        self.train_accumulated_buffers.pop(buffer_id, None)
        self.train_accumulated_buffer_size.pop(buffer_id, None)
        self.train_accumulated_buffers[buffer_id] = {}
        self.train_accumulated_buffer_size[buffer_id] = 0

        if processed_data is None or processed_size == 0:
            tq.kv_clear(keys=original_keys, partition_id=batch_meta.partition_id)
            return False, None

        # Partial clear: recover kept keys from the processor's uid column and
        # rebuild EntryInfo inventory + a tighter KVBatchMeta.
        request_ids = tu.get_non_tensor_data(processed_data, "uid")
        kept_keys = [str(request_id) for request_id in request_ids]
        dropped_keys = [k for k in original_keys if k not in set(kept_keys)]
        if dropped_keys:
            tq.kv_clear(keys=dropped_keys, partition_id=batch_meta.partition_id)
        tq.kv_batch_put(
            keys=kept_keys, partition_id=batch_meta.partition_id, fields=processed_data
        )

        prompt_entry_infos = self.extract_entry_infos_from_td(processed_data)
        for entry_info in prompt_entry_infos:
            model_version = (
                min(entry_info.model_version)
                if isinstance(entry_info.model_version, list)
                else entry_info.model_version
            )
            self.train_accumulated_buffers[buffer_id].setdefault(model_version, []).append(entry_info)
            self.train_accumulated_buffer_size[buffer_id] += 1

        tags = [
            {
                "parent_id": entry_info.prompt_id,
                "uid": entry_info.prompt_id * self.rollout_n + (
                    entry_info.request_idx[0]
                    if isinstance(entry_info.request_idx, list)
                    else entry_info.request_idx
                ),
                "version_tag": (
                    entry_info.model_version[0]
                    if isinstance(entry_info.model_version, list)
                    else entry_info.model_version
                ),
                "rollout_instance_id": entry_info.rollout_instance_id,
            }
            for entry_info in prompt_entry_infos
        ]
        new_meta = KVBatchMeta(
            keys=kept_keys,
            tags=tags,
            partition_id=batch_meta.partition_id,
            fields=None,
        )

        return False, new_meta

    def extract_entry_infos_from_td(self, data: TensorDict) -> list[EntryInfo]:
        """Extract EntryInfo objects from TensorDict.

        This method extracts EntryInfo objects from the non-tensor batch
        information in the provided TensorDict.

        Args:
            data (TensorDict): The data from which to extract EntryInfo objects
        Returns:
            List[EntryInfo]: List of extracted EntryInfo objects
        """
        is_validate = tu.get_non_tensor_data(data, "validate", default=False)
        rollout_n = self.val_rollout_n if is_validate else self.rollout_n
        entry_infos_map: dict[int, EntryInfo] = {}
        if rollout_n > 1:
            parent_ids = tu.get_non_tensor_data(data, "parent_id")
            rollout_instance_ids = tu.get_non_tensor_data(data, "rollout_instance_id")
            request_ids = tu.get_non_tensor_data(data, "uid")
            model_versions = tu.get_non_tensor_data(data, "version_tag")
            for parent_id, rollout_instance_id, request_id, model_version in zip(
                parent_ids, rollout_instance_ids, request_ids, model_versions
            ):
                if parent_id in entry_infos_map:
                    entry_info = entry_infos_map[parent_id]
                    if isinstance(entry_info.request_idx, list):
                        entry_info.request_idx.append(request_id % rollout_n)
                    else:
                        entry_info.request_idx = [
                            entry_info.request_idx,
                            request_id % rollout_n,
                        ]
                    if isinstance(entry_info.model_version, list):
                        entry_info.model_version.append(model_version)
                    else:
                        entry_info.model_version = [
                            entry_info.model_version,
                            model_version,
                        ]
                else:
                    entry_info = EntryInfo(
                        rollout_instance_id=rollout_instance_id,
                        request_idx=request_id % rollout_n,
                        prompt_id=parent_id,
                        model_version=model_version,
                        is_validate=is_validate,
                    )
                    entry_infos_map[parent_id] = entry_info
        else:
            request_ids = tu.get_non_tensor_data(data, "uid")
            model_versions = tu.get_non_tensor_data(data, "version_tag")
            rollout_instance_ids = tu.get_non_tensor_data(data, "rollout_instance_id")
            for request_id, model_version, rollout_instance_id in zip(
                request_ids, model_versions, rollout_instance_ids
            ):
                entry_info = EntryInfo(
                    rollout_instance_id=rollout_instance_id,
                    request_idx=0,
                    prompt_id=request_id,
                    model_version=model_version,
                    is_validate=is_validate,
                )
                entry_infos_map[request_id] = entry_info
        return list(entry_infos_map.values())   

    def log_ready_buffer(self, buffer_id: int, is_validate: bool = False):
        """Log the ready buffer.

        Args:
            buffer_id (int): The ID of the buffer that is ready.
            is_validate (bool): Whether the buffer is for validation data.
        """
        logged_ready_buffer_ids = (
            self.logged_ready_val_buffer_ids if is_validate else self.logged_ready_train_buffer_ids
        )
        if buffer_id not in logged_ready_buffer_ids:
            log_single_event(
                f"{'Train' if not is_validate else 'Validate'} Buffer {buffer_id} is ready",
                psrl_logger,
                event_type=EventType.BUFFER_READY,
            )
            logged_ready_buffer_ids.add(buffer_id)

    async def handle_ready_buffer(self, buffer_id: int, is_validate: bool = False):
        """
        Handle the ready buffer.

        Args:
            buffer_id (int): The ID of the buffer that is ready.
            is_validate (bool): Whether the buffer is for validation data.
        """
        # Check whether there exists ready buffer for training
        self.log_ready_buffer(buffer_id, is_validate)

        psrl_logger.info(f"Checking staleness and aborting requests for buffer {buffer_id}.")
        if not is_validate:
            await self.ps_manager_handle.handle_ready_buffer.remote(buffer_id)
            # NOTE(linsh): the aborted requests have been cleared from tq in ps manager

        data_buffers = self.val_data_buffers if is_validate else self.train_data_buffers
        _buffer_waiters = self._val_buffer_waiters if is_validate else self._train_buffer_waiters
        if not data_buffers:
            return
        min_ready_buffer_id = min(data_buffers.keys())

         # Wake all Futures waiting for this buffer
        if min_ready_buffer_id in _buffer_waiters:
            batch: KVBatchMeta = self.consume_buffer(min_ready_buffer_id, is_validate=is_validate)
            assert len(_buffer_waiters[min_ready_buffer_id]) == 1, (
                f"Expected only one waiter for buffer {min_ready_buffer_id}, "
                f"but found {len(_buffer_waiters[min_ready_buffer_id])}."
            )
            # Set the result for all futures
            for fut in _buffer_waiters[min_ready_buffer_id]:
                if not fut.done():
                    fut.set_result(batch)
            # Remove the key after waking all waiters
            del _buffer_waiters[min_ready_buffer_id]

            if is_validate:
                await self.ps_manager_handle.maybe_delete_buffer.remote(min_ready_buffer_id, is_validate)
        else:
            psrl_logger.warning(f"No waiters found for buffer {buffer_id} when trying to awake.")

    async def handle_waiting_buffer(self, buffer_id: int):
        """Handle the waiting buffer."""
        # WIP(lhy): Implement the retry and truncate strategy
        if self.config.psrl.proactive_filter_strategy.method is None:
            return
        if self.config.psrl.proactive_filter_strategy.method == "retry":
            gap = self.ready_entries_per_buffer - self.train_accumulated_buffer_size[buffer_id]
            if gap == 0:
                return
            assert gap > 0, f"Gap should be greater than 0, but got {gap}"
            if gap <= self.config.psrl.proactive_filter_strategy.threshold:
                psrl_logger.info(
                    f"Trying to abort the rest {gap} entries in buffer {buffer_id} "
                    f"and move some occupied entries from other buffers to make it ready."
                )
                # Guarantee other buffers have enough entries to make it ready
                total_available_entries = 0
                for other_buffer_id in sorted(list(self.train_accumulated_buffers.keys()), reverse=True):
                    if other_buffer_id == buffer_id:
                        break
                    total_available_entries += sum(
                        len(self.train_accumulated_buffers[other_buffer_id][model_version])
                        for model_version in self.train_accumulated_buffers[other_buffer_id].keys()
                    )
                if total_available_entries < gap:
                    psrl_logger.info(
                        f"Not enough entries in other buffers to make buffer {buffer_id} ready, "
                        f"the gap is {gap}, but only {total_available_entries} entries are available"
                    )
                    return
                psrl_logger.info(
                    f"Aborting the rest {gap} entries in buffer {buffer_id} "
                    f"and moving some occupied entries from other buffers to make it ready."
                )
                # First, abort the reserved requests in the buffer
                aborted_entry_num, _ = await self.ps_manager_handle.abort_reserved_requests.remote(
                    buffer_id
                )
                # NOTE(linsh): the aborted requests have been cleared from tq in ps manager

                # Then, move the occupied entries from other buffers to the buffer
                total_moved_entries = 0
                moved_occupied_entry_infos: list[EntryInfo] = []
                for other_buffer_id in sorted(list(self.train_accumulated_buffers.keys()), reverse=True):
                    if other_buffer_id == buffer_id:
                        break
                    for model_version in sorted(list(self.train_accumulated_buffers[other_buffer_id].keys())):
                        moved_entry_infos: list[EntryInfo] = []
                        for entry_info in self.train_accumulated_buffers[other_buffer_id][model_version]:
                            moved_entry_infos.append(entry_info)
                            total_moved_entries += 1
                            if total_moved_entries == gap:
                                break
                        self.train_accumulated_buffers[buffer_id].setdefault(model_version, [])
                        for moved_entry_info in moved_entry_infos:
                            moved_occupied_entry_infos.append(moved_entry_info)
                            self.train_accumulated_buffers[buffer_id][model_version].append(moved_entry_info)
                            self.train_accumulated_buffer_size[buffer_id] += 1
                            self.train_accumulated_buffers[other_buffer_id][model_version].remove(moved_entry_info)
                            self.train_accumulated_buffer_size[other_buffer_id] -= 1
                        if total_moved_entries == gap:
                            break
                    for model_version in list(self.train_accumulated_buffers[other_buffer_id].keys()):
                        if len(self.train_accumulated_buffers[other_buffer_id][model_version]) == 0:
                            self.train_accumulated_buffers[other_buffer_id].pop(model_version)
                    if total_moved_entries == gap:
                        break
                for other_buffer_id in list(self.train_accumulated_buffers.keys()):
                    if len(self.train_accumulated_buffers[other_buffer_id]) == 0:
                        self.train_accumulated_buffers.pop(other_buffer_id)
                        self.train_accumulated_buffer_size.pop(other_buffer_id)
                # Finally, notify the PS manager to move the occupied entries to the buffer
                await self.ps_manager_handle.move_occupied_entries.remote(moved_occupied_entry_infos, buffer_id)
                psrl_logger.info(
                    f"Moved {total_moved_entries} occupied entries (the total gap is {gap}) "
                    f"from other buffers to buffer {buffer_id}."
                )
                for _ in range(aborted_entry_num):
                    await self._retry_data()

        elif self.config.psrl.proactive_filter_strategy.method == "truncate":
            raise NotImplementedError("Truncate strategy is not implemented yet.")

    async def wait_for_training_batch(self, buffer_id: int) -> KVBatchMeta:
        """Await a training batch for a specific buffer ID."""
        await self.ps_manager_handle.ensure_train_buffer_exists.remote(buffer_id)

        if buffer_id in self.train_data_buffers:
            # If the buffer is ready, return immediately
            psrl_logger.info(f"Buffer {buffer_id} is ready, returning immediately.")
            return self.consume_buffer(buffer_id)

        # WIP(lhy): Support more consumption strategies
        # 1. Truncate if buffer status is STUCK
        # 2. Abort the RESERVED entry if buffer status is STUCK and move some OCCUPIED entries from other buffers
        if buffer_id in self.train_accumulated_buffers:
            async with AsyncBusyPollingRayLock(self.ps_manager_handle):
                await self.handle_waiting_buffer(buffer_id)

                if self.train_accumulated_buffer_size[buffer_id] == self.ready_entries_per_buffer:
                    prompt_entry_infos = []
                    for model_version in sorted(list(self.train_accumulated_buffers[buffer_id].keys())):
                        prompt_entry_infos.extend(self.train_accumulated_buffers[buffer_id][model_version])
                    batch_meta = self.entry_infos_to_kv_batch_meta(prompt_entry_infos, is_validate=False)
                    # Apply buffer post-processing if exists and add to data_buffers
                    add_buffer = self.maybe_add_buffer(buffer_id, batch_meta)
                    if add_buffer:
                        psrl_logger.info(
                            f"Buffer {buffer_id} is READY with {len(self.train_data_buffers[buffer_id])} entries."
                        )
                        await self.handle_ready_buffer(buffer_id)
                        self.train_accumulated_buffers.pop(buffer_id)
                        self.train_accumulated_buffer_size.pop(buffer_id)
                        psrl_logger.info(
                            f"Buffer {buffer_id} is ready after the abort "
                            f"and truncate strategy, returning immediately."
                        )
                        return self.consume_buffer(buffer_id)

        # If the buffer is still not ready after the abort and truncate strategy, wait for it to be ready
        psrl_logger.info(f"Buffer {buffer_id} is not ready, waiting for it to be ready.")
        fut = asyncio.get_event_loop().create_future()
        self._train_buffer_waiters.setdefault(buffer_id, []).append(fut)
        batch_meta = await fut
        return batch_meta

    async def wait_for_validation_batch(self, buffer_id: int) -> KVBatchMeta:
        """Await a validation batch, returning a ``KVBatchMeta``."""
        await self.ps_manager_handle.ensure_validate_buffer_exists.remote()

        if buffer_id in self.val_data_buffers:
            # If the buffer is ready, return immediately
            psrl_logger.info(f"Validate buffer {buffer_id} is ready, returning immediately.")
            return self.consume_buffer(buffer_id, is_validate=True)

        # TODO(lhy): support more consumption strategies, now only support waiting for the buffer to be ready
        # 1. Partial rollout if buffer status is STUCK
        # 2. Truncate if buffer status is STUCK
        # 3. Drop the RESERVED entry if buffer status is STUCK and move some OCCUPIED entries from other buffers

        psrl_logger.info(f"Validate buffer {buffer_id} is not ready, waiting for it to be ready.")
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._val_buffer_waiters.setdefault(buffer_id, []).append(fut)
        return await fut

    async def generate_validate_sequences(self, batch: KVBatchMeta) -> int:
        """Dispatch a validation batch; returns the val buffer id."""
        prompt_num = len(batch) // self.val_rollout_n
        prompt_batch_metas = batch.chunk(prompt_num)
        for prompt_batch_meta in prompt_batch_metas:
            request_ids = [int(key) for key in prompt_batch_meta.keys]
            await self.ps_manager_handle.add_request.remote(request_ids, is_validate=True)
            await self.put_data(prompt_batch_meta, is_validate=True)
        self._val_buffer_id += 1

        return self._val_buffer_id - 1

    def log_buffer(self, buffer_id: int, is_validate: bool = False):
        """Log a histogram of ``version_tag`` values for the given buffer."""
        data_buffer = self.val_data_buffers if is_validate else self.train_data_buffers
        assert buffer_id in data_buffer, (
            f"Buffer {buffer_id} not found in {'val' if is_validate else 'train'} buffers."
        )

        version_tags = [tag.get("version_tag", -1) for tag in data_buffer[buffer_id].tags]

        # Count different version_tags
        version_tag_counts = Counter(version_tags)
        total_count = len(version_tags)

        # Calculate staleness for each version_tag
        staleness_dict = {
            version_tag: (None if is_validate else buffer_id - version_tag)
            for version_tag in version_tag_counts.keys()
        }

        psrl_logger.info(
            f"{'VALIDATION' if is_validate else 'TRAINING'} Buffer {buffer_id} version tag distribution:"
        )
        for version_tag in sorted(version_tag_counts.keys()):
            count = version_tag_counts[version_tag]
            percentage = (count / total_count) * 100
            staleness = staleness_dict[version_tag]
            psrl_logger.info(
                f"version_tag={version_tag}: count={count} ({percentage:.2f}%), staleness={staleness}"
            )

    def consume_buffer(self, buffer_id: int, is_validate: bool = False) -> KVBatchMeta:
        """
        Consume (retrieve and remove) all data from the specified buffer.

        Args:
            buffer_id (int): The ID of the buffer to consume.
            is_validate (bool): Whether the buffer is for validation data.
        Returns:
            KVBatchMeta: The concatenated data from the buffer.
        Raises:
            AssertionError: If the buffer is not in READY state.
        """

        self.log_buffer(buffer_id, is_validate)
        buffer = (
            self.val_data_buffers.pop(buffer_id, None) if is_validate else self.train_data_buffers.pop(buffer_id, None)
        )
        assert buffer is not None, f"Buffer {buffer_id} not found or already consumed."
        # NOTE(linsh): we will delete buffer during aborting requests of specific versions
        # This is because the inflight requests of the remaining entries
        # in the buffer can still be utilized for training
        return buffer