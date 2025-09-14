import os
import logging
import asyncio
import numpy as np
from omegaconf import DictConfig

import ray

from verl import DataProto

from psrl.workers.ps.request_status_tracker import RequestStatus
from psrl.utils.logger import DualOutputHandler, get_worker_info, log_single_event, EventType, deprecated

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

@ray.remote
class PSRL_AgentLoopManager:

    def __init__(
        self,
        config: DictConfig,
        data_queue,
        agent_loop_workers,
        ps_manager_handle,
    ):
        """Initialize agent loop manager.
        Agent loop manager that manages a group of agent loop workers.
        Handles data distribution, versioning, and coordination between workers.

        Args:
            config (DictConfig): Configuration containing training and rollout settings.
            data_queue: Queue for receiving input data.
            agent_loop_workers: List of agent loop worker instances.
            ps_manager_handle: Handle to the parameter server manager.
        """
        self.config = config
        self.staleness = self.config.psrl.staleness
        if self.config.psrl.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n

        self.data_queue = data_queue
        self.agent_loop_workers = agent_loop_workers
        self.ps_manager_handle = ps_manager_handle

        self._request_counter = 0 # For version tag setting
        
        self._dispatch_idx = 0
        self.running_loop = None
        self.busy_loop_task = None
        self.stop_busy_loop_task = False
        
        self.curr_ps_version_tag = 0
        
        # Build logger
        self.log_prefix = f"AgentLoopManager"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
    
    def start_busy_loop(self):
        """Start the busy loop for continuous data processing from the queue."""
        if self.busy_loop_task is not None and not self.busy_loop_task.done():
            return
        
        # Start the busy loop of agent loop workers
        futures = []
        for worker in self.agent_loop_workers:
            futures.append(worker.start_busy_loop.remote())
        ray.get(futures)

        # Start the background task to process data
        self.running_loop = asyncio.get_running_loop()
        self.busy_loop_task = self.running_loop.create_task(self._dispatch_data())

    def stop_busy_loop(self):
        """Stop the busy loop and wait for all tasks to complete."""
        if not self.busy_loop_task or self.busy_loop_task.done():
            return

        self.stop_busy_loop_task = True
        # Wait for the background task to finish
        self.running_loop.run_until_complete(self.busy_loop_task)

        # Stop the busy loop of agent loop workers
        futures = []
        for worker in self.agent_loop_workers:
            futures.append(worker.stop_busy_loop.remote())
        ray.get(futures)

    async def _dispatch_data(self):
        """Main dispatch loop that processes data from the queue and routes to workers."""
        while not self.stop_busy_loop_task:
            if not self.data_queue.empty():
                data = self.data_queue.get_nowait()
                
                # Receive END signal to stop processing data queue
                if data is None:
                    self.stop_busy_loop_task = True
                    continue

                # Set version tag for each request
                batch_size = len(data)
                request_list = data.chunk(batch_size)
                version_tags = []
                for request in request_list:
                    version_tag = self.set_version_tag(request)
                    version_tags.append(version_tag)
                data.non_tensor_batch["version_tag"] = np.array(version_tags)

                # Wait for version update in ps
                max_version_tag = np.max(version_tags)
                if max_version_tag > self.curr_ps_version_tag:
                    psrl_logger.debug(f"Waiting for ps model version: {max_version_tag}")
                    await self.ps_manager_handle.wait_for_ps_model_version.remote(max_version_tag)
                    self.curr_ps_version_tag = max_version_tag
                    psrl_logger.debug(f"ps model version updated to {self.curr_ps_version_tag}, continue to dispatch")

                # Dispatch data to agent loop workers
                await self._inner_dispatch_data(data)
            await asyncio.sleep(0) # Yield control to the event loop

    def set_version_tag(self, request):
        """
        Set the version tag for the request based on the current staleness and request counter.
        
        NOTE: Currently it's a naive greedy implementation that increments the version tag
        for each request. This may not be optimal in a real-world scenario.
        """
        if self.config.psrl.redundant_rollout.enable:
            buffer_size = self.config.psrl.redundant_rollout.redundant_global_batch_size * self.rollout_n
        else:
            buffer_size = self.config.psrl.staleness_buffer_entries * self.rollout_n

        version_tag = max(self._request_counter - self.staleness * buffer_size, 0) // buffer_size
        self._request_counter += 1
        return version_tag

    async def _inner_dispatch_data(self, data: DataProto):
        """Dispatch data to agent loop workers in a round-robin manner.
        Args:
            data (DataProto): Input data.
        """

        # Update request status from PENDING to RUNNING
        request_ids = data.non_tensor_batch["uid"]
        if "version_tag" in data.non_tensor_batch:
            version_tags = data.non_tensor_batch["version_tag"]
        else:
            version_tags = data.non_tensor_batch["max_version_limit"]
        update_status_success = await self.ps_manager_handle.update_request_status.remote(
            request_ids.tolist(),
            RequestStatus.RUNNING,
            model_version=version_tags.tolist(),
        )
        dispatch_request_idxs = [i for i, success in enumerate(update_status_success) if success]
        if not dispatch_request_idxs:
            return

        dispatch_data = data.select_idxs(dispatch_request_idxs)
        dispatch_plan = self.get_dispatch_plan(dispatch_data)
        
        for worker_index, worker_data in dispatch_plan.items():
            if not worker_data:
                continue
            
            # Dispatch data to the corresponding worker
            if self.config.psrl.gen_mode == "stream":
                # Dispatch `rollout_n` requests
                for i in range(self.rollout_n):
                    self.agent_loop_workers[worker_index].add_agent_program.remote(worker_data[i:i+1])
            else:
                self.agent_loop_workers[worker_index].add_agent_program.remote(worker_data)

    def get_dispatch_plan(self, data: DataProto) -> dict[int, DataProto]:
        """Create a dispatch plan for distributing data across workers.
        
        Args:
            data (DataProto): Data to be distributed.
            
        Returns:
            dict[int, DataProto]: Mapping of worker index to assigned data.
        """
        dispatch_plan = {}
        sample_num = len(data) // self.rollout_n
        for i in range(sample_num):
            sample_idx = i * self.rollout_n
            # Round-robin dispatching
            worker_index = (self._dispatch_idx + i) % len(self.agent_loop_workers)
            if worker_index not in dispatch_plan:
                dispatch_plan[worker_index] = []
            dispatch_plan[worker_index].append(data[sample_idx: (sample_idx + self.rollout_n)])
        
        # Convert lists to DataProto
        for worker_index, data in dispatch_plan.items():
            dispatch_plan[worker_index] = DataProto.concat(data)
        self._dispatch_idx = (self._dispatch_idx + sample_num) % len(self.agent_loop_workers)
        return dispatch_plan

    def retry_request(self, buffer_id: int):
        """Notify the agent loop manager to retry processing requests associated with a specific buffer ID.
        
        Args:
            buffer_id (int): The buffer ID whose requests need to be retried.
        """
        if self.running_loop and not self.stop_busy_loop_task:
            self.running_loop.call_soon_threadsafe(self._retry_request, buffer_id)
        else:
            psrl_logger.warning("Busy loop of the agent loop manager has stopped, the retry operation will be skipped")

    def _retry_request(self, max_version_limit: int):
        """Retry processing requests with a specific version limit."""
        # Get request from the data queue and re-dispatch among instance with versions in [buffer_id - staleness, buffer_id]
        retry_num = self.alg_rollout_n if self.config.psrl.gen_mode == "stream" else 1
        for i in range(retry_num):
            if not self.data_queue.empty():
                data = self.data_queue.get_nowait()
                if data is None:
                    raise ValueError("Data queue should not contain None when retrying requests.")

                data.non_tensor_batch["max_version_limit"] = np.array([max_version_limit] * len(data), dtype=int)
                psrl_logger.debug(f"Retrying to dispatch new requests with ids: {data.non_tensor_batch['uid']}")

                asyncio.run_coroutine_threadsafe(self._inner_dispatch_data(data), self.running_loop)
