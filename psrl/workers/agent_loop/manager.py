import os
import logging
import asyncio
import numpy as np
from omegaconf import DictConfig

import ray

from verl import DataProto

from psrl.workers.ps.request_status_tracker import RequestStatus

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

@ray.remote
class PSRL_AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers."""

    def __init__(
        self,
        config: DictConfig,
        data_queue,
        agent_loop_workers,
        ps_manager_handle,
    ):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): trainer config.
        """
        self.config = config
        self.staleness = self.config.psrl.staleness
        if self.config.psrl.rollout_test.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.rollout_test.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.rollout_test.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n

        self.data_queue = data_queue
        self.agent_loop_workers = agent_loop_workers
        self.ps_manager_handle = ps_manager_handle

        self._request_counter = 0 # For version tag setting
        
        self._dispatch_idx = 0
        self.busy_loop_task = None
        self.stop_busy_loop_task = False
    
    def start_busy_loop(self):
        """Start a busy loop to continuously process data from the queue."""
        if self.busy_loop_task is not None and not self.busy_loop_task.done():
            return

        # Start the background task to process data
        running_loop = asyncio.get_running_loop()
        self.busy_loop_task = running_loop.create_task(self._dispatch_data())

    def stop_busy_loop(self):
        """Stop the busy loop."""
        if not self.busy_loop_task or self.busy_loop_task.done():
            return

        self.stop_busy_loop_task = True
        # Wait for the background task to finish
        running_loop = asyncio.get_running_loop()
        running_loop.run_until_complete(self.busy_loop_task)

    async def _dispatch_data(self):
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

                # Dispatch data to agent loop workers
                self._inner_dispatch_data(data)
            await asyncio.sleep(0) # Yield control to the event loop

    def set_version_tag(self, request):
        """
        Set the version tag for the request based on the current staleness and request counter.
        
        NOTE: Currently it's a naive greedy implementation that increments the version tag
        for each request. This may not be optimal in a real-world scenario.
        """
        if self.config.psrl.rollout_test.redundant_rollout.enable:
            buffer_size = self.config.psrl.rollout_test.redundant_rollout.redundant_global_batch_size * self.rollout_n
        else:
            buffer_size = self.config.psrl.staleness_buffer_entries * self.rollout_n

        version_tag = max(self._request_counter - self.staleness * buffer_size, 0) // buffer_size
        self._request_counter += 1
        return version_tag

    def _inner_dispatch_data(self, data: DataProto) -> dict[int, DataProto]:
        """Dispatch data to agent loop workers in a round-robin manner.
        Args:
            data (DataProto): Input data.
        Returns:
            dict[int, DataProto]: A dictionary mapping worker index to DataProto.
        """

        # Update request status from PENDING to RUNNING
        request_ids = data.non_tensor_batch["uid"]
        update_status_success = ray.get(self.ps_manager_handle.update_request_status.remote(
            request_ids.tolist(),
            RequestStatus.RUNNING,
        ))
        dispatch_request_idxs = [i for i, success in enumerate(update_status_success) if success]
        if not dispatch_request_idxs:
            return

        dispatch_data = data.select_idxs(dispatch_request_idxs)
        dispatch_plan = self.get_dispatch_plan(dispatch_data)
        
        futures = []
        for worker_index, worker_data in dispatch_plan.items():
            if not worker_data:
                continue
            
            # Dispatch data to the corresponding worker
            futures.append(self.agent_loop_workers[worker_index].add_agent_program.remote(worker_data))
        ray.get(futures)

    def get_dispatch_plan(self, data: DataProto) -> dict[int, DataProto]:
        dispatch_plan = {}
        batch_size = len(data)
        for i in range(batch_size):
            # Round-robin dispatching
            worker_index = (self._dispatch_idx + i) % len(self.agent_loop_workers)
            if worker_index not in dispatch_plan:
                dispatch_plan[worker_index] = []
            dispatch_plan[worker_index].append(data[i:i+1])
        
        # Convert lists to DataProto
        for worker_index, data in dispatch_plan.items():
            dispatch_plan[worker_index] = DataProto.concat(data)
        self._dispatch_idx = (self._dispatch_idx + batch_size) % len(self.agent_loop_workers)
        return dispatch_plan
