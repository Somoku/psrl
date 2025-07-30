import os
import logging
import numpy as np
from typing import Any
from threading import Thread

import torch

import ray

from verl import DataProto
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.torch_functional import pad_2d_list_to_length

from psrl.utils.logger import log_dual_events, EventType
from psrl.utils.server.command import Command, CommandType, CommandExtension
from psrl.workers.request_manager.request_status_manager import RequestStatus, RequestStatusManager

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

@ray.remote
class RewardServer(CommandExtension):
    def __init__(
        self,
        config,
        tokenizer,
        ps_manager_handle,
        rollout_queue,
        request_status_manager,
        reward_fn=None,
        use_rm=False,
    ):
        """
        Initialize the reward server with the given configuration, tokenizer and communication handles.
        
        Args:
            config: Configuration object containing server settings.
            tokenizer: Tokenizer for processing text data.
            ps_manager_handle: Handle to the parameter server for communication.
            rollout_queue: Queue for receiving rollout data from rollout server.
            request_status_manager: Manager for tracking request statuses.
            reward_fn: Optional function for computing rewards.
            use_rm: Boolean indicating whether to use a reward model.
        """
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer
        if self.config.psrl.rollout_test.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.rollout_test.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.rollout_test.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n
        assert self.rollout_n >= self.alg_rollout_n, \
            f"Rollout n {self.rollout_n} must be greater than or equal to alg_rollout_n {self.alg_rollout_n}."

        # Queues for data transfer between workers
        self.rollout_queue = rollout_queue
        
        # Server state management
        self.server_running = False
        self.reward_paused = False
        self.skipping_rollout_queue = False
        
        # Reward model configuration
        self.use_rm = use_rm
        self.reward_fn = reward_fn
        self.reward_futures = []
        self.request_id_to_future = {}
        psrl_logger.debug(f"Reward configuration: use_rm={use_rm}, reward_fn_type={type(reward_fn).__name__ if reward_fn else None}")
        
        # Background event handler
        self._threads = []
        
        # Communication handles
        self.ps_manager_handle = ps_manager_handle
        self.request_status_manager = request_status_manager
        
    def start_server(self):
        """
        Start the reward server and begin processing requests.
        
        This method initializes the server and starts the background event handler
        for processing rollout data and computing rewards.
        """
        if self.server_running:
            psrl_logger.debug("Server already running, ignoring start_server call")
            return
        
        self.server_running = True
        
        # Start the background event handler thread
        psrl_logger.debug("Creating background event handler thread")
        event_handler = Thread(
            target=self._background_event_handler,
            name="reward_event_thread",
            daemon=True,
        )
        
        event_handler.start()
        self._threads = [event_handler]
        psrl_logger.debug("Background event handler thread started")
    
    def shutdown_server(self):
        """Shutdown the reward server gracefully."""
        if not self.server_running:
            psrl_logger.debug("Server not running, ignoring shutdown_server call")
            return
        
        psrl_logger.info("Shutdown reward server...")
        self.server_running = False

        # Stop the background event handler
        for thread in self._threads:
            psrl_logger.debug(f"Waiting for thread {thread.name} to finish")
            if thread.is_alive():
                thread.join(timeout=10)
                if thread.is_alive():
                    psrl_logger.warning(f"Thread {thread.name} did not finish within timeout")
        self._threads = []

        psrl_logger.info("Reward server shutdown.")

    def stop_server(self):
        """Stop the reward server and pause reward processing if it is running."""
        if not self.reward_paused:
            psrl_logger.info("Stopping reward server...")
            self.exec_command(Command(CommandType.STOP))
            psrl_logger.info("Reward server stopped.")

    def resume_server(self):
        """Resume the reward server if it was paused."""
        if self.reward_paused:
            psrl_logger.info("Resuming reward server...")
            self.exec_command(Command(CommandType.RESUME))
            psrl_logger.info("Reward server processing resumed")
    
    def _background_event_handler(self):
        """
        Background event handler for processing commands and rollout data.
        
        This method runs in a separate process and handles:
        1. Command processing (ABORT, STOP, RESUME, etc.)
        2. Rollout data processing from the rollout queue (logprobs, etc.)
        3. Reward computation and status updates.
        4. Occupy requests in the PS worker.
        """
        psrl_logger.debug("Starting background event handler thread in RewardServer")
        while self.server_running:
            # Command processing
            if not self.command_queue.empty():
                # Get command from the queue
                psrl_logger.debug("Command queue is not empty, try to get commands")
                command = self.command_queue.get()

                assert isinstance(command, Command), f"Expected Command, got {type(command)}"

                # Unpack command attributes
                command_type = command.type
                command_id = command.get_kwargs()["id"]
                command_args = command.get_args()

                psrl_logger.debug(f"Command: type = {command_type}, kwargs = {command.get_kwargs()}, args = {command_args}")
                
                result = None
                
                # Process the command based on its type
                if command_type == CommandType.STOP:
                    psrl_logger.debug("Begin to pause reward processing")
                    self.reward_paused = True # No more new requests will be processed
                    psrl_logger.debug(f"Received STOP command, paused reward processing")
                elif command_type == CommandType.RESUME:
                    psrl_logger.debug("Begin to resume reward processing")
                    self.reward_paused = False
                    psrl_logger.debug("Resumed reward processing")
                elif command_type == CommandType.ABORT:
                    assert "parent_ids" in command_args or "uids" in command_args, \
                        "Abort command must contain either 'parent_ids' or 'uids' in args."
                    parent_ids = command_args.get("parent_ids", None)
                    uids = command_args.get("uids", None)

                    if parent_ids is None and uids is None:
                        raise ValueError("Abort command must contain either 'parent_ids' or 'uids' in args.")
                    
                    psrl_logger.debug(f"Received ABORT command with parent_ids: {parent_ids}, uids: {uids}")
                    
                    if not isinstance(parent_ids, (list, type(None))):
                        parent_ids = [parent_ids]
                    if not isinstance(uids, (list, type(None))):
                        uids = [uids]

                    # Collect all requests to be aborted
                    abort_request_uids = set()
                    # Step 1. Get child requests from parent_ids
                    if parent_ids is not None:
                        parent_ids = set(parent_ids) # Ensure uniqueness
                        psrl_logger.debug(f"Getting child requests for {len(parent_ids)} parent_ids")
                        child_uids = ray.get(self.request_status_manager.get_recorded_child_requests.remote(list(parent_ids)))
                        psrl_logger.debug(f"Found {len(child_uids)} child requests for the parent_ids")
                        abort_request_uids.update(child_uids)
                    # Step 2. Get requests from uids
                    if uids is not None:
                        uids = set(uids)
                        psrl_logger.debug(f"Adding {len(uids)} direct uids to abort list")
                        abort_request_uids.update(uids)

                    psrl_logger.debug(f"Total of {len(abort_request_uids)} requests to abort")

                    # Abort requests in the reward server
                    # request_id -> reward_future
                    # 1. Kill running reward computation futures
                    aborted_count = 0
                    for abort_request_id in abort_request_uids:
                        psrl_logger.debug(f"Checking if request_id {abort_request_id} has a running reward computation")
                        future_data = self.request_id_to_future.pop(abort_request_id, None)
                        if future_data is not None:
                            _, reward_future = future_data
                            psrl_logger.debug(f"Killing reward computation future for request_id {abort_request_id}")
                            ray.kill(reward_future, no_restart=True)
                            aborted_count += 1
                    
                    psrl_logger.debug(f"Aborted {aborted_count} running reward computations")
                    
                    # 2. Remove from the request tracker (update_status)
                    update_status_success = ray.get(self.request_status_manager.update_status.remote(list(abort_request_uids), RequestStatus.REWARD_COMPLETED))
                    assert all(not status for status in update_status_success), "Update status should not be successful for aborted requests."
                    result = aborted_count
                else:
                    raise ValueError(f"Unknown command type: {command_type}")
                
                # Post process the command
                psrl_logger.debug(f"Completing command {command_id} with result: {result}")
                self._complete_command(command_id, result)

            # Data processing
            # Process requests in the rollout queue
            if not self.rollout_queue.empty() and not self.reward_paused and not self.skipping_rollout_queue:
                psrl_logger.debug("Rollout queue is not empty, processing rollout data")
                rollout_data = self.rollout_queue.get_nowait()
                
                assert rollout_data is not None, "Data from rollout queue should not be None"
                assert len(rollout_data) == 1, "Rollout data should contain exactly one request"
                
                request_ids = rollout_data.non_tensor_batch["uid"]
                psrl_logger.debug(f"Processing rollout data for request_id: {request_ids[0]}")
                
                # Update the request status to REWARD_RUNNING
                psrl_logger.debug(f"Updating status for request_id {request_ids[0]} to REWARD_RUNNING")
                update_status_success = ray.get(self.request_status_manager.update_status.remote(request_ids.tolist(), RequestStatus.REWARD_RUNNING))
                if not update_status_success[0]:
                    psrl_logger.debug(f"Failed to update status for request_id {request_ids[0]}, skipping")
                    continue

                rollout_data.non_tensor_batch.pop("raw_prompt_ids")
                rollout_data.non_tensor_batch.pop("raw_response_ids")
                
                # Rollout log probs processing
                if self.config.psrl.log_prob.enable_inference_engine_log_prob:
                    psrl_logger.debug("Processing inference engine log probs")
                    device = rollout_data.batch["input_ids"].device
                    rollout_log_probs = rollout_data.non_tensor_batch.pop("rollout_log_probs", None)
                    assert rollout_log_probs is not None, "rollout_log_probs should not be None"
                    psrl_logger.debug(f"log probs shape before padding: {rollout_log_probs.shape}")
                    rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=self.config.gen_actor_rollout_ref.rollout.response_length).to(device)
                    rollout_log_probs = rollout_log_probs.to(torch.float32)
                    rollout_data.batch["rollout_log_probs"] = rollout_log_probs
                    psrl_logger.debug(f"Processed log probs shape: {rollout_log_probs.shape}")
                
                if self.rollout_n > 1:
                    sample_ids = rollout_data.non_tensor_batch["parent_id"]
                else:
                    sample_ids = rollout_data.non_tensor_batch["uid"]
                rollout_instance_ids = rollout_data.non_tensor_batch["rollout_instance_id"]
                version_tags = rollout_data.non_tensor_batch["version_tag"]

                for i, (sample_id, request_id, rollout_instance_id, version_tag) in enumerate(zip(sample_ids, request_ids, rollout_instance_ids, version_tags)):
                    request_data = ray.get(self.request_status_manager.get_request_data_from_buffer.remote(sample_id))
                    if request_data is None:
                        # If request data is None, it means the request has been aborted or not found.
                        assert self.rollout_n > 1, "Request data should not be None when rollout_n is 1."
                        continue
                    psrl_logger.debug(f"Got request data with keys: {list(request_data.batch.keys()) if hasattr(request_data, 'batch') else 'N/A'}")
                    response_data = rollout_data[i:i+1]
                    merge_request_data = response_data.union(request_data)
                    psrl_logger.debug(f"Merged request data with keys: {list(merge_request_data.batch.keys()) if hasattr(merge_request_data, 'batch') else 'N/A'}")
                    
                    batch_keys_to_pop = ["prompts", "attention_mask", "responses"]
                    non_tensor_batch_keys_to_pop = ["reward_model"]
                    if "extra_info" in merge_request_data.non_tensor_batch:
                        non_tensor_batch_keys_to_pop.append("extra_info")
                    if "data_source" in merge_request_data.non_tensor_batch:
                        non_tensor_batch_keys_to_pop.append("data_source")
                        
                    psrl_logger.debug(f"Popping keys for reward input - batch: {batch_keys_to_pop}, non_tensor: {non_tensor_batch_keys_to_pop}")
                    reward_input = merge_request_data.pop(
                        batch_keys=batch_keys_to_pop,
                        non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                    )
                    
                    # Reward computation
                    if self.use_rm:
                        pass
                    elif self.config.reward_model.launch_reward_fn_async:
                        with log_dual_events("Launch async reward model score", psrl_logger, event_type=EventType.OTHER):
                            future_reward = compute_reward_async.remote(reward_input, self.config, self.tokenizer)
                            psrl_logger.debug("Merging reward_input back into merge_request_data")
                            merge_request_data.union(reward_input)
                            
                            if self.config.psrl.log_prob.enable_inference_engine_log_prob:
                                psrl_logger.debug(f"Storing future for request_id: {request_id}")
                                self.request_id_to_future[request_id] = (merge_request_data, future_reward)
                                self.reward_futures.append(future_reward)
                                psrl_logger.debug(f"Now tracking {len(self.reward_futures)} reward futures")
                            else:
                                psrl_logger.debug("Directly storing future reward reference")
                                merge_request_data.non_tensor_batch["reward"] = np.array([future_reward])
                    else:
                        with log_dual_events("Compute reward model score", psrl_logger, event_type=EventType.OTHER):
                            reward_tensor, reward_extra_infos_dict = compute_reward(reward_input, self.reward_fn)
                            psrl_logger.debug(f"Got reward tensor: {reward_tensor} and {len(reward_extra_infos_dict)} extra info items")
                            
                            merge_request_data.union(reward_input)
                            merge_request_data.batch["reward"] = reward_tensor
                            merge_request_data.meta_info["reward_extra_info_keys"] = list(reward_extra_infos_dict.keys())
                            merge_request_data.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                            psrl_logger.debug("Added reward data to merge_request_data")

                        # Update the request status to REWARD_COMPLETED
                        update_status_success = ray.get(self.request_status_manager.update_status.remote(int(request_id), RequestStatus.REWARD_COMPLETED))
                        complete_request_idxs = [
                            i for i, success in enumerate(update_status_success) if success
                        ]
                        
                        if complete_request_idxs:
                            with log_dual_events("Occupy requests", psrl_logger, event_type=EventType.OTHER):
                                future = self.ps_manager_handle.store_and_maybe_occupy_rollout_instance_request.remote(
                                    rollout_instance_id=int(rollout_instance_id),
                                    request_id=int(request_id),
                                    version_tag=version_tag,
                                    data=merge_request_data,
                                    parent_id=sample_id if self.rollout_n > 1 else None,
                                )
                                psrl_logger.debug(f"Occupy future for request {request_id} with rollout_instance_id {rollout_instance_id} and version_tag {version_tag}")
                                ray.get(future)

            # Check if any reward futures are ready
            if (
                self.config.psrl.log_prob.enable_inference_engine_log_prob and
                self.config.reward_model.launch_reward_fn_async
            ):
                psrl_logger.debug(f"Checking {len(self.reward_futures)} reward futures for completion")
                ready_rewards, self.reward_futures = ray.wait(self.reward_futures)
                psrl_logger.debug(f"{len(ready_rewards)} rewards are ready, {len(self.reward_futures)} still in progress")

                running_requests = set(self.request_id_to_future.keys())
                finished_request_data = []
                
                # Process the ready rewards
                for request_id in running_requests:
                    request_data, reward_future = self.request_id_to_future[request_id]
                    if reward_future in ready_rewards:
                        self.request_id_to_future.pop(request_id, None)
                        psrl_logger.debug("Getting reward tensor and extra info from future")
                        reward_tensor, reward_extra_infos_dict = ray.get(reward_future)
                        psrl_logger.debug(f"Adding reward tensor: {reward_tensor} and {len(reward_extra_infos_dict)} extra info items to request data")
                        
                        request_data.batch["reward"] = reward_tensor
                        request_data.meta_info["reward_extra_info_keys"] = list(reward_extra_infos_dict.keys())
                        request_data.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                        finished_request_data.append(request_data)
                        psrl_logger.debug(f"Added request_id {request_id} to finished_request_data")
                
                if finished_request_data:
                    finished_request_data = DataProto.concat(finished_request_data)
                    request_ids = finished_request_data.non_tensor_batch["uid"]
                    
                    # Update the request status to REWARD_COMPLETED
                    psrl_logger.debug(f"Updating status for {len(request_ids)} requests to REWARD_COMPLETED")
                    update_status_success = ray.get(self.request_status_manager.update_status.remote(request_ids.tolist(), RequestStatus.REWARD_COMPLETED))
                    complete_request_idxs = [
                        i for i, success in enumerate(update_status_success) if success
                    ]
                    psrl_logger.debug(f"{len(complete_request_idxs)}/{len(request_ids)} requests successfully updated to REWARD_COMPLETED")
                    
                    # If requests are completed, occupy them in the PS worker
                    if complete_request_idxs:
                        complete_request_data = finished_request_data[complete_request_idxs]
                        psrl_logger.debug(f"Processing {len(complete_request_data)} completed requests")
                        
                        occupy_futures = []
                        for i in range(len(complete_request_data)):
                            request_data = complete_request_data[i:i+1]
                            request_id = request_data.non_tensor_batch["uid"][0]
                            sample_id = request_data.non_tensor_batch["parent_id"][0] if self.rollout_n > 1 else request_id
                            rollout_instance_id = request_data.non_tensor_batch["rollout_instance_id"][0]
                            version_tag = request_data.non_tensor_batch["version_tag"][0]
                            
                            psrl_logger.debug(f"Occupying request_id {request_id} from rollout_instance_id {rollout_instance_id}")
                            occupy_futures.append(
                                self.ps_manager_handle.store_and_maybe_occupy_rollout_instance_request.remote(
                                    rollout_instance_id=int(rollout_instance_id),
                                    request_id=int(request_id),
                                    version_tag=version_tag,
                                    data=request_data,
                                    parent_id=sample_id if self.rollout_n > 1 else None,
                            ))
                        psrl_logger.debug(f"Waiting for {len(occupy_futures)} occupy operations to complete")
                        with log_dual_events("Occupy requests", psrl_logger, event_type=EventType.OTHER):
                            ray.get(occupy_futures)
                        psrl_logger.debug("All occupy operations completed")

        psrl_logger.info("Background event handler of reward server has finished.")
