import os
import logging
import asyncio
import torch
import ray
import numpy as np
from queue import Queue
from typing import Dict, List, Union, Set
from threading import Thread
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import pad_2d_list_to_length

from psrl.utils.dataset.utils import _pre_process_inputs
from psrl.utils.logger import log_data_protocol, log_single_event, log_dual_events, EventType, DualOutputHandler
from psrl.utils.server.command import Command, CommandType, CommandExtension
from psrl.workers.ps.request_status_tracker import RequestStatus
from psrl.workers.ps.staleness_controller import EntryInfo

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class RewardServer(CommandExtension):
    def __init__(
        self,
        config,
        tokenizer,
        processor,
        ps_manager_handle,
        agent_loop_manager,
        rollout_queue_size,
        reward_fn=None,
        use_rm=False,
        group_post_process_fn=None,
        buffer_post_process_fn=None,
    ):
        """Initialize the reward server for processing rollout data and computing rewards.
        
        The reward server receives rollout data from rollout workers, computes rewards
        using either rule-based functions or reward models, and sends the results
        to the parameter server for training.
        
        Args:
            config: Configuration object containing server settings and hyperparameters
            tokenizer: Tokenizer for processing text data and converting tokens
            processor: Processor for processing multi-modal data
            ps_manager_handle: Handle to the parameter server for status updates and communication
            agent_loop_manager: Handle to the agent loop manager
            rollout_queue_size: Size of the queue for receiving rollout data from agent loop workers
            reward_fn (optional): Custom function for computing rewards. Defaults to None
            use_rm (bool): Whether to use a reward model for computing rewards. Defaults to False
            group_post_process_fn (Optional[callable]): Optional function to post-process 
                grouped entry data before occupying the buffer
            buffer_post_process_fn (Optional[callable]): Optional function to post-process 
                ready buffer data
        """
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.group_post_process_fn = group_post_process_fn
        self.buffer_post_process_fn = buffer_post_process_fn
        if self.config.psrl.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n
        assert self.rollout_n >= self.alg_rollout_n, \
            f"Rollout n {self.rollout_n} must be greater than or equal to alg_rollout_n {self.alg_rollout_n}."

        # Queues for data transfer between workers
        self.rollout_queue = asyncio.Queue(maxsize=rollout_queue_size)
        
        # Server state management
        self.reward_paused = False
        self.skipping_rollout_queue = False
        
        # Reward model configuration
        self.use_rm = use_rm
        self.reward_fn = reward_fn
        self.reward_futures = []
        self.request_id_to_future = {}
        
        # Background event handler
        self.running_loop = None
        self.busy_loop_task = None
        self.stop_busy_loop_task = False
        
        # Communication handles
        self.ps_manager_handle = ps_manager_handle
        
        # Data
        self.request_buffer = {} # Maps sample IDs to request DataProto (for merging with rollout data)
        self.data_pool: Dict[EntryInfo, DataProto] = {} # Maps EntryInfo to stored/occupied DataProto
        self.data_buffers: Dict[int, DataProto] = {} # data of READY buffer in ps manager
        self.max_ready_buffer_id = -1 # Max buffer id that has been processed and logged as READY
        
        # Set of buffer ids that have been logged as ready, to avoid duplicate logging
        self.logged_ready_buffer_ids: Set[int] = set()
        
        # Waiting lists for training batches
        self._buffer_waiters: Dict[int, List[asyncio.Future]] = {}  # Maps buffer IDs to a set of futures waiting for that buffer
        
        # Track finished child requests for Group Sampling
        self.rollout_request_tracker: Dict[Union[str, int], List[EntryInfo]] = {} # Maps parent request ids to "occupied" child entries
        
        # Agent Loop Manager reference
        self.agent_loop_manager = agent_loop_manager
        
        # Build logger
        self.log_prefix = "RewardServer"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        psrl_logger.info(f"Initialized RewardServer.")
        
    def add_requests(self, sample_id_to_request_data: Dict[int, DataProto]):
        self.request_buffer.update(sample_id_to_request_data)
        
    def remove_requests(self, sample_ids: List[int]):
        for sample_id in sample_ids:
            self.request_buffer.pop(sample_id, None)

    def start_busy_loop(self):
        """Start the reward server and begin processing requests.
        
        This method initializes the server state and starts the background event handler
        task for processing rollout data and computing rewards. The server will run
        until explicitly stopped.
        """
        if self.busy_loop_task is not None and not self.busy_loop_task.done():
            return

        # Start the background task to process data
        self.running_loop = asyncio.get_running_loop()
        self.busy_loop_task = self.running_loop.create_task(self._background_event_handler())
        self.busy_loop_task.add_done_callback(lambda f: f.result()) # To avoid silent error in async tasks
    
    async def stop_busy_loop(self):
        """Shutdown the reward server gracefully.
        
        This method stops the server and waits for all background tasks
        to complete before returning.
        """
        if not self.busy_loop_task or self.busy_loop_task.done():
            return

        self.stop_busy_loop_task = True
        # Wait for the background task to finish
        await self.busy_loop_task

    def stop_server(self):
        """Stop the reward server and pause reward processing.
        
        This method sends a STOP command to pause reward processing
        without shutting down the server completely.
        """
        if not self.reward_paused:
            self.exec_command(Command(CommandType.STOP))

    def resume_server(self):
        """Resume the reward server if it was paused.
        
        This method sends a RESUME command to restart reward processing
        after it was paused by stop_server().
        """
        if self.reward_paused:
            self.exec_command(Command(CommandType.RESUME))
            
    def _pre_process(self, inputs: DataProto) -> DataProto:
        """Pre-process the generated outputs to create properly formatted tensors.
        
        This method handles padding, attention masks, position IDs, and multi-modal inputs
        to ensure compatibility with the training pipeline.
        
        Args:
            inputs (DataProto): Raw generation outputs.
            
        Returns:
            DataProto: Formatted data ready for training.
        """
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        log_data_protocol(inputs, psrl_logger, self.log_prefix + " before preprocess data from rollout queue", level=logging.DEBUG)

        # prompts
        self.tokenizer.padding_side = "left"
        if "raw_prompt_ids" not in inputs.non_tensor_batch:
            batch_size = len(inputs)
            raw_prompt_ids = np.array(
                [_pre_process_inputs(self.tokenizer.pad_token_id, inputs.batch["input_ids"][i]) for i in range(batch_size)], dtype=object
            )
        else:
            raw_prompt_ids = inputs.non_tensor_batch["raw_prompt_ids"]

        prompt_output = self.tokenizer.pad(
            [{"input_ids": raw_prompt_id} for raw_prompt_id in raw_prompt_ids],
            padding="max_length",
            max_length=self.config.gen_actor_rollout_ref.rollout.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        prompt_ids, prompt_attention_mask = prompt_output["input_ids"], prompt_output["attention_mask"]

        # responses
        raw_response_ids = inputs.non_tensor_batch.pop("raw_response_ids", None)
        assert raw_response_ids is not None, "raw_response_ids must be provided in the input batch"
        self.tokenizer.padding_side = "right"
        outputs = self.tokenizer.pad(
            [{"input_ids": raw_response_id} for raw_response_id in raw_response_ids],
            padding="max_length",
            max_length=self.config.gen_actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        response_ids, response_attention_mask = outputs["input_ids"], outputs["attention_mask"]

        # response_mask
        response_masks = inputs.non_tensor_batch.pop("response_mask", None)
        assert response_masks is not None, "response_masks must be provided in the input batch"
        outputs = self.tokenizer.pad(
            [{"input_ids": response_mask} for response_mask in response_masks],
            padding="max_length",
            max_length=self.config.gen_actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        # [bsz, response_length], each row is [1, 1, ..., 1, 0, 0, ..., 0] 
        # Currently no tool call, it is the same as the response_attention_mask
        # Only need to note that it exclude the eos token
        response_mask = outputs["input_ids"] 

        assert response_ids.shape == response_mask.shape, (
            f"mismatch in response_ids and response_mask shape: {response_ids.shape} vs {response_mask.shape}"
        )
        
        response_mask = response_mask * response_attention_mask
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        # Handle multi-modal inputs and position_ids calculation
        # Only support Qwen2VLImageProcessor for multi-modal processing currently
        # TODO(verl): support other multi-modal inputs
        multi_modal_inputs = None
        if (
            self.processor is not None
            and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
        ):
            from verl.models.transformers.qwen2_vl import get_rope_index

            images = inputs.non_tensor_batch["multi_modal_data"].get("image", None)
            current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
            multi_modal_inputs = self.processor(text=[current_text], images=images, return_tensors="pt")
            multi_modal_inputs.pop("input_ids", None)
            multi_modal_inputs.pop("attention_mask", None)

            # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
            # because np.array() only keeps the keys for BatchFeature.
            multi_modal_inputs = dict(multi_modal_inputs)

            image_grid_thw = multi_modal_inputs.get("image_grid_thw")
            video_grid_thw = multi_modal_inputs.get("video_grid_thw")
            second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts")

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids.squeeze(0),
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask.squeeze(0),
            ).unsqueeze(0)  # (1, 3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)  

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                "position_ids": position_ids,  # [bsz, prompt_length + response_length]
            },
            batch_size=len(input_ids),
        )
        # Rollout log probs processing
        if self.config.psrl.log_prob.enable_rollout_engine_log_prob:
            device = batch["input_ids"].device
            rollout_log_probs = inputs.non_tensor_batch.pop("rollout_log_probs", None)
            assert rollout_log_probs is not None, "rollout_log_probs should not be None"
            rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=self.config.gen_actor_rollout_ref.rollout.response_length).to(device)
            rollout_log_probs = rollout_log_probs.to(torch.float32)
            batch["rollout_log_probs"] = rollout_log_probs
            
        inputs.non_tensor_batch.pop("raw_prompt_ids", None)
        inputs.non_tensor_batch.pop("raw_response_ids", None)
        non_tensor_batch = inputs.non_tensor_batch
        if multi_modal_inputs is not None:
            non_tensor_batch["multi_modal_inputs"] = multi_modal_inputs

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    
    async def put_data(self, data: DataProto):
        """Put objectref of rollout data into the reward server's processing queue.
        
        This method is used by agent loop workers to send generated rollout data
        to the reward server for reward computation.
        
        Args:
            data (DataProto): DataProto containing the rollout data.
        """
        await self.rollout_queue.put(data)

    async def _background_event_handler(self):
        """
        Background event handler for processing commands and rollout data.
        
        This method runs in a separate process and handles:
        1. Command processing (ABORT, STOP, RESUME, etc.)
        2. Rollout data processing from the rollout queue (logprobs, etc.)
        3. Reward computation and status updates.
        4. Occupy requests in the PS worker.
        """
        while not self.stop_busy_loop_task:
            # Command processing
            if not self.command_queue.empty():
                # Get command from the queue
                command = self.command_queue.get_nowait()

                assert isinstance(command, Command), f"Expected Command, got {type(command)}"

                # Unpack command attributes
                command_type = command.type
                command_id = command.get_kwargs()["id"]
                command_args = command.get_args()
                psrl_logger.debug(f"Receive command: type = {command_type}, kwargs = {command.get_kwargs()}, args = {command_args}")

                result = None
                
                # Process the command based on its type
                if command_type == CommandType.STOP:
                    self.reward_paused = True # No more new requests will be processed
                elif command_type == CommandType.RESUME:
                    self.reward_paused = False
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
                        child_uids = await self.ps_manager_handle.get_recorded_child_requests.remote(list(parent_ids))
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
                        future_data = self.request_id_to_future.pop(abort_request_id, None)
                        if future_data is not None:
                            _, reward_future = future_data
                            ray.kill(reward_future, no_restart=True)
                            aborted_count += 1
                    
                    psrl_logger.debug(f"Aborted {aborted_count} running reward computations")
                    
                    # 2. Remove from the request tracker (update_status)
                    update_status_success = await self.ps_manager_handle.update_request_status.remote(list(abort_request_uids), RequestStatus.REWARD_COMPLETED)
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
                rollout_data = self.rollout_queue.get_nowait()
                
                with log_dual_events("Process rollout data", psrl_logger, level=logging.DEBUG, event_type=EventType.OTHER):
                    assert rollout_data is not None, "Data from rollout queue should not be None"
                    # assert len(rollout_data) == 1, "Rollout data should contain exactly one request"
                    rollout_data = self._pre_process(rollout_data)
                    psrl_logger.debug(f"Rollout data after pre-process, "
                                      f"prompt length: {(rollout_data.batch['prompts'] != self.tokenizer.pad_token_id).sum(dim=-1)}, "
                                      f"response length: {(rollout_data.batch['responses'] != self.tokenizer.pad_token_id).sum(dim=-1)}, "
                                      f"attention_mask sum: {rollout_data.batch['attention_mask'].sum(dim=-1)}")
                    request_ids = rollout_data.non_tensor_batch["uid"]
                    # print(f"Reward server received request ids: {request_ids}")
                    
                    # Update the request status to REWARD_RUNNING
                    update_status_success = await self.ps_manager_handle.update_request_status.remote(request_ids.tolist(), RequestStatus.REWARD_RUNNING)
                    if not update_status_success[0]:
                        continue
                    
                    if self.rollout_n > 1:
                        sample_ids = rollout_data.non_tensor_batch["parent_id"]
                    else:
                        sample_ids = rollout_data.non_tensor_batch["uid"]
                    rollout_instance_ids = rollout_data.non_tensor_batch["rollout_instance_id"]
                    version_tags = rollout_data.non_tensor_batch["version_tag"]

                with log_dual_events(f"Compute reward for samples {sample_ids} and requests {request_ids}", psrl_logger, level=logging.DEBUG, event_type=EventType.OTHER):
                    for i, (sample_id, request_id, rollout_instance_id, version_tag) in enumerate(zip(sample_ids, request_ids, rollout_instance_ids, version_tags)):
                        request_data = self.request_buffer.get(sample_id, None)
                        assert request_data is not None, "Request data should not be None."
                        '''
                        if request_data is None:
                            # If request data is None, it means the request has been aborted or not found.
                            assert self.rollout_n > 1, "Request data should not be None when rollout_n is 1."
                            continue
                        '''
                        response_data = rollout_data[i:i+1]
                        merge_request_data = response_data.union(request_data)
                        
                        batch_keys_to_pop = ["prompts", "attention_mask", "responses"]
                        non_tensor_batch_keys_to_pop = ["reward_model"]
                        if "extra_info" in merge_request_data.non_tensor_batch:
                            non_tensor_batch_keys_to_pop.append("extra_info")
                        if "data_source" in merge_request_data.non_tensor_batch:
                            non_tensor_batch_keys_to_pop.append("data_source")
                            
                        reward_input = merge_request_data.pop(
                            batch_keys=batch_keys_to_pop,
                            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                        )
                        
                        # Reward computation
                        if self.use_rm:
                            pass
                        elif self.config.reward_model.launch_reward_fn_async:
                            with log_dual_events("Launch async reward model score", psrl_logger, level=logging.DEBUG, event_type=EventType.OTHER):
                                future_reward = compute_reward_async.remote(reward_input, self.config, self.tokenizer)
                                merge_request_data.union(reward_input)
                                # TODO(lhy): currently seems that cannot overlap with log prob recomputing.
                                if self.config.psrl.log_prob.enable_rollout_engine_log_prob:
                                    self.request_id_to_future[request_id] = (merge_request_data, future_reward)
                                    self.reward_futures.append(future_reward)
                                else:
                                    merge_request_data.non_tensor_batch["reward"] = np.array([future_reward])
                        else:
                            with log_dual_events("Compute reward model score", psrl_logger, level=logging.DEBUG, event_type=EventType.OTHER):
                                reward_tensor, reward_extra_infos_dict = compute_reward(reward_input, self.reward_fn)
                                merge_request_data.union(reward_input)
                                merge_request_data.batch["reward"] = reward_tensor
                                merge_request_data.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            # Update the request status to REWARD_COMPLETED
                            update_status_success = await self.ps_manager_handle.update_request_status.remote(int(request_id), RequestStatus.REWARD_COMPLETED)
                            complete_request_idxs = [
                                i for i, success in enumerate(update_status_success) if success
                            ]
                            
                            if complete_request_idxs:
                                await self.occupy_requests(merge_request_data[complete_request_idxs])

            # Check if any reward futures are ready
            if self.config.reward_model.launch_reward_fn_async:
                ready_rewards, self.reward_futures = ray.wait(self.reward_futures)
                running_requests = set(self.request_id_to_future.keys())
                finished_request_data = []
                
                # Process the ready rewards
                for request_id in running_requests:
                    request_data, reward_future = self.request_id_to_future[request_id]
                    if reward_future in ready_rewards:
                        self.request_id_to_future.pop(request_id, None)
                        reward_tensor, reward_extra_infos_dict = await reward_future
                        
                        request_data.batch["reward"] = reward_tensor
                        request_data.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                        finished_request_data.append(request_data)
                
                if finished_request_data:
                    finished_request_data = DataProto.concat(finished_request_data)
                    request_ids = finished_request_data.non_tensor_batch["uid"]
                    
                    # Update the request status to REWARD_COMPLETED
                    update_status_success = await self.ps_manager_handle.update_request_status.remote(request_ids.tolist(), RequestStatus.REWARD_COMPLETED)
                    complete_request_idxs = [
                        i for i, success in enumerate(update_status_success) if success
                    ]
                    
                    # If requests are completed, occupy them in the PS worker
                    if complete_request_idxs:
                        complete_request_data = finished_request_data[complete_request_idxs]
                        await self.occupy_requests(complete_request_data)
            await asyncio.sleep(0)

        psrl_logger.info("Background event handler of reward server has finished.")

    async def occupy_requests(self, request_data: DataProto):
        """
        Try to occupy the requests in the PS worker and manage the data buffers.
        This method attempts to occupy the requests in the PS worker by communicating
        with the PS manager. It also manages the data buffers and handles group post-processing
        if applicable.
        
        Args:
            request_data (DataProto): DataProto containing the requests to be occupied.
        """
        
        # Add data to the data pool
        for i in range(len(request_data)):
            self.add_to_data_pool(
                EntryInfo(
                    rollout_instance_id=int(request_data.non_tensor_batch["rollout_instance_id"][i]),
                    request_id=int(request_data.non_tensor_batch["uid"][i]),
                    model_version=request_data.non_tensor_batch["version_tag"][i],
                ),
                request_data[i:i+1]
            )
        
        # Occupy requests in the PS worker and try to awake waiters if READY buffers are formed
        if self.rollout_n > 1:
            sample_ids = request_data.non_tensor_batch["parent_id"].tolist()
            for i, sample_id in enumerate(sample_ids):
                if sample_id not in self.rollout_request_tracker:
                    self.rollout_request_tracker[sample_id] = []
                entry_info = EntryInfo(
                    rollout_instance_id=int(request_data.non_tensor_batch["rollout_instance_id"][i]),
                    request_id=int(request_data.non_tensor_batch["uid"][i]),
                    model_version=request_data.non_tensor_batch["version_tag"][i],
                )
                self.rollout_request_tracker[sample_id].append(entry_info)
                psrl_logger.debug(f"Store data for prompt {sample_id} with info {entry_info}, "
                                    f"request num: {len(self.rollout_request_tracker[sample_id])}")

            # Group post process
            unique_sample_ids = set(sample_ids)
            for sample_id in unique_sample_ids:
                if len(self.rollout_request_tracker[sample_id]) >= self.alg_rollout_n:
                    psrl_logger.debug(f"Reached/Required: ({len(self.rollout_request_tracker[sample_id])}/{self.alg_rollout_n}) samples for prompt {sample_id}")
                    entry_infos = self.rollout_request_tracker.pop(sample_id)
                    psrl_logger.debug(f"Popped entry_infos from rollout_request_tracker for sample_id {sample_id}, entry count: {len(entry_infos)}")
                    
                    all_child_ids = set(range(self.rollout_n))
                    stored_child_ids = set([int(entry_info.request_id) % self.rollout_n for entry_info in entry_infos])
                    abort_child_ids = all_child_ids - stored_child_ids
                    psrl_logger.debug(f"Stored child IDs: {stored_child_ids}, Abort child IDs: {abort_child_ids}")
                    
                    # Notify the request status manager to abort the child requests
                    abort_entries = []
                    if abort_child_ids:
                        psrl_logger.debug(f"Aborting child requests {abort_child_ids} for sample {sample_id}.")
                        self.ps_manager_handle.abort_requests.remote(list(abort_child_ids))
                        # Clear the reserved entries for the aborted requests
                        abort_entries = [entry_info for entry_info in entry_infos if hash(entry_info) in abort_child_ids]
                    
                    occupy_group_data = True
                    alg_entry_infos = entry_infos[:self.alg_rollout_n]
                    if self.group_post_process_fn:
                        occupy_group_data = await self._group_post_process(sample_id, alg_entry_infos)
                    
                    if not occupy_group_data:
                        abort_entries.extend(entry_infos)
                    else:
                        abort_entries.extend(entry_infos[self.alg_rollout_n:])
                        occupy_futures = []
                        for entry_info in alg_entry_infos:
                            occupy_futures.append(
                                self.ps_manager_handle.maybe_occupy_rollout_instance_request.remote(
                                    rollout_instance_id=entry_info.rollout_instance_id,
                                    request_id=entry_info.request_id,
                                    version_tag=entry_info.model_version,
                                    parent_id=sample_id
                            ))
                        with log_dual_events("Occupy requests", psrl_logger, event_type=EventType.OTHER):
                            results = await asyncio.gather(*occupy_futures)
                        
                        valid_results = [result for result in results if result[0] is not None]
                        assert len(valid_results) <= 1, "At most one request should succeed in making the buffer READY in group sampling"
                        
                        if len(valid_results) > 0:
                            buffer_id, buffer_entries = valid_results[0]
                            data_buffer = self.get_buffer_from_data_pool(buffer_entries)
                            self.data_buffers[buffer_id] = data_buffer
                            self.try_awake_waiters()
                    
                    if abort_entries:
                        await self.ps_manager_handle.clear_reserved_entries.remote(abort_entries)
                        self.remove_from_data_pool(abort_entries)
        else:
            occupy_futures = []
            for i in range(len(request_data)):
                request_data = request_data[i:i+1]
                request_id = request_data.non_tensor_batch["uid"][0]
                sample_id = request_data.non_tensor_batch["parent_id"][0] if self.rollout_n > 1 else request_id
                rollout_instance_id = request_data.non_tensor_batch["rollout_instance_id"][0]
                version_tag = request_data.non_tensor_batch["version_tag"][0]
                
                occupy_futures.append(
                    self.ps_manager_handle.maybe_occupy_rollout_instance_request.remote(
                        rollout_instance_id=int(rollout_instance_id),
                        request_id=int(request_id),
                        version_tag=version_tag,
                        parent_id=None,
                ))
            with log_dual_events("Occupy requests", psrl_logger, event_type=EventType.OTHER):
                results = await asyncio.gather(*occupy_futures)
            for result in results:
                if result[0] is not None:
                    buffer_id, buffer_entries = result
                    data_buffer = self.get_buffer_from_data_pool(buffer_entries)
                    await self.maybe_add_buffer(buffer_id, data_buffer)
                    self.try_awake_waiters()

    async def maybe_add_buffer(self, buffer_id, data_buffer):
        """
        Apply buffer post-processing function if defined and add the buffer to data_buffers.
        
        Args:
            buffer_id (int): The ID of the buffer to be added.
            data_buffer (DataProto): The data buffer to be potentially post-processed and added.
        """
        add_buffer = True
        if self.buffer_post_process_fn:
            add_buffer = await self._buffer_post_process(buffer_id, data_buffer)
        
        if add_buffer:
            self.data_buffers[buffer_id] = data_buffer
        else:
            await self.ps_manager_handle.clear_buffer.remote()

    async def _group_post_process(self, parent_id: int, entry_infos: List[EntryInfo]) -> bool:
        """Apply post-processing function to a group of entry infos.
        
        This method retrieves data from the data pool for each entry, applies
        the group post-processing function, and stores the processed data back.
        
        Args:
            parent_id (int): The parent request id for the group
            entry_infos (List[EntryInfo]): List of entry info objects to process
        
        Returns:
            bool: whether the group data is reserved
        """
        assert self.group_post_process_fn is not None, "Group post-processing function is not set."

        data_list = [self.pop_from_data_pool(entry_info) for entry_info in entry_infos]
        group_data = DataProto.concat(data_list)
        processed_group_data = self.group_post_process_fn(group_data)
        
        if not processed_group_data:
            psrl_logger.info(f"Post-processing function returned empty data for parent {parent_id}. Retrying later.")
            # Clear the reserved entries for the group
            await self.ps_manager_handle.clear_reserved_entries.remote(entry_infos)
            min_pending_buffer = await self.ps_manager_handle.get_min_pending_buffer.remote()

            # Notify agent loop manager to retry new requests
            self.notify_request_retry(min_pending_buffer)
            return False
        else:
            processed_group_data_list = processed_group_data.chunk(len(processed_group_data))
            self.update_data_pool({entry_info: processed_group_data for entry_info, processed_group_data in zip(entry_infos, processed_group_data_list)})
            return True

    async def _buffer_post_process(self, buffer_id: int, buffer_data: DataProto) -> bool:
        """Apply post-processing function to a full buffer of data.
        
        This method applies the buffer post-processing function to the data
        in the buffer and stores the processed data if valid.
        
        Args:
            buffer_id (int): The ID of the buffer to process
            buffer_data (DataProto): The data in the buffer to be processed
        Returns:
            bool: whether the buffer data is reserved
        """
        assert self.buffer_post_process_fn is not None, "Buffer post-processing function is not set."
        
        processed_buffer_data = self.buffer_post_process_fn(buffer_data)
        # NOTE(linsh): Current implementation will retry with new global batch if any sample is filtered
        if not processed_buffer_data or len(processed_buffer_data) < self.num_entries:
            psrl_logger.info(f"Post-processing function returned "
                             f"{len(processed_buffer_data) if processed_buffer_data else 0} requests "
                             f"for buffer {buffer_id}, less than {self.num_entries}. Retrying later.")
            uid_of_processed_buffer = processed_buffer_data.non_tensor_batch["uid"].tolist()
            abort_entries = []
            for entry_id in range(self.num_entries):
                data = buffer_data[entry_id:entry_id+1]
                if int(data.non_tensor_batch["uid"][0]) not in uid_of_processed_buffer:
                    abort_entries.append(entry_id)
            await self.ps_manager_handle.clear_occupied_entries.remote(abort_entries)
            min_pending_buffer = await self.ps_manager_handle.get_min_pending_buffer.remote()

            # Notify agent loop manager to retry new requests
            self.notify_request_retry(min_pending_buffer)
            return False
        else:
            self.data_buffers[buffer_id] = processed_buffer_data
            return True

    def log_ready_buffer(self, buffer_id: int):
        """Log the ready buffer."""
        if buffer_id not in self.logged_ready_buffer_ids:
            log_single_event(f"Buffer {buffer_id} is ready", psrl_logger, event_type=EventType.BUFFER_READY)
            self.logged_ready_buffer_ids.add(buffer_id)

    def try_awake_waiters(self):
        """Check for ready buffers and wake up waiters.
        
        This method checks if there are any new ready buffers for training and
        processes them by waking up waiters and handling staleness control.
        It also sends interruption commands for partial rollout if enabled.
        """
        # Check whether there exists ready buffer for training
        max_ready_buffer_id = max(self.data_buffers.keys())
        min_ready_buffer_id = min(self.data_buffers.keys())
        if (
            max_ready_buffer_id is not None and
            max_ready_buffer_id > self.max_ready_buffer_id
        ):
            self.log_ready_buffer(max_ready_buffer_id)
            self.max_ready_buffer_id = max_ready_buffer_id
        
        ray.get(self.ps_manager_handle.check_abort_and_interrupt_rollout.remote())
        
        if min_ready_buffer_id is not None:
            self.process_ready_buffer(min_ready_buffer_id)

    def process_ready_buffer(self, min_ready_buffer_id: int):
        """
        Notify the rollout server to check abortion and interruption when there is a ready buffer.
        
        This method is called when a buffer is ready to be processed.
        - Abortion: when a buffer is full, all requests with version_tag equal to `buffer_id - S` should be aborted.
        - Interruption: check the workload of each rollout instance and whether the abortion led by interruption will
        influence the training process, to determine whether to interrupt the rollout instance.
        
        Args:
            min_ready_buffer_id (int): The minimum ready buffer ID to process
        """
        # If there are ready buffers, wake up the waiters for the minimum ready buffer
        self._awake_training_batch_waiters(min_ready_buffer_id)

    async def wait_for_training_batch(
        self,
        buffer_id: int
    ) -> DataProto:
        """Await a training batch for a specific buffer ID."""
        await self.ps_manager_handle.ensure_buffer_exists.remote(buffer_id)
        
        if buffer_id in self.data_buffers:
            # If the buffer is ready, return immediately
            psrl_logger.info(f"Buffer {buffer_id} is ready, returning immediately.")
            return self.consume_buffer(buffer_id)
        
        # TODO(lhy): support more consumption strategies, now only support waiting for the buffer to be ready
        # 1. Partial rollout if buffer status is STUCK
        # 2. Truncate if buffer status is STUCK
        # 3. Drop the RESERVED entry if buffer status is STUCK and move some OCCUPIED entries from other buffers 
        
        psrl_logger.info(f"Buffer {buffer_id} is not ready, waiting for it to be ready.")
        fut = asyncio.get_event_loop().create_future()
        if buffer_id not in self._buffer_waiters:
            self._buffer_waiters[buffer_id] = []
        self._buffer_waiters[buffer_id].append(fut)
        result = await fut
        # Once resumed, return
        return result

    def _awake_training_batch_waiters(self, buffer_id: int):
        """Wake up all training waiters for a specific buffer.
        
        When a buffer becomes ready, this method consumes the buffer data
        and sets the result for all futures waiting for this buffer.
        
        Args:
            buffer_id (int): The buffer ID to wake waiters for
        """
        # Wake all Futures waiting for this buffer
        if buffer_id in self._buffer_waiters:
            buffer_data = self.consume_buffer(buffer_id)
            assert len(self._buffer_waiters[buffer_id]) == 1, \
                f"Expected only one waiter for buffer {buffer_id}, but found {len(self._buffer_waiters[buffer_id])}."
            # Set the result for all futures
            for fut in self._buffer_waiters[buffer_id]:
                if not fut.done():
                    fut.set_result(buffer_data)
            # Remove the key after waking all waiters
            del self._buffer_waiters[buffer_id]
        else:
            psrl_logger.warning(f"No waiters found for buffer {buffer_id} when trying to awake.")

    def notify_request_retry(self, waiting_buffer_id: int):
        """
        Notify the agent loop manager to retry new requests asynchronously.
        
        Args:
            waiting_buffer_id (int): The buffer ID that is currently waiting for processing.
        """
        assert self.agent_loop_manager is not None, "Agent Loop Manager is not set."

        psrl_logger.info(f"Notifying agent loop manager to retry new requests for buffer ID {waiting_buffer_id}")
        ray.get(self.agent_loop_manager.retry_request.remote(waiting_buffer_id))

    # ------- DATA POOL MANAGEMENT -------

    def consume_buffer(self, buffer_id: int) -> DataProto:
        """
        Consume (retrieve and remove) all data from the specified buffer.

        Args:
            buffer_id (int): The ID of the buffer to consume.
        Returns:
            DataProto: The concatenated data from the buffer.
        Raises:
            AssertionError: If the buffer is not in READY state.
        """
        
        buffer = self.data_buffers.pop(buffer_id, None)
        assert buffer is not None, f"Buffer {buffer_id} not found or already consumed."
        self.ps_manager_handle.delete_buffer.remote(buffer_id)
        return buffer

    def get_buffer_from_data_pool(self, entry_infos: List[EntryInfo]) -> DataProto:
        """Retrieve data buffers from the internal data pool based on entry information.
        
        This method is used to fetch specific data buffers that have been stored
        in the reward server's internal data pool.
        
        Args:
            entry_infos (List[EntryInfo]): List of EntryInfo objects specifying which buffers to retrieve.
            
        Returns:
            List[DataProto]: List of DataProto objects corresponding to the requested buffers.
        """
        data_list = []
        for entry_info in entry_infos:
            data = self.data_pool.pop(entry_info, None)
            assert data is not None, f"Buffer for entry {entry_info} not found in data pool."
            data_list.append(data)
        return DataProto.concat(data_list)

    def add_to_data_pool(
        self,
        entry_info: EntryInfo,
        data: DataProto,
    ):
        """
        Add rollout data to the group data pool for a specific entry.

        Args:
            entry_info (EntryInfo): The entry metadata.
            data (DataProto): The data to add.
        Raises:
            AssertionError: If data for the entry already exists.
        """
        assert entry_info not in self.data_pool, f"Data pool already has data for entry info {entry_info}"

        self.data_pool[entry_info] = data

    def update_data_pool(self, entry_info_to_data: Dict[EntryInfo, DataProto]):
        """Update the internal data pool with new data buffers.
        
        This method adds new data buffers to the reward server's internal data pool,
        which can later be retrieved using get_buffer_from_data_pool().
        
        Args:
            entry_info_to_data (Dict[EntryInfo, DataProto]): Dictionary mapping EntryInfo objects to their corresponding DataProto buffers.
        """
        self.data_pool.update(entry_info_to_data)
    
    def pop_from_data_pool(
        self,
        entry_info: EntryInfo,
    ) -> DataProto:
        """
        Pop rollout data from the group data pool for a specific entry.

        Args:
            entry_info (EntryInfo): The entry metadata.
        Returns:
            DataProto: The popped data.
        Raises:
            AssertionError: If data for the entry does not exist.
        """
        assert entry_info in self.data_pool, f"Data pool must have data for entry info {entry_info}"

        return self.data_pool.pop(entry_info)
    
    def get_from_data_pool(
        self,
        entry_info: EntryInfo,
    ) -> DataProto:
        """
        Retrieve rollout data from the group data pool for a specific entry.

        Args:
            entry_info (EntryInfo): The entry metadata.
        Returns:
            DataProto: The retrieved data.
        Raises:
            AssertionError: If data for the entry does not exist.
        """
        assert entry_info in self.data_pool, f"Data pool must have data for entry info {entry_info}"

        return self.data_pool[entry_info]
    
    def remove_from_data_pool(
        self,
        entry_infos: Union[EntryInfo, List[EntryInfo]],
    ):
        """
        Delete data from the group data pool for a specific entry.

        Args:
            entry_info (EntryInfo): The entry metadata.
        Raises:
            AssertionError: If data for the entry does not exist.
        """
        if not isinstance(entry_infos, list):
            entry_infos = [entry_infos]
        for entry_info in entry_infos:
            assert entry_info in self.data_pool, f"Data pool must have data for entry info {entry_info}"
            del self.data_pool[entry_info]