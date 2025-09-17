import os
import logging
import torch
import ray
import numpy as np
from threading import Thread
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import pad_2d_list_to_length

from psrl.utils.dataset.utils import _pre_process_inputs
from psrl.utils.logger import log_dual_events, EventType, DualOutputHandler
from psrl.utils.server.command import Command, CommandType, CommandExtension
from psrl.workers.ps.request_status_tracker import RequestStatus

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class RewardServer(CommandExtension):
    def __init__(
        self,
        config,
        tokenizer,
        processor,
        ps_manager_handle,
        rollout_queue,
        reward_fn=None,
        use_rm=False,
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
            rollout_queue: Queue for receiving rollout data from agent loop workers
            reward_fn (optional): Custom function for computing rewards. Defaults to None
            use_rm (bool): Whether to use a reward model for computing rewards. Defaults to False
        """
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        if self.config.psrl.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.redundant_rollout.alg_rollout_n
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
        
        # Background event handler
        self._threads = []
        
        # Communication handles
        self.ps_manager_handle = ps_manager_handle
        
        # Build logger
        self.log_prefix = "RewardServer"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        psrl_logger.info(f"Initialized RewardServer.")
        
    def start_busy_loop(self):
        """Start the reward server and begin processing requests.
        
        This method initializes the server state and starts the background event handler
        thread for processing rollout data and computing rewards. The server will run
        until explicitly stopped.
        """
        if self.server_running:
            return
        
        self.server_running = True
        
        # Start the background event handler thread
        event_handler = Thread(
            target=self._background_event_handler,
            name="reward_event_thread",
            daemon=True,
        )
        
        event_handler.start()
        self._threads = [event_handler]
    
    def stop_busy_loop(self):
        """Shutdown the reward server gracefully.
        
        This method stops the server and waits for all background threads
        to complete before returning.
        """
        if not self.server_running:
            return
        
        self.server_running = False

        # Stop the background event handler
        for thread in self._threads:
            if thread.is_alive():
                thread.join()
        self._threads = []

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
        response_mask = outputs["input_ids"] # [bsz, response_length], each row is [1, 1, ..., 1, 0, 0, ..., 0] (0 is the padding)

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
            position_ids = compute_position_id_with_mask(attention_mask)  # (1, seq_len)

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
        non_tensor_batch = inputs.non_tensor_batch
        if multi_modal_inputs is not None:
            non_tensor_batch["multi_modal_inputs"] = multi_modal_inputs

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    
    def _background_event_handler(self):
        """
        Background event handler for processing commands and rollout data.
        
        This method runs in a separate process and handles:
        1. Command processing (ABORT, STOP, RESUME, etc.)
        2. Rollout data processing from the rollout queue (logprobs, etc.)
        3. Reward computation and status updates.
        4. Occupy requests in the PS worker.
        """
        while self.server_running:
            # Command processing
            if not self.command_queue.empty():
                # Get command from the queue
                command = self.command_queue.get()

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
                        child_uids = ray.get(self.ps_manager_handle.get_recorded_child_requests.remote(list(parent_ids)))
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
                    update_status_success = ray.get(self.ps_manager_handle.update_request_status.remote(list(abort_request_uids), RequestStatus.REWARD_COMPLETED))
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
                rollout_data = self.rollout_queue.get()
                
                with log_dual_events("Process rollout data", psrl_logger, event_type=EventType.OTHER):
                    assert rollout_data is not None, "Data from rollout queue should not be None"
                    # assert len(rollout_data) == 1, "Rollout data should contain exactly one request"
                    rollout_data = self._pre_process(rollout_data)
                    request_ids = rollout_data.non_tensor_batch["uid"]
                    # print(f"Reward server received request ids: {request_ids}")
                    
                    # Update the request status to REWARD_RUNNING
                    update_status_success = ray.get(self.ps_manager_handle.update_request_status.remote(request_ids.tolist(), RequestStatus.REWARD_RUNNING))
                    if not update_status_success[0]:
                        continue

                    rollout_data.non_tensor_batch.pop("raw_prompt_ids", None)
                    rollout_data.non_tensor_batch.pop("raw_response_ids", None)
                    
                    # Rollout log probs processing
                    if self.config.psrl.log_prob.enable_inference_engine_log_prob:
                        device = rollout_data.batch["input_ids"].device
                        rollout_log_probs = rollout_data.non_tensor_batch.pop("rollout_log_probs", None)
                        assert rollout_log_probs is not None, "rollout_log_probs should not be None"
                        rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=self.config.gen_actor_rollout_ref.rollout.response_length).to(device)
                        rollout_log_probs = rollout_log_probs.to(torch.float32)
                        rollout_data.batch["rollout_log_probs"] = rollout_log_probs
                    
                    if self.rollout_n > 1:
                        sample_ids = rollout_data.non_tensor_batch["parent_id"]
                    else:
                        sample_ids = rollout_data.non_tensor_batch["uid"]
                    rollout_instance_ids = rollout_data.non_tensor_batch["rollout_instance_id"]
                    version_tags = rollout_data.non_tensor_batch["version_tag"]

                with log_dual_events(f"Compute reward for samples {sample_ids} and requests {request_ids}", psrl_logger, event_type=EventType.OTHER):
                    for i, (sample_id, request_id, rollout_instance_id, version_tag) in enumerate(zip(sample_ids, request_ids, rollout_instance_ids, version_tags)):
                        request_data = ray.get(self.ps_manager_handle.get_request_data_from_buffer.remote(sample_id))
                        if request_data is None:
                            # If request data is None, it means the request has been aborted or not found.
                            assert self.rollout_n > 1, "Request data should not be None when rollout_n is 1."
                            continue
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
                                
                                if self.config.psrl.log_prob.enable_inference_engine_log_prob:
                                    self.request_id_to_future[request_id] = (merge_request_data, future_reward)
                                    self.reward_futures.append(future_reward)
                                else:
                                    merge_request_data.non_tensor_batch["reward"] = np.array([future_reward])
                        else:
                            with log_dual_events("Compute reward model score", psrl_logger, level=logging.DEBUG, event_type=EventType.OTHER):
                                reward_tensor, reward_extra_infos_dict = compute_reward(reward_input, self.reward_fn)
                                
                                merge_request_data.union(reward_input)
                                merge_request_data.batch["reward"] = reward_tensor
                                merge_request_data.meta_info["reward_extra_info_keys"] = list(reward_extra_infos_dict.keys())
                                merge_request_data.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            # Update the request status to REWARD_COMPLETED
                            update_status_success = ray.get(self.ps_manager_handle.update_request_status.remote(int(request_id), RequestStatus.REWARD_COMPLETED))
                            complete_request_idxs = [
                                i for i, success in enumerate(update_status_success) if success
                            ]
                            
                            if complete_request_idxs:
                                with log_dual_events("Occupy requests", psrl_logger, level=logging.DEBUG, event_type=EventType.OTHER):
                                    future = self.ps_manager_handle.store_and_maybe_occupy_rollout_instance_request.remote(
                                        rollout_instance_id=int(rollout_instance_id),
                                        request_id=int(request_id),
                                        version_tag=version_tag,
                                        data=merge_request_data,
                                        parent_id=sample_id if self.rollout_n > 1 else None,
                                    )
                                    ray.get(future)

            # Check if any reward futures are ready
            if (
                self.config.psrl.log_prob.enable_inference_engine_log_prob and
                self.config.reward_model.launch_reward_fn_async
            ):
                ready_rewards, self.reward_futures = ray.wait(self.reward_futures)

                running_requests = set(self.request_id_to_future.keys())
                finished_request_data = []
                
                # Process the ready rewards
                for request_id in running_requests:
                    request_data, reward_future = self.request_id_to_future[request_id]
                    if reward_future in ready_rewards:
                        self.request_id_to_future.pop(request_id, None)
                        reward_tensor, reward_extra_infos_dict = ray.get(reward_future)
                        
                        request_data.batch["reward"] = reward_tensor
                        request_data.meta_info["reward_extra_info_keys"] = list(reward_extra_infos_dict.keys())
                        request_data.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                        finished_request_data.append(request_data)
                
                if finished_request_data:
                    finished_request_data = DataProto.concat(finished_request_data)
                    request_ids = finished_request_data.non_tensor_batch["uid"]
                    
                    # Update the request status to REWARD_COMPLETED
                    update_status_success = ray.get(self.ps_manager_handle.update_request_status.remote(request_ids.tolist(), RequestStatus.REWARD_COMPLETED))
                    complete_request_idxs = [
                        i for i, success in enumerate(update_status_success) if success
                    ]
                    
                    # If requests are completed, occupy them in the PS worker
                    if complete_request_idxs:
                        complete_request_data = finished_request_data[complete_request_idxs]
                        
                        occupy_futures = []
                        for i in range(len(complete_request_data)):
                            request_data = complete_request_data[i:i+1]
                            request_id = request_data.non_tensor_batch["uid"][0]
                            sample_id = request_data.non_tensor_batch["parent_id"][0] if self.rollout_n > 1 else request_id
                            rollout_instance_id = request_data.non_tensor_batch["rollout_instance_id"][0]
                            version_tag = request_data.non_tensor_batch["version_tag"][0]
                            
                            occupy_futures.append(
                                self.ps_manager_handle.store_and_maybe_occupy_rollout_instance_request.remote(
                                    rollout_instance_id=int(rollout_instance_id),
                                    request_id=int(request_id),
                                    version_tag=version_tag,
                                    data=request_data,
                                    parent_id=sample_id if self.rollout_n > 1 else None,
                            ))
                        with log_dual_events("Occupy requests", psrl_logger, event_type=EventType.OTHER):
                            ray.get(occupy_futures)

        psrl_logger.info("Background event handler of reward server has finished.")
