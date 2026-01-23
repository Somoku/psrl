import asyncio
import logging
import os
from collections import Counter

import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from verl import DataProto
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import pad_2d_list_to_length

from psrl.utils.dataset.utils import _pre_process_inputs
from psrl.utils.logger import (
    DualOutputHandler,
    EventType,
    log_data_protocol,
    log_dual_events,
    log_single_event,
)
from psrl.utils.ray import AsyncBusyPollingRayLock
from psrl.workers.ps.request_status_tracker import PSRL_RequestStatus
from psrl.workers.ps.staleness_controller import EntryInfo

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class PSRL_AgentLoopManager:
    def __init__(
        self,
        config: DictConfig,
        data_queue_size: int,
        agent_loop_workers,
        ps_manager_handle,
        group_post_process_fn=None,
        buffer_post_process_fn=None,
    ):
        """Initialize agent loop manager.
        Agent loop manager that manages a group of agent loop workers.
        Handles data distribution, versioning, and coordination between workers.

        Args:
            config (DictConfig): Configuration containing training and rollout settings.
            data_queue_size (int): Size of the data queue.
            agent_loop_workers: List of agent loop worker instances.
            ps_manager_handle: Handle to the parameter server manager.
            group_post_process_fn (Optional[callable]): Optional function to post-process
                grouped entry data before occupying the buffer
            buffer_post_process_fn (Optional[callable]): Optional function to post-process
                ready buffer data
        """
        self.config = config
        model_path = config.gen_actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.gen_actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = hf_processor(local_path, trust_remote_code=True)

        self.staleness = self.config.psrl.staleness
        self.group_post_process_fn = group_post_process_fn
        self.buffer_post_process_fn = buffer_post_process_fn
        if self.config.psrl.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n

        if self.config.psrl.redundant_rollout.enable:
            self.entries_per_buffer = self.config.psrl.redundant_rollout.redundant_global_batch_size
            self.ready_entries_per_buffer = self.config.psrl.redundant_rollout.alg_global_batch_size
        else:
            self.entries_per_buffer = self.config.psrl.staleness_buffer_entries
            self.ready_entries_per_buffer = self.config.psrl.staleness_buffer_entries

        self.data_queue = asyncio.Queue(maxsize=data_queue_size)
        # result_queue_size = self.entries_per_buffer * self.rollout_n * (self.staleness + 1)
        # self.result_queue = asyncio.Queue(maxsize=result_queue_size)
        self.result_queue = asyncio.Queue()
        self.agent_loop_workers = agent_loop_workers
        self.ps_manager_handle = ps_manager_handle

        self._request_counter = 0  # For version tag setting

        self._dispatch_idx = 0
        self.running_loop = None
        self.dispatch_task = None
        self.collect_task = None
        self.stop_dispatch_task = False
        self.stop_collect_task = False

        self.curr_ps_version_tag = 0

        # Data
        self.data_pool: dict[int, DataProto] = {}  # Maps request_id to stored/occupied DataProto
        self.data_buffers: dict[int, DataProto] = {}  # data of READY buffer in ps manager
        self.accumulated_buffers: dict[
            int, dict[int, list[EntryInfo]]
        ] = {}  # Maps buffer_id to dict of model_version to READY entry_info list
        self.accumulated_buffer_size: dict[int, int] = {}  # Maps buffer id to current accumulated size
        self.abort_occupied_entries: dict[int, list[int]] = {}

        # Set of buffer ids that have been logged as ready, to avoid duplicate logging
        self.logged_ready_buffer_ids: set[int] = set()

        # Waiting lists for training batches
        self._buffer_waiters: dict[
            int, list[asyncio.Future]
        ] = {}  # Maps buffer IDs to a set of futures waiting for that buffer

        # Track finished child requests for Group Sampling
        self.rollout_request_tracker: dict[
            str | int, list[EntryInfo]
        ] = {}  # Maps parent request ids to "occupied" child entries

        # Build logger
        self.log_prefix = "AgentLoopManager"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))

    async def start_busy_loop(self):
        """Start the busy loop for continuous data processing from the queue."""
        if (
            self.dispatch_task is not None
            and not self.dispatch_task.done()
            or self.collect_task is not None
            and not self.collect_task.done()
        ):
            return

        # Start the busy loop of agent loop workers
        futures = []
        for worker in self.agent_loop_workers:
            futures.append(worker.start_busy_loop.remote())
        await asyncio.gather(*futures)

        # Start the background task to process data
        self.running_loop = asyncio.get_running_loop()
        self.dispatch_task = self.running_loop.create_task(self._dispatch_data())
        self.dispatch_task.add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks
        self.collect_task = self.running_loop.create_task(self._collect_results())
        self.collect_task.add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks

    async def stop_busy_loop(self):
        """Stop the busy loop and wait for all tasks to complete."""
        if (not self.dispatch_task or self.dispatch_task.done()) and (
            not self.collect_task or self.collect_task.done()
        ):
            return

        self.stop_dispatch_task = True
        self.stop_collect_task = True
        # Wait for the background task to finish
        await asyncio.gather(self.dispatch_task, self.collect_task)

        # Stop the busy loop of agent loop workers
        futures = []
        for worker in self.agent_loop_workers:
            futures.append(worker.stop_busy_loop.remote())
        await asyncio.gather(*futures)

    async def put_data(self, data: DataProto):
        """Put data into the manager's data queue."""
        await self.data_queue.put(data)

    def _post_process(self, inputs: DataProto) -> DataProto:
        """Post-process the generated outputs to create properly formatted tensors.

        This method handles padding, attention masks, position IDs, multi-modal inputs
        and routed experts to ensure compatibility with the training pipeline.

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

        log_data_protocol(
            inputs,
            psrl_logger,
            self.log_prefix + " before preprocess data from rollout queue",
            level=logging.DEBUG,
        )

        # prompts
        self.tokenizer.padding_side = "left"
        if "raw_prompt_ids" not in inputs.non_tensor_batch:
            batch_size = len(inputs)
            raw_prompt_ids = np.array(
                [
                    _pre_process_inputs(self.tokenizer.pad_token_id, inputs.batch["input_ids"][i])
                    for i in range(batch_size)
                ],
                dtype=object,
            )
            ## psrl_logger.info("Remove left padding from input ids to get raw_prompt_ids")
        else:
            raw_prompt_ids = inputs.non_tensor_batch["raw_prompt_ids"]

        ## psrl_logger.info("Left pad prompt ids begin")
        prompt_output = self.tokenizer.pad(
            [{"input_ids": raw_prompt_id} for raw_prompt_id in raw_prompt_ids],
            padding="max_length",
            max_length=self.config.gen_actor_rollout_ref.rollout.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        prompt_ids, prompt_attention_mask = (
            prompt_output["input_ids"],
            prompt_output["attention_mask"],
        )

        # responses
        raw_response_ids = inputs.non_tensor_batch.pop("raw_response_ids", None)
        assert raw_response_ids is not None, "raw_response_ids must be provided in the input batch"
        self.tokenizer.padding_side = "right"
        ## psrl_logger.info("Right pad response ids begin")
        outputs = self.tokenizer.pad(
            [{"input_ids": raw_response_id} for raw_response_id in raw_response_ids],
            padding="max_length",
            max_length=self.config.gen_actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        response_ids, response_attention_mask = (
            outputs["input_ids"],
            outputs["attention_mask"],
        )

        # response_mask
        response_masks = inputs.non_tensor_batch.pop("response_mask", None)
        assert response_masks is not None, "response_masks must be provided in the input batch"
        ## psrl_logger.info("Right pad response masks begin")
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

        ## psrl_logger.info("Multiply response mask and response attention mask begin")
        response_mask = response_mask * response_attention_mask
        ## psrl_logger.info("Concat prompt attention mask and response attention mask begin")
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        ## psrl_logger.info("Concat prompt ids and response ids begin")
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        # Handle multi-modal inputs and position_ids calculation
        # Only support Qwen2VLImageProcessor for multi-modal processing currently
        # TODO(verl): support other multi-modal inputs
        multi_modal_inputs = None
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
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
            ## psrl_logger.info("Compute position ids with attention mask begin")
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
            ## psrl_logger.info("Rollout log probs processing begin")
            device = batch["input_ids"].device
            rollout_log_probs = inputs.non_tensor_batch.pop("rollout_log_probs", None)
            assert rollout_log_probs is not None, "rollout_log_probs should not be None"
            rollout_log_probs = pad_2d_list_to_length(
                rollout_log_probs,
                -1,
                max_length=self.config.gen_actor_rollout_ref.rollout.response_length,
            ).to(device)
            rollout_log_probs = rollout_log_probs.to(torch.float32)
            batch["rollout_log_probs"] = rollout_log_probs

        # Routed experts processing
        if self.config.gen_actor_rollout_ref.rollout.enable_rollout_routing_replay:
            device = batch["input_ids"].device
            bsz, total_length = input_ids.shape

            experts_array_list = inputs.non_tensor_batch.pop("routed_experts", None)
            assert experts_array_list is not None, "routed_experts should not be None"
            assert len(experts_array_list) == bsz, "len of experts_array_list should be equal to bsz"
            _, layer_num, topk_num = experts_array_list[0].shape
            dtype = torch.from_numpy(np.array([], dtype=experts_array_list[0].dtype)).dtype

            routed_experts = torch.zeros(bsz, total_length, layer_num, topk_num, dtype=dtype)

            for i in range(bsz):
                experts_array = experts_array_list[i]
                actual_length = experts_array.shape[0]
                experts_tensor = torch.from_numpy(experts_array)

                raw_prompt_id = raw_prompt_ids[i]

                # Calculate start position: left padding means original prompt starts at the end
                start_pos = prompt_ids.shape[1] - len(raw_prompt_id)
                end_pos = min(start_pos + actual_length, total_length)

                # Add boundary checks for robustness
                if start_pos < 0 or end_pos > total_length:
                    raise ValueError(
                        f"Invalid position range: start_pos={start_pos}, "
                        f"end_pos={end_pos}, total_length={total_length}"
                    )

                routed_experts[i, start_pos:end_pos] = experts_tensor

            routed_experts = routed_experts.to(device)
            batch["routed_experts"] = routed_experts

        inputs.non_tensor_batch.pop("raw_prompt_ids", None)
        inputs.non_tensor_batch.pop("raw_response_ids", None)
        non_tensor_batch = inputs.non_tensor_batch
        if multi_modal_inputs is not None:
            non_tensor_batch["multi_modal_inputs"] = multi_modal_inputs

        meta_info = {}
        # Reward processing
        if not self.config.reward_model.launch_reward_fn_async:
            ## psrl_logger.info("Reward processing begin")
            scores = inputs.non_tensor_batch.pop("reward_scores", None).tolist()
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores  # [bsz, response_length]

            # add reward_extra_info to non_tensor_batch
            reward_extra_infos = inputs.non_tensor_batch.pop("reward_extra_infos", None)
            reward_extra_keys = list(reward_extra_infos[0].keys())
            for key in reward_extra_keys:
                non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])
            meta_info = {"reward_extra_keys": reward_extra_keys}

        ## psrl_logger.info("Return data proto")
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

    async def _dispatch_data(self):
        """Main dispatch loop that processes data from the queue and routes to workers."""
        while not self.stop_dispatch_task:
            while not self.data_queue.empty():
                data = await self.data_queue.get()

                # Receive END signal to stop processing data queue
                if data is None:
                    self.stop_dispatch_task = True
                    continue

                psrl_logger.debug(f"Got {len(data)} requests from data queue")

                # Set version tag for each request
                batch_size = len(data)
                static_version_tags = [self.get_new_static_version_tag() for _ in range(batch_size)]

                # Wait for version update in ps
                max_version_tag = np.max(static_version_tags)
                if max_version_tag > self.curr_ps_version_tag:
                    psrl_logger.debug(f"Waiting for ps model version: {max_version_tag}")
                    # Busy polling until the PS worker has the needed model version
                    while (
                        await self.ps_manager_handle.get_ps_model_version.remote(debug_info="agent_loop_manager")
                    ) < max_version_tag:
                        await asyncio.sleep(0.1)
                    self.curr_ps_version_tag = max_version_tag
                    psrl_logger.info(f"ps model version updated to {self.curr_ps_version_tag}, continue to dispatch")

                if self.config.psrl.routing_strategy.enable_dynamic_version_tag:
                    dynamic_version_tags = [-1 for _ in range(batch_size)]
                    data.non_tensor_batch["version_tag"] = np.array(dynamic_version_tags)
                else:
                    data.non_tensor_batch["version_tag"] = np.array(static_version_tags)
                # psrl_logger.debug(
                #     f"Dispatching data to agent loop workers, total {len(data)} requests "
                #     f"with version tag {data.non_tensor_batch['version_tag']}"
                # )

                # Dispatch data to agent loop workers
                await self._inner_dispatch_data(data)
            await asyncio.sleep(0)  # Yield control to the event loop

    def get_new_static_version_tag(self):
        """
        Get the new static version tag based on the current staleness and request counter.
        This is a naive implementation that increments the version tag for each request.
        If support dynamic version tag, this version tag is only used to determine
        when to put the request into the data queue.
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
            # Guarantee the version tags are valid (>=0)
            version_tags = np.maximum(data.non_tensor_batch["min_version_limit"] - self.staleness, 0)
        update_status_success = await self.ps_manager_handle.update_request_status.remote(
            request_ids.tolist(),
            PSRL_RequestStatus.RUNNING,
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
                    self.agent_loop_workers[worker_index].add_agent_program.remote(worker_data[i : i + 1])
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
        prompt_to_worker = {}
        if self.rollout_n > 1:
            assert "parent_id" in data.non_tensor_batch, "parent_id not found in data"
            prompt_ids = data.non_tensor_batch["parent_id"].tolist()
        else:
            assert "uid" in data.non_tensor_batch, "uid not found in data"
            prompt_ids = data.non_tensor_batch["uid"].tolist()
        # Round-robin dispatching
        for i, prompt_id in enumerate(prompt_ids):
            if prompt_id in prompt_to_worker:
                worker_index = prompt_to_worker[prompt_id]
            else:
                worker_index = (self._dispatch_idx + prompt_id) % len(self.agent_loop_workers)
                prompt_to_worker[prompt_id] = worker_index
            if worker_index not in dispatch_plan:
                dispatch_plan[worker_index] = []
            dispatch_plan[worker_index].append(data[i : (i + 1)])

        # Convert lists to DataProto
        for worker_index, data in dispatch_plan.items():
            dispatch_plan[worker_index] = DataProto.concat(data)
        self._dispatch_idx = (self._dispatch_idx + len(prompt_to_worker)) % len(self.agent_loop_workers)
        return dispatch_plan

    async def put_result(self, result: DataProto):
        """Put result data into the manager's result queue."""
        await self.result_queue.put(result)
        # psrl_logger.info(f"Put result {result.non_tensor_batch['uid']} into result queue")

    async def _collect_results(self):
        """Main collection loop that gathers results from workers."""
        while not self.stop_collect_task:
            # psrl_logger.info(f"Collecting results from result queue with size {self.result_queue.qsize()}")
            while not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                # psrl_logger.info(f"Got requests {result.non_tensor_batch['uid']} from result queue")

                # Process the collected result
                result = self._post_process(result)
                # Occupy requests in PS worker
                # psrl_logger.info(f"Post-processed requests {result.non_tensor_batch['uid']}")
                await self.occupy_requests(result)
                # psrl_logger.info(f"Occupied requests {result.non_tensor_batch['uid']}")
            await asyncio.sleep(0)  # Yield control to the event loop
        psrl_logger.info("Stop collecting results")

    async def occupy_requests(self, request_data: DataProto):
        """
        Try to occupy the requests in the PS worker and manage the data buffers.
        This method attempts to occupy the requests in the PS worker by communicating
        with the PS manager. It also manages the data buffers and handles group post-processing
        if applicable.

        Args:
            request_data (DataProto): DataProto containing the requests to be occupied.
        """
        # psrl_logger.info(f"Occupying requests {request_data.non_tensor_batch['uid']}")
        # Ensure the whole occupation process is atomic from the PS manager side
        async with AsyncBusyPollingRayLock(self.ps_manager_handle):
            # Add data to the data pool
            for i in range(len(request_data)):
                self.add_to_data_pool(
                    int(request_data.non_tensor_batch["uid"][i]),
                    request_data[i : i + 1],
                )

            retry_buffer_ids = set()  # Buffer IDs that need to abort OCCUPY entries and retry
            ready_buffer_ids = set()  # Buffer IDs that are READY after occupation
            accumulate_entry_data_list = []  # Whether to accumulate data for each prompt entry
            occupy_futures = []
            abort_request_ids = [] # Used to abort requests in the data pool
            # Occupy requests in the PS worker and try to awake waiters if READY buffers are formed
            if self.rollout_n > 1:
                sample_ids = request_data.non_tensor_batch["parent_id"].tolist()
                for i, sample_id in enumerate(sample_ids):
                    if sample_id not in self.rollout_request_tracker:
                        self.rollout_request_tracker[sample_id] = []
                    entry_info = EntryInfo(
                        rollout_instance_id=int(request_data.non_tensor_batch["rollout_instance_id"][i]),
                        request_idx=int(request_data.non_tensor_batch["uid"][i]) % self.rollout_n,
                        prompt_id=int(request_data.non_tensor_batch["parent_id"][i]),
                        model_version=request_data.non_tensor_batch["version_tag"][i],
                    )
                    self.rollout_request_tracker[sample_id].append(entry_info)
                    psrl_logger.debug(
                        f"Store data for prompt {sample_id} with info {entry_info}, "
                        f"request num: {len(self.rollout_request_tracker[sample_id])}"
                    )

                # Group post process
                unique_sample_ids = set(sample_ids)
                prompt_to_occupy_requests = {}
                for sample_id in unique_sample_ids:
                    if len(self.rollout_request_tracker[sample_id]) >= self.alg_rollout_n:
                        psrl_logger.debug(
                            f"Reached/Required: ({len(self.rollout_request_tracker[sample_id])}/{self.alg_rollout_n}) "
                            f"samples for prompt {sample_id}"
                        )
                        entry_infos = self.rollout_request_tracker.pop(sample_id)
                        psrl_logger.debug(
                            f"Popped entry_infos from rollout_request_tracker for sample_id {sample_id}, "
                            f"entry count: {len(entry_infos)}"
                        )

                        all_child_idxs = set(range(self.rollout_n))
                        stored_child_idxs = {entry_info.request_idx for entry_info in entry_infos}
                        abort_child_idxs = all_child_idxs - stored_child_idxs
                        abort_child_ids = [sample_id * self.rollout_n + idx for idx in abort_child_idxs]
                        stored_child_ids = [sample_id * self.rollout_n + idx for idx in stored_child_idxs]
                        psrl_logger.debug(f"Stored child IDs: {stored_child_ids}, Abort child IDs: {abort_child_ids}")

                        # Notify the request status manager to abort the child requests
                        if abort_child_ids:
                            psrl_logger.info(f"Aborting child requests {abort_child_ids} for sample {sample_id}.")
                            with log_dual_events(
                                f"Abort {len(abort_child_ids)} requests",
                                psrl_logger,
                                level=logging.INFO,
                                event_type=EventType.OTHER,
                            ):
                                await self.ps_manager_handle.abort_requests.remote(
                                    list(abort_child_ids), blocking=False
                                )

                        # Abort the extra entries beyond alg_rollout_n
                        abort_request_ids.extend(
                            [
                                sample_id * self.rollout_n + entry_info.request_idx
                                for entry_info in entry_infos[self.alg_rollout_n :]
                            ]
                        )

                        alg_entry_infos = entry_infos[: self.alg_rollout_n]
                        accumulate_group_data = True
                        if self.group_post_process_fn:
                            accumulate_group_data = await self._group_post_process(alg_entry_infos)

                        if not accumulate_group_data and self.config.psrl.retry_bound == -1:
                            # Retry immediately and no occupation
                            # NOTE(linsh): data has been popped from data pool in `_group_post_process`
                            psrl_logger.info(
                                f"Post-processing function returned empty data for "
                                f"prompt {sample_id}. Retrying immediately."
                            )
                            # Clear the reserved entries for the group entry
                            await self.ps_manager_handle.clear_reserved_entries.remote(sample_id)
                            min_pending_buffer = await self.ps_manager_handle.get_min_pending_buffer.remote()

                            # Notify agent loop manager to retry new requests
                            await self.retry_request(min_version_limit=min_pending_buffer)
                        else:
                            accumulate_entry_data_list.append(accumulate_group_data)

                            prompt_to_occupy_requests[sample_id] = alg_entry_infos
                            request_ids = [
                                sample_id * self.rollout_n + entry_info.request_idx for entry_info in alg_entry_infos
                            ]
                            occupy_futures.append(
                                self.ps_manager_handle.occupy_rollout_instance_request.remote(
                                    prompt_id=sample_id,
                                    request_ids=request_ids,
                                    accumulate_sample=accumulate_group_data,
                                )
                            )
            # Without group sampling (e.g., PPO)
            # Every request is a accumulate data
            else:
                for i in range(len(request_data)):
                    request = request_data[i : i + 1]
                    request_id = int(request.non_tensor_batch["uid"][0])
                    accumulate_data = True
                    accumulate_entry_data_list.append(accumulate_data)

                    occupy_futures.append(
                        self.ps_manager_handle.occupy_rollout_instance_request.remote(
                            prompt_id=request_id,
                            accumulate_sample=accumulate_data,
                        )
                    )

            if not occupy_futures:
                return

            with log_dual_events(
                "Occupy requests",
                psrl_logger,
                level=logging.DEBUG,
                event_type=EventType.OTHER,
            ):
                results = await asyncio.gather(*occupy_futures)

            for result, accumulate_entry_data in zip(results, accumulate_entry_data_list):
                buffer_id, occupy_num, prompt_entry_info = result
                # If occupy failed due to READY status, the requests must be aborted already
                # Just continue
                if buffer_id is None:
                    # request_ids = prompt_entry_info.get_all_requests(self.rollout_n)
                    # psrl_logger.info(
                    #     f"Failed to occupy prompt {prompt_entry_info}, aborting requests {request_ids}."
                    # )
                    # abort_request_ids.extend(request_ids)
                    continue

                psrl_logger.debug(
                    f"Successfully occupied prompt {prompt_entry_info} into "
                    f"buffer {buffer_id} with occupy_num {occupy_num}."
                )

                if self.rollout_n > 1:
                    alg_entry_infos = prompt_to_occupy_requests.pop(prompt_entry_info.prompt_id, None)
                    request_ids = [
                        prompt_entry_info.prompt_id * self.rollout_n + entry_info.request_idx
                        for entry_info in alg_entry_infos
                    ]
                else:
                    request_ids = [prompt_entry_info.prompt_id + prompt_entry_info.request_idx]

                # Accumulate data or mark for abort based on accumulate_entry_data
                if accumulate_entry_data:
                    if buffer_id not in self.accumulated_buffers:
                        self.accumulated_buffers[buffer_id] = {}
                        self.accumulated_buffer_size[buffer_id] = 0
                    model_version = prompt_entry_info.get_entry_version()
                    if model_version not in self.accumulated_buffers[buffer_id]:
                        self.accumulated_buffers[buffer_id][model_version] = []
                    self.accumulated_buffers[buffer_id][model_version].append(prompt_entry_info)
                    self.accumulated_buffer_size[buffer_id] += 1
                    psrl_logger.info(
                        f"Accumulated buffer {buffer_id} size: "
                        f"{self.accumulated_buffer_size[buffer_id]}/{self.ready_entries_per_buffer}"
                    )
                    # Check if the buffer is the earliest waiting buffer
                    # If so, handle the waiting buffer using the abort and truncate strategy
                    if self._buffer_waiters:
                        min_waiter_buffer_id = min(self._buffer_waiters.keys())
                        if min_waiter_buffer_id == buffer_id:
                            await self.handle_waiting_buffer(buffer_id)
                else:
                    abort_request_ids.extend(request_ids)
                    if self.config.psrl.retry_bound >= 0:
                        if buffer_id not in self.abort_occupied_entries:
                            self.abort_occupied_entries[buffer_id] = []
                        self.abort_occupied_entries[buffer_id].append(entry_info.prompt_id)

                # Check for READY or RETRY buffers
                if (
                    self.config.psrl.retry_bound >= 0
                    and self.accumulated_buffer_size[buffer_id] < self.ready_entries_per_buffer
                    and occupy_num == self.entries_per_buffer - self.config.psrl.retry_bound
                ):
                    retry_buffer_ids.add(buffer_id)
                elif (
                    self.accumulated_buffer_size[buffer_id] == self.ready_entries_per_buffer
                    and buffer_id not in ready_buffer_ids
                ):
                    psrl_logger.info(f"Add buffer {buffer_id} to ready_buffer_ids")
                    ready_buffer_ids.add(buffer_id)

            if abort_request_ids:
                self.remove_from_data_pool(abort_request_ids, guarantee_exists=True)

            # Process READY buffers
            for buffer_id in sorted(list(ready_buffer_ids)):
                # Collect all prompt entry infos for the buffer
                prompt_entry_infos = []
                for model_version in sorted(list(self.accumulated_buffers[buffer_id].keys())):
                    prompt_entry_infos.extend(self.accumulated_buffers[buffer_id][model_version])
                # Get the data buffer from the data pool
                data_buffer = self.get_buffer_from_data_pool(prompt_entry_infos)
                # Apply buffer post-processing if exists and add to data_buffers
                add_buffer = self.maybe_add_buffer(buffer_id, data_buffer)
                if add_buffer:
                    psrl_logger.info(f"Buffer {buffer_id} is READY with {len(self.data_buffers[buffer_id])} entries.")
                    await self.handle_ready_buffer(buffer_id)
                    self.remove_buffer_from_data_pool(prompt_entry_infos)
                    self.accumulated_buffers.pop(buffer_id)
                    self.accumulated_buffer_size.pop(buffer_id)

            # Process RETRY buffers
            for retry_buffer_id in retry_buffer_ids:
                if self.config.psrl.gen_mode == "batch":
                    assert self.config.psrl.retry_bound == 0, "For batch mode, retry_bound must be 0."
                    await self.ps_manager_handle.clear_buffer.remote(retry_buffer_id)
                    min_pending_buffer = await self.ps_manager_handle.get_min_pending_buffer.remote()
                    # Notify agent loop manager to retry new requests
                    await self.retry_request(min_version_limit=min_pending_buffer)
                elif self.config.psrl.retry_bound >= 0:
                    # retry num = retry_ratio * num of failed OCCUPY entries
                    retry_prompt_num = (
                        self.entries_per_buffer - self.config.psrl.retry_bound - self.ready_entries_per_buffer
                    ) * self.config.psrl.retry_ratio
                    if retry_prompt_num > 0:
                        # the last retry_prompt_num prompts to retry
                        psrl_logger.debug(f"Retrying {retry_prompt_num} prompts from full buffer {retry_buffer_id}.")
                        # Clear the last `retry_prompt_num` occupied entries for retry with RESERVE
                        abort_occupied_entries = self.abort_occupied_entries[retry_buffer_id][-retry_prompt_num:]
                        self.abort_occupied_entries[retry_buffer_id] = self.abort_occupied_entries[retry_buffer_id][
                            :-retry_prompt_num
                        ]
                        await self.ps_manager_handle.clear_occupied_entries.remote(abort_occupied_entries)
                        min_pending_buffer = await self.ps_manager_handle.get_min_pending_buffer.remote()
                        # Notify agent loop manager to retry new requests
                        await self.retry_request(min_version_limit=min_pending_buffer)

    def maybe_add_buffer(self, buffer_id, data_buffer) -> bool:
        """
        Apply buffer post-processing function if defined and add the buffer to data_buffers.

        Args:
            buffer_id (int): The ID of the buffer to be added.
            data_buffer (DataProto): The data buffer to be potentially post-processed and added.
        Returns:
            bool: whether the buffer was added to data_buffers.
        """
        add_buffer = True
        if self.buffer_post_process_fn:
            add_buffer, data_buffer = self._buffer_post_process(buffer_id, data_buffer)

        if add_buffer:
            self.data_buffers[buffer_id] = data_buffer
            psrl_logger.debug(f"Buffer {buffer_id} is added to data_buffers after post-processing.")
        return add_buffer

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

        request_ids = [entry_info.prompt_id * self.rollout_n + entry_info.request_idx for entry_info in entry_infos]
        data_list = [self.pop_from_data_pool(request_id) for request_id in request_ids]
        group_data = DataProto.concat(data_list)
        processed_group_data = self.group_post_process_fn(group_data)

        if not processed_group_data:
            return False
        else:
            processed_group_data_list = processed_group_data.chunk(len(processed_group_data))
            self.update_data_pool(
                {
                    request_id: processed_group_data
                    for request_id, processed_group_data in zip(request_ids, processed_group_data_list)
                }
            )
            return True

    def _buffer_post_process(self, buffer_id: int, buffer_data: DataProto) -> tuple[bool, DataProto | None]:
        """Apply post-processing function to a full buffer of data.

        This method applies the buffer post-processing function to the data
        in the buffer and stores the processed data if valid.

        Args:
            buffer_id (int): The ID of the buffer to process
            buffer_data (DataProto): The data in the buffer to be processed
        Returns:
            Tuple[bool, Optional[DataProto]]: A tuple where the first element indicates
            whether to add the buffer to data_buffers, and the second element is the
            processed DataProto or None if not added.
        """
        assert self.buffer_post_process_fn is not None, "Buffer post-processing function is not set."

        processed_buffer_data = self.buffer_post_process_fn(buffer_data)
        if not processed_buffer_data or len(processed_buffer_data) < len(buffer_data):
            # Clear entries from accumulated_data_buffer
            self.accumulated_buffers.pop(buffer_id, None)
            self.accumulated_buffer_size.pop(buffer_id, None)
            self.accumulated_buffers[buffer_id] = {}
            self.accumulated_buffer_size[buffer_id] = 0
            if processed_buffer_data is not None:
                prompt_entry_infos = self.extract_entry_infos_from_data(processed_buffer_data)
                for entry_info in prompt_entry_infos:
                    model_version = (
                        min(entry_info.model_version)
                        if isinstance(entry_info.model_version, list)
                        else entry_info.model_version
                    )
                    if model_version not in self.accumulated_buffers[buffer_id]:
                        self.accumulated_buffers[buffer_id][model_version] = []
                    self.accumulated_buffers[buffer_id][model_version].append(entry_info)
                    self.accumulated_buffer_size[buffer_id] += 1
            return False, None
        else:
            return True, processed_buffer_data

    def extract_entry_infos_from_data(self, data: DataProto) -> list[EntryInfo]:
        """Extract EntryInfo objects from DataProto.

        This method extracts EntryInfo objects from the non-tensor batch
        information in the provided DataProto.

        Args:
            data (DataProto): The data from which to extract EntryInfo objects
        Returns:
            List[EntryInfo]: List of extracted EntryInfo objects
        """
        entry_infos = {}
        if self.rollout_n > 1:
            parent_ids = data.non_tensor_batch["parent_id"].tolist()
            rollout_instance_ids = data.non_tensor_batch["rollout_instance_id"].tolist()
            request_ids = data.non_tensor_batch["uid"].tolist()
            model_versions = data.non_tensor_batch["version_tag"].tolist()
            for parent_id, rollout_instance_id, request_id, model_version in zip(
                parent_ids, rollout_instance_ids, request_ids, model_versions
            ):
                if parent_id in entry_infos:
                    entry_info = entry_infos[parent_id]
                    if isinstance(entry_info.request_idx, list):
                        entry_info.request_idx.append(request_id % self.rollout_n)
                    else:
                        entry_info.request_idx = [
                            entry_info.request_idx,
                            request_id % self.rollout_n,
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
                        request_idx=request_id % self.rollout_n,
                        prompt_id=parent_id,
                        model_version=model_version,
                    )
                    entry_infos.append(entry_info)
        else:
            request_ids = data.non_tensor_batch["uid"].tolist()
            model_versions = data.non_tensor_batch["version_tag"].tolist()
            rollout_instance_ids = data.non_tensor_batch["rollout_instance_id"].tolist()
            for request_id, model_version, rollout_instance_id in zip(
                request_ids, model_versions, rollout_instance_ids
            ):
                entry_info = EntryInfo(
                    rollout_instance_id=rollout_instance_id,
                    request_idx=0,
                    prompt_id=request_id,
                    model_version=model_version,
                )
                entry_infos.append(entry_info)
        return entry_infos

    def log_ready_buffer(self, buffer_id: int):
        """Log the ready buffer."""
        if buffer_id not in self.logged_ready_buffer_ids:
            log_single_event(
                f"Buffer {buffer_id} is ready",
                psrl_logger,
                event_type=EventType.BUFFER_READY,
            )
            self.logged_ready_buffer_ids.add(buffer_id)

    async def handle_ready_buffer(self, buffer_id: int):
        """
        Handle the ready buffer.
        
        Args:
            buffer_id (int): The ID of the buffer that is ready.
        """
        # Check whether there exists ready buffer for training
        self.log_ready_buffer(buffer_id)

        psrl_logger.info(f"Checking staleness and aborting requests for buffer {buffer_id}.")
        aborted_request_ids = await self.ps_manager_handle.handle_ready_buffer.remote(buffer_id)
        if aborted_request_ids:
            self.remove_from_data_pool(aborted_request_ids)

        min_ready_buffer_id = min(self.data_buffers.keys(), default=None)
        if min_ready_buffer_id is not None:
            # Wake all Futures waiting for this buffer
            if min_ready_buffer_id in self._buffer_waiters:
                buffer_data = self.consume_buffer(min_ready_buffer_id)
                assert len(self._buffer_waiters[min_ready_buffer_id]) == 1, (
                    f"Expected only one waiter for buffer {min_ready_buffer_id}, but found {len(self._buffer_waiters[min_ready_buffer_id])}."
                )
                # Set the result for all futures
                for fut in self._buffer_waiters[min_ready_buffer_id]:
                    if not fut.done():
                        fut.set_result(buffer_data)
                # Remove the key after waking all waiters
                del self._buffer_waiters[buffer_id]
            else:
                psrl_logger.warning(f"No waiters found for buffer {buffer_id} when trying to awake.")
                
    async def handle_waiting_buffer(self, buffer_id: int):
        """Handle the waiting buffer."""
        # WIP(lhy): Implement the abort and truncate strategy
        if self.config.psrl.abort_and_truncate_strategy.method == "abort":
            gap = self.ready_entries_per_buffer - self.accumulated_buffer_size[buffer_id]
            if gap == 0:
                return
            assert gap > 0, f"Gap should be greater than 0, but got {gap}"
            if gap <= self.config.psrl.abort_and_truncate_strategy.threshold:
                psrl_logger.info(f"Trying to abort the rest {gap} entries in buffer {buffer_id} and move some occupied entries from other buffers to make it ready.")
                # Guarantee other buffers have enough entries to make it ready
                total_available_entries = 0
                for other_buffer_id in sorted(list(self.accumulated_buffers.keys()), reverse=True):
                    if other_buffer_id == buffer_id:
                        break
                    total_available_entries += sum(len(self.accumulated_buffers[other_buffer_id][model_version]) for model_version in self.accumulated_buffers[other_buffer_id].keys())
                if total_available_entries < gap:
                    psrl_logger.info(f"Not enough entries in other buffers to make buffer {buffer_id} ready, the gap is {gap}, but only {total_available_entries} entries are available")
                    return
                psrl_logger.info(f"Aborting the rest {gap} entries in buffer {buffer_id} and moving some occupied entries from other buffers to make it ready.")
                # First, abort the reserved requests in the buffer
                aborted_entry_num, aborted_request_ids = await self.ps_manager_handle.abort_reserved_requests.remote(buffer_id)
                if aborted_request_ids:
                    self.remove_from_data_pool(aborted_request_ids)
                # Then, move the occupied entries from other buffers to the buffer
                total_moved_entries = 0
                moved_occupied_entry_infos = list()
                for other_buffer_id in sorted(list(self.accumulated_buffers.keys()), reverse=True):
                    if other_buffer_id == buffer_id:
                        break
                    for model_version in sorted(list(self.accumulated_buffers[other_buffer_id].keys())):
                        moved_entry_infos = []
                        for entry_info in self.accumulated_buffers[other_buffer_id][model_version]:
                            moved_entry_infos.append(entry_info)
                            total_moved_entries += 1
                            if total_moved_entries == gap:
                                break
                        if model_version not in self.accumulated_buffers[buffer_id]:
                            self.accumulated_buffers[buffer_id][model_version] = []
                        for moved_entry_info in moved_entry_infos:
                            moved_occupied_entry_infos.append(moved_entry_info)
                            self.accumulated_buffers[buffer_id][model_version].append(moved_entry_info)
                            self.accumulated_buffer_size[buffer_id] += 1
                            self.accumulated_buffers[other_buffer_id][model_version].remove(moved_entry_info)
                            self.accumulated_buffer_size[other_buffer_id] -= 1
                        if total_moved_entries == gap:
                            break
                    for model_version in list(self.accumulated_buffers[other_buffer_id].keys()):
                        if len(self.accumulated_buffers[other_buffer_id][model_version]) == 0:
                            self.accumulated_buffers[other_buffer_id].pop(model_version)
                    if total_moved_entries == gap:
                        break
                for other_buffer_id in list(self.accumulated_buffers.keys()):
                    if len(self.accumulated_buffers[other_buffer_id]) == 0:
                        self.accumulated_buffers.pop(other_buffer_id)
                        self.accumulated_buffer_size.pop(other_buffer_id)
                # Finally, notify the PS manager to move the occupied entries to the buffer 
                await self.ps_manager_handle.move_occupied_entries.remote(moved_occupied_entry_infos, buffer_id)
                psrl_logger.info(f"Moved {total_moved_entries} occupied entries (the total gap is {gap}) from other buffers to buffer {buffer_id}.")
                await self.retry_request(retry_num=aborted_entry_num)
                            
        elif self.config.psrl.abort_and_truncate_strategy.method == "truncate":
            raise NotImplementedError("Truncate strategy is not implemented yet.")
        
        else:
            pass

    async def wait_for_training_batch(self, buffer_id: int) -> DataProto:
        """Await a training batch for a specific buffer ID."""
        await self.ps_manager_handle.ensure_buffer_exists.remote(buffer_id)

        if buffer_id in self.data_buffers:
            # If the buffer is ready, return immediately
            psrl_logger.info(f"Buffer {buffer_id} is ready, returning immediately.")
            return self.consume_buffer(buffer_id)

        # WIP(lhy): Support more consumption strategies
        # 1. Truncate if buffer status is STUCK
        # 2. Abort the RESERVED entry if buffer status is STUCK and move some OCCUPIED entries from other buffers
        if buffer_id in self.accumulated_buffers:
            async with AsyncBusyPollingRayLock(self.ps_manager_handle):
                await self.handle_waiting_buffer(buffer_id)
                
                if self.accumulated_buffer_size[buffer_id] == self.ready_entries_per_buffer:
                    prompt_entry_infos = []
                    for model_version in sorted(list(self.accumulated_buffers[buffer_id].keys())):
                        prompt_entry_infos.extend(self.accumulated_buffers[buffer_id][model_version])
                    # Get the data buffer from the data pool
                    data_buffer = self.get_buffer_from_data_pool(prompt_entry_infos)
                    # Apply buffer post-processing if exists and add to data_buffers
                    add_buffer = self.maybe_add_buffer(buffer_id, data_buffer)
                    if add_buffer:
                        psrl_logger.info(f"Buffer {buffer_id} is READY with {len(self.data_buffers[buffer_id])} entries.")
                        await self.handle_ready_buffer(buffer_id)
                        self.remove_buffer_from_data_pool(prompt_entry_infos)
                        self.accumulated_buffers.pop(buffer_id)
                        self.accumulated_buffer_size.pop(buffer_id)
                        psrl_logger.info(f"Buffer {buffer_id} is ready after the abort and truncate strategy, returning immediately.")
                        return self.consume_buffer(buffer_id)

        # If the buffer is still not ready after the abort and truncate strategy, wait for it to be ready
        psrl_logger.info(f"Buffer {buffer_id} is not ready, waiting for it to be ready.")
        fut = asyncio.get_event_loop().create_future()
        if buffer_id not in self._buffer_waiters:
            self._buffer_waiters[buffer_id] = []
        self._buffer_waiters[buffer_id].append(fut)
        result = await fut
        # Once resumed, return
        return result

    # TODO(lhy): current logic may cause the request have no place to RESERVE
    # If other requests OCCUPY the place to RESERVE, the request will have no place to RESERVE
    # So we may need to reserve here in advance, rather than in the router
    async def retry_request(self, min_version_limit: int = None, retry_num: int = 1):
        """Notify the agent loop manager to retry processing requests associated with a specific buffer ID.

        Args:
            min_version_limit (int): The buffer ID whose requests need to be retried.
            retry_num (int): The number of retries to attempt.
        """
        if self.running_loop and not self.stop_dispatch_task:
            for _ in range(retry_num):
                if not self.data_queue.empty():
                    data = await self.data_queue.get()
                    if data is None:
                        raise ValueError("Data queue should not contain None when retrying requests.")

                    if min_version_limit is not None:
                        data.non_tensor_batch["min_version_limit"] = np.array([min_version_limit] * len(data), dtype=int)
                        psrl_logger.info(
                            f"Retry {len(data)} requests with minimum version limit {min_version_limit}"
                        )
                    else:
                        assert self.config.psrl.routing_strategy.enable_dynamic_version_tag, "Dynamic version tag is not enabled"
                        data.non_tensor_batch["version_tag"] = np.array([-1] * len(data), dtype=int)
                        psrl_logger.info(
                            f"Retry {len(data)} requests with dynamic version tag (i.e., -1)"
                        )

                    await self._inner_dispatch_data(data)
        else:
            psrl_logger.warning("Busy loop of the agent loop manager has stopped, the retry operation will be skipped")

    # ------- DATA POOL MANAGEMENT -------

    def log_buffer(self, buffer_id: int):
        """Log the buffer version tag distribution and staleness.

        Args:
            buffer_id (int): The ID of the buffer to log.
        """
        assert buffer_id in self.data_buffers, f"Buffer {buffer_id} not found in data buffers."
        version_tags = self.data_buffers[buffer_id].non_tensor_batch["version_tag"].tolist()

        # Count different version_tags
        version_tag_counts = Counter(version_tags)
        total_count = len(version_tags)

        # Calculate staleness for each version_tag
        staleness_dict = {}
        for version_tag in version_tag_counts.keys():
            staleness = buffer_id - version_tag
            staleness_dict[version_tag] = staleness

        # Log statistics
        psrl_logger.info(f"Buffer {buffer_id} version tag distribution:")
        for version_tag in sorted(version_tag_counts.keys()):
            count = version_tag_counts[version_tag]
            percentage = (count / total_count) * 100
            staleness = staleness_dict[version_tag]
            psrl_logger.info(f"version_tag={version_tag}: count={count} ({percentage:.2f}%), staleness={staleness}")

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

        self.log_buffer(buffer_id)
        buffer = self.data_buffers.pop(buffer_id, None)
        assert buffer is not None, f"Buffer {buffer_id} not found or already consumed."
        # NOTE(linsh): we will delete buffer during aborting requests of specific versions
        # This is because the inflight requests of the remaining entries
        # in the buffer can still bu utilized for training
        return buffer

    def get_buffer_from_data_pool(self, entry_infos: list[EntryInfo]) -> DataProto:
        """Retrieve data buffers from the internal data pool based on entry information.

        This method is used to fetch specific data buffers that have been stored
        in the reward manager's internal data pool.

        Args:
            entry_infos (List[EntryInfo]): List of EntryInfo objects specifying which buffers to retrieve.

        Returns:
            List[DataProto]: List of DataProto objects corresponding to the requested buffers.
        """
        data_list = []
        for entry_info in entry_infos:
            prompt_id = entry_info.prompt_id
            request_idxs = entry_info.request_idx
            if not isinstance(request_idxs, list):
                request_idxs = [request_idxs]
            assert len(request_idxs) == self.alg_rollout_n, (
                f"EntryInfo for prompt {prompt_id} has {len(request_idxs)} request indices, "
                f"expected {self.alg_rollout_n}."
            )

            for request_idx in request_idxs:
                request_id = prompt_id * self.rollout_n + request_idx
                data = self.data_pool.get(request_id, None)
                assert data is not None, (
                    f"Buffer for request {request_id} (idx {request_idx} for prompt {prompt_id}) "
                    f"not found in data pool."
                )
                data_list.append(data)
        return DataProto.concat(data_list)

    def remove_buffer_from_data_pool(self, entry_infos: list[EntryInfo]):
        """Remove data buffers from the internal data pool based on entry information.

        This method is used to delete specific data buffers that have been stored
        in the reward manager's internal data pool.

        Args:
            entry_infos (List[EntryInfo]): List of EntryInfo objects specifying which buffers to remove.
        """
        for entry_info in entry_infos:
            prompt_id = entry_info.prompt_id
            request_idxs = entry_info.request_idx
            if not isinstance(request_idxs, list):
                request_idxs = [request_idxs]
            for request_idx in request_idxs:
                request_id = prompt_id * self.rollout_n + request_idx
                if request_id in self.data_pool:
                    del self.data_pool[request_id]

    def add_to_data_pool(
        self,
        request_id: int,
        data: DataProto,
    ):
        """
        Add rollout data to the group data pool for a specific entry.

        Args:
            request_id (int): The request ID.
            data (DataProto): The data to add.
        Raises:
            AssertionError: If data for the entry already exists.
        """
        assert request_id not in self.data_pool, f"Data pool already has data for request ID {request_id}"

        self.data_pool[request_id] = data

    def update_data_pool(self, entry_info_to_data: dict[int, DataProto]):
        """Update the internal data pool with new data buffers.

        This method adds new data buffers to the reward manager's internal data pool,
        which can later be retrieved using get_buffer_from_data_pool().

        Args:
            entry_info_to_data (Dict[int, DataProto]):
                Dictionary mapping request IDs to their corresponding DataProto buffers.
        """
        self.data_pool.update(entry_info_to_data)

    def pop_from_data_pool(
        self,
        request_id: int,
    ) -> DataProto:
        """
        Pop rollout data from the group data pool for a specific entry.

        Args:
            request_id (int): The request ID.
        Returns:
            DataProto: The popped data.
        Raises:
            AssertionError: If data for the entry does not exist.
        """
        assert request_id in self.data_pool, f"Data pool must have data for request ID {request_id}"

        return self.data_pool.pop(request_id)

    def get_from_data_pool(
        self,
        request_id: int,
    ) -> DataProto:
        """
        Retrieve rollout data from the group data pool for a specific entry.

        Args:
            request_id (int): The request ID.
        Returns:
            DataProto: The retrieved data.
        Raises:
            AssertionError: If data for the entry does not exist.
        """
        assert request_id in self.data_pool, f"Data pool must have data for request ID {request_id}"

        return self.data_pool[request_id]

    def remove_from_data_pool(
        self,
        request_ids: int | list[int] | set[int],
        guarantee_exists: bool = False,
    ):
        """
        Delete data from the group data pool for a specific entry.

        Args:
            request_ids (Union[int, List[int], Set[int]]): The request ID.
            guarantee_exists (bool): Whether to guarantee that the data exists in the data pool.
        Raises:
            AssertionError: If data for the entry does not exist.
        """
        if not isinstance(request_ids, (list, set)):
            request_ids = [request_ids]
        for request_id in request_ids:
            if guarantee_exists:
                assert request_id in self.data_pool, f"Data pool must have data for request ID {request_id}"
                del self.data_pool[request_id]
            else:
                if request_id in self.data_pool:
                    del self.data_pool[request_id]
