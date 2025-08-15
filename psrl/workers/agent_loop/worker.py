import os
import logging
import asyncio
import hydra
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from collections import deque

import ray

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

from psrl.workers.agent_loop.utils import DummyConfig, AgentLoopOutput, _agent_loop_registry
from psrl.workers.ps.request_status_tracker import RequestStatus
from psrl.workers.agent_loop.router import RolloutRouter
from psrl.utils.logger import DualOutputHandler, get_worker_info, log_single_event, EventType, deprecated

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

@ray.remote
class PSRL_AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(
        self,
        config: DictConfig,
        tokenizer,
        ps_manager_handle,
        rollout_wg_list,
        rollout_queue,
        processor=None,
    ):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): YAML config.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        
        self.rollout_router = RolloutRouter(
            config,
            ps_manager_handle,
            rollout_wg_list,
        )
        self.ps_manager_handle = ps_manager_handle
        self.rollout_wg_list = rollout_wg_list
        self.rollout_queue = rollout_queue
        
        self.agent_programs = set()
        self.pending_program_queue = deque()

        self.running_loop = asyncio.get_running_loop()
        self.busy_loop_task = None
        self.stop_busy_loop_task = False

        # Register agent loop configs from file
        agent_loop_config_path = config.agent.agent_loop_config_path
        if agent_loop_config_path:
            agent_loop_configs = OmegaConf.load(agent_loop_config_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config

    def add_agent_program(self, data: DataProto):
        if isinstance(data, DataProto):
            batch_size = len(data)
            if batch_size > 0:
                programs = data.chunk(batch_size)
                for program in programs:
                    self.pending_program_queue.append(program)
            else:
                raise ValueError("Cannot add empty DataProto to agent loop worker.")
        elif data is None:
            self.pending_program_queue.append(None)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}. Expected DataProto or None.")

    def start_busy_loop(self):
        if self.busy_loop_task is not None and not self.busy_loop_task.done():
            return

        # Start the background task to process data
        self.busy_loop_task = self.running_loop.create_task(self._launch_agent_loop())

    def stop_busy_loop(self):
        if not self.busy_loop_task or self.busy_loop_task.done():
            return

        self.stop_busy_loop_task = True
        # Wait for the background task to finish
        self.running_loop.run_until_complete(self.busy_loop_task)

    async def _launch_agent_loop(self):
        while not self.stop_busy_loop_task:
            if len(self.pending_program_queue) > 0:
                program = self.pending_program_queue.popleft()
                if program is None:
                    self.stop_busy_loop_task = True
                    continue
                await self.generate_sequences(program)
            await asyncio.sleep(0)

    def _create_task_done_callback(self, task):
        self.agent_programs.discard(task)

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        # by default, we assume it's a generation-only agent
        agent_names = batch.non_tensor_batch.pop("agent_name", np.array(["generate_only_agent"] * len(batch), dtype=object))

        batch_size = len(batch)
        request_list = batch.chunk(batch_size)
        for agent_name, request in zip(agent_names, request_list, strict=True):
            task = asyncio.create_task(
                self._run_agent_loop(agent_name, request)
            )
            task.add_done_callback(self._create_task_done_callback(task))
            self.agent_programs.add(task)

    async def _run_agent_loop(
        self,
        agent_name: str,
        request: DataProto,
    ):
        assert agent_name in _agent_loop_registry, (
            f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
        )
        agent_loop_config = _agent_loop_registry[agent_name]
        
        agent_loop = hydra.utils.instantiate(
            config=agent_loop_config,
            trainer_config=DummyConfig(config=self.config),
            rollout_router=self.rollout_router,
            ps_manager_handle=self.ps_manager_handle,
            tokenizer=self.tokenizer,
        )
        output = await agent_loop.run(request)
        
        if output is not None:
            request_ids = request.non_tensor_batch["uid"]
            update_status_success = await self.ps_manager_handle.update_request_status.remote(
                request_ids.tolist(),
                RequestStatus.COMPLETED,
            )
            dispatch_request_idxs = [i for i, success in enumerate(update_status_success) if success]
            if update_status_success[0]:
                output = self._post_process(output)
                self.rollout_queue.put(output)

    def update_engine_status(self, engine_status: dict):
        """Update the engine status received from RolloutCoordinator."""
        async def _update_status():
            self.rollout_router.update_engine_status(engine_status)
            # Log some key metrics
            instances = engine_status.get("instances", {})
            total_queue_size = sum(
                inst_status.get("waiting_and_running_queue_size", 0) 
                for inst_status in instances.values()
            )
            print(f"Updated engine status: {len(instances)} instances, total queue size: {total_queue_size}")
        
        # Schedule the async update
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_update_status())
        except RuntimeError:
            # If no event loop is running, create one
            asyncio.run(_update_status())

    def get_engine_status(self):
        """Get the latest engine status."""
        return self.rollout_router.latest_engine_status

    def _post_process(self, inputs: DataProto) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # prompts
        self.tokenizer.padding_side = "left"
        prompt_output = self.tokenizer.pad(
            [{"input_ids": input.batch["input_ids"]} for input in inputs],
            padding="max_length",
            max_length=self.config.gen_actor_rollout_ref.rollout.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        prompt_ids, prompt_attention_mask = prompt_output["input_ids"], prompt_output["attention_mask"]

        # responses
        self.tokenizer.padding_side = "right"
        outputs = self.tokenizer.pad(
            [{"input_ids": input.non_tensor_batch.pop("raw_response_ids")} for input in inputs],
            padding="max_length",
            max_length=self.config.gen_actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        response_ids, response_attention_mask = outputs["input_ids"], outputs["attention_mask"]

        # response_mask
        outputs = self.tokenizer.pad(
            [{"input_ids": input.non_tensor_batch.pop("response_mask")} for input in inputs],
            padding="max_length",
            max_length=self.config.gen_actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        response_mask = outputs["input_ids"]

        assert response_ids.shape == response_mask.shape, (
            f"mismatch in response_ids and response_mask shape: {response_ids.shape} vs {response_mask.shape}"
        )
        
        response_mask = response_mask * response_attention_mask
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        # Handle multi-modal inputs and position_ids calculation
        # Only support Qwen2VLImageProcessor for multi-modal processing currently
        # TODO: support other multi-modal inputs
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
