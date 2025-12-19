import asyncio
import logging
import os
from collections import deque

import hydra
import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from verl import DataProto
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask

from psrl.utils.dataset.utils import _pre_process_inputs
from psrl.utils.logger import DualOutputHandler, EventType, log_dual_events
from psrl.workers.agent_loop.loops.utils import AGENT_LOOP_REGISTRY, DummyConfig
from psrl.workers.ps.request_status_tracker import PSRL_RequestStatus

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@ray.remote
class PSRL_AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(
        self,
        config: DictConfig,
        ps_manager_handle,
        rollout_router,
        rollout_wg_list,
    ):
        """Initialize agent loop worker.

        Args:
            config (DictConfig): Configuration containing model and rollout settings.
            ps_manager_handle: Handle to the parameter server manager.
            rollout_router: Handle to the rollout router actor.
            rollout_wg_list: List of rollout worker groups.
            rollout_queue: Queue for storing completed rollout results.
        """
        self.config = config
        model_path = config.gen_actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.gen_actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = hf_processor(local_path, trust_remote_code=True)

        self.rollout_router = rollout_router
        self.ps_manager_handle = ps_manager_handle
        self.rollout_wg_list = rollout_wg_list
        self.agent_loop_manager = None
        self.reward_manager = None

        self.agent_programs = set()
        self.pending_program_queue = deque()

        self.running_loop = None
        self.busy_loop_task = None
        self.stop_busy_loop_task = False

        # Register agent loop configs from file
        agent_loop_config_path = config.gen_actor_rollout_ref.rollout.agent.agent_loop_config_path
        if agent_loop_config_path:
            agent_loop_configs = OmegaConf.load(agent_loop_config_path)
            for agent_loop_config in agent_loop_configs:
                AGENT_LOOP_REGISTRY[agent_loop_config.name] = agent_loop_config

        # Build logger
        # TODO(lhy): support >1 workers
        self.log_prefix = "AgentLoopWorker"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))

    def set_agent_loop_manager(self, agent_loop_manager: ray.actor.ActorHandle):
        """Set the agent loop manager handle for communication.

        Args:
            agent_loop_manager: Handle to the agent loop manager actor.
        """
        self.agent_loop_manager = agent_loop_manager

    def set_reward_manager(self, reward_manager: ray.actor.ActorHandle):
        """Set the reward manager handle for sending processed data.

        Args:
            reward_manager: Handle to the reward manager actor.
        """
        self.reward_manager = reward_manager

    def add_agent_program(self, data: DataProto):
        """Add a new agent program to the pending queue for processing.

        Args:
            data (DataProto or None): Data to process, or None to signal termination.
        """
        if isinstance(data, DataProto):
            # Prioritize retry requests
            if "min_version_limit" in data.non_tensor_batch:
                self.pending_program_queue.appendleft(data)
            else:
                self.pending_program_queue.append(data)
        elif data is None:
            self.pending_program_queue.append(None)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}. Expected DataProto or None.")

    def start_busy_loop(self):
        """Start the busy loop to continuously process agent programs from the queue."""
        if self.busy_loop_task is not None and not self.busy_loop_task.done():
            return

        # Start the background task to process data
        self.running_loop = asyncio.get_running_loop()
        self.busy_loop_task = self.running_loop.create_task(self._launch_agent_loop())
        self.busy_loop_task.add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks

    def stop_busy_loop(self):
        """Stop the busy loop and wait for the current task to complete."""
        if not self.busy_loop_task or self.busy_loop_task.done():
            return

        self.stop_busy_loop_task = True
        # Wait for the background task to finish
        self.running_loop.run_until_complete(self.busy_loop_task)

    async def _launch_agent_loop(self):
        """Main loop that processes agent programs from the pending queue."""
        while not self.stop_busy_loop_task:
            if len(self.pending_program_queue) > 0:
                program = self.pending_program_queue.popleft()
                if program is None:
                    self.stop_busy_loop_task = True
                    continue
                await self.generate_trajectories(program)
            await asyncio.sleep(0)

    def _create_task_done_callback(self, task):
        """Create a callback function to handle task completion."""

        def task_done_callback(future):
            try:
                future.result()  # This will raise an exception if the task failed
            except Exception as e:
                psrl_logger.error(f"Task {task} failed with exception: {e}")
            finally:
                self.agent_programs.discard(task)

        return task_done_callback

    async def generate_trajectories(self, batch: DataProto) -> DataProto:
        """Generate trajectories using the specified agent type based on configuration.

        This method only create the task (agent_loop) and add the task to the agent_programs set.
        But the task is not await here so different agent_loop can be run in parallel.

        Args:
            batch (DataProto): Input batch containing prompts and metadata.

        Returns:
            DataProto: Generated trajectories and associated data.
        """
        # by default, we assume it's a generation-only agent
        if self.config.psrl.gen_mode == "batch":
            default_agent_name = "batch_generate_only_agent"
        elif self.config.psrl.gen_mode == "stream":
            default_agent_name = "generate_only_agent"

        agent_names = batch.non_tensor_batch.pop(
            "agent_name", np.array([default_agent_name] * len(batch), dtype=object)
        )
        assert np.all(agent_names == agent_names[0]), "All agent names must be the same for generation-only agents."
        agent_name = agent_names[0]

        task = asyncio.create_task(self._run_agent_loop(agent_name, batch))
        task.add_done_callback(self._create_task_done_callback(task))
        self.agent_programs.add(task)

    async def _run_agent_loop(
        self,
        agent_name: str,
        requests: DataProto,
    ):
        """Execute the specified agent loop on the given requests.

        Args:
            agent_name (str): Name of the agent loop to run.
            requests (DataProto): Input requests to process.
        """
        assert agent_name in AGENT_LOOP_REGISTRY, (
            f"Agent loop {agent_name} not registered, registered agent loops: {AGENT_LOOP_REGISTRY.keys()}"
        )
        agent_loop_config = AGENT_LOOP_REGISTRY[agent_name]

        agent_loop = hydra.utils.instantiate(
            config=agent_loop_config,
            trainer_config=DummyConfig(config=self.config),
            rollout_router=self.rollout_router,
            reward_manager=self.reward_manager,
            ps_manager_handle=self.ps_manager_handle,
            tokenizer=self.tokenizer,
        )

        with log_dual_events(
            f"Agent loop with requests {requests.non_tensor_batch['uid']}",
            psrl_logger,
            level=logging.DEBUG,
            event_type=EventType.GEN,
        ):
            output = await agent_loop.run(requests)

        if output is not None:
            assert isinstance(output, DataProto), f"Output must be a DataProto for now (got {type(output)})"
            request_ids = requests.non_tensor_batch["uid"]
            with log_dual_events(
                "Update request status",
                psrl_logger,
                level=logging.DEBUG,
                event_type=EventType.OTHER,
            ):
                update_status_success = await self.ps_manager_handle.update_request_status.remote(
                    request_ids.tolist(),
                    PSRL_RequestStatus.COMPLETED,
                )
            with log_dual_events(
                f"Put requests {request_ids} into result queue",
                psrl_logger,
                level=logging.DEBUG,
                event_type=EventType.OTHER,
            ):
                dispatch_request_idxs = [i for i, success in enumerate(update_status_success) if success]
                if dispatch_request_idxs:
                    output = output.select_idxs(dispatch_request_idxs)
                    # NOTE(lhy): The DataProto will be huge and slow to transfer when putting into
                    # the result queue, so we process the data inside the reward manager
                    # output = self._post_process(output)
                    await self.agent_loop_manager.put_result.remote(output)

    # NOTE(lhy): This method is moved to the reward manager
    def _post_process(self, inputs: DataProto) -> DataProto:
        """Post-process the generated outputs to create properly formatted tensors.

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
                [
                    _pre_process_inputs(self.tokenizer.pad_token_id, inputs.batch["input_ids"][i])
                    for i in range(batch_size)
                ],
                dtype=object,
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
        prompt_ids, prompt_attention_mask = (
            prompt_output["input_ids"],
            prompt_output["attention_mask"],
        )

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
        response_ids, response_attention_mask = (
            outputs["input_ids"],
            outputs["attention_mask"],
        )

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
