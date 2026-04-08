import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import ray
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local

from psrl.utils.dataset.utils import _pre_process_inputs
from psrl.utils.logger import (
    DualOutputHandler,
    EventType,
    log_data_protocol,
    log_dual_events,
)
from psrl.utils.server.command import Command, CommandExtension, CommandType
from psrl.workers.ps.request_status_tracker import PSRL_RequestStatus
from psrl.workers.reward.reward_loop import load_reward_manager
from psrl.workers.reward.reward_model import PSRL_RewardModelManager

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@dataclass
class RewardSpec:
    reward_manager_type: str
    reward_fn_name: str
    reward_model_name: str | None

    def __hash__(self):
        return hash((self.reward_manager_type, self.reward_fn_name, self.reward_model_name))

    def __eq__(self, other):
        if not isinstance(other, RewardSpec):
            return NotImplemented
        return (
            self.reward_manager_type == other.reward_manager_type
            and self.reward_fn_name == other.reward_fn_name
            and self.reward_model_name == other.reward_model_name
        )

    def key(self) -> str:
        """Return a unique string key for logging/tracking."""
        return f"{self.reward_manager_type}/{self.reward_fn_name}/{self.reward_model_name}"


class RewardLoopManager(CommandExtension):
    def __init__(
        self,
        config,
        tokenizer,
        processor,
        reward_model_configs: list[DictConfig],
        reward_model_to_manager: dict[str, PSRL_RewardModelManager],
        ps_manager_handle=None,
    ):
        """Initialize the reward manager for processing rollout data and computing rewards.

        The reward manager receives rollout data from rollout workers, computes rewards
        using either rule-based functions or reward models, and sends the results
        to the parameter server for training.  It also handles validation reward
        computation via ``compute_score_for_validation``.

        Args:
            config: Configuration object containing server settings and hyperparameters
            tokenizer: Tokenizer for processing text data and converting tokens
            processor: Processor for processing multi-modal data
            ps_manager_handle: Handle to the parameter server for status updates and communication
        """
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.reward_model_to_manager = reward_model_to_manager

        if self.config.psrl.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n
        assert self.rollout_n >= self.alg_rollout_n, (
            f"Rollout n {self.rollout_n} must be greater than or equal to alg_rollout_n {self.alg_rollout_n}."
        )

        # Reward model configuration
        self.request_id_to_future = {}
        self.request_id_to_reward = {}
        self.request_id_to_data_source = {}

        # Reward normalization
        self.reward_normalization = self.config.reward.reward_normalization
        self.request_id_to_group = {}

        # Background event handler
        self.running_loop = None
        self.command_loop_task = None
        self.stop_command_loop_task = False

        # Communication handles
        self.ps_manager_handle = ps_manager_handle

        # Data
        self.request_buffer = {}  # Maps sample IDs to request DataProto (for merging with rollout data)

        self.reward_model_configs = reward_model_configs

        # Reward loop managers
        self.reward_spec_to_manager: dict[RewardSpec, RewardManagerBase] = {}

        self._init_reward_fn()

        # Build logger
        self.log_prefix = "RewardLoopManager"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        psrl_logger.info("Initialized RewardLoopManager.")

    def _init_reward_fn(self):
        """Initialize the reward function and related components.

        This method sets up the reward loop manager based on the configuration,
        including loading tokenizers and reward model routers as needed.
        """
        input_tokenizer_local_path = copy_to_local(self.config.train_actor_rollout_ref.model.path)
        self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path, trust_remote_code=True)

        for reward_model_config in self.reward_model_configs:
            psrl_logger.info(f"Initializing reward function for {reward_model_config}")

            reward_manager_cfg = reward_model_config.reward_manager
            reward_manager_type = reward_model_config.reward_loop_type
            reward_fn_configs = reward_model_config.reward_fn
            reward_model_name = reward_model_config.get("reward_model_name", None)
            reward_loop_kwargs = reward_model_config.get("reward_loop_kwargs", {})

            for reward_fn_config in reward_fn_configs:
                reward_fn_name = reward_fn_config.get("name", None)
                reward_spec = RewardSpec(
                    reward_manager_type=reward_manager_type,
                    reward_fn_name=reward_fn_name,
                    reward_model_name=reward_model_name,
                )

                # If the reward task already has a manager,
                # skip initialization to avoid duplicate managers for the same reward task.
                if reward_spec in self.reward_spec_to_manager:
                    continue

                if reward_manager_type == "gen" and (
                    reward_model_name is None or reward_model_name not in self.reward_model_to_manager
                ):
                    raise ValueError(f"Reward model manager for {reward_model_name} not found")

                reward_model_manager = self.reward_model_to_manager.get(reward_model_name, None)
                reward_manager = load_reward_manager(
                    self.config,
                    self.input_tokenizer,
                    reward_manager_cfg=reward_manager_cfg,
                    reward_fn_config=reward_fn_config,
                    reward_model_manager=reward_model_manager,
                    **reward_loop_kwargs,
                )
                self.reward_spec_to_manager[reward_spec] = reward_manager

    def _resolve_reward_manager(
        self, reward_model_dicts: list[dict]
    ) -> tuple[list[str], list[RewardManagerBase], list[float]]:
        """Resolve reward loop keys, managers, and coefficients from reward_model_dicts.

        This is the single source of truth for extracting per-request reward loop info.

        Args:
            reward_model_dicts: List of reward model config dicts from non_tensor_batch.

        Returns:
            Tuple of (keys, loops, coefs) — all parallel lists of equal length.
        """
        reward_spec_keys, reward_managers, coefs = [], [], []
        for reward_model_dict in reward_model_dicts:
            reward_spec = RewardSpec(
                reward_manager_type=reward_model_dict.get("reward_loop_type", "naive"),
                reward_fn_name=reward_model_dict.get("reward_fn", "default"),
                reward_model_name=reward_model_dict.get("reward_model_name", None),
            )
            reward_spec_keys.append(reward_spec.key())
            reward_managers.append(self.reward_spec_to_manager.get(reward_spec, None))
            coefs.append(reward_model_dict.get("reward_coef", 1.0))
        return reward_spec_keys, reward_managers, coefs

    def add_requests(self, sample_id_to_request_data: dict[int, DataProto]):
        self.request_buffer.update(sample_id_to_request_data)

    def remove_requests(self, sample_ids: list[int]):
        for sample_id in sample_ids:
            self.request_buffer.pop(sample_id, None)

    def start_busy_loop(self):
        """Start the reward manager and begin processing requests.

        This method initializes the server state and starts the background event handler
        task for processing rollout data and computing rewards. The server will run
        until explicitly stopped.
        """
        if self.command_loop_task is not None and not self.command_loop_task.done():
            return

        # Start the background task to process data
        self.running_loop = asyncio.get_running_loop()
        self.command_loop_task = self.running_loop.create_task(self._command_event_handler())
        self.command_loop_task.add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks

    async def stop_busy_loop(self):
        """Shutdown the reward manager gracefully.

        This method stops the command loop task and waits for it
        to complete before returning.
        """
        if not self.command_loop_task or self.command_loop_task.done():
            return

        self.stop_command_loop_task = True
        # Wait for the background task to finish
        await asyncio.gather(self.command_loop_task)

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

        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        # Handle multi-modal inputs and position_ids calculation
        # Only support Qwen2VLImageProcessor for multi-modal processing currently
        # TODO(verl): support other multi-modal inputs
        multi_modal_inputs = None
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            images = inputs.non_tensor_batch["multi_modal_data"].get("image", None)
            current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
            multi_modal_inputs = self.processor(text=[current_text], images=images, return_tensors="pt")
            multi_modal_inputs.pop("input_ids", None)
            multi_modal_inputs.pop("attention_mask", None)

            # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
            # because np.array() only keeps the keys for BatchFeature.
            multi_modal_inputs = dict(multi_modal_inputs)

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
            },
            batch_size=len(input_ids),
        )

        inputs.non_tensor_batch.pop("raw_prompt_ids", None)
        inputs.non_tensor_batch.pop("raw_response_ids", None)
        non_tensor_batch = inputs.non_tensor_batch
        if multi_modal_inputs is not None:
            non_tensor_batch["multi_modal_inputs"] = multi_modal_inputs

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=inputs.meta_info)

    async def _command_event_handler(self):
        """Background task to handle incoming commands for the reward manager.

        This method continuously listens for commands from the command queue
        and processes them accordingly. It supports commands such as aborting
        reward computations for specific requests.
        """
        while not self.stop_command_loop_task:
            # Command processing
            if not self.command_queue.empty():
                # Get command from the queue
                command = self.command_queue.get_nowait()

                assert isinstance(command, Command), f"Expected Command, got {type(command)}"

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
                    assert "parent_ids" in command_args or "uids" in command_args, (
                        "Abort command must contain either 'parent_ids' or 'uids' in args."
                    )
                    parent_ids = command_args.get("parent_ids", None)
                    uids = command_args.get("uids", None)
                    is_validate = command_args.get("is_validate", False)

                    assert not is_validate, "Eval data should not be aborted in reward manager."

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
                        parent_ids = set(parent_ids)  # Ensure uniqueness
                        psrl_logger.debug(f"Getting child requests for {len(parent_ids)} parent_ids")
                        child_uids = await self.ps_manager_handle.get_recorded_child_requests.remote(
                            list(parent_ids), is_validate
                        )
                        psrl_logger.debug(f"Found {len(child_uids)} child requests for the parent_ids")
                        abort_request_uids.update(child_uids)
                    # Step 2. Get requests from uids
                    if uids is not None:
                        uids = set(uids)
                        abort_request_uids.update(uids)

                    psrl_logger.debug(f"Total of {len(abort_request_uids)} requests to abort")
                    # Abort requests in the reward loop manager
                    # 0. Remove data_source from the request_id_to_data_source
                    for abort_request_id in abort_request_uids:
                        self.request_id_to_group.pop(abort_request_id, None)
                        self.request_id_to_data_source.pop(abort_request_id, None)

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
                    update_status_success = await self.ps_manager_handle.update_request_status.remote(
                        list(abort_request_uids),
                        PSRL_RequestStatus.REWARD_COMPLETED,
                        is_validate=is_validate,
                    )
                    assert all(not status for status in update_status_success), (
                        "Update status should not be successful for aborted requests."
                    )
                    result = aborted_count
                else:
                    raise ValueError(f"Unknown command type: {command_type}")

                # Post process the command
                psrl_logger.debug(f"Completing command {command_id} with result: {result}")

            await asyncio.sleep(0)
        psrl_logger.info("Command event handler of reward manager has finished.")

    def normalize_reward(self, request_id_to_reward: dict[int, dict]) -> dict[int, dict]:
        """Normalize the reward for the given request_id_to_reward.

        Args:
            request_id_to_reward (dict[int, dict]): Mapping from request IDs to reward scores and extra info.
        Returns:
            Dict[int, dict]: Mapping from request IDs to normalized reward scores and extra info.
        """
        for request_id, reward in request_id_to_reward.items():
            reward["reward_extra_info"]["data_source"] = self.request_id_to_data_source.pop(request_id)
        if self.reward_normalization != "batch" and self.reward_normalization != "group":
            return request_id_to_reward

        group_rewards_dicts = {}
        # Store original reward structure (dict or float) for each request_id
        original_rewards = {}
        for request_id, reward in request_id_to_reward.items():
            reward_value = reward["reward_score"]
            reward["reward_extra_info"]["original_reward_score"] = reward_value
            original_rewards[request_id] = reward

            group_id = self.request_id_to_group[request_id]
            if group_id not in group_rewards_dicts:
                group_rewards_dicts[group_id] = {
                    "request_ids": [],
                    "rewards": [],
                }
            group_rewards_dicts[group_id]["request_ids"].append(request_id)
            group_rewards_dicts[group_id]["rewards"].append(reward_value)
            self.request_id_to_group.pop(request_id, None)
        for group_id, group_rewards_dict in group_rewards_dicts.items():
            request_ids = group_rewards_dict["request_ids"]
            rewards = np.array(group_rewards_dict["rewards"])
            psrl_logger.info(f"Rewards for group {group_id}: {len(rewards)}")
            norm_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            for request_id, norm_reward in zip(request_ids, norm_rewards):
                original_rewards[request_id]["reward_score"] = float(norm_reward)
                request_id_to_reward[request_id] = original_rewards[request_id]
        return request_id_to_reward

    async def compute_score(self, reward_inputs: DataProto) -> dict[int, dict]:
        """
        Compute the reward score for the given inputs.

        Args:
            reward_inputs (DataProto): Input data for reward computation.
        Returns:
            Dict[int, dict]: Mapping from request IDs to reward scores and extra info.
            For async reward computation, the result will be fetched later via
            `wait_for_reward_of_requests` in the main trainer.
        """
        is_validate = reward_inputs.meta_info.get("validate", False)
        # Skip reward computation during validation for outside reward functions
        if is_validate:
            return {}

        rollout_n = self.rollout_n
        # Data processing
        with log_dual_events(
            "Process reward input",
            psrl_logger,
            level=logging.DEBUG,
            event_type=EventType.OTHER,
        ):
            assert reward_inputs is not None, "Reward input should not be None"
            # assert len(rollout_data) == 1, "Rollout data should contain exactly one request"
            reward_inputs = self._pre_process(reward_inputs)
            psrl_logger.debug(
                f"Reward input after pre-process, "
                f"prompt length: {(reward_inputs.batch['prompts'] != self.tokenizer.pad_token_id).sum(dim=-1)}, "
                f"response length: {(reward_inputs.batch['responses'] != self.tokenizer.pad_token_id).sum(dim=-1)}, "
                f"attention_mask sum: {reward_inputs.batch['attention_mask'].sum(dim=-1)}"
            )
            request_ids = reward_inputs.non_tensor_batch["uid"]
            # is_validate is already known and unchanged after _pre_process

            # Update the request status to REWARD_RUNNING
            update_status_success = await self.ps_manager_handle.update_request_status.remote(
                request_ids.tolist(),
                PSRL_RequestStatus.REWARD_RUNNING,
                is_validate=is_validate,
            )
            if not update_status_success[0]:
                return None

            if rollout_n > 1:
                sample_ids = reward_inputs.non_tensor_batch["parent_id"]
            else:
                sample_ids = reward_inputs.non_tensor_batch["uid"]

        # Compute reward
        results = {}
        with log_dual_events(
            f"Compute reward for samples {sample_ids} and requests {request_ids}",
            psrl_logger,
            level=logging.DEBUG,
            event_type=EventType.OTHER,
        ):
            for i, (sample_id, request_id) in enumerate(zip(sample_ids, request_ids)):
                request_data = self.request_buffer.get(sample_id, None)
                assert request_data is not None, "Request data should not be None."
                reward_input = reward_inputs[i : i + 1]
                reward_input = reward_input.union(request_data)

                # Reward normalization group assignment
                if self.reward_normalization == "batch":
                    group_id = reward_input[0].non_tensor_batch["data_source"]
                    self.request_id_to_group[request_id] = group_id
                elif self.reward_normalization == "group":
                    group_id = reward_input[0].non_tensor_batch["parent_id"]
                    self.request_id_to_group[request_id] = group_id
                self.request_id_to_data_source[request_id] = reward_input[0].non_tensor_batch["data_source"]

                if self.config.reward.launch_reward_fn_async:
                    # Launch async reward computation
                    with log_dual_events(
                        "Launch async reward model score",
                        psrl_logger,
                        level=logging.DEBUG,
                        event_type=EventType.OTHER,
                    ):
                        asyncio.create_task(self._async_reward_task(reward_input))
                else:
                    with log_dual_events(
                        "Compute reward model score",
                        psrl_logger,
                        level=logging.DEBUG,
                        event_type=EventType.OTHER,
                    ):
                        # TODO(zyf): need to support batchify reward computation for sync mode
                        result = await self._compute_score(reward_input)
                        # Update the request status to REWARD_COMPLETED
                        update_status_success = await self.ps_manager_handle.update_request_status.remote(
                            int(request_id),
                            PSRL_RequestStatus.REWARD_COMPLETED,
                            is_validate=is_validate,
                        )
                        complete_request_idxs = [i for i, success in enumerate(update_status_success) if success]
                        if complete_request_idxs:
                            results[request_id] = result

            if not self.config.reward.launch_reward_fn_async:
                results = self.normalize_reward(results)

        return results

    async def _compute_score(self, reward_input: DataProto) -> dict:
        """Run all reward loops for a single sample in parallel and aggregate the results.

        Resolves the reward loops from the input's ``reward_model_dicts``, executes them
        concurrently, and returns a unified result dict with weighted reward score,
        per-loop extra info, and per-loop metrics.

        Args:
            reward_input: A single-sample DataProto slice that already contains
                ``reward_model_dicts`` in its non_tensor_batch (i.e. after ``.union``
                with request_data for training, or directly for validation).

        Returns:
            dict with keys:
                - ``reward_score``: weighted sum of all loop scores (float)
                - ``reward_extra_info``: ``{loop_key: extra_info_dict, ...}``
                - ``reward_metrics``: ``{loop_key: metrics_dict, ...}``
        """
        reward_model_dicts = reward_input[0].non_tensor_batch["reward_model_dicts"]
        reward_spec_keys, reward_managers, reward_coefs = self._resolve_reward_manager(reward_model_dicts)

        futures = [asyncio.create_task(reward_manager.run_single(reward_input)) for reward_manager in reward_managers]
        results = await asyncio.gather(*futures)

        reward_score = sum(r["reward_score"] * c for r, c in zip(results, reward_coefs))
        reward_extra_info_dict = {key: r["reward_extra_info"] for key, r in zip(reward_spec_keys, results)}
        reward_metrics_dict = {key: r.get("reward_metrics", {}) for key, r in zip(reward_spec_keys, results)}

        return {
            "reward_score": reward_score,
            "reward_extra_info": reward_extra_info_dict,
            "reward_metrics": reward_metrics_dict,
        }

    async def compute_score_for_validation(self, reward_inputs: DataProto) -> dict[int, dict]:
        """Compute reward scores for a validation batch.

        Mirrors the logic of ``compute_score`` for the training path but omits
        PS status updates and reward normalisation, which are not applicable to
        validation.  Supports multiple reward models per sample in the same way
        as the training path: ``reward_model_dicts`` in the non_tensor_batch
        controls which loops run and their coefficients.

        Args:
            reward_inputs: Validation batch.  Must already carry
                ``reward_model_dicts`` in its non_tensor_batch; unlike the
                training path, no ``request_buffer`` look-up or ``_pre_process``
                step is needed because the batch is prepared by the trainer.

        Returns:
            Dict mapping request_id -> reward result dict
            (``reward_score``, ``reward_extra_info``, ``reward_metrics``).
        """
        assert reward_inputs is not None, "Reward input should not be None"
        request_ids = reward_inputs.non_tensor_batch["uid"]

        with log_dual_events(
            f"Compute reward for requests {request_ids}",
            psrl_logger,
            level=logging.DEBUG,
            event_type=EventType.OTHER,
        ):
            results = {}
            for i, request_id in enumerate(request_ids):
                reward_input = reward_inputs[i : i + 1]

                if self.config.reward.launch_reward_fn_async:
                    with log_dual_events(
                        "Launch async reward model score",
                        psrl_logger,
                        level=logging.DEBUG,
                        event_type=EventType.OTHER,
                    ):
                        asyncio.create_task(self._async_reward_task(reward_input))
                else:
                    with log_dual_events(
                        "Compute reward model score",
                        psrl_logger,
                        level=logging.DEBUG,
                        event_type=EventType.OTHER,
                    ):
                        psrl_logger.info(f"{request_id=} sync validation reward task")
                        result = await self._compute_score(reward_input)
                        results[request_id] = result

        if self.config.reward.launch_reward_fn_async:
            return await self.wait_for_reward_of_requests(request_ids.tolist(), is_validate=True)
        return results

    async def _async_reward_task(self, reward_input: DataProto):
        """Fire-and-forget async task: compute reward and store the result for later retrieval.

        Derives all reward loop configuration from ``reward_input`` itself via
        ``_compute_score``, so callers only need to pass the data.
        Works for both training and validation paths.
        """
        request_id = reward_input.non_tensor_batch["uid"][0]
        result = await self._compute_score(reward_input)
        await self.set_reward_for_requests({request_id: result})

    async def wait_for_reward_of_requests(self, request_ids: list[int], is_validate: bool = False):
        """Wait for the reward results of the specified requests.

        This method blocks until the reward results for all specified request IDs
        are available, either from previously computed rewards or from ongoing
        reward computation tasks.
        """
        request_id_to_reward = {}
        futures_to_wait = {}

        for request_id in request_ids:
            if request_id in self.request_id_to_reward:
                request_id_to_reward[request_id] = self.request_id_to_reward.pop(request_id)
            elif request_id in self.request_id_to_future:
                futures_to_wait[request_id] = self.request_id_to_future[request_id]
            else:
                fut = asyncio.get_running_loop().create_future()
                self.request_id_to_future[request_id] = fut
                futures_to_wait[request_id] = fut

        if futures_to_wait:
            results = await asyncio.gather(*futures_to_wait.values())
            for request_id, reward in zip(futures_to_wait.keys(), results):
                request_id_to_reward[request_id] = reward

        for request_id in request_ids:
            self.request_id_to_future.pop(request_id, None)

        # return request_id_to_reward
        if not is_validate:
            return self.normalize_reward(request_id_to_reward)
        else:
            return request_id_to_reward

    async def set_reward_for_requests(self, request_id_to_reward: dict[int, Any]):
        """Set the reward for the specified request IDs."""
        for request_id, reward in request_id_to_reward.items():
            if request_id in self.request_id_to_future:
                fut = self.request_id_to_future[request_id]
                if not fut.done():
                    fut.set_result(reward)
            else:
                self.request_id_to_reward[request_id] = reward
