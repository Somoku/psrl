import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import ray
import torch
from omegaconf import DictConfig
import transfer_queue as tq
from transfer_queue import KVBatchMeta
from tensordict import TensorDict
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils import tensordict_utils as tu
from verl.utils.model import compute_position_id_with_mask
from verl.utils.tensordict_utils import list_of_dict_to_tensordict
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local

from psrl.utils.dataset.utils import _pre_process_inputs
from psrl.utils.logger import (
    DualOutputHandler,
    EventType,
    log_dual_events,
)
from psrl.utils.server.command import Command, CommandExtension, CommandType
from psrl.workers.ps.request_status_tracker import PSRL_RequestStatus
from psrl.workers.reward.reward_loop import load_reward_manager
from psrl.workers.reward.reward_model import RewardModelManager

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
        reward_model_to_manager: dict[str, RewardModelManager],
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
            # reward_fn can be either a string name (from dataset defaults) or a
            # list-of-config-dicts (from system config). Normalise to string.
            raw_reward_fn = reward_model_dict.get("reward_fn", "default")
            if isinstance(raw_reward_fn, list) and len(raw_reward_fn) > 0:
                reward_fn_name = raw_reward_fn[0].get("name", "default") if isinstance(raw_reward_fn[0], dict) else str(raw_reward_fn[0])
            elif isinstance(raw_reward_fn, dict):
                reward_fn_name = raw_reward_fn.get("name", "default")
            else:
                reward_fn_name = str(raw_reward_fn)

            # reward_model_name: treat string "null" the same as Python None
            raw_model_name = reward_model_dict.get("reward_model_name", None)
            if raw_model_name == "null":
                raw_model_name = None

            reward_spec = RewardSpec(
                reward_manager_type=reward_model_dict.get("reward_loop_type", "naive"),
                reward_fn_name=reward_fn_name,
                reward_model_name=raw_model_name,
            )
            manager = self.reward_spec_to_manager.get(reward_spec, None)

            # Fallback: if not found by exact match, try progressively looser matching.
            # This handles mismatches between dataset-default reward_model_dicts
            # (e.g. reward_loop_type="naive", reward_fn="default") and the actual
            # system-configured reward spec.
            if manager is None:
                # 1st fallback: match by reward_loop_type only
                candidates = [
                    (spec, mgr) for spec, mgr in self.reward_spec_to_manager.items()
                    if spec.reward_manager_type == reward_spec.reward_manager_type
                ]
                if len(candidates) == 0:
                    # 2nd fallback: if only one manager registered globally, use it
                    all_managers = list(self.reward_spec_to_manager.items())
                    if len(all_managers) == 1:
                        candidates = all_managers
                    elif len(all_managers) > 1:
                        psrl_logger.warning(
                            f"No reward spec match for {reward_spec} among {[s for s, _ in all_managers]}. "
                            f"Using the first registered manager as fallback."
                        )
                        candidates = [all_managers[0]]
                if candidates:
                    reward_spec, manager = candidates[0]

            reward_spec_keys.append(reward_spec.key())
            reward_managers.append(manager)
            coefs.append(reward_model_dict.get("reward_coef", 1.0))
        return reward_spec_keys, reward_managers, coefs

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

    def _compute_multi_modal_inputs(self, inputs: TensorDict, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute multi-modal inputs with image and video.

        Mirrors ``PSRL_AgentLoopManager._compute_multi_modal_inputs`` but omits
        ``images_seqlens`` computation (not needed by reward functions).
        """
        multi_modal_inputs = {}
        if self.processor is None:
            return multi_modal_inputs

        images = tu.get(inputs, "multi_modal_data", {})[0].get("images", None)
        videos = tu.get(inputs, "multi_modal_data", {})[0].get("videos", None)
        # split the videos and according metadatas
        if videos is not None:
            videos, video_metadatas = zip(*videos, strict=False)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None
        current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
        multi_modal_inputs = self.processor(
            text=[current_text],
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            return_tensors="pt",
            do_sample_frames=False,
        )
        multi_modal_inputs.pop("input_ids", None)
        multi_modal_inputs.pop("attention_mask", None)

        # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
        # because np.array() only keeps the keys for BatchFeature.
        multi_modal_inputs = dict(multi_modal_inputs.convert_to_tensors("pt"))
        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            images_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0])
            multi_modal_inputs["images_seqlens"] = images_seqlens
        return multi_modal_inputs

    def _compute_position_ids(self, input_ids, attention_mask, multi_modal_inputs) -> torch.Tensor:
        """Compute position ids for multi-modal inputs."""
        if self.processor is None:
            return compute_position_id_with_mask(attention_mask)  # (1, seq_len)

        multi_modal_kwargs = {
            "image_grid_thw": multi_modal_inputs.get("image_grid_thw"),
            "video_grid_thw": multi_modal_inputs.get("video_grid_thw"),
        }
        # For transformers>=5.3.0, mm_token_type_ids is only used to calculate position ids.
        if multi_modal_inputs.pop("mm_token_type_ids", None) is not None:
            mm_token_type_ids = torch.zeros_like(input_ids)
            mm_token_type_ids[0][input_ids[0] == self.processor.image_token_id] = 1
            mm_token_type_ids[0][input_ids[0] == self.processor.video_token_id] = 2
            multi_modal_kwargs["mm_token_type_ids"] = mm_token_type_ids

        # Model's get_rope_index has been dynamically bind to the processor.
        vision_position_ids, _ = self.processor.get_rope_index(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **multi_modal_kwargs,
        )
        vision_position_ids = vision_position_ids.transpose(0, 1)  # (3, 1, seq_len) => (1, 3, seq_len)

        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        text_position_ids = text_position_ids.unsqueeze(0)
        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
        return position_ids

    def pre_process(self, inputs: TensorDict) -> TensorDict:
        prompts = tu.get(inputs, "prompts")
        responses = tu.get(inputs, "responses")
        # prompts and responses are 1D unpadded tensors stored per-sample in TQ.
        input_ids = torch.cat([prompts, responses], dim=0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.int64) # No padding
        # _compute_multi_modal_inputs and _compute_position_ids expect [1, seq_len].
        multi_modal_inputs = self._compute_multi_modal_inputs(inputs, input_ids.unsqueeze(0))
        position_ids = self._compute_position_ids(
            input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
        ).squeeze(0)
        inputs["position_ids"] = position_ids
        return inputs

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
                            self.request_id_to_reward[abort_request_id] = None  # Mark reward as None for aborted requests

                    psrl_logger.debug(f"Aborted {aborted_count} running reward computations")
                    # 2. Remove from the request tracker (update_status)
                    update_status_success = await self.ps_manager_handle.update_request_status.remote(
                        list(abort_request_uids),
                        PSRL_RequestStatus.REWARD_COMPLETED,
                        is_validate=is_validate,
                    )
                    # update_request_status returns bool when single request, list[bool] when multiple
                    if isinstance(update_status_success, bool):
                        update_status_success = [update_status_success]
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

    def normalize_reward(self, reward_batch: KVBatchMeta) -> KVBatchMeta:
        """Normalize the reward for the given reward_batch.

        Args:
            reward_batch (KVBatchMeta): Batch of reward data.
        Returns:
            KVBatchMeta: Normalized reward batch.
        """
        fields = ["uid", "reward_extra_info", "reward_score", "data_source"]
        data = tq.kv_batch_get(keys=reward_batch.keys, partition_id=reward_batch.partition_id, select_fields=fields)

        uids = data["uid"]
        reward_extra_infos = data["reward_extra_info"]
        data_sources = data["data_source"]
        reward_scores = data["reward_score"]

        for reward_extra_info, data_source, reward_score in zip(reward_extra_infos, data_sources, reward_scores):
            reward_extra_info["data_source"] = data_source
            reward_extra_info["original_reward_score"] = reward_score

        if self.reward_normalization != "batch" and self.reward_normalization != "group":
            data["reward_extra_info"] = reward_extra_infos
            tq.kv_batch_put(keys=reward_batch.keys, partition_id=reward_batch.partition_id, fields=data.select("reward_extra_info"))
            return reward_batch

        group_rewards_dicts = {}
        for uid, reward_extra_info, reward_score in zip(uids, reward_extra_infos, reward_scores):
            reward_extra_info["original_reward_score"] = reward_score
            group_id = self.request_id_to_group[uid]
            if group_id not in group_rewards_dicts:
                group_rewards_dicts[group_id] = {
                    "uids": [],
                    "rewards": [],
                }
            group_rewards_dicts[group_id]["uids"].append(uid)
            group_rewards_dicts[group_id]["rewards"].append(reward_score)
            self.request_id_to_group.pop(uid, None)

        for group_id, group_rewards_dict in group_rewards_dicts.items():
            uids = group_rewards_dict["uids"]
            rewards = np.array(group_rewards_dict["rewards"])
            psrl_logger.info(f"Rewards for group {group_id}: {len(rewards)}")
            norm_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            for uid, norm_reward in zip(uids, norm_rewards):
                reward_scores[uids.index(uid)] = float(norm_reward)
        
        # Put updated reward scores and extra info back to the batch
        data["reward_extra_info"] = reward_extra_infos
        data["reward_score"] = reward_scores

        # Update rm_scores
        response_mask = tq.kv_batch_get(
            keys=reward_batch.keys,
            partition_id=reward_batch.partition_id,
            select_fields=["response_mask"],
        )["response_mask"]
        rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
        for i, reward_score in enumerate(reward_scores):
            rm_scores[i][-1] = reward_score
        data["rm_scores"] = rm_scores

        tq.kv_batch_put(keys=reward_batch.keys, partition_id=reward_batch.partition_id, fields=data.select("reward_extra_info", "reward_score", "rm_scores"))
        return reward_batch

    async def compute_score(self, reward_inputs: TensorDict) -> dict[str, Any]:
        """
        Compute the reward score for the given inputs.

        Args:
            reward_inputs (_post_process_and_merge_reward): Input data for reward computation.
        Returns:
            dict: A dictionary containing the computed reward score and any additional information.
        """
        reward_inputs = self.pre_process(reward_inputs)

        is_validate = tu.get(reward_inputs, "validate", False)
        request_ids = tu.get(reward_inputs, "uid")

        # Update the request status to REWARD_RUNNING
        update_status_success = await self.ps_manager_handle.update_request_status.remote(
            request_ids,
            PSRL_RequestStatus.REWARD_RUNNING,
            is_validate=is_validate,
        )
        if not update_status_success:
            # Mark reward as None for requests that fail to update status
            self.request_id_to_reward[request_ids[0]] = None
            return None

        # Compute reward
        result = None
        with log_dual_events(
            f"Compute reward for requests {request_ids}",
            psrl_logger,
            level=logging.DEBUG,
            event_type=EventType.OTHER,
        ):
            for i, request_id in enumerate(request_ids):
                reward_input = reward_inputs[i : i + 1]
                parent_id = tu.get(reward_input, "parent_id")[0]
                data_source = tu.get(reward_input, "data_source")[0]

                # Reward normalization group assignment
                if self.reward_normalization == "batch":
                    group_id = data_source
                    self.request_id_to_group[request_id] = group_id
                elif self.reward_normalization == "group":
                    group_id = parent_id
                    self.request_id_to_group[request_id] = group_id
                self.request_id_to_data_source[request_id] = data_source

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
                            request_id,
                            PSRL_RequestStatus.REWARD_COMPLETED,
                            is_validate=is_validate,
                        )
                        if not update_status_success:
                            result = None

        return result

    async def _compute_score(self, reward_input: TensorDict) -> dict[str, Any]:
        """Run all reward loops for a single sample in parallel and aggregate the results.

        Resolves the reward loops from the input's ``reward_model_dicts``, executes them
        concurrently, and returns a unified result dict with weighted reward score,
        per-loop extra info, and per-loop metrics.

        Args:
            reward_input: A single-sample TensorDict slice that already contains
                ``reward_model_dicts`` in its non_tensor_batch (i.e. after ``.union``
                with request_data for training, or directly for validation).

        Returns:
            dict with keys:
                - ``reward_score``: weighted sum of all loop scores (float)
                - ``reward_extra_info``: ``{loop_key: extra_info_dict, ...}``
                - ``reward_metrics``: ``{loop_key: metrics_dict, ...}``
        """
        assert len(reward_input) == 1, "Reward input for _compute_score should contain exactly one sample"

        request_ids = tu.get(reward_input, "uid")
        keys = [str(request_id) for request_id in request_ids]
        partition_id = "val" if tu.get(reward_input, "validate", False) else "train"
        fields = ["reward_model_dicts"]

        data = await tq.async_kv_batch_get(keys=keys, partition_id=partition_id, select_fields=fields)

        reward_model_dicts = data["reward_model_dicts"]
        reward_spec_keys, reward_managers, reward_coefs = self._resolve_reward_manager(reward_model_dicts)

        # Return reward data instead of meta
        futures = [asyncio.create_task(reward_manager.run_single(reward_input)) for reward_manager in reward_managers]
        results = await asyncio.gather(*futures)

        reward_score = sum(r["reward_score"] * c for r, c in zip(results, reward_coefs))
        reward_extra_info_dict = {key: r["reward_extra_info"] for key, r in zip(reward_spec_keys, results)}
        reward_metrics_dict = {key: r.get("reward_metrics", {}) for key, r in zip(reward_spec_keys, results)}

        # TODO(linsh): consider multi-step rewards from multi-sources
        result = {
            "reward_score": reward_score,
            "reward_extra_info": reward_extra_info_dict,
            "reward_metrics": reward_metrics_dict,
        }

        return result

    async def _async_reward_task(self, reward_input: TensorDict):
        """Fire-and-forget async task: compute reward and store the result for later retrieval.

        Derives all reward loop configuration from ``reward_input`` itself via
        ``_compute_score``, so callers only need to pass the data.
        Works for both training and validation paths.
        """
        assert len(reward_input) == 1, "Async reward task should be launched with a single-sample batch"

        uid = tu.get(reward_input, "uid")[0]
        result = await self._compute_score(reward_input)
        await self.set_reward_for_requests({uid: result})

    async def wait_for_reward_of_requests(self, requests: KVBatchMeta) -> KVBatchMeta:
        """Wait for the reward results of the specified requests.

        This method blocks until the reward results for all specified request IDs
        are available, either from previously computed rewards or from ongoing
        reward computation tasks.
        """
        request_ids = [int(key) for key in requests.keys]
        is_validate = requests.partition_id == "val"

        request_id_to_reward: dict[int, KVBatchMeta] = {}
        futures_to_wait = {}

        # Gather available rewards and futures for the requested IDs.
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

        fields = []
        for request_id in request_ids:
            fields.append(request_id_to_reward[request_id])
            self.request_id_to_future.pop(request_id, None)

        await tq.async_kv_batch_put(
            keys=requests.keys,
            partition_id="train" if not is_validate else "val",
            fields=list_of_dict_to_tensordict(fields),
        )

        if not is_validate:
            return self.normalize_reward(requests)
        else:
            return requests

    async def set_reward_for_requests(self, request_id_to_reward: dict[int, dict[str, Any]]):
        """Set the reward for the specified request IDs."""
        for request_id, reward in request_id_to_reward.items():
            if request_id in self.request_id_to_future:
                fut = self.request_id_to_future[request_id]
                if not fut.done():
                    fut.set_result(reward)
            else:
                self.request_id_to_reward[request_id] = reward
