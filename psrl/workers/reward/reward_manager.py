import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import ray
import torch
import transfer_queue as tq
from omegaconf import DictConfig
from tensordict import TensorDict
from transfer_queue import KVBatchMeta
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils import hf_tokenizer
from verl.utils import tensordict_utils as tu
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

from psrl.utils.logger import (
    DualOutputHandler,
    EventType,
    log_dual_events,
)
from psrl.utils.server.command import Command, CommandExtension, CommandType
from psrl.workers.ps.request_status_tracker import PSRL_RequestStatus
from psrl.workers.reward.reward_loop import load_reward_manager
from psrl.workers.reward.reward_model import RewardModelManager

psrl_logger = logging.getLogger(__name__)
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
        reward_loop_workers: list[ray.actor.ActorHandle] | None = None,
    ):
        """Initialize the reward manager for processing rollout data and computing rewards.

        The reward manager receives rollout data from rollout workers, computes rewards
        using either rule-based functions or reward models, and sends the results
        to the transfer queue.

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

        if self.config.psrl.rollout_coordination.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.rollout_coordination.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.rollout_coordination.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n
        assert self.rollout_n >= self.alg_rollout_n, (
            f"Rollout n {self.rollout_n} must be greater than or equal to alg_rollout_n {self.alg_rollout_n}."
        )
        self.val_rollout_n = self.config.train_actor_rollout_ref.rollout.val_kwargs.n

        # Reward model configuration
        self.request_id_to_n_trajectory = {}
        self.request_key_to_future: dict[str, asyncio.Future] = {}
        self.request_key_to_reward = {}
        self.request_id_to_data_source = {}
        self.request_id_to_request_keys: dict[Any, list[str]] = {}
        self.request_key_to_worker: dict[str, int] = {}
        self.request_key_to_attempt: dict[str, int] = {}

        # Reward normalization
        self.reward_normalization = self.config.reward.reward_normalization
        self.request_id_to_group = {}

        # Background event handler
        self.running_loop = None
        self.command_loop_task = None
        self.collect_task = None
        self.dispatch_task: asyncio.Task | None = None
        self.stop_command_loop_task = False
        self.stop_collect_task = False
        self.stop_dispatch_task = False

        worker_cfg = self.config.reward.reward_loop_worker
        self.result_queue = asyncio.Queue(maxsize=worker_cfg.result_queue_size)
        self.result_drain_batch_size = worker_cfg.result_drain_batch_size
        self.reward_loop_workers = reward_loop_workers or []
        self.use_reward_loop_workers = worker_cfg.enable and len(self.reward_loop_workers) > 0
        self.dispatch_idx = 0
        self.attempt_counter = 0
        self.dispatch_queue: asyncio.Queue = asyncio.Queue()

        # Communication handles
        self.ps_manager_handle = ps_manager_handle

        self.reward_model_configs = reward_model_configs

        # Reward loop managers
        self.reward_spec_to_manager: dict[RewardSpec, RewardManagerBase] = {}

        if not self.use_reward_loop_workers:
            self._init_reward_fn()

        # Build logger
        self.log_prefix = "RewardLoopManager"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        psrl_logger.info("Initialized RewardLoopManager.")

    def start_busy_loop(self):
        """Start the reward manager and begin processing requests.

        This method initializes the server state and starts the background event handler
        task for processing rollout data and computing rewards. The server will run
        until explicitly stopped.
        """
        if self.command_loop_task is not None and not self.command_loop_task.done():
            return

        for worker in self.reward_loop_workers:
            worker.start_busy_loop.remote()

        # Start the background task to process data
        self.stop_command_loop_task = False
        self.stop_collect_task = False
        self.running_loop = asyncio.get_running_loop()
        self.command_loop_task = self.running_loop.create_task(self._command_event_handler())
        self.command_loop_task.add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks
        if self.use_reward_loop_workers:
            self.collect_task = self.running_loop.create_task(self._collect_results())
            self.collect_task.add_done_callback(lambda f: f.result())
            self.stop_dispatch_task = False
            self.dispatch_task = self.running_loop.create_task(self._dispatch_loop())
            self.dispatch_task.add_done_callback(lambda f: f.result())

    async def stop_busy_loop(self):
        """Shutdown the reward manager gracefully.

        This method stops the command loop task and waits for it
        to complete before returning.
        """
        if (
            (not self.command_loop_task or self.command_loop_task.done())
            and (not self.collect_task or self.collect_task.done())
            and (not self.dispatch_task or self.dispatch_task.done())
        ):
            return

        self.stop_command_loop_task = True
        self.stop_collect_task = True
        self.stop_dispatch_task = True
        # Wait for the background tasks to finish
        tasks = [task for task in (self.command_loop_task, self.collect_task, self.dispatch_task) if task is not None]
        await asyncio.gather(*tasks)
        await asyncio.gather(*[worker.stop_busy_loop.remote() for worker in self.reward_loop_workers])

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
                    # 1. Cancel running reward computation futures/tasks.
                    aborted_count = 0
                    worker_to_abort_keys: dict[int, list[str]] = {}
                    for abort_request_id in abort_request_uids:
                        n_trajectory = self.request_id_to_n_trajectory.pop(abort_request_id, 1)
                        if n_trajectory > 1:
                            reward_keys = [f"{abort_request_id}_{i}" for i in range(n_trajectory)]
                        else:
                            reward_keys = [str(abort_request_id)]

                        for key in reward_keys:
                            self.request_key_to_attempt.pop(key, None)
                            worker_index = self.request_key_to_worker.pop(key, None)
                            if worker_index is not None:
                                worker_to_abort_keys.setdefault(worker_index, []).append(key)

                            fut = self.request_key_to_future.pop(key, None)
                            if fut is not None and not fut.done():
                                fut.set_result(None)
                                aborted_count += 1
                            self.request_key_to_reward[key] = None

                    if worker_to_abort_keys:
                        await asyncio.gather(
                            *[
                                self.reward_loop_workers[worker_index].abort_requests.remote(keys)
                                for worker_index, keys in worker_to_abort_keys.items()
                            ],
                            return_exceptions=True,
                        )

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

    async def normalize_reward(self, reward_batch: KVBatchMeta) -> KVBatchMeta:
        """Normalize the reward for the given reward_batch.

        Args:
            reward_batch (KVBatchMeta): Batch of reward data.
        Returns:
            KVBatchMeta: Normalized reward batch.
        """
        fields = ["uid", "reward_score", "reward_extra_info"]
        data = await tq.async_kv_batch_get(
            keys=reward_batch.keys, partition_id=reward_batch.partition_id, select_fields=fields
        )

        uids = tu.get(data, "uid")
        reward_extra_infos = tu.get(data, "reward_extra_info")
        uid_to_data_source = {uid: self.request_id_to_data_source.pop(uid) for uid in set(uids)}
        data_sources = [uid_to_data_source[uid] for uid in uids]
        reward_scores = tu.get(data, "reward_score")

        for reward_extra_info, data_source, reward_score in zip(reward_extra_infos, data_sources, reward_scores):
            reward_extra_info["data_source"] = data_source
            reward_extra_info["original_reward_score"] = reward_score

        if self.reward_normalization != "batch" and self.reward_normalization != "group":
            data["reward_extra_info"] = reward_extra_infos
            await tq.async_kv_batch_put(
                keys=reward_batch.keys,
                partition_id=reward_batch.partition_id,
                fields=data.select("reward_extra_info"),
            )
            return reward_batch

        # final trajectory of each uid: uid => (trajectory_index, row_index)
        final_trajectories: dict[str, tuple[int, int]] = {}
        row_session_keys = []
        for i, key in enumerate(reward_batch.keys):
            fields = key.rsplit("_", 1)
            if len(fields) == 2:
                uid_key, index = fields[0], int(fields[1])
                if uid_key not in final_trajectories or final_trajectories[uid_key][0] < index:
                    final_trajectories[uid_key] = (index, i)
            else:
                uid_key, index = key, 0
                final_trajectories[uid_key] = (index, i)
            row_session_keys.append(uid_key)

        # final trajectory indices in batch data
        final_indices = []
        uid_key_to_local_index = {}
        for uid, (_, row_index) in final_trajectories.items():
            final_indices.append(row_index)
            uid_key_to_local_index[uid] = len(final_indices) - 1
        row_to_local_index = [uid_key_to_local_index[uid_key] for uid_key in row_session_keys]

        # Group final-trajectory rewards by group_id.
        group_rewards_dicts = {}
        for local_index, row_index in enumerate(final_indices):
            uid = uids[row_index]
            if uid not in self.request_id_to_group:
                continue
            group_id = self.request_id_to_group[uid]
            if group_id not in group_rewards_dicts:
                group_rewards_dicts[group_id] = {
                    "local_indices": [],
                    "rewards": [],
                }
            group_rewards_dicts[group_id]["local_indices"].append(local_index)
            group_rewards_dicts[group_id]["rewards"].append(reward_scores[row_index])
            self.request_id_to_group.pop(uid, None)

        # Normalize rewards for each group
        final_reward_scores = [0.0] * len(final_indices)
        for group_id, group_rewards_dict in group_rewards_dicts.items():
            local_indices = group_rewards_dict["local_indices"]
            rewards = np.array(group_rewards_dict["rewards"])
            psrl_logger.info(f"Rewards for group {group_id}: {len(rewards)}")
            norm_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            for local_index, norm_reward in zip(local_indices, norm_rewards):
                final_reward_scores[local_index] = float(norm_reward)

        # Scatter normalized final rewards to all trajectories with the same uid.
        normalized_reward_scores = [final_reward_scores[local_index] for local_index in row_to_local_index]
        reward_scores = np.asarray(normalized_reward_scores, dtype=reward_scores.dtype)

        # Put updated reward scores and extra info back to the batch
        data["reward_extra_info"] = reward_extra_infos
        data["reward_score"] = reward_scores

        # Update rm_scores
        response_mask = await tq.async_kv_batch_get(
            keys=reward_batch.keys,
            partition_id=reward_batch.partition_id,
            select_fields=["response_mask"],
        )["response_mask"]
        rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
        rm_scores[:, -1] = torch.as_tensor(reward_scores, dtype=rm_scores.dtype, device=rm_scores.device)
        data["rm_scores"] = rm_scores

        await tq.async_kv_batch_put(
            keys=reward_batch.keys,
            partition_id=reward_batch.partition_id,
            fields=data.select("reward_extra_info", "reward_score", "rm_scores"),
        )
        return reward_batch

    @staticmethod
    def _default_reward_result() -> dict[str, Any]:
        return {
            "reward_score": 0.0,
            "reward_extra_info": {},
            "reward_metrics": {},
            "response_len": 1,
        }

    def _next_attempt_id(self) -> int:
        self.attempt_counter += 1
        return self.attempt_counter

    async def _dispatch_loop(self):
        """Background task that drains dispatch_queue and sends requests to workers."""
        while not self.stop_dispatch_task:
            # Drain all available requests from the queue in one batch
            batch: list[TensorDict] = []
            while not self.dispatch_queue.empty():
                try:
                    batch.append(self.dispatch_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            if not batch:
                await asyncio.sleep(0)
                continue

            await self._dispatch_reward_requests(batch)
        psrl_logger.info("Dispatch loop has finished.")

    async def _dispatch_reward_requests(self, requests: list[TensorDict]) -> None:
        requests_by_worker: dict[int, list[TensorDict]] = {}
        worker_num = len(self.reward_loop_workers)
        for request in requests:
            worker_index = self.dispatch_idx % worker_num
            self.dispatch_idx += 1
            requests_by_worker.setdefault(worker_index, []).append(request)
            request_keys = tu.get(request, "request_keys")[0]
            for key in request_keys:
                self.request_key_to_worker[key] = worker_index

        tasks = []
        for worker_index, worker_requests in requests_by_worker.items():
            tasks.append(self.reward_loop_workers[worker_index].add_reward_requests.remote(worker_requests))
        await asyncio.gather(*tasks)

    async def put_reward_result(self, result: dict):
        await self.result_queue.put(result)

    async def _collect_results(self):
        while not self.stop_collect_task:
            results: list[dict] = []
            while not self.result_queue.empty():
                results.append(self.result_queue.get_nowait())

            if not results:
                await asyncio.sleep(0)
                continue

            await self._handle_reward_results(results)

        psrl_logger.info("Reward result collector has finished.")

    async def _handle_reward_results(self, results: list[dict]):
        valid_results = []
        for result in results:
            request_keys = result.get("request_keys")
            if not request_keys:
                continue
            expected_attempt = self.request_key_to_attempt.get(request_keys[0])
            if expected_attempt != result["attempt_id"]:
                psrl_logger.debug(
                    "Drop stale reward result uid=%s attempt=%s expected=%s",
                    result["uid"],
                    result["attempt_id"],
                    expected_attempt,
                )
                continue
            valid_results.append(result)

        if not valid_results:
            return

        for is_validate in (False, True):
            group = [r for r in valid_results if r["is_validate"] == is_validate]
            if not group:
                continue
            uids = [r["uid"] for r in group]
            await self.ps_manager_handle.update_request_status.remote(
                uids,
                PSRL_RequestStatus.REWARD_COMPLETED,
                is_validate=is_validate,
            )
            for result in group:
                reward = result["result"] if result.get("error") is None else None
                if result.get("error") is not None:
                    psrl_logger.error(
                        "Reward worker %s failed uid=%s keys=%s error=%s",
                        result["worker_id"],
                        result["uid"],
                        result["request_keys"],
                        result["error"],
                    )
                if reward is not None:
                    reward["response_len"] = result["response_len"]
                await self._set_reward_for_request_keys(result["request_keys"], reward)

    async def _set_reward_for_request_keys(self, request_keys: list[str], reward: dict[str, Any] | None):
        for request_key in request_keys:
            per_key_reward = reward.copy() if reward is not None else None
            self.request_key_to_attempt.pop(request_key, None)
            self.request_key_to_worker.pop(request_key, None)
            if request_key in self.request_key_to_future:
                fut = self.request_key_to_future[request_key]
                if not fut.done():
                    fut.set_result(per_key_reward)
            else:
                self.request_key_to_reward[request_key] = per_key_reward

    async def compute_score(
        self,
        reward_inputs: TensorDict,
    ) -> dict[str, Any] | None:
        """Compute reward or dispatch reward computation to reward loop workers."""
        if not self.use_reward_loop_workers:
            return await self.compute_score_local(reward_inputs)

        is_validate = tu.get(reward_inputs, "validate", False)
        request_ids = tu.get(reward_inputs, "uid")
        n_trajectories = tu.get(reward_inputs, "n_trajectory")

        rollout_n = self.val_rollout_n if is_validate else self.rollout_n
        request_keys = []
        dispatch_items: list[TensorDict] = []

        with log_dual_events(
            f"Compute reward for requests {request_ids}",
            psrl_logger,
            level=logging.DEBUG,
            event_type=EventType.OTHER,
        ):
            update_status_success = await self.ps_manager_handle.update_request_status.remote(
                request_ids,
                PSRL_RequestStatus.REWARD_RUNNING,
                is_validate=is_validate,
            )
            if not isinstance(update_status_success, list):
                update_status_success = [update_status_success]

            for i, (request_id, n_trajectory, status_ok) in enumerate(
                zip(request_ids, n_trajectories, update_status_success)
            ):
                reward_input = reward_inputs[i : i + 1]
                prompt_key = "parent_id" if rollout_n > 1 else "uid"
                prompt_id = tu.get(reward_input, prompt_key)[0]
                data_source = tu.get(reward_input, "data_source")[0]
                if n_trajectory > 1:
                    current_request_keys = [f"{request_id}_{j}" for j in range(n_trajectory)]
                else:
                    current_request_keys = [str(request_id)]
                request_keys.extend(current_request_keys)

                self.request_id_to_n_trajectory[request_id] = n_trajectory
                self.request_id_to_request_keys[request_id] = current_request_keys
                if self.reward_normalization == "batch":
                    self.request_id_to_group[request_id] = data_source
                elif self.reward_normalization == "group":
                    self.request_id_to_group[request_id] = prompt_id
                self.request_id_to_data_source[request_id] = data_source

                attempt_id = self._next_attempt_id()
                for request_key in current_request_keys:
                    if status_ok:
                        existing_fut = self.request_key_to_future.get(request_key)
                        if existing_fut is None or existing_fut.done():
                            fut = asyncio.get_running_loop().create_future()
                            self.request_key_to_future[request_key] = fut
                        self.request_key_to_attempt[request_key] = attempt_id
                    else:
                        psrl_logger.warning(
                            "compute_score:update_request_status(REWARD_RUNNING) failed for uid=%s "
                            "(aborted?), returning None.",
                            request_id,
                        )
                        existing_fut = self.request_key_to_future.pop(request_key, None)
                        if existing_fut is not None and not existing_fut.done():
                            existing_fut.set_result(None)
                        self.request_key_to_reward[request_key] = None

                if status_ok:
                    tu.assign_non_tensor_stack(reward_input, "request_keys", [current_request_keys])
                    tu.assign_non_tensor_stack(reward_input, "attempt_id", [attempt_id])
                    dispatch_items.append(reward_input)

            if dispatch_items:
                if self.config.reward.launch_reward_fn_async:
                    # Async mode: enqueue for background dispatch (non-blocking).
                    for item in dispatch_items:
                        self.dispatch_queue.put_nowait(item)
                else:
                    # Sync mode: dispatch directly and wait for results.
                    await self._dispatch_reward_requests(dispatch_items)

            if self.config.reward.launch_reward_fn_async:
                return None

            # NOTE(linsh): currently sync reward only supports single request.
            sync_keys = tu.get(dispatch_items[0], "request_keys")[0] if dispatch_items else request_keys
            sync_results = []
            for request_key in sync_keys:
                if request_key in self.request_key_to_reward:
                    sync_results.append(self.request_key_to_reward.pop(request_key))
                else:
                    sync_results.append(await self.request_key_to_future[request_key])
                    self.request_key_to_future.pop(request_key, None)

        return sync_results[0] if sync_results else None

    async def wait_for_reward_of_requests(self, requests: KVBatchMeta) -> KVBatchMeta:
        """Wait for the reward results of the specified requests.

        This method blocks until the reward results for all specified request IDs
        are available, either from previously computed rewards or from ongoing
        reward computation tasks.
        """
        request_keys = requests.keys
        is_validate = requests.partition_id == "val"
        partition_id = "val" if is_validate else "train"
        # Padding from `_balance_batch` in `ray_trainer.py`
        padding_keys: set[str] = {
            key for key, tag in zip(requests.keys, requests.tags) if tag.get("is_padding", False)
        }
        # Reward may be ready in group post processing
        ready_keys = {
            key for key, tag in zip(requests.keys, requests.tags) if not is_validate and tag.get("reward_ready", False)
        }
        request_keys_to_put = [key for key in request_keys if key not in ready_keys]
        request_key_to_reward: dict[str, dict] = {}
        futures_to_wait = {}

        # Gather available rewards and futures for the requested IDs.
        for request_key in request_keys_to_put:
            if request_key in padding_keys:
                request_key_to_reward[request_key] = {
                    "reward_score": 0.0,
                    "reward_extra_info": {},
                    "reward_metrics": {},
                    "response_len": 1,
                }
            elif request_key in self.request_key_to_reward:
                request_key_to_reward[request_key] = self.request_key_to_reward.pop(request_key)
            elif request_key in self.request_key_to_future:
                futures_to_wait[request_key] = self.request_key_to_future[request_key]
            else:
                fut = asyncio.get_running_loop().create_future()
                self.request_key_to_future[request_key] = fut
                futures_to_wait[request_key] = fut

        if futures_to_wait:
            results = await asyncio.gather(*futures_to_wait.values())
            for request_key, reward in zip(futures_to_wait.keys(), results):
                request_key_to_reward[request_key] = reward

        fields = []
        for request_key in request_keys_to_put:
            reward = request_key_to_reward[request_key]
            if reward is None:
                reward = self._default_reward_result()
            fields.append(reward)
            self.request_key_to_future.pop(request_key, None)

        for i, f in enumerate(fields):
            raw_response_len = f.pop("response_len", 1)
            if raw_response_len <= 0:
                psrl_logger.error(
                    "wait_for_reward_of_requests: response_len=%d for key=%s, "
                    "reward_score=%s, reward_keys=%s",
                    raw_response_len,
                    request_keys_to_put[i] if i < len(request_keys_to_put) else "?",
                    f.get("reward_score"),
                    list(f.keys()),
                )
            response_len = max(raw_response_len, 1)
            rm_score_tensor = torch.zeros(response_len, dtype=torch.float32)
            rm_score_tensor[-1] = float(f.get("reward_score", 0.0))
            f["rm_scores"] = rm_score_tensor

        if request_keys_to_put:
            await tq.async_kv_batch_put(
                keys=request_keys_to_put,
                partition_id=partition_id,
                fields=list_of_dict_to_tensordict(fields),
            )

        if not is_validate and self.reward_normalization:
            requests = await self.normalize_reward(requests)

        return requests

    async def wait_for_reward_ready(self, request_keys: list[str]) -> None:
        """Wait until reward results are available for the given keys and write them to TQ.

        This is used by AgentLoopManager's ``_group_post_process`` which needs the
        ``reward_score`` field available in TQ for filtering. The AgentLoopManager
        marks successfully processed keys in the resulting batch metadata so the
        trainer can skip the redundant TQ put.
        """
        futures_to_wait: dict[str, asyncio.Future] = {}
        resolved: dict[str, dict] = {}
        for request_key in request_keys:
            if request_key in self.request_key_to_reward:
                resolved[request_key] = self.request_key_to_reward[request_key]
            elif request_key in self.request_key_to_future:
                fut = self.request_key_to_future[request_key]
                if fut.done():
                    resolved[request_key] = fut.result()
                else:
                    futures_to_wait[request_key] = fut
            else:
                # No future and no result — create a future so reward dispatch can resolve it.
                fut = asyncio.get_running_loop().create_future()
                self.request_key_to_future[request_key] = fut
                futures_to_wait[request_key] = fut

        if futures_to_wait:
            results = await asyncio.gather(*futures_to_wait.values())
            for request_key, result in zip(futures_to_wait.keys(), results):
                resolved[request_key] = result

        fields = []
        for request_key in request_keys:
            reward = resolved.get(request_key)
            if reward is None:
                reward = self._default_reward_result()
            f = reward.copy()
            response_len = max(f.pop("response_len", 1), 1)
            rm_score_tensor = torch.zeros(response_len, dtype=torch.float32)
            rm_score_tensor[-1] = float(f.get("reward_score", 0.0))
            f["rm_scores"] = rm_score_tensor
            fields.append(f)

        if request_keys:
            await tq.async_kv_batch_put(
                keys=request_keys,
                partition_id="train",
                fields=list_of_dict_to_tensordict(fields),
            )
            for request_key in request_keys:
                self.request_key_to_reward.pop(request_key, None)
                self.request_key_to_future.pop(request_key, None)

    # TODO(linsh): we may remove methods below and put computation to reward workers

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
            reward_model_name = reward_model_config.get("reward_model_name", "null")
            reward_kwargs = reward_model_config.get("reward_kwargs", {})

            for reward_fn_config in reward_fn_configs:
                reward_fn_name = reward_fn_config.get("name", "null")
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
                    **reward_kwargs,
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
            # list-of-config-dicts (from system config or OmegaConf ListConfig).
            # Normalise to a plain string reward function name.
            raw_reward_fn = reward_model_dict.get("reward_fn", "default")
            if isinstance(raw_reward_fn, list) and len(raw_reward_fn) > 0:
                reward_fn_name = (
                    raw_reward_fn[0].get("name", "default")
                    if isinstance(raw_reward_fn[0], dict)
                    else str(raw_reward_fn[0])
                )
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
                    (spec, mgr)
                    for spec, mgr in self.reward_spec_to_manager.items()
                    if spec.reward_manager_type == reward_spec.reward_manager_type
                ]
                if len(candidates) == 0:
                    # 2nd fallback: if only one manager registered globally, use it
                    all_managers = list(self.reward_spec_to_manager.items())
                    if len(all_managers) == 1:
                        candidates = all_managers
                    elif len(all_managers) > 1:
                        psrl_logger.info(
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

    def _compute_multi_modal_inputs(self, data: TensorDict, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute multi-modal inputs with image and video.

        Mirrors ``PSRL_AgentLoopManager._compute_multi_modal_inputs`` but omits
        ``images_seqlens`` computation (not needed by reward functions).
        """
        multi_modal_inputs = {}
        if self.processor is None:
            return multi_modal_inputs

        images = tu.get(data, "multi_modal_data", {})[0].get("images", None)
        videos = tu.get(data, "multi_modal_data", {})[0].get("videos", None)
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
        """Add attention_mask and position_ids to TensorDict's tensor batch."""
        prompts = tu.get(inputs, "prompts").squeeze(0)  # [1, prompt_len] -> [prompt_len]
        responses = tu.get(inputs, "responses").squeeze(0)  # [1, response_len] -> [response_len]
        # prompts and responses are 1D unpadded tensors stored per-sample in TQ.
        input_ids = torch.cat([prompts, responses], dim=0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.int64)  # No padding
        # _compute_multi_modal_inputs and _compute_position_ids expect [1, seq_len].
        multi_modal_inputs = self._compute_multi_modal_inputs(inputs, input_ids.unsqueeze(0))
        position_ids = self._compute_position_ids(
            input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
        )
        inputs["attention_mask"] = attention_mask.unsqueeze(0)
        inputs["position_ids"] = position_ids
        return inputs

    async def compute_score_local(
        self,
        reward_inputs: TensorDict,
    ) -> dict[str, Any] | None:
        """
        Compute the reward score for the given inputs.
        Compatibility path used when reward loop workers are disabled.
        """
        is_validate = tu.get(reward_inputs, "validate", False)
        request_ids = tu.get(reward_inputs, "uid")
        n_trajectories = tu.get(reward_inputs, "n_trajectory")
        rollout_n = self.val_rollout_n if is_validate else self.rollout_n
        request_keys = []
        for request_id, n_trajectory in zip(request_ids, n_trajectories):
            self.request_id_to_n_trajectory[request_id] = n_trajectory
            if n_trajectory > 1:
                request_keys.extend([f"{request_id}_{i}" for i in range(n_trajectory)])
            else:
                request_keys.append(str(request_id))

        # Update the request status to REWARD_RUNNING
        update_status_success = await self.ps_manager_handle.update_request_status.remote(
            request_ids,
            PSRL_RequestStatus.REWARD_RUNNING,
            is_validate=is_validate,
        )
        if not update_status_success:
            # Mark reward as None for requests that fail to update status
            for request_key in request_keys:
                self.request_key_to_reward[request_key] = None
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
                prompt_id_key = "parent_id" if rollout_n > 1 else "uid"
                prompt_id = tu.get(reward_input, prompt_id_key)[0]
                data_source = tu.get(reward_input, "data_source")[0]

                # Reward normalization group assignment
                if self.reward_normalization == "batch":
                    self.request_id_to_group[request_id] = data_source
                elif self.reward_normalization == "group":
                    self.request_id_to_group[request_id] = prompt_id
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
                        reward_input = self.pre_process(reward_input)
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

    async def _compute_score(
        self,
        reward_data: TensorDict,
    ) -> dict[str, Any]:
        """Run all reward loops for a single sample in parallel and aggregate the results.

        Resolves the reward loops from the input's ``reward_model_dicts``, executes them
        concurrently, and returns a unified result dict with weighted reward score,
        per-loop extra info, and per-loop metrics.

        Args:
            reward_data: A single-sample TensorDict that contains all fields needed
                by reward loops (tensors in batch, metadata in non_tensor_batch).

        Returns:
            dict with keys:
                - ``reward_score``: weighted sum of all loop scores (float)
                - ``reward_extra_info``: ``{loop_key: extra_info_dict, ...}``
                - ``reward_metrics``: ``{loop_key: metrics_dict, ...}``
        """
        assert len(reward_data) == 1, "Reward input for _compute_score should contain exactly one sample"

        reward_model_dicts = tu.get(reward_data, "reward_model_dicts")[0]
        reward_spec_keys, reward_managers, reward_coefs = self._resolve_reward_manager(reward_model_dicts)

        futures = [asyncio.create_task(reward_manager.run_single(reward_data)) for reward_manager in reward_managers]
        results = await asyncio.gather(*futures)

        reward_score = float(np.sum([r["reward_score"] * c for r, c in zip(results, reward_coefs)]))
        reward_extra_info_dict = {key: r["reward_extra_info"] for key, r in zip(reward_spec_keys, results)}
        reward_metrics_dict = {key: r.get("reward_metrics", {}) for key, r in zip(reward_spec_keys, results)}

        result = {
            "reward_score": reward_score,
            "reward_extra_info": reward_extra_info_dict,
            "reward_metrics": reward_metrics_dict,
        }

        return result

    async def _async_reward_task(self, reward_data: TensorDict):
        """Fire-and-forget async task: compute reward and store the result for later retrieval.

        Derives all reward loop configuration from ``reward_data`` itself via
        ``_compute_score``, so callers only need to pass the data.
        Works for both training and validation paths.
        """
        assert len(reward_data) == 1, "Async reward task should be launched with a single-sample batch"

        reward_data = self.pre_process(reward_data)
        uid = tu.get(reward_data, "uid")[0]

        result = await self._compute_score(reward_data)

        n_trajectory = tu.get(reward_data, "n_trajectory", [1])[0]
        result["response_len"] = reward_data["responses"].shape[-1]

        # Broadcast to all trajectories
        trajectory_to_results = {}
        if n_trajectory > 1:
            for i in range(n_trajectory):
                trajectory_to_results[f"{uid}_{i}"] = result
        else:
            trajectory_to_results[str(uid)] = result

        await self.set_reward_for_requests(trajectory_to_results)

    async def set_reward_for_requests(self, request_key_to_reward: dict[str, dict[str, Any]]):
        """Set the reward for the specified request IDs."""
        for request_key, reward in request_key_to_reward.items():
            per_key_reward = reward.copy() if reward is not None else None
            if request_key in self.request_key_to_future:
                fut = self.request_key_to_future[request_key]
                if not fut.done():
                    fut.set_result(per_key_reward)
            else:
                self.request_key_to_reward[request_key] = per_key_reward
