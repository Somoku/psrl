import asyncio
from abc import ABC, abstractmethod

import numpy as np
import ray
from omegaconf import DictConfig
from transformers import AutoTokenizer
from verl import DataProto

from psrl.utils.profiling.collector import TurnProfilingCollector
from psrl.utils.rollout.trajectory_writer import TrajectoryWriter
from psrl.workers.agent_loop.loops.utils import DummyConfig, TerminateReason


class AgentLoopBase(ABC):
    _class_initialized = False

    def __init__(
        self,
        trainer_config: DummyConfig,
        rollout_router: ray.actor.ActorHandle,
        reward_manager: ray.actor.ActorHandle,
        ps_manager_handle: ray.actor.ActorHandle,
        tokenizer: AutoTokenizer,
        **kwargs,
    ):
        """Initialize agent loop instance.
        Base class for agent loops that process requests and interact with LLM servers.

        Args:
            trainer_config (DummyConfig): Wrapper containing trainer configuration.
            rollout_router (ray.actor.ActorHandle): Router for distributing requests to LLM servers.
            ps_manager_handle (ray.actor.ActorHandle): Handle to parameter server manager.
            tokenizer (AutoTokenizer): Tokenizer for processing text messages.
            **kwargs: Additional keyword arguments.
        """
        self.init_class(trainer_config.config, **kwargs)
        self.config = trainer_config.config
        self.rollout_router = rollout_router
        self.reward_manager = reward_manager
        self.ps_manager_handle = ps_manager_handle
        self.tokenizer = tokenizer
        self.loop = asyncio.get_running_loop()
        self.traj_writer = TrajectoryWriter.from_config(self.config)

    @classmethod
    def init_class(cls, config: DictConfig, **kwargs):
        """Perform heavy initialization work shared across all instances.

        This method is called only once per class to avoid redundant initialization.

        Args:
            config (DictConfig): Configuration object containing training settings.
            **kwargs: Additional keyword arguments from configuration.
        """
        if cls._class_initialized:
            return
        cls._class_initialized = True

        cls.prompt_length = config.gen_actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.gen_actor_rollout_ref.rollout.response_length

    def _post_process_and_merge_reward(self, reward_result: dict[int, dict], outputs: DataProto) -> DataProto:
        """Merge the computed reward results back into the output DataProto.

        This method updates the output data with the reward scores and any additional
        information returned by the reward model.

        Args:
            reward_result (Dict[int, dict]): Computed reward results indexed by data item.
            outputs (DataProto): Original output data to be updated.

        Returns:
            DataProto: Updated output data with reward information.
        """
        if outputs.meta_info.get("validate", False):
            return outputs  # Skip merging for validation data

        filtered_request_ids = list(reward_result.keys())
        filtered_request_idxs = [
            idx for idx, uid in enumerate(outputs.non_tensor_batch["uid"].tolist()) if uid in filtered_request_ids
        ]
        outputs = outputs.select_idxs(filtered_request_idxs)
        request_ids = outputs.non_tensor_batch["uid"].tolist()

        rewards = []
        reward_extra_infos = []
        for request_id in request_ids:
            assert request_id in reward_result, f"Missing reward result for request ID: {request_id}"
            result = reward_result[request_id]
            rewards.append(result["reward_score"])
            extra_info = result.get("reward_extra_info", {})
            reward_extra_infos.append(extra_info)
        outputs.non_tensor_batch["reward_scores"] = np.array(rewards)
        outputs.non_tensor_batch["reward_extra_infos"] = np.array(reward_extra_infos, dtype=object)
        return outputs

    @abstractmethod
    async def run(
        self,
        request: DataProto,
        profiling_collector: TurnProfilingCollector | None = None,
    ) -> tuple[DataProto | None, TerminateReason]:
        """Execute the agent loop for the given request.

        Args:
            request (DataProto): Input request to process.
            profiling_collector: Per-trajectory profiling collector, or None if disabled.

        Returns:
            Tuple of (output DataProto or None, TerminateReason).

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def run_with_termination_handling(
        self,
        request: DataProto,
        raise_on_error: bool = True,
        profiling_collector: TurnProfilingCollector | None = None,
    ) -> tuple[DataProto | None, TerminateReason]:
        """Run the agent loop with termination event handling.

        This method wraps the run method to catch termination events and handle them appropriately.
        It enables timeouts and error handling based on the provided configuration.

        Args:
            request (DataProto): Input request to process.
            raise_on_error (bool): Whether to raise exceptions on errors.
            profiling_collector: Per-trajectory profiling collector, or None if disabled.

        Returns:
            DataProto: Processed response data.
        """
        uid = int(request.non_tensor_batch["uid"][0])
        try:
            coro = self.run(request, profiling_collector)
            output, terminate_reason = await asyncio.wait_for(
                coro, timeout=self.config.gen_actor_rollout_ref.rollout.agent.trajectory_timeout
            )
            if output is not None and isinstance(output, DataProto):
                return output, terminate_reason
            elif not raise_on_error:
                return None, TerminateReason.UNKNOWN
            else:
                raise RuntimeError("Agent loop run did not return a valid DataProto output.")
        except asyncio.TimeoutError:
            return None, TerminateReason.TIMEOUT
        except Exception as e:
            if not raise_on_error:
                import logging

                psrl_logger = logging.getLogger(__file__)
                psrl_logger.error(
                    f"Exception in agent_loop.run for request {request.non_tensor_batch.get('uid', 'N/A')}",
                    exc_info=True,
                )
                return None, TerminateReason.ERROR
            raise e
        finally:
            await self.rollout_router.kv_unregister.remote(uid)
