import asyncio
from abc import ABC, abstractmethod

import numpy as np
import ray
from omegaconf import DictConfig
from transformers import AutoTokenizer
from verl import DataProto

from psrl.workers.agent_loop.loops.utils import DummyConfig
from psrl.workers.agent_loop.router import RolloutRouter
from psrl.workers.reward.reward_manager import RewardManager


class AgentLoopBase(ABC):
    _class_initialized = False

    def __init__(
        self,
        trainer_config: DummyConfig,
        rollout_router: RolloutRouter,
        reward_manager: RewardManager,
        ps_manager_handle: ray.actor.ActorHandle,
        tokenizer: AutoTokenizer,
        **kwargs,
    ):
        """Initialize agent loop instance.
        Base class for agent loops that process requests and interact with LLM servers.

        Args:
            trainer_config (DummyConfig): Wrapper containing trainer configuration.
            rollout_router (RolloutRouter): Router for distributing requests to LLM servers.
            ps_manager_handle (ray.actor.ActorHandle): Handle to parameter server manager.
            tokenizer (AutoTokenizer): Tokenizer for processing text messages.
            **kwargs: Additional keyword arguments.
        """
        self.init_class(trainer_config.config, tokenizer, **kwargs)
        self.config = trainer_config.config
        self.rollout_router = rollout_router
        self.reward_manager = reward_manager
        self.ps_manager_handle = ps_manager_handle
        self.tokenizer = tokenizer
        self.loop = asyncio.get_running_loop()

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer, **kwargs):
        """Perform heavy initialization work shared across all instances.

        This method is called only once per class to avoid redundant initialization.

        Args:
            config (DictConfig): Configuration object containing training settings.
            tokenizer (AutoTokenizer): Tokenizer for processing text messages.
            **kwargs: Additional keyword arguments from configuration.
        """
        if cls._class_initialized:
            return
        cls._class_initialized = True

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
    async def run(self, request: DataProto) -> DataProto:
        """Execute the agent loop for the given request.

        Args:
            request (DataProto): Input request to process.

        Returns:
            DataProto: Processed response data.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
