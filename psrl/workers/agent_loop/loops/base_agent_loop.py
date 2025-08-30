import asyncio
from abc import ABC, abstractmethod
from omegaconf import DictConfig
from transformers import AutoTokenizer

import ray

from verl import DataProto

from psrl.workers.agent_loop.loops.utils import DummyConfig
from psrl.workers.agent_loop.router import RolloutRouter

class AgentLoopBase(ABC):

    _class_initialized = False

    def __init__(
        self,
        trainer_config: DummyConfig,
        rollout_router: RolloutRouter,
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
