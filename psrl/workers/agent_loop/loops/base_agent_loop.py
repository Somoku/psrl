import asyncio
from abc import ABC, abstractmethod
from omegaconf import DictConfig
from transformers import AutoTokenizer

import ray

from verl import DataProto

from psrl.workers.agent_loop.utils import DummyConfig, AgentLoopOutput
from psrl.workers.agent_loop.router import RolloutRouter

class AgentLoopBase(ABC):
    """An agent loop takes a input message, chat with OpenAI compatible LLM server and interact with various
    environments."""

    _class_initialized = False

    def __init__(
        self,
        trainer_config: DummyConfig,
        rollout_router: RolloutRouter,
        ps_manager_handle: ray.actor.ActorHandle,
        tokenizer: AutoTokenizer,
        **kwargs,
    ):
        """Initialize agent loop, each sample will have its own loop instance.

        Args:
            trainer_config (_DummyConfig): trainer config.
            rollout_router (RolloutRouter): Rollout router to route requests to different LLM servers.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
        """
        self.init_class(trainer_config.config, tokenizer, **kwargs)
        self.config = trainer_config.config
        self.rollout_router = rollout_router
        self.ps_manager_handle = ps_manager_handle
        self.tokenizer = tokenizer
        self.loop = asyncio.get_running_loop()

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer, **kwargs):
        """This is used to do heavy initialization work that should shared across all instances. It's only called once.

        Args:
            config (DictConfig): trainer config.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            **kwargs: extra kwargs from config file passed in by `hydra.utils.instantiate`.
        """
        if cls._class_initialized:
            return
        cls._class_initialized = True

    @abstractmethod
    async def run(self, request: DataProto) -> DataProto:
        raise NotImplementedError
