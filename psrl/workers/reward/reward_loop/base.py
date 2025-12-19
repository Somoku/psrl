# Modified from verl/experimental/reward/reward_loop/base.py
import asyncio
from abc import ABC, abstractmethod

from omegaconf import DictConfig
from transformers import AutoTokenizer
from verl import DataProto


class RewardLoopManagerBase(ABC):
    _class_initialized = False

    def __init__(self, config: DictConfig, tokenizer: AutoTokenizer, is_validate: bool = False):
        """Initialize agent loop.

        Args:
            config (DictConfig): YAML config.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            is_validate (bool): Whether in validation mode.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.is_validate = is_validate
        self.loop = asyncio.get_running_loop()
        self.init_class(config, tokenizer)

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer):
        """Initialize class state shared across all instances."""
        if cls._class_initialized:
            return
        cls._class_initialized = True

    @abstractmethod
    async def run_single(self, data: DataProto):
        raise NotImplementedError
