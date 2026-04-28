# Modified from verl/experimental/reward/reward_loop/base.py
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from omegaconf import DictConfig
from tensordict import TensorDict
from transformers import AutoTokenizer
from verl.utils.ray_utils import get_event_loop

RawRewardFn = Callable[..., Any] | None


class RewardManagerBase(ABC):
    _class_initialized = False

    def __init__(self, config: DictConfig, tokenizer: AutoTokenizer, compute_score: RawRewardFn):
        """Initialize agent loop.

        Args:
            config (DictConfig): YAML config.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            compute_score (RawRewardFn): Function to compute rewards.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.compute_score = compute_score
        self.loop = get_event_loop()
        self.init_class(config, tokenizer)

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer):
        """Initialize class state shared across all instances."""
        if cls._class_initialized:
            return
        cls._class_initialized = True

    @abstractmethod
    async def run_single(self, data: TensorDict):
        raise NotImplementedError
