# Modified from verl/experimental/reward/reward_loop/base.py
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch

from omegaconf import DictConfig
from tensordict import TensorDict
from transformers import AutoTokenizer
from verl.utils.ray_utils import get_event_loop
from verl.utils import tensordict_utils as tu

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

    @classmethod
    def assemble_rm_scores(cls, data: TensorDict, scores: list[float]) -> torch.Tensor:
        """Assemble per-sample reward scores into the ``rm_scores`` tensor for a batch.

        Args:
            data: The concatenated batch passed to :meth:`run_single`.
                ``data.batch["prompts"]``, ``data.batch["responses"]`` and
                ``data.batch["attention_mask"]`` are expected to be present
                for the default implementation.
            scores: List of scalar reward scores, one per sample in ``data``.

        Returns:
            torch.Tensor: The ``rm_scores`` tensor with leading dimension equal to
            ``len(data)``.
        """
        prompt_length = tu.get(data, "prompts").size(1)
        valid_response_length = tu.get(data, "attention_mask")[:, prompt_length:].sum(dim=1)
        rm_scores = torch.zeros_like(tu.get(data, "responses"), dtype=torch.float32)
        rm_scores[torch.arange(rm_scores.size(0), device=rm_scores.device), valid_response_length - 1] = (
            rm_scores.new_tensor(scores)
        )
        return rm_scores