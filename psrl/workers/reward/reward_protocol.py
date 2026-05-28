from dataclasses import dataclass
from typing import Any


@dataclass
class RewardModelRuntimeInfo:
    """Runtime view required by GenRewardManager."""

    gateway_url: str
    reward_model_tokenizer: Any

    def get_gateway_url(self) -> str:
        return self.gateway_url

    def get_reward_model_tokenizer(self):
        return self.reward_model_tokenizer
