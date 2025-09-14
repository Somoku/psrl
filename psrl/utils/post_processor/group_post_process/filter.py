import torch
import numpy as np
from typing import Optional
from omegaconf import DictConfig

from verl import DataProto
from ..base import BaseGroupPostProcessor, GroupPostProcessorRegistry

@GroupPostProcessorRegistry.register("null")
class NoFilterProcessor(BaseGroupPostProcessor):
    """
    Group post-processor that performs no filtering.
    
    This processor simply returns all data unchanged.
    """
    
    def __call__(self, data: DataProto) -> Optional[DataProto]:
        """
        Return data unchanged.
        
        Args:
            data (DataProto): The grouped data to be processed.
            
        Returns:
            DataProto: The unchanged data.
        """
        return data

@GroupPostProcessorRegistry.register("dynamic_sampling_filter")
class DynamicSamplingFilterProcessor(BaseGroupPostProcessor):
    """
    Group post-processor that filters data based on reward variance.
    
    This processor filters out data groups where the reward variance is zero,
    which indicates that all samples in the group have the same reward value.
    """
    
    def __call__(self, data: DataProto) -> Optional[DataProto]:
        """
        Filter data based on reward variance.
        
        Args:
            data (DataProto): The grouped data to be processed.
            
        Returns:
            Optional[DataProto]: The data if variance > 0, None otherwise.
        """
        metric_name = self.config.algorithm.filter_groups.metric
        if metric_name == "seq_final_reward":
            # Turn to numpy for easier filtering
            data.non_tensor_batch["seq_final_reward"] = (
                data.batch["token_level_rewards"].sum(dim=-1).numpy()
            )
        elif metric_name == "seq_reward":
            data.non_tensor_batch["seq_reward"] = (
                data.batch["token_level_scores"].sum(dim=-1).numpy()
            )

        rewards = data.non_tensor_batch[metric_name]
        if torch.tensor(rewards, dtype=torch.float).std() > 0.0:
            return data
        else:
            return None
