from enum import Enum

from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.single_controller.ray import RayResourcePool

class PSRL_Role(Enum):
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6

class PSRL_ResourcePoolManager(ResourcePoolManager):
    """
    Support multiple instances of the same role
    """
    mapping: dict[PSRL_Role, list[str]]
    
    def get_resource_pool(self, role: PSRL_Role, instance_id: int = 0) -> RayResourcePool:
        """Get the resource pool of the worker_cls for the given instance_id."""
        return self.resource_pool_dict[self.mapping[role][instance_id]]
