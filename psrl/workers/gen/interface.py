import ray
from dataclasses import dataclass

@dataclass
class GenInterface:
    """Info for the PSRL GenWorker."""
    rollout_instance_id: int
    ps_manager_handle: ray.actor.ActorHandle
