import ray
from dataclasses import dataclass

@dataclass
class TrainInterface:
    """Info for the PSRL TrainWorker."""
    ps_handle: ray.actor.ActorHandle