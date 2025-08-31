from .base_train_worker import TrainInterface

# NOTE(ls): Backend-specific worker will be lazily imported

__all__ = [
    "TrainInterface",
]