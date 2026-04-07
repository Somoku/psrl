import enum
from dataclasses import dataclass


class KVCacheBackend(enum.Enum):
    """
    Backend for KV cache offloading.
    """

    CPU = "cpu"
    DISK = "disk"
    REMOTE = "remote"


@dataclass
class KVCacheStatus:
    """
    Status of KV cache offloading for the engine.
    """

    # Whether LMCache offloading is enabled.
    enabled: bool
    # Current backend in use.
    backend: KVCacheBackend | None = None
    # Total offload buffer size in GiB.
    offload_size_gb: float = 0.0
