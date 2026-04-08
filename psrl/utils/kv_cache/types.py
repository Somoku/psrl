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


@dataclass
class TrajectoryCacheInfo:
    """
    Snapshot of KV cache usage for a single trajectory's token sequence.

    All fields reflect the *cached prefix* only — the longest contiguous run
    of chunks/blocks that actually exist in the target store.  Operations like
    `pin` and `clear` act on the same prefix.
    """

    total_tokens: int
    """Length of the token sequence passed in."""

    # --- LMCache backend side ---
    lmcache_cached_chunks: int
    """Number of LMCache chunks present in the backend."""

    lmcache_cached_tokens: int
    """Tokens covered by those chunks."""

    lmcache_bytes: int
    """Bytes those chunks occupy in the backend allocator."""

    lmcache_total_bytes: int
    """Total backend allocator capacity in bytes."""

    lmcache_usage_pct: float
    """`lmcache_bytes / lmcache_total_bytes`."""

    # --- vLLM GPU prefix cache side ---
    gpu_cached_blocks: int
    """KVCacheBlocks with matching token hash present in the GPU prefix cache."""

    gpu_cached_tokens: int
    """Tokens covered by those blocks."""

    gpu_total_blocks: int
    """Total GPU KV block pool size."""

    gpu_usage_pct: float
    """`gpu_cached_blocks / gpu_total_blocks`."""

    # --- PSRL-managed pin state ---
    gpu_pinned: bool
    """Whether PSRL has pinned the GPU blocks for this trajectory."""

    backend_pinned: bool
    """Whether PSRL has pinned the backend chunks for this trajectory."""
