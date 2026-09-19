"""torch_memory_saver (TMS) sleep-mode backend.

Replaces vLLM's default ``CuMemAllocator`` sleep mechanism with
``torch_memory_saver``, which keeps virtual addresses stable across
pause/resume. Allocations are tagged so that weights, the KV cache and the CUDA
graph pool can be released and restored as separate regions.

Registration happens in ``vllm_patches.register_patches`` via
``SleepModeBackendFactory.register_backend``; selection happens through
``ModelConfig.sleep_mode_backend == "tms"``, which PSRL sets on the engine args.

The ``graph`` region only exists when ``PSRL_VLLM_PATCHES`` is ``TMS:GRAPH``,
because that is what makes the CUDA-graph capture patches route the capture pool
into a tagged region: ``TMSCUDAGraphWrapperPatch`` (v1 runner),
``TMSCudaGraphManagerPatch`` (v2 runner FULL graphs) and
``vllm_patches.patches.breakable_cudagraph`` (breakable graphs on either runner).
When it is absent we must not pause or resume it.
"""

from __future__ import annotations

import logging
import os

from vllm.device_allocator.sleep_mode_backend import SleepModeBackend

logger = logging.getLogger(__name__)

WEIGHTS_TAG = "weights"
KV_CACHE_TAG = "kv_cache"
GRAPH_TAG = "graph"


def _graph_region_enabled() -> bool:
    """Whether the graph capture patches registered a ``graph`` region."""
    return os.environ.get("PSRL_VLLM_PATCHES", "").strip() == "TMS:GRAPH"


class TMSBackend(SleepModeBackend):
    """Free and restore GPU state through ``torch_memory_saver``."""

    def __init__(self) -> None:
        super().__init__()
        # Track each region independently: a wake may restore weights and the
        # KV cache in either order across separate ``wake_up`` calls.
        self._weights_suspended = False
        self._kv_cache_suspended = False
        self._graph_suspended = False

    # -- Capability introspection --

    @classmethod
    def is_supported(cls) -> bool:
        try:
            import torch_memory_saver  # noqa: F401
        except ImportError:
            logger.warning("torch_memory_saver is not installed; the 'tms' sleep-mode backend is unavailable.")
            return False
        return True

    @classmethod
    def preserves_communicators(cls) -> bool:
        # TMS only releases the allocations it manages; NCCL communicator
        # buffers live outside them, so no communicator re-init is needed.
        return True

    @classmethod
    def preserves_compiled_artifacts(cls) -> bool:
        # Regions are remapped at their original virtual addresses, so compiled
        # kernels and captured CUDA graphs stay valid.
        return True

    @classmethod
    def preserves_graphs_with_communicators(cls) -> bool:
        return True

    # -- Lifecycle --

    def suspend(self, level: int = 1) -> None:
        """Release weights, the KV cache, and (when enabled) the graph pool.

        ``level`` is accepted for interface compatibility. torch_memory_saver
        always offloads the weights region to host memory when CPU backup is
        enabled for it (``additional_config.enable_weights_cpu_backup``); the
        KV cache is discarded on every level, matching vLLM's sleep semantics.
        """
        from torch_memory_saver import torch_memory_saver

        self._state = "SUSPENDED"
        torch_memory_saver.pause(WEIGHTS_TAG)
        torch_memory_saver.pause(KV_CACHE_TAG)
        self._weights_suspended = True
        self._kv_cache_suspended = True
        if _graph_region_enabled():
            torch_memory_saver.pause(GRAPH_TAG)
            self._graph_suspended = True

    def resume(self, tags: list[str] | None = None) -> None:
        """Restore the requested regions.

        The graph pool is resumed implicitly once weights and the KV cache are
        both mapped again, so callers never need to pass ``"graph"``. vLLM's
        ``Executor.wake_up`` only accepts tags it recorded in ``sleeping_tags``
        (``{"weights", "kv_cache"}``), so keeping the tag out of the public wake
        path avoids patching the executor.
        """
        from torch_memory_saver import torch_memory_saver

        self._state = "RESUMING"
        wake_weights = tags is None or WEIGHTS_TAG in tags
        wake_kv_cache = tags is None or KV_CACHE_TAG in tags

        if wake_weights:
            torch_memory_saver.resume(WEIGHTS_TAG)
            self._weights_suspended = False
        if wake_kv_cache:
            torch_memory_saver.resume(KV_CACHE_TAG)
            self._kv_cache_suspended = False

        if self._graph_suspended and not self._weights_suspended and not self._kv_cache_suspended:
            torch_memory_saver.resume(GRAPH_TAG)
            self._graph_suspended = False

        if not (self._weights_suspended or self._kv_cache_suspended):
            self._state = "RUNNING"
