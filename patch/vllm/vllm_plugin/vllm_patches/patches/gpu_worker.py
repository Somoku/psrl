import logging
import os
from contextlib import AbstractContextManager, nullcontext

from vllm.v1.worker.gpu_worker import Worker

from vllm_patches.core import min_vllm_version, vLLMPatch

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@min_vllm_version("0.29.0")
class TMSMemoryPoolPatch(vLLMPatch[Worker]):
    """Route vLLM's sleep-mode allocations through torch_memory_saver.

    Only the allocation-scope hook is replaced. Suspend/resume is owned by
    ``vllm_patches.backends.tms.TMSBackend`` (selected with
    ``ModelConfig.sleep_mode_backend = "tms"``), and the level-2 buffer handling
    stays in vLLM's own ``Worker.sleep``/``Worker.wake_up``.

    vLLM's ``Worker.initialize_from_config`` and ``Worker.load_model`` already
    allocate the KV cache and the weights through this context manager, so no
    further Worker methods need overriding. Overriding
    ``initialize_from_config`` here (as an earlier revision did) bypasses
    vLLM's ``record_kv_cache_layout`` bookkeeping.
    """

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        """Get the memory pool context manager if sleep mode is enabled."""
        if not self.vllm_config.model_config.enable_sleep_mode:
            return nullcontext()

        from torch_memory_saver import torch_memory_saver

        enable_weights_cpu_backup = self.vllm_config.additional_config.get("enable_weights_cpu_backup", False)
        return torch_memory_saver.region(
            tag=tag,
            enable_cpu_backup=enable_weights_cpu_backup,
        )
