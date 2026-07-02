import logging
import os

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


def apply_engine_core_patches() -> None:
    """
    Add PSRL forwarding methods to `EngineCore`.

    The vLLM v1 UTILITY handler dispatches `call_utility_async` calls via
    ``getattr(self, method_name)`` on the `EngineCore` instance.  The GPU block
    pool helpers (`psrl_get_gpu_cache_info`, `psrl_pin_gpu`, `psrl_unpin_gpu`)
    live on `RolloutScheduler`, which is assigned to `EngineCore.scheduler`.
    Since `EngineCore` does not inherit from `RolloutScheduler`, these methods
    are not directly reachable via `getattr(engine_core, ...)`.

    This patch adds thin forwarding methods to `EngineCore` that delegate each
    call to `self.scheduler`.  The patch is idempotent and guarded by a sentinel
    attribute so it is safe to call multiple times.
    """
    _patch_engine_core_psrl_methods()


def _patch_engine_core_psrl_methods() -> None:
    """
    Attach `psrl_get_gpu_cache_info`, `psrl_pin_gpu`, and `psrl_unpin_gpu`
    forwarding methods to `EngineCore`.
    """
    from vllm.v1.engine.core import EngineCore

    _SENTINEL = "_psrl_gpu_methods_patched"

    if getattr(EngineCore, _SENTINEL, False):
        psrl_logger.debug("EngineCore PSRL GPU forwarding methods already patched, skipping.")
        return

    def psrl_get_gpu_cache_info(self, tokens: list) -> dict:
        return self.scheduler.psrl_get_gpu_cache_info(tokens)

    def psrl_pin_gpu(self, tokens: list) -> int:
        return self.scheduler.psrl_pin_gpu(tokens)

    def psrl_unpin_gpu(self, tokens: list) -> int:
        return self.scheduler.psrl_unpin_gpu(tokens)

    EngineCore.psrl_get_gpu_cache_info = psrl_get_gpu_cache_info
    EngineCore.psrl_pin_gpu = psrl_pin_gpu
    EngineCore.psrl_unpin_gpu = psrl_unpin_gpu
    setattr(EngineCore, _SENTINEL, True)

    psrl_logger.info("Patched EngineCore: added psrl_get_gpu_cache_info, psrl_pin_gpu, psrl_unpin_gpu.")
