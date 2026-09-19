import logging
import os

logger = logging.getLogger(__name__)


class PatchManager:
    """Manages registration and application of vLLM patches."""

    def __init__(self):
        self.available_patches: dict[str, type] = {}
        self.applied_patches: list[str] = []

    def register(self, name: str, patch_class: type):
        """Register a patch for later application."""
        self.available_patches[name] = patch_class
        logger.info(f"Registered patch: {name}")

    def apply_patch(self, name: str) -> bool:
        """Apply a single patch by name."""
        if name not in self.available_patches:
            logger.error(f"Unknown patch: {name}")
            return False

        try:
            self.available_patches[name].apply()
            self.applied_patches.append(name)
            return True
        except Exception as e:
            logger.error(f"Failed to apply {name}: {e}")
            return False


# Global manager instance
manager = PatchManager()


def register_patches():
    """
    Main entry point called by vLLM's plugin system.
    This function is invoked automatically when vLLM starts.
    """
    logger.info("=" * 60)
    logger.info("Initializing vLLM Custom Patches Plugin")
    logger.info("=" * 60)

    _install_weight_layouts()
    _install_sleep_mode_backend()
    _apply_tms_patches_from_env()
    _apply_always_on_patches()

    logger.info("=" * 60)


def _install_weight_layouts() -> None:
    """Register the per-model vLLM -> HF weight layouts used by NIXL sync."""
    from vllm_patches.weight_layouts import (
        install_weight_layout_registry_hook,
        register_weight_layouts,
    )

    install_weight_layout_registry_hook()
    register_weight_layouts()


def _install_sleep_mode_backend() -> None:
    """Register the torch_memory_saver sleep backend.

    Registration stores a module path and class name and imports them lazily, so
    ``torch_memory_saver`` is only required once a worker actually selects the
    ``tms`` backend through ``ModelConfig.sleep_mode_backend``.
    """
    from vllm.device_allocator.sleep_mode_backend import SleepModeBackendFactory

    try:
        SleepModeBackendFactory.register_backend(
            "tms",
            "vllm_patches.backends.tms",
            "TMSBackend",
        )
    except ValueError:
        # vLLM's own `load_general_plugins` is idempotent per process, so this
        # should not happen; stay tolerant in case a process imports the plugin
        # through more than one path.
        logger.debug("Sleep-mode backend 'tms' is already registered.")
        return
    logger.info("Registered sleep-mode backend: tms")


def _apply_tms_patches_from_env() -> None:
    """Apply the torch_memory_saver patches requested by ``PSRL_VLLM_PATCHES``.

    Supported values:
      - ``TMS``: memory-pool routing only.
      - ``TMS:GRAPH``: memory-pool routing plus CUDA-graph capture through the
        ``graph`` region.

    The TMS patch modules import ``torch_memory_saver`` at import time, so they
    are imported here rather than at plugin load: a process that never selects
    TMS must not require the package.
    """
    env_patches = os.environ.get("PSRL_VLLM_PATCHES", "").strip()

    if env_patches not in ("TMS", "TMS:GRAPH"):
        if env_patches:
            logger.warning(
                "Unknown PSRL_VLLM_PATCHES value %r; no TMS patches applied. "
                "Supported values are 'TMS' and 'TMS:GRAPH'.",
                env_patches,
            )
        else:
            logger.info("No custom patches specified (PSRL_VLLM_PATCHES not set)")
        return

    try:
        import torch_memory_saver  # noqa: F401
    except ImportError:
        logger.error(
            "PSRL_VLLM_PATCHES is set to %r, but 'torch_memory_saver' is not "
            "installed. Install the pinned torch_memory_saver build before "
            "enabling TMS.",
            env_patches,
        )
        raise

    from vllm_patches.patches.cuda_graph import TMSCUDAGraphWrapperPatch
    from vllm_patches.patches.cudagraph_utils import TMSCudaGraphManagerPatch
    from vllm_patches.patches.gpu_worker import TMSMemoryPoolPatch

    manager.register("TMSMemoryPoolPatch", TMSMemoryPoolPatch)
    manager.register("TMSCUDAGraphWrapperPatch", TMSCUDAGraphWrapperPatch)
    manager.register("TMSCudaGraphManagerPatch", TMSCudaGraphManagerPatch)

    patch_names = ["TMSMemoryPoolPatch"]
    if env_patches == "TMS:GRAPH":
        patch_names += ["TMSCUDAGraphWrapperPatch", "TMSCudaGraphManagerPatch"]

    logger.info("Applying TMS patches: %s", patch_names)
    for name in patch_names:
        manager.apply_patch(name)

    if env_patches == "TMS:GRAPH":
        # Breakable CUDA graphs bypass `torch.cuda.graph()` entirely, so they
        # need their own hook rather than a `vLLMPatch` subclass.
        from vllm_patches.patches.breakable_cudagraph import (
            apply_breakable_cudagraph_patch,
        )

        apply_breakable_cudagraph_patch()

    logger.info(f"Successfully applied: {manager.applied_patches}")


def _apply_always_on_patches() -> None:
    """Apply patches that are idempotent and cheap when their feature is off."""
    from vllm_patches.patches.attn_hijack import apply_attn_hijack_patch
    from vllm_patches.patches.block_table import apply_block_table_clamp_patch
    from vllm_patches.patches.engine_core import apply_engine_core_patches
    from vllm_patches.patches.prompt_token_clamp import (
        apply_prompt_token_clamp_patch,
    )

    # Reachable from `call_utility_async("psrl_pin_gpu" / "psrl_unpin_gpu")`.
    apply_engine_core_patches()
    # Correctness guard for the v2 runner's block table; a no-op on v1.
    apply_block_table_clamp_patch()
    # Profiling/analysis hooks, gated at runtime by env / metric type.
    apply_attn_hijack_patch()
    apply_prompt_token_clamp_patch()
