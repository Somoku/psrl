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

    def apply_from_env(self):
        """
        Apply patches specified in PSRL_VLLM_PATCHES environment variable.

        Format: PSRL_VLLM_PATCHES="PatchOne,PatchTwo"
        Supported values:
          - Comma-separated patch names (e.g., "PatchA,PatchB")
          - "TMS" to apply TMSWorkerPatch
          - "TMS:GRAPH" to apply TMSWorkerPatch, TMSCUDAGraphWrapperPatch, and TMSCudaGraphManagerPatch
        """
        env_patches = os.environ.get("PSRL_VLLM_PATCHES", "").strip()

        if not env_patches:
            logger.info("No custom patches specified (PSRL_VLLM_PATCHES not set)")
            return

        if env_patches in ("TMS", "TMS:GRAPH"):
            # assert `torch_memory_saver` is installed
            try:
                import torch_memory_saver  # noqa: F401
            except ImportError:
                logger.error(
                    "PSRL_VLLM_PATCHES is set to apply TMS patches, "
                    "but 'torch_memory_saver' is not installed. "
                    "Please install it via 'pip install torch_memory_saver==0.0.9'."
                )
                raise
            if env_patches == "TMS:GRAPH":
                patch_names = [
                    "TMSWorkerPatch",
                    "TMSCUDAGraphWrapperPatch",
                    "TMSCudaGraphManagerPatch",
                    "TMSExecutorPatch",
                ]
            else:
                patch_names = ["TMSWorkerPatch"]
            logger.info("Applying TMS patches.")

        for name in patch_names:
            self.apply_patch(name)

        logger.info(f"Successfully applied: {self.applied_patches}")


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

    # Import and register all available patches
    from vllm_patches.patches.cuda_graph import TMSCUDAGraphWrapperPatch
    from vllm_patches.patches.cudagraph_utils import TMSCudaGraphManagerPatch
    from vllm_patches.patches.executor import TMSExecutorPatch
    from vllm_patches.patches.gpu_worker import TMSWorkerPatch

    manager.register("TMSWorkerPatch", TMSWorkerPatch)
    manager.register("TMSCudaGraphManagerPatch", TMSCudaGraphManagerPatch)
    manager.register("TMSCUDAGraphWrapperPatch", TMSCUDAGraphWrapperPatch)
    manager.register("TMSExecutorPatch", TMSExecutorPatch)

    # Apply patches based on environment configuration
    manager.apply_from_env()

    logger.info("=" * 60)
