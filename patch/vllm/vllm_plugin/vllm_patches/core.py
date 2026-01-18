import logging
from types import MethodType, ModuleType

import vllm
from packaging import version

logger = logging.getLogger(__name__)

PatchTarget = type | ModuleType


class vLLMPatch:
    """
    Base class for creating clean, surgical patches to vLLM classes.

    Usage:
        class MyPatch(vLLMPatch[TargetClass]):
            def new_method(self):
                return "patched behavior"

        MyPatch.apply()
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "_patch_target"):
            raise TypeError(f"{cls.__name__} must be defined as vLLMPatch[Target]")

    @classmethod
    def __class_getitem__(cls, target: PatchTarget) -> type:
        if not isinstance(target, (type, ModuleType)):
            raise TypeError(f"Can only patch classes or modules, not {type(target)}")

        return type(f"{cls.__name__}[{target.__name__}]", (cls,), {"_patch_target": target})

    @classmethod
    def apply(cls):
        """Apply this patch to the target class/module."""
        if cls is vLLMPatch:
            raise TypeError("Cannot apply base vLLMPatch class directly")

        target = cls._patch_target

        # Track which patches have been applied
        if not hasattr(target, "_applied_patches"):
            target._applied_patches = {}

        special_methods = ["_maybe_get_memory_pool_context"]

        for name, attr in cls.__dict__.items():
            if name.startswith("_") and name not in special_methods or name in ("apply",):
                continue

            if name in target._applied_patches:
                existing = target._applied_patches[name]
                raise ValueError(f"{target.__name__}.{name} already patched by {existing}")

            target._applied_patches[name] = cls.__name__

            # Handle classmethods
            if isinstance(attr, MethodType):
                attr = MethodType(attr.__func__, target)

            setattr(target, name, attr)
            action = "replaced" if hasattr(target, name) else "added"
            logger.info(f"✓ {cls.__name__} {action} {target.__name__}.{name}")


def min_vllm_version(version_str: str):
    """
    Decorator to specify minimum vLLM version required for a patch.

    Usage:
        @min_vllm_version("0.9.1")
        class MyPatch(vLLMPatch[SomeClass]):
            pass
    """

    def decorator(cls):
        original_apply = cls.apply

        @classmethod
        def checked_apply(cls):
            current = version.parse(vllm.__version__)
            minimum = version.parse(version_str)

            if current < minimum:
                logger.warning(
                    f"Skipping {cls.__name__}: requires vLLM >= {version_str}, but found {vllm.__version__}"
                )
                return

            original_apply()

        cls.apply = checked_apply
        cls._min_version = version_str
        return cls

    return decorator
