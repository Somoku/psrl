<<<<<<< HEAD:psrl/utils/common/dynamic_import.py
import importlib
import inspect
=======
import base64
import pickle
>>>>>>> dev/smg:psrl/utils/common/utils.py
from typing import Any


def import_class_from_string(class_string: str) -> type:
    """
    Import a class from a string in the format 'module.path:ClassName'.

    Args:
        class_string: String in format 'module.path:ClassName'

    Returns:
        The imported class

    Raises:
        ImportError: If the module or class cannot be imported
        ValueError: If the string format is invalid
    """
    import importlib

    if ":" not in class_string:
        raise ValueError(f"Class string must be in format 'module.path:ClassName', got: {class_string}")

    module_path, class_name = class_string.split(":", 1)

    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_path}': {e}") from e
    except AttributeError as e:
        raise ImportError(f"Could not find class '{class_name}' in module '{module_path}': {e}") from e
