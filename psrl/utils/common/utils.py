import base64
import importlib
import inspect
import pickle
from typing import Any


def b64_dumps(obj: Any) -> str:
    """Serialize a Python object to a base64 string.

    Note: this is intended for trusted, in-cluster traffic only. Pickle is not
    safe for untrusted inputs.
    """
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return base64.b64encode(payload).decode("ascii")


def b64_loads(payload_b64: str) -> Any:
    """Deserialize a base64 string back to a Python object."""
    payload = base64.b64decode(payload_b64.encode("ascii"))
    return pickle.loads(payload)


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


def lazy_import_to_globals(
    module_path: str,
    name: str,
    alias: str | None = None,
    target_globals: dict[str, Any] | None = None,
) -> Any:
    """
    Lazily import an object from a module and store it in the global namespace.
    Equivalent to
        - from module_path import name as alias
        - from module_path import name

    Args:
        module_path (str): The dot-separated path of the module to import from.
        name (str): The name of the object to import from the module.
        alias (str | None): An optional alias to store the object under in globals().
            If None, the object is stored under its original name.
        target_globals (dict[str, Any] | None): The globals dictionary to store the imported object in.
            If None, uses the caller's global namespace.
    """
    if target_globals is None:
        frame = inspect.currentframe()
        # currentframe() -> this function; f_back -> caller
        caller_frame = frame.f_back if frame is not None else None
        target_globals = caller_frame.f_globals if caller_frame is not None else globals()

    g = target_globals
    key = alias or name
    obj = g.get(key, None)
    if obj is not None:
        return obj

    mod = importlib.import_module(module_path)
    obj = getattr(mod, name)
    g[key] = obj
    return obj


def lazy_import_many_to_globals(module_path: str, names: list[str]) -> None:
    """
    Lazily import multiple objects from a module and store them in the global namespace.
    Equivalent to
        - from module_path import name1, name2, ..., nameN

    Args:
        module_path (str): The dot-separated path of the module to import from.
        names (list[str]): A list of names of the objects to import from the module.
    """
    frame = inspect.currentframe()
    caller_frame = frame.f_back if frame is not None else None
    target_globals = caller_frame.f_globals if caller_frame is not None else globals()

    for name in names:
        lazy_import_to_globals(module_path, name, target_globals=target_globals)
