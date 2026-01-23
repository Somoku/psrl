import importlib
import inspect
from typing import Any


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
