from typing import Any


def lazy_import_to_globals(module_path: str, name: str, *, alias: str | None = None) -> Any:
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
    """
    g = globals()
    key = alias or name
    obj = g.get(key, None)
    if obj is not None:
        return obj

    import importlib

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
    for name in names:
        lazy_import_to_globals(module_path, name)
