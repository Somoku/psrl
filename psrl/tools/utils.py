import inspect
import typing
from typing import Any, cast


# Copied from rllm/rllm/tools/utils.py
def function_to_dict(func):
    """
    Converts a function into a dictionary representation suitable for JSON Schema.

    Parameters:
        func (function): The function to convert.

    Returns:
        dict: A dictionary representing the function in JSON Schema format.
    """
    # Get the function name
    func_name = func.__name__

    # Get the docstring
    docstring = func.__doc__ or ""

    # Get the function signature
    sig = inspect.signature(func)

    # Initialize the parameters dictionary
    params = {"type": "object", "properties": {}, "required": []}
    required_list = cast(list[str], params["required"])  # Type hint for mypy
    properties_dict = cast(dict[str, Any], params["properties"])  # Type hint for mypy

    # Map Python types to JSON Schema types
    type_mapping = {
        int: "integer",
        float: "number",
        bool: "boolean",
        str: "string",
        dict: "object",
        list: "array",
    }

    for param_name, param in sig.parameters.items():
        # Get the type annotation
        annotation = param.annotation

        param_type = "string"  # Default type
        param_description = ""

        # Determine the JSON Schema type and description
        if annotation != inspect.Parameter.empty:
            # Handle Annotated types
            origin = typing.get_origin(annotation)
            if origin is typing.Annotated:
                args = typing.get_args(annotation)
                base_type = args[0]
                metadata = args[1:]
                param_type = type_mapping.get(base_type, "string")
                # Assuming the first metadata argument is the description
                if metadata:
                    param_description = metadata[0]
            else:
                param_type = type_mapping.get(annotation, "string")
        # Add the parameter to properties
        param_schema = {"type": param_type}
        if param_description:
            param_schema["description"] = param_description
        properties_dict[param_name] = param_schema

        # Add to required if there's no default value
        if param.default == inspect.Parameter.empty:
            required_list.append(param_name)

    # Build the final dictionary
    function_dict = {
        "type": "function",
        "function": {
            "name": func_name,
            "description": docstring.strip().split("\n")[0],  # First line of docstring
            "parameters": params,
        },
    }

    return function_dict
