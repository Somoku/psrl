from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
from collections import defaultdict

# Modified from rllm/rllm/tools/tool_base.py
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from pydantic import BaseModel, model_validator

from psrl.tools.utils import function_to_dict


class ToolResponse(BaseModel):
    """Structured response from a tool execution.

    Mirrors verl's ``ToolResponse``: text/image/video are cleanly separated
    so that callers (e.g. ToolEnvironment) can build multimodal messages
    without re-interpreting raw dicts.  image and video must always be lists
    when non-None — a ``@model_validator`` enforces this contract.
    """

    id: str | None = None
    text: str | None = None
    image: list[Any] | None = None
    video: list[Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _validate_list_fields(cls, values):
        for field in ("image", "video"):
            val = values.get(field)
            if val is not None and not isinstance(val, list):
                raise ValueError(
                    f"ToolResponse.{field} must be a list, got {type(val)}. "
                    f"Wrap single items: [{field}=obj] → [{field}=[obj]]."
                )
        return values

    def is_empty(self) -> bool:
        return not self.text and not self.image and not self.video

    def is_text_only(self) -> bool:
        return bool(self.text) and not self.image and not self.video

@dataclass
class ToolCall:
    """
    Represents a call to a tool with its name and arguments, extracted
    from a language model's output.
    """

    name: str
    arguments: dict[str, Any]

    def to_dict(self):
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class ToolOutput:
    name: str
    output: dict = field(default_factory=lambda: defaultdict())
    error: str | None = None
    metadata: dict | None = None

    def __str__(self) -> str:
        """Convert the tool output to a string representation.

        Returns:
            str: JSON string for dict/list outputs, error message if failed,
                 or string representation of the output value
        """
        if self.error:
            return f"Error: {self.error}"
        elif self.output is None:
            return ""
        elif isinstance(self.output, list | dict):
            import json

            return json.dumps(self.output)
        else:
            return str(self.output)

    def to_string(self) -> str:
        """
        Convert the tool output to a string representation.

        Example usage:
            tool_output = ToolOutput(name="calculator", output=42)
            result_string = tool_output.to_string()
        """
        return str(self)


class Tool:
    """
    Abstract base class for all tools that provides a common interface.

    All tools should inherit from this class and implement either:
    - forward() for synchronous tools
    - async_forward() for asynchronous tools
    """

    _registry: dict[str, type[Tool]] = {}

    def __init__(self, name: str | None = None, description: str | None = None, function: Callable | None = None):
        """
        Initialize the tool with a name.

        Args:
            name (str): The name of the tool, used for tool registry and calling.
            description (str): A description of the tool's purpose and functionality.
            function (Callable, optional): Function to convert to tool format.
        """
        self.name = name
        self.description = description
        self.function = function

        if function is not None:
            self._json = function_to_dict(function)
            self.name = self._json["function"]["name"]
            self.description = self._json["function"]["description"]
        else:
            assert name is not None, "Name is required for Tool class."
            assert description is not None, "Description is required for Tool class."
            self._json = None

    @property
    def json(self) -> dict[str, Any]:
        """
        Return the tool's information in a standardized format for tool registration.

        Returns:
            Dict[str, Any]: Tool information including name, description, and parameters.
                Expected format:
                {
                    "type": "function",
                    "function": {
                        "name": self.name,
                        "description": "Description of what the tool does",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                # Parameter definitions
                            },
                            "required": ["list", "of", "required", "parameters"]
                        }
                    }
                }
        """
        assert self._json is not None, (
            "Tool must be initialized with a function to get json representation if no custom json is provided."
        )
        return self._json

    def forward(self, *args, **kwargs) -> ToolOutput:
        """
        Synchronous implementation of the tool functionality.

        This method executes the tool synchronously. Override this method for
        synchronous tools, or provide a function during initialization.

        Args:
            *args: Positional arguments for the tool
            **kwargs: Keyword arguments for the tool

        Returns:
            ToolOutput: The result of the tool execution

        Raises:
            NotImplementedError: If neither a function is provided nor this
                                method is overridden
        """
        if self.function is not None:
            try:
                output, reward = self.function(*args, **kwargs)
                return ToolOutput(name=self.name or "unknown", output=output), reward
            except Exception as e:
                return ToolOutput(name=self.name or "unknown", error=f"{type(e).__name__} - {str(e)}"), None
        else:
            raise NotImplementedError("Tool must implement forward() method or provide a function")

    async def async_forward(self, *args, **kwargs) -> ToolOutput:
        """
        Asynchronous implementation of the tool functionality.

        This method executes the tool asynchronously. Override this method for
        async tools. By default, it delegates to the synchronous forward() method.

        Args:
            *args: Positional arguments for the tool
            **kwargs: Keyword arguments for the tool

        Returns:
            ToolOutput: The result of the tool execution
        """
        return self.forward(*args, **kwargs)

    def __call__(self, *args, use_async=None, **kwargs):
        """
        Make the tool instance callable. Automatically detects whether to use sync or async implementation.

        This method intelligently routes calls to either forward() or async_forward() based on:
        1. Explicit use_async parameter if provided
        2. Auto-detection of which methods are implemented
        3. The execution context (whether called from async or sync code)

        Args:
            *args: Positional arguments to pass to the implementation
            use_async: Optional explicit flag to force async (True) or sync (False) execution.
                      If None (default), auto-detects based on implementation availability.
            **kwargs: Keyword arguments to pass to the implementation

        Returns:
            ToolOutput or Coroutine[ToolOutput]:
                - If async_forward is called, returns a coroutine that must be awaited
                - If forward is called, returns ToolOutput directly

        Raises:
            NotImplementedError: If the requested execution mode is not implemented

        Example:
            # Async context
            result = await tool(completion="...", test_cases={...})

            # Sync context (if forward is implemented)
            result = tool(completion="...", test_cases=..., use_async=False)
        """
        # Check which implementations are available
        has_custom_async = self.__class__.async_forward is not Tool.async_forward
        has_custom_sync = self.__class__.forward is not Tool.forward

        # Explicit routing based on use_async flag
        if use_async is True:
            if has_custom_async:
                return self.async_forward(*args, **kwargs)
            else:
                raise NotImplementedError(
                    f"Tool {self.__class__.__name__} does not implement async_forward() "
                    f"but use_async=True was specified."
                )
        elif use_async is False:
            if has_custom_sync:
                return self.forward(*args, **kwargs)
            else:
                raise NotImplementedError(
                    f"Tool {self.__class__.__name__} does not implement forward() but use_async=False was specified."
                )

        # Auto-detect: prefer async if available, fall back to sync
        if has_custom_async:
            return self.async_forward(*args, **kwargs)
        elif has_custom_sync:
            return self.forward(*args, **kwargs)
        else:
            raise NotImplementedError(
                f"Tool {self.__class__.__name__} must implement either forward() or async_forward()."
            )

    @property
    def has_async_forward(self) -> bool:
        """
        Check if the tool has an asynchronous implementation.

        Returns:
            bool: True if async_forward is implemented, False otherwise.
        """
        return self.__class__.async_forward is not Tool.async_forward

    @classmethod
    def get_tool(cls, tool_name: str, *args, **kwargs) -> Tool:
        """
        Get a tool instance by name from the registry.

        Args:
            tool_name: The name of the tool to retrieve
            *args: Positional arguments to pass to the tool constructor
            **kwargs: Keyword arguments to pass to the tool constructor

        Returns:
            Tool: An instance of the requested tool

        Raises:
            ValueError: If the tool is not found in the registry
        """
        if tool_name not in cls._registry:
            raise ValueError(f"Unknown tool: {tool_name}")
        return cls._registry[tool_name](*args, **kwargs)

    @classmethod
    def register(cls, tool_name: str):
        """
        Decorator to register a tool class with the registry.

        Args:
            tool_name: The name to register the tool under

        Returns:
            The decorator function

        Example:
            @Tool.register("my_tool")
            class MyTool(Tool):
                def forward(self, x):
                    return x * 2
        """

        def decorator(subclass: type[Tool]) -> type[Tool]:
            cls._registry[tool_name] = subclass
            return subclass

        return decorator


class ToolGroup(Tool):
    """A collection of tools grouped together for batch management.

    This class allows organizing multiple tools into a single group, making it
    easier to manage and access related tools. It provides methods for adding,
    removing, and retrieving tools by name.

    Attributes:
        tools: Dictionary mapping tool names to Tool instances
    """

    def __init__(self, tools: list[Tool], name: str = "tool_group", description: str = "Group of tools"):
        """Initialize a ToolGroup with a list of tools.

        Args:
            tools: List of Tool instances to include in the group
            name: Name of the tool group (default: "tool_group")
            description: Description of the tool group (default: "Group of tools")
        """
        super().__init__(name=name, description=description)
        self.tools = {tool.name: tool for tool in tools}

    @property
    def json(self) -> list[dict[str, Any]]:
        """
        Return the tool group's information in a standardized format for tool registration.

        Returns:
            list: List of JSON dictionaries representing each tool in the group
        """
        return [tool.json for tool in self.tools.values()]

    def get_tool(self, name: str) -> Tool | None:
        """
        Retrieve a tool by name from the group.

        Args:
            name (str): The name of the tool to retrieve.
        Returns:
            Tool | None: The tool instance if found, else None.
        """
        return self.tools.get(name)

    def add_tool(self, tool: Tool) -> None:
        """
        Add a new tool to the group.

        Args:
            tool (Tool): The tool instance to add.
        """
        self.tools[tool.name] = tool

    def remove_tool(self, name: str) -> None:
        """
        Remove a tool from the group by name.

        Args:
            name: The name of the tool to remove

        Note:
            Silently ignores if the tool name doesn't exist in the group
        """
        if name in self.tools:
            del self.tools[name]

    def list_tools(self) -> list[str]:
        """
        List all tool names in the group.

        Returns:
            list: A list of tool names currently in the group
        """
        return list(self.tools.keys())

    def has_tool(self, name: str) -> bool:
        """
        Check if a tool with the given name exists in the group.

        Args:
            name: The name of the tool to check

        Returns:
            bool: True if the tool exists, False otherwise
        """
        return name in self.tools


def initialize_tools_from_config(config_path: str) -> list[Tool]:
    """Initialize tools from a configuration file.

    Loads tool configurations from a YAML/OmegaConf file and instantiates
    the corresponding Tool objects. Supports both dynamic loading from
    Python modules and registry-based tool instantiation.

    This method supports:
    - native tools (dynamic import or registry)
    - MCP tools (discover remote tools from MCP servers and wrap as Tool instances)

    Args:
        config_path: Path to the tool configuration file (YAML format)

    Returns:
        list: List of initialized Tool instances

    Raises:
        ImportError: If a tool class cannot be imported
        ValueError: If configuration is invalid

    Example:
        tools = initialize_tools_from_config("configs/tools.yaml")
    """
    tools_config = OmegaConf.load(config_path)
    tools: list[Tool] = []

    # Lazy initialization for MCP support (only if needed).
    tmp_event_loop: asyncio.AbstractEventLoop | None = None
    thread: threading.Thread | None = None

    def _get_mcp_event_loop() -> asyncio.AbstractEventLoop:
        nonlocal tmp_event_loop, thread
        if tmp_event_loop is None:
            tmp_event_loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=tmp_event_loop.run_forever,
                name="psrl mcp tool list fetcher",
                daemon=True,
            )
            thread.start()
        return tmp_event_loop

    def _run_coro(coro):
        loop = _get_mcp_event_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()

    # Cache managers per servers_config_path so config can include multiple MCP blocks cheaply.
    mcp_managers: dict[str, MCPClientManager] = {}

    try:
        for tool_config in tools_config.tools:
            tool_param_dict = OmegaConf.to_container(tool_config.get("params", {}), resolve=True)
            tool_type = tool_param_dict.get("type", "native")

            # ---- MCP tools (discover remote tools and expand into a list of tools) ----
            if tool_type == "mcp":
                from psrl.tools.mcp_clients.manager import MCPClientManager
                from psrl.tools.mcp_tool import MCPTool

                servers_cfg = tool_param_dict.get("mcp_servers_config_path")
                if not servers_cfg:
                    raise ValueError(
                        "MCP tool config missing 'mcp_servers_config_path'. "
                        "Example: params: {type: mcp, mcp_servers_config_path: ./mcp_server.json}"
                    )

                servers_cfg_path = Path(servers_cfg)
                if not servers_cfg_path.is_absolute():
                    servers_cfg_path = Path(config_path).parent / servers_cfg_path

                tool_selected_list = tool_param_dict.get("tool_selected_list")

                if tool_selected_list is not None and not isinstance(tool_selected_list, list):
                    raise ValueError("MCP 'tool_selected_list' must be a list of tool names")

                rate_limit = float(tool_param_dict.get("rate_limit", 120.0))
                timeout = tool_param_dict.get("timeout")
                timeout_f = float(timeout) if timeout is not None else None

                mgr_key = str(servers_cfg_path)
                manager = mcp_managers.get(mgr_key, None)
                if manager is None:
                    manager = MCPClientManager(
                        servers_config_path=str(servers_cfg_path),
                        rate_limit=rate_limit,
                        timeout=30.0,
                    )
                    mcp_managers[mgr_key] = manager

                try:
                    schemas = _run_coro(manager.fetch_tool_schemas(tool_selected_list))
                except FileNotFoundError as e:
                    raise FileNotFoundError(
                        f"MCP servers config not found: {servers_cfg_path}. "
                        'Please create a JSON like {"mcpServers": {"name": {"url": "http://..."}}}.'
                    ) from e
                except Exception as e:
                    raise RuntimeError(
                        "Failed to discover MCP tool schemas. "
                        "Check MCP server URL/auth_token, server availability, and fastmcp installation. "
                        f"Root error: {type(e).__name__}: {e}"
                    ) from e

                if not schemas:
                    raise ValueError(
                        "No MCP tools discovered. "
                        "If you set tool_selected_list, ensure it matches server tool names; "
                        "otherwise omit it or set it to null."
                    )

                # Create MCPTool instances for each discovered schema.
                for schema in schemas:
                    tools.append(MCPTool(manager=manager, tool_schema=schema, timeout=timeout_f))
                continue

            # ---- Native tools ----
            # Dynamic loading from path
            if "path" in tool_config and "class_name" in tool_config:
                tool_path = Path(tool_config["path"])
                class_name = tool_config["class_name"]

                if not tool_path.is_absolute():
                    # Assume path is relative to the config file's directory
                    config_dir = Path(config_path).parent
                    tool_path = config_dir / tool_path

                if not tool_path.exists():
                    raise FileNotFoundError(f"Tool file not found at: {tool_path}")

                # Dynamically import the module
                spec = importlib.util.spec_from_file_location(tool_path.stem, tool_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not create module spec for {tool_path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)

                # Get the class and instantiate
                if not hasattr(module, class_name):
                    raise AttributeError(f"Class '{class_name}' not found in module {tool_path}")
                tool_cls = getattr(module, class_name)
                tool = tool_cls(**tool_param_dict)
            # Fallback to registry
            else:
                tool_cls_name = tool_config.get("tool_name")
                if tool_cls_name is None:
                    raise ValueError(
                        f"Tool configuration must contain 'tool_name' or 'path' and 'class_name': {tool_config}"
                    )
                tool = Tool.get_tool(tool_cls_name, **tool_param_dict)

            tools.append(tool)
        return tools
    finally:
        # Properly cleanup event loop if it was created.
        if tmp_event_loop is not None:
            tmp_event_loop.call_soon_threadsafe(tmp_event_loop.stop)
            if thread is not None and thread.is_alive():
                thread.join(timeout=5.0)
            tmp_event_loop.close()
