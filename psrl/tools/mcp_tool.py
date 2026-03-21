import asyncio
import logging
from typing import Any

from psrl.tools.base import Tool, ToolOutput
from psrl.tools.mcp_clients.manager import MCPClientManager
from psrl.tools.mcp_clients.schema import MCPToolSchema

psrl_logger = logging.getLogger(__name__)


@Tool.register("mcp")
class MCPTool(Tool):
    """MCP remote tool.

    This class is instantiated per MCP tool schema discovered from servers.
    It is designed to be compatible with `ToolEnvironment._call_tool`, which
    expects `ToolOutput.output` to be a dict containing at least `content`.
    """

    def __init__(
        self,
        manager: MCPClientManager,
        tool_schema: dict[str, Any] | MCPToolSchema,
        timeout: float | None = None,
    ):
        if isinstance(tool_schema, MCPToolSchema):
            schema_obj = tool_schema
            self._json = schema_obj.to_openai_function_schema()
            name = schema_obj.name
            description = schema_obj.description
        else:
            # Expect OpenAI function schema format.
            fn = tool_schema.get("function", {})
            name = fn.get("name")
            description = fn.get("description")
            if not name or description is None:
                raise ValueError(f"Invalid MCP tool schema: {tool_schema}")
            self._json = tool_schema

        super().__init__(name=name, description=description)

        self._manager = manager
        self._timeout = timeout

    async def async_forward(self, *args, **kwargs) -> ToolOutput:
        """Call the underlying remote MCP tool.

        Returns:
            ToolOutput: The tool output containing 'content' and optional metadata.
        """
        try:
            if args:
                raise TypeError("MCPTool only supports keyword arguments")
            coro = self._manager.call_tool(self.name, kwargs)
            if self._timeout is not None:
                text, metadata = await asyncio.wait_for(coro, timeout=float(self._timeout))
            else:
                text, metadata = await coro
            output = {
                "content": text,
                "score": 0.0,
                "metadata": metadata,
            }
            return ToolOutput(name=self.name or "mcp", output=output)
        except TimeoutError:
            # NOTE: timeout is handled by server/transport in most MCP stacks;
            # we keep the field for future per-call timeout control.
            msg = (
                f"Error when executing tool '{self.name}': request timed out after {self._timeout}s. "
                "Check MCP server availability / auth token / rate limit."
            )
            psrl_logger.warning(msg)
            output = {
                "content": msg,
                "score": 0.0,
                "metadata": {
                    "api_request_error": msg,
                    "error_type": "timeout",
                    "timeout": self._timeout,
                },
            }
            return ToolOutput(name=self.name or "mcp", output=output)
        except Exception as e:
            raise RuntimeError(f"Error when executing tool '{self.name}': {type(e).__name__} - {e}. ") from e
