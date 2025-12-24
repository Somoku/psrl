from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MCPToolSchema:
    """Internal normalized schema for a single MCP tool."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai_function_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function-calling schema format."""
        openai_schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                "strict": False,
            },
        }
        if not openai_schema["function"]["parameters"].get("required", None):
            openai_schema["function"]["parameters"]["required"] = []
        return openai_schema
