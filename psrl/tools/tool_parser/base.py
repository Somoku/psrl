from abc import ABC, abstractmethod

from psrl.tools.base import ToolCall


class ToolParser(ABC):
    _registry: dict[str, type["ToolParser"]] = {}

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    @abstractmethod
    def extract_tool_calls(self, responses_ids: list[int]) -> tuple[str, list[ToolCall]]:
        """Extract tool calls from the responses.

        Args:
            responses_ids (List[int]): The ids of the responses.

        Returns:
            List[ToolCall]: Extracted tool calls.
        """
        raise NotImplementedError

    @classmethod
    def get_tool_parser(cls, name: str, tokenizer):
        if name not in cls._registry:
            raise ValueError(f"Unknown tool parser: {name}")
        return cls._registry[name](tokenizer)

    @classmethod
    def register(cls, name: str):
        def decorator(subclass: type[ToolParser]) -> type[ToolParser]:
            cls._registry[name] = subclass
            return subclass

        return decorator
