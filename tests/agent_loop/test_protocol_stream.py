"""Streaming wire-shape tests for the harness response renderers."""

from __future__ import annotations

import json
from typing import Any

import pytest

from psrl.workers.gen.harness.protocol import responses_stream_response


async def _collect_events(response) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    async for chunk in response.body_iterator:
        text = chunk.decode() if isinstance(chunk, bytes) else chunk
        for frame in text.split("\n\n"):
            if not frame.strip():
                continue
            lines = frame.splitlines()
            name = next(line[len("event: ") :] for line in lines if line.startswith("event: "))
            data = next(json.loads(line[len("data: ") :]) for line in lines if line.startswith("data: "))
            events.append((name, data))
    return events


def _completed(output: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": "actor",
        "status": "in_progress",
        "output": output,
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }


@pytest.mark.asyncio
async def test_custom_tool_call_renders_native_input_events() -> None:
    """A custom tool call streams `custom_tool_call_input` events, not `arguments`.

    Codex reads the free-form payload from `response.custom_tool_call_input.delta`
    and from the final `output_item.done` item; emitting `function_call_arguments`
    events here would leave the patch empty.
    """
    item = {
        "type": "custom_tool_call",
        "id": "ctc_1",
        "call_id": "call_1",
        "name": "apply_patch",
        "input": "*** Begin Patch",
        "status": "in_progress",
    }
    response = responses_stream_response(_completed([item]), headers={})

    events = await _collect_events(response)
    names = [name for name, _ in events]

    assert "response.custom_tool_call_input.delta" in names
    assert "response.custom_tool_call_input.done" in names
    assert "response.function_call_arguments.delta" not in names

    delta = next(data for name, data in events if name == "response.custom_tool_call_input.delta")
    assert delta["delta"] == "*** Begin Patch"
    assert delta["call_id"] == "call_1"

    done = next(data for name, data in events if name == "response.output_item.done")
    assert done["item"]["type"] == "custom_tool_call"
    assert done["item"]["input"] == "*** Begin Patch"

    assert names[-1] == "response.completed"


@pytest.mark.asyncio
async def test_function_call_still_uses_arguments_events() -> None:
    item = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_2",
        "name": "exec_command",
        "arguments": '{"cmd":"ls"}',
        "status": "in_progress",
    }
    response = responses_stream_response(_completed([item]), headers={})

    names = [name for name, _ in await _collect_events(response)]

    assert "response.function_call_arguments.delta" in names
    assert "response.custom_tool_call_input.delta" not in names
