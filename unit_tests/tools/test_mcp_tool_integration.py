import json
from pathlib import Path

import pytest  # noqa: F401
from psrl.tools.base import initialize_tools_from_config  # noqa: F401


class _FakeManager:
    def __init__(self):
        self.calls = []

    async def call_tool(self, tool_name, parameters):
        self.calls.append((tool_name, parameters))
        return f"echo:{parameters.get('q')}", {"status": "success"}


class _SlowManager:
    async def call_tool(self, tool_name, parameters):  # noqa: ARG001
        await __import__("asyncio").sleep(0.2)
        return "never", {}


class _DummyClient:
    """A minimal async context manager used to populate MCPClientManager mappings.

    We don't actually call it in these unit tests; it simply acts as a placeholder
    to validate that schema discovery can set up tool -> client mapping.
    """

    async def __aenter__(self):  # noqa: D401
        return self

    async def __aexit__(self, exc_type, exc, tb):  # noqa: D401
        return False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mcp_tool_output_contract(monkeypatch, tmp_path: Path):
    """Ensure MCPTool returns ToolOutput.output with {content, score, metadata}.

    We don't require fastmcp in unit tests; we patch schema discovery to return a dummy schema
    and patch the MCPClientManager instance used by MCPTool to a fake manager.
    """

    # Create minimal mcp servers config (won't be read due to patch below, but required by loader)
    mcp_server_json = tmp_path / "mcp_server.json"
    mcp_server_json.write_text(json.dumps({"mcpServers": {"dummy": {"url": "http://localhost"}}}))

    tool_cfg = tmp_path / "tools.yaml"
    tool_cfg.write_text(
        """
        tools:
          - tool_name: mcp
            params:
              type: mcp
              mcp_servers_config_path: ./mcp_server.json
              tool_selected_list: [dummy_echo]
              rate_limit: 999
              timeout: 5
        """
    )

    # Patch discovery/initialization to avoid importing/initializing fastmcp.
    from psrl.tools.mcp_clients.manager import MCPClientManager  # noqa: F401

    async def _noop_initialize(self):
        self._initialized = True  # noqa: SLF001
        return None

    async def _fake_fetch(self, tool_selected_list=None):  # noqa: ARG001
        from psrl.tools.mcp_clients.schema import MCPToolSchema

        # Populate mapping as real discovery would.
        self._tool_client_mapping["dummy_echo"] = _DummyClient()  # noqa: SLF001

        return [
            MCPToolSchema(
                name="dummy_echo",
                description="dummy echo",
                parameters={
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            )
        ]

    monkeypatch.setattr(MCPClientManager, "initialize", _noop_initialize)
    monkeypatch.setattr(MCPClientManager, "fetch_tool_schemas", _fake_fetch)

    tools = initialize_tools_from_config(str(tool_cfg))
    assert len(tools) == 1
    tool = tools[0]

    # Ensure schema discovery created tool->client mapping (manager layer contract).
    assert "dummy_echo" in tool._manager._tool_client_mapping  # noqa: SLF001

    # Replace real manager with fake manager for call_tool.
    tool._manager = _FakeManager()  # noqa: SLF001

    tool_output, reward = await tool(q="hi")
    assert reward is None or reward == 0.0

    assert isinstance(tool_output.output, dict)
    assert tool_output.output["content"] == "echo:hi"
    assert tool_output.output["metadata"]["status"] == "success"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mcp_tool_timeout(monkeypatch, tmp_path: Path):
    mcp_server_json = tmp_path / "mcp_server.json"
    mcp_server_json.write_text(json.dumps({"mcpServers": {"dummy": {"url": "http://localhost"}}}))

    tool_cfg = tmp_path / "tools.yaml"
    tool_cfg.write_text(
        """
        tools:
          - tool_name: mcp
            params:
              type: mcp
              mcp_servers_config_path: ./mcp_server.json
              tool_selected_list: [dummy_echo]
              rate_limit: 999
              timeout: 0.05
        """
    )

    from psrl.tools.mcp_clients.manager import MCPClientManager

    async def _noop_initialize(self):
        self._initialized = True  # noqa: SLF001
        return None

    async def _fake_fetch(self, tool_selected_list=None):  # noqa: ARG001
        from psrl.tools.mcp_clients.schema import MCPToolSchema

        self._tool_client_mapping["dummy_echo"] = _DummyClient()  # noqa: SLF001

        return [
            MCPToolSchema(
                name="dummy_echo",
                description="dummy echo",
                parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            )
        ]

    monkeypatch.setattr(MCPClientManager, "initialize", _noop_initialize)
    monkeypatch.setattr(MCPClientManager, "fetch_tool_schemas", _fake_fetch)
    tools = initialize_tools_from_config(str(tool_cfg))
    tool = tools[0]

    tool._manager = _SlowManager()  # noqa: SLF001
    tool_output, reward = await tool(q="hi")

    assert reward == 0.0
    assert tool_output.output["metadata"]["error_type"] == "timeout"
    assert "timed out" in tool_output.output["content"]
