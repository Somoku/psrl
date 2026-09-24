"""The node-side forwarder that makes one callback URL resolve from any node.

A sandbox reaches its callback through its own node's gateway alias. Without a
forwarder that resolves to whichever node the sandbox landed on rather than to the
worker that owns the trajectory, so the node listens and forwards instead.
"""

from __future__ import annotations

import asyncio

import pytest
from psrl.sandbox.node_agent import CallbackForwarder, CallbackTarget

pytestmark = pytest.mark.cpu_test


async def _echo_server() -> tuple[asyncio.AbstractServer, int]:
    """A stand-in for the caller's session server, which answers every line it reads."""

    async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while line := await reader.readline():
                writer.write(b"pong\n" if line == b"ping\n" else b"?\n")
                await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            return
        finally:
            writer.close()

    server = await asyncio.start_server(echo, host="127.0.0.1", port=0)
    sockets = server.sockets or ()
    return server, int(sockets[0].getsockname()[1])


def test_a_callback_target_must_be_an_authority() -> None:
    assert CallbackTarget.parse("10.0.0.7:8080") == CallbackTarget("10.0.0.7", 8080)
    assert CallbackTarget.parse("10.0.0.7:8080").as_str() == "10.0.0.7:8080"
    with pytest.raises(ValueError, match="host:port"):
        CallbackTarget.parse("10.0.0.7")
    with pytest.raises(ValueError, match="host:port"):
        CallbackTarget.parse(":8080")


async def test_the_forwarder_carries_a_conversation_to_the_target() -> None:
    # This is the whole point: the sandbox talks to the node, and the node talks to the
    # caller's worker.
    server, port = await _echo_server()
    forwarder = CallbackForwarder(CallbackTarget("127.0.0.1", port))
    node_port = await forwarder.start()

    reader, writer = await asyncio.open_connection("127.0.0.1", node_port)
    writer.write(b"ping\n")
    await writer.drain()
    assert await reader.readline() == b"pong\n"
    writer.close()

    await forwarder.close()
    server.close()
    await server.wait_closed()


async def test_starting_a_forwarder_twice_keeps_the_same_listener() -> None:
    # One forwarder per caller, so a second sandbox from the same worker does not open a
    # second port on the node.
    server, port = await _echo_server()
    forwarder = CallbackForwarder(CallbackTarget("127.0.0.1", port))

    first = await forwarder.start()
    second = await forwarder.start()

    assert first == second
    await forwarder.close()
    server.close()
    await server.wait_closed()


async def test_a_forwarder_whose_target_is_gone_closes_the_connection() -> None:
    # A node that cannot reach the caller must not leave a sandbox hanging on a read.
    probe = await asyncio.start_server(lambda reader, writer: None, host="127.0.0.1", port=0)
    dead_port = int((probe.sockets or ())[0].getsockname()[1])
    probe.close()
    await probe.wait_closed()

    forwarder = CallbackForwarder(CallbackTarget("127.0.0.1", dead_port))
    node_port = await forwarder.start()
    reader, writer = await asyncio.open_connection("127.0.0.1", node_port)

    assert await asyncio.wait_for(reader.read(), timeout=2.0) == b""

    writer.close()
    await forwarder.close()


async def test_closing_the_forwarder_stops_listening() -> None:
    server, port = await _echo_server()
    forwarder = CallbackForwarder(CallbackTarget("127.0.0.1", port))
    node_port = await forwarder.start()

    await forwarder.close()

    assert forwarder.port is None
    with pytest.raises(OSError):
        await asyncio.open_connection("127.0.0.1", node_port)
    server.close()
    await server.wait_closed()
