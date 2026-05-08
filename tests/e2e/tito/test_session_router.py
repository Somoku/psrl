import json

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import Response as FastAPIResponse
from httpx import ASGITransport
from psrl.workers.gen_dplb.session_router import SessionRouter

# ---------------------------------------------------------------------------
# Mock SMG server
# ---------------------------------------------------------------------------
mock_smg = FastAPI()
captured: dict = {}


@mock_smg.post("/tito/sessions")
async def mock_create():
    return {"session_id": "test-sid-123"}


@mock_smg.get("/tito/sessions/{sid}")
async def mock_get(sid: str):
    return {"session_id": sid, "accumulated_token_ids": [1, 2, 3], "records": []}


@mock_smg.delete("/tito/sessions/{sid}")
async def mock_delete(sid: str):
    return FastAPIResponse(status_code=204)


@mock_smg.post("/v1/chat/completions")
async def mock_chat(request: Request):
    body = json.loads(await request.body())
    headers = dict(request.headers)
    captured["chat_body"] = body
    captured["chat_headers"] = headers
    return {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}


@mock_smg.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def mock_catchall(path: str, request: Request):
    captured["proxy_path"] = path
    captured["proxy_headers"] = dict(request.headers)
    return {"proxied": True}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clear_captured():
    captured.clear()


@pytest.fixture()
def router():
    """Build a SessionRouter whose internal httpx client talks to mock_smg."""
    sr = SessionRouter(smg_url="http://mock-smg")
    # Replace the real client with one backed by the mock ASGI app
    sr.client = httpx.AsyncClient(transport=ASGITransport(app=mock_smg), base_url="http://mock-smg")
    return sr


@pytest.fixture()
def client(router):
    """httpx client that talks to the SessionRouter's FastAPI app."""
    return httpx.AsyncClient(transport=ASGITransport(app=router.app), base_url="http://testserver")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_create_session(client):
    resp = await client.post("/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "test-sid-123"


@pytest.mark.asyncio
async def test_get_session(client):
    resp = await client.get("/sessions/my-session")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "my-session"
    assert data["accumulated_token_ids"] == [1, 2, 3]


@pytest.mark.asyncio
async def test_delete_session(client):
    resp = await client.delete("/sessions/my-session")
    assert resp.status_code == 204


@pytest.mark.asyncio
async def test_chat_completions_forces_logprobs(client):
    payload = {"model": "my-model", "messages": [{"role": "user", "content": "hi"}]}
    resp = await client.post("/sessions/sid-abc/v1/chat/completions", json=payload)
    assert resp.status_code == 200

    # The body forwarded to mock SMG must have logprobs=True
    assert captured["chat_body"]["logprobs"] is True
    # Original fields preserved
    assert captured["chat_body"]["model"] == "my-model"


@pytest.mark.asyncio
async def test_chat_completions_overrides_logprobs_false(client):
    """Even if the caller explicitly sets logprobs=false, the router forces True."""
    payload = {"model": "m", "messages": [], "logprobs": False}
    resp = await client.post("/sessions/sid-abc/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert captured["chat_body"]["logprobs"] is True


@pytest.mark.asyncio
async def test_chat_completions_injects_session_header(client):
    payload = {"model": "m", "messages": []}
    await client.post("/sessions/sid-xyz/v1/chat/completions", json=payload)
    assert captured["chat_headers"]["x-smg-tito-session-id"] == "sid-xyz"
    assert captured["chat_headers"]["x-smg-tito-trajectory-id"] == "0"


@pytest.mark.asyncio
async def test_session_proxy_injects_header(client):
    resp = await client.post("/sessions/sid-999/v1/some/other/endpoint", content=b"hello")
    assert resp.status_code == 200
    assert captured["proxy_headers"]["x-smg-tito-session-id"] == "sid-999"
    assert captured["proxy_path"] == "v1/some/other/endpoint"


@pytest.mark.asyncio
async def test_turn_counter_increments_on_chat_completion(router, client):
    """Turn counter increments exactly once per successful chat completion."""
    payload = {"model": "m", "messages": []}
    state = await router._ensure_state("sid-abc")
    assert state.turn == 0

    await client.post("/sessions/sid-abc/v1/chat/completions", json=payload)
    assert state.turn == 1

    await client.post("/sessions/sid-abc/v1/chat/completions", json=payload)
    assert state.turn == 2


@pytest.mark.asyncio
async def test_delete_during_new_request_returns_409(router, client):
    """After session is closed, new chat completion requests get 409."""
    state = await router._ensure_state("sid-close")
    async with state.lock:
        state.closing = True

    payload = {"model": "m", "messages": []}
    resp = await client.post("/sessions/sid-close/v1/chat/completions", json=payload)
    assert resp.status_code == 409
    assert "session is closing" in resp.text
