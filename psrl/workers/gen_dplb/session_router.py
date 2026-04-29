import asyncio
import json
import logging
from dataclasses import dataclass, field

import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from psrl.utils.common.http_utils import (
    HttpResponse,
    create_aiohttp_client,
    raw_request,
)

psrl_logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SessionState:
    """Local concurrency state for one TITO session."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    drained: asyncio.Event = field(default_factory=asyncio.Event)
    closing: bool = False
    inflight: int = 0
    turn: int = 0

    def __post_init__(self) -> None:
        if self.inflight == 0:
            self.drained.set()
        else:
            self.drained.clear()

class SessionRouter:
    def __init__(
        self,
        smg_url: str,
        client_concurrency: int = 1024,
    ):
        self.smg_url = smg_url.rstrip("/")
        self.app = FastAPI()
        self.client: aiohttp.ClientSession | None = None
        self.client_concurrency = client_concurrency
        self.states: dict[str, SessionState] = {}
        self.states_lock = asyncio.Lock()
        self.setup_routes()
        self.app.router.on_shutdown.append(self.aclose)

    def setup_routes(self):
        self.app.post("/sessions")(self.create_session)
        self.app.get("/sessions/{sid}")(self.get_session)
        self.app.delete("/sessions/{sid}")(self.delete_session)
        self.app.post("/sessions/{sid}/v1/chat/completions")(self.session_chat_completions)
        self.app.api_route(
            "/sessions/{sid}/{path:path}",
            methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        )(self.session_proxy)

    async def aclose(self):
        """Close the router-owned aiohttp client."""
        if self.client is not None and not self.client.closed:
            await self.client.close()

    async def create_session(self) -> Response:
        result = await self._request_upstream("POST", "v1/tito/sessions")
        if result.status < 400:
            session_id = self.extract_session_id(result)
            if session_id is not None:
                await self._ensure_state(session_id)
        return self.build_response(result)

    async def get_session(self, sid: str) -> Response:
        result = await self._request_upstream("GET", f"v1/tito/sessions/{sid}")
        return self.build_response(result)

    async def delete_session(self, sid: str) -> Response:
        state = await self._ensure_state(sid)
        async with state.lock:
            state.closing = True

        await state.drained.wait()

        result = await self._request_upstream("DELETE", f"v1/tito/sessions/{sid}")

        async with self.states_lock:
            if self.states.get(sid) is state:
                self.states.pop(sid, None)
        return self.build_response(result)

    async def session_chat_completions(self, sid: str, request: Request) -> Response:
        state = await self._ensure_state(sid)
        async with state.lock:
            if state.closing:
                return JSONResponse(status_code=409, content={"error": "session is closing"})
            expected_turn = state.turn
            state.inflight += 1
            state.drained.clear()

        try:
            headers = self.add_session_headers(request, sid)
            result = await self._request_upstream(
                "POST",
                "v1/chat/completions",
                content=await request.body(),
                headers=headers,
            )
            if result.status < 400:
                async with state.lock:
                    if state.turn != expected_turn:
                        psrl_logger.warning(
                            "SessionRouter stale write detected for sid=%s expected_turn=%s current_turn=%s.",
                            sid,
                            expected_turn,
                            state.turn,
                        )
                        return
                    state.turn += 1
            return self.build_response(result)
        finally:
            async with state.lock:
                state.inflight = max(0, state.inflight - 1)
                if state.inflight == 0:
                    state.drained.set()

    async def session_proxy(self, sid: str, path: str, request: Request) -> Response:
        headers = self.add_session_headers(request, sid)
        result = await self._request_upstream(
            request.method,
            path,
            content=await request.body(),
            headers=headers,
        )
        return self.build_response(result)

    async def _ensure_state(self, sid: str) -> SessionState:
        async with self.states_lock:
            state = self.states.get(sid)
            if state is None:
                state = SessionState()
                self.states[sid] = state
            return state

    async def _request_upstream(
        self,
        method: str,
        path: str,
        *,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResponse:
        url = f"{self.smg_url}/{path.lstrip('/')}"
        try:
            return await raw_request(
                method,
                url,
                content=content,
                headers=headers,
                client=await self._ensure_client(),
                max_retries=1,
            )
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            psrl_logger.warning(
                "SessionRouter upstream transport error for %s %s: %s.",
                method,
                path,
                exc,
            )
            body = json.dumps(
                {
                    "error": f"backend transport error: {type(exc).__name__}: {exc}",
                }
            ).encode()
            return HttpResponse(
                status=502,
                body=body,
                headers={"content-type": "application/json"},
            )

    async def _ensure_client(self) -> aiohttp.ClientSession:
        if self.client is None or self.client.closed:
            self.client = create_aiohttp_client(concurrency=self.client_concurrency)
        return self.client

    @staticmethod
    def build_response(result: HttpResponse) -> Response:
        content_type = result.headers.get("content-type", "")
        if not result.body or result.status in (204, 304):
            return Response(
                content=result.body,
                status_code=result.status,
                headers=result.headers,
                media_type=content_type or None,
            )
        if content_type.startswith("application/json"):
            return JSONResponse(
                content=json.loads(result.body or b"{}"),
                status_code=result.status,
                headers=result.headers,
            )
        return Response(
            content=result.body,
            status_code=result.status,
            headers=result.headers,
            media_type=content_type or None,
        )

    @staticmethod
    def extract_session_id(result: HttpResponse) -> str | None:
        session_id = result.json().get("session_id")
        return session_id if isinstance(session_id, str) else None

    @staticmethod
    def add_session_headers(request: Request, sid: str) -> dict[str, str]:
        headers = request.headers
        headers["x-smg-tito-session-id"] = sid
        return headers
