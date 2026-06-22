import asyncio
import json
import logging
import os
from dataclasses import dataclass, field

import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from psrl.utils.common.http_utils import (
    HttpResponse,
    create_aiohttp_client,
    filter_http_headers,
    request_raw,
)
from psrl.utils.logger import DualOutputHandler
from psrl.workers.gen.smg_adapter import TITO_SESSIONS_PATH

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@dataclass(slots=True)
class SessionState:
    """Local concurrency state for one TITO session."""

    headers: dict[str, str] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    drained: asyncio.Event = field(default_factory=asyncio.Event)
    closing: bool = False
    inflight: int = 0
    # Per-trajectory turn tracking: trajectory_id → current_turn
    trajectory_turns: dict[int, int] = field(default_factory=dict)
    base_worker_id: str | None = None
    target_dp_rank: str | None = None

    def __post_init__(self) -> None:
        if self.inflight == 0:
            self.drained.set()
        else:
            self.drained.clear()

    def get_trajectory_turn(self, trajectory_id: int) -> int:
        """Get the current turn for a trajectory, initializing to 0 if new."""
        return self.trajectory_turns.get(trajectory_id, 0)

    def advance_trajectory_turn(self, trajectory_id: int) -> None:
        """Advance the turn counter for a specific trajectory."""
        current = self.trajectory_turns.get(trajectory_id, 0)
        self.trajectory_turns[trajectory_id] = current + 1


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

    async def create_session(self, request: Request) -> Response:
        result = await self._request_upstream("POST", TITO_SESSIONS_PATH)
        if result.status < 400:
            session_id = self.extract_session_id(result)
            if session_id is not None:
                state = await self._ensure_state(session_id)
                async with state.lock:
                    state.headers = self.session_headers(request.headers)
        return self.build_response(result)

    async def get_session(self, sid: str) -> Response:
        result = await self._request_upstream("GET", f"{TITO_SESSIONS_PATH}/{sid}")
        state = self.states.get(sid)
        if state is not None:
            async with state.lock:
                if state.base_worker_id is not None:
                    result.headers["x-base-worker-id"] = state.base_worker_id
                if state.target_dp_rank is not None:
                    result.headers["x-target-dp-rank"] = state.target_dp_rank
        return self.build_response(result)

    async def delete_session(self, sid: str) -> Response:
        state = await self._ensure_state(sid)
        async with state.lock:
            state.closing = True

        await state.drained.wait()

        result = await self._request_upstream("DELETE", f"{TITO_SESSIONS_PATH}/{sid}")

        async with self.states_lock:
            if self.states.get(sid) is state:
                self.states.pop(sid, None)
        return self.build_response(result)

    async def session_chat_completions(self, sid: str, request: Request) -> Response:
        """Handle a chat completion request within a TITO session."""
        state = await self._ensure_state(sid)

        # Extract trajectory_id from request headers (set in add_session_headers)
        trajectory_id = self._extract_trajectory_id(request)

        async with state.lock:
            if state.closing:
                return JSONResponse(status_code=409, content={"error": "session is closing"})
            expected_turn = state.get_trajectory_turn(trajectory_id)
            session_headers = state.headers
            base_worker_id = state.base_worker_id
            target_dp_rank = state.target_dp_rank
            state.inflight += 1
            state.drained.clear()

        headers = self.add_session_headers(request, sid, session_headers)
        if base_worker_id is not None:
            headers["x-base-worker-id"] = base_worker_id
        if target_dp_rank is not None:
            headers["x-target-dp-rank"] = target_dp_rank
        result: HttpResponse | None = None
        try:
            result = await self._request_upstream(
                "POST",
                "v1/chat/completions",
                content=await request.body(),
                headers=headers,
            )
        finally:
            # Single combined critical section: close out inflight bookkeeping
            # and, on success, advance the trajectory's turn counter.
            async with state.lock:
                state.inflight = max(0, state.inflight - 1)
                if state.inflight == 0:
                    state.drained.set()
                if result is not None and result.status < 400:
                    base_worker_id = result.headers.get("x-base-worker-id")
                    target_dp_rank = result.headers.get("x-target-dp-rank")
                    if base_worker_id is not None and target_dp_rank is not None:
                        state.base_worker_id = base_worker_id
                        state.target_dp_rank = target_dp_rank

                    # Check for out-of-order response arrival
                    current_turn = state.get_trajectory_turn(trajectory_id)
                    if current_turn != expected_turn:
                        # This is a turn skew: response arrived out of order.
                        # With trajectory-aware tracking, this is expected during
                        # partial rollout re-dispatch and doesn't block progress.
                        psrl_logger.debug(
                            "SessionRouter turn skew for sid=%s trajectory_id=%s "
                            "expected_turn=%s current_turn=%s. "
                            "This is expected during re-dispatch (hybrid fix active).",
                            sid,
                            trajectory_id,
                            expected_turn,
                            current_turn,
                        )

                    state.advance_trajectory_turn(trajectory_id)

        return self.build_response(result)

    async def session_proxy(self, sid: str, path: str, request: Request) -> Response:
        state = await self._ensure_state(sid)
        async with state.lock:
            session_headers = state.headers
        headers = self.add_session_headers(request, sid, session_headers)
        result = await self._request_upstream(
            request.method,
            path,
            content=await request.body(),
            headers=headers,
        )
        return self.build_response(result)

    async def _ensure_state(self, sid: str) -> SessionState:
        state = self.states.get(sid)
        if state is not None:
            return state
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
            return await request_raw(
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
        existing = self.client
        if existing is None:
            self.client = create_aiohttp_client(concurrency=self.client_concurrency)
            return self.client
        is_closed = getattr(existing, "closed", None)
        if is_closed is None:
            is_closed = getattr(existing, "is_closed", False)
        if is_closed:
            self.client = create_aiohttp_client(concurrency=self.client_concurrency)
        return self.client

    @staticmethod
    def _extract_trajectory_id(request: Request) -> int:
        """Extract trajectory_id from x-smg-tito-trajectory-id header.

        Returns:
            int: The trajectory_id, or 0 if not present or invalid.
        """
        trajectory_id_str = request.headers.get("x-smg-tito-trajectory-id", "0")
        return int(trajectory_id_str)

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
    def session_headers(headers) -> dict[str, str]:
        """Capture immutable routing metadata supplied when a session is created."""
        allowed = {
            "x-base-worker-id",
            "x-is-sticky",
            "x-is-validate",
            "x-prompt-id",
            "x-request-id",
            "x-smg-tito-trajectory-id",
            "x-target-dp-rank",
            "x-version-tag",
        }
        return {key.lower(): value for key, value in headers.items() if key.lower() in allowed}

    @staticmethod
    def add_session_headers(
        request: Request,
        sid: str,
        session_headers: dict[str, str] | None = None,
    ) -> dict[str, str]:
        headers = filter_http_headers(request.headers)
        headers.update(session_headers or {})
        headers["x-smg-tito-session-id"] = sid
        headers.setdefault("x-smg-tito-trajectory-id", "0")
        return headers
