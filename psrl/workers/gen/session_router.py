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
from psrl.workers.gen.smg_adapter import TITO_SESSIONS_PATH, TRAJECTORY_ID_STRATEGIES

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

# Hang-state values (see SessionState.hang_state).
SESSION_RUNNING = "running"
SESSION_HUNG = "hung"

# Session status values, derived from inflight: a trajectory is either inferring
# on vLLM/SMG (generate) or between turns / calling the environment (env).
STATUS_GENERATE = "generate"
STATUS_ENV = "env"

SESSION_ID_HEADER = "x-smg-tito-session-id"
TRAJECTORY_ID_HEADER = "x-smg-tito-trajectory-id"


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
    # Version tag pinned by SMG on the first routed turn. Once set (non-`-1`),
    # it is carried into every subsequent turn so re-routes filter on a version
    # at least as fresh as the instance that first served this trajectory.
    version_tag: str | None = None
    # --- Hang/continue scheduling (ThunderAgent port) ---
    # Accumulated token footprint of the session (max prompt+completion observed).
    total_tokens: int = 0
    # "running" | "hung": whether the coordinator has hung this session.
    hang_state: str = SESSION_RUNNING
    # Set by the coordinator when a hang is requested while a turn is in flight;
    # converted to hung at the next turn boundary (deferred hang for generate).
    marked_for_hang: bool = False
    # Set means "may proceed". A hung session blocks at the next turn entry until
    # the coordinator continues it (sets the event). Initialized set in __post_init__.
    continue_event: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        if self.inflight == 0:
            self.drained.set()
        else:
            self.drained.clear()
        # A fresh session is running: allow it to proceed.
        self.continue_event.set()

    @property
    def status(self) -> str:
        """generate while a turn is in flight, else env (between turns)."""
        return STATUS_GENERATE if self.inflight > 0 else STATUS_ENV

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
        trajectory_id_strategy: str = "manual",
    ):
        trajectory_id_strategy = trajectory_id_strategy.lower()
        if trajectory_id_strategy not in TRAJECTORY_ID_STRATEGIES:
            choices = ", ".join(sorted(TRAJECTORY_ID_STRATEGIES))
            raise ValueError(f"Invalid trajectory_id_strategy {trajectory_id_strategy!r}; expected one of: {choices}.")
        self.smg_url = smg_url.rstrip("/")
        self.app = FastAPI()
        self.client: aiohttp.ClientSession | None = None
        self.client_concurrency = client_concurrency
        self.trajectory_id_strategy = trajectory_id_strategy
        self.states: dict[str, SessionState] = {}
        self.states_lock = asyncio.Lock()
        self.setup_routes()
        self.app.router.on_shutdown.append(self.aclose)

    def setup_routes(self):
        self.app.post("/sessions")(self.create_session)
        self.app.get("/sessions/{sid}")(self.get_session)
        self.app.delete("/sessions/{sid}")(self.delete_session)
        self.app.post("/sessions/{sid}/v1/chat/completions")(self.session_chat_completions)
        # Coordinator-facing hang/continue control plane.
        self.app.get("/control/sessions")(self.control_list_sessions)
        self.app.post("/control/hang")(self.control_hang)
        self.app.post("/control/continue")(self.control_continue)
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
            state = await self._ensure_state(session_id)
            async with state.lock:
                state.headers = self.session_headers(request.headers, session_id)
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
            # Unblock any turn hung at the hang point so the session can drain
            # and be torn down instead of deadlocking on continue_event.
            state.hang_state = SESSION_RUNNING
            state.marked_for_hang = False
            state.continue_event.set()

        await state.drained.wait()

        result = await self._request_upstream("DELETE", f"{TITO_SESSIONS_PATH}/{sid}")

        async with self.states_lock:
            if self.states.get(sid) is state:
                self.states.pop(sid, None)
        return self.build_response(result)

    async def session_chat_completions(self, sid: str, request: Request) -> Response:
        """Handle a chat completion request within a TITO session."""
        state = await self._ensure_state(sid)

        # SMG owns trajectory resolution in auto mode, so the SessionRouter only
        # tracks caller-selected IDs in manual mode.
        trajectory_id = (
            int(request.headers.get(TRAJECTORY_ID_HEADER, "0")) if self.trajectory_id_strategy == "manual" else None
        )

        # Hang point: block at the entry of the next turn while the session is
        # hung. Any in-flight turn (generate on SMG or an env step between turns)
        # has already returned, so the whole session quiesces here without
        # aborting in-flight work. Await the event OUTSIDE the lock so the
        # coordinator can flip hang_state/continue_event concurrently.
        while True:
            async with state.lock:
                if state.closing:
                    return JSONResponse(status_code=409, content={"error": "session is closing"})
                if state.hang_state != SESSION_HUNG:
                    break
                continue_event = state.continue_event
            await continue_event.wait()

        async with state.lock:
            if state.closing:
                return JSONResponse(status_code=409, content={"error": "session is closing"})
            session_headers = state.headers.copy()
            base_worker_id = state.base_worker_id
            target_dp_rank = state.target_dp_rank
            version_tag = state.version_tag
            state.inflight += 1
            state.drained.clear()

        headers = self.add_session_headers(request, session_headers)
        if base_worker_id is not None:
            headers["x-base-worker-id"] = base_worker_id
        if target_dp_rank is not None:
            headers["x-target-dp-rank"] = target_dp_rank
        if version_tag is not None:
            headers["x-version-tag"] = version_tag

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
                    version_tag = result.headers.get("x-version-tag")
                    if version_tag is not None:
                        state.version_tag = version_tag

                    # Update the session token footprint from the usage block.
                    # usage.prompt_tokens already includes the full accumulated
                    # TITO context, so prompt+completion is the live footprint.
                    self._update_total_tokens(state, result)

                    if trajectory_id is not None:
                        state.advance_trajectory_turn(trajectory_id)

                # Deferred-hang conversion: a hang requested mid-turn takes effect
                # now that this trajectory has returned and the session is idle.
                if state.marked_for_hang and state.inflight == 0:
                    state.marked_for_hang = False
                    state.hang_state = SESSION_HUNG
                    state.continue_event.clear()

        return self.build_response(result)

    async def session_proxy(self, sid: str, path: str, request: Request) -> Response:
        state = await self._ensure_state(sid)
        async with state.lock:
            session_headers = state.headers.copy()
        headers = self.add_session_headers(request, session_headers)
        result = await self._request_upstream(
            request.method,
            path,
            content=await request.body(),
            headers=headers,
        )
        return self.build_response(result)

    # -------------------------------------------------------------------------
    # Hang/continue control plane (coordinator-facing)
    # -------------------------------------------------------------------------

    async def control_list_sessions(self) -> Response:
        """Return a snapshot of every live session for the coordinator scheduler."""
        # Snapshot the states dict first so we don't hold states_lock while
        # acquiring per-session locks.
        items = list(self.states.items())
        sessions = []
        for sid, state in items:
            async with state.lock:
                sessions.append(
                    {
                        "session_id": sid,
                        "base_worker_id": state.base_worker_id,
                        "target_dp_rank": state.target_dp_rank,
                        "status": state.status,
                        "hang_state": state.hang_state,
                        "inflight": state.inflight,
                        "total_tokens": state.total_tokens,
                        "marked_for_hang": state.marked_for_hang,
                        "closing": state.closing,
                    }
                )
        return JSONResponse(content={"sessions": sessions})

    async def control_hang(self, request: Request) -> Response:
        """Hang the given sessions.

        Body: ``[{"session_id": ...}, ...]``. If a session is idle (env, no
        in-flight turn) it is hung immediately; if a turn is in flight
        (generate) the hang is deferred and applied at the next turn boundary.
        """
        payload = await self._read_control_ids(request)
        applied, deferred, missing = [], [], []
        for sid in payload:
            state = self.states.get(sid)
            if state is None:
                missing.append(sid)
                continue
            async with state.lock:
                if state.closing:
                    missing.append(sid)
                    continue
                if state.inflight == 0:
                    state.hang_state = SESSION_HUNG
                    state.marked_for_hang = False
                    state.continue_event.clear()
                    applied.append(sid)
                else:
                    state.marked_for_hang = True
                    deferred.append(sid)
        return JSONResponse(content={"hung": applied, "deferred": deferred, "missing": missing})

    async def control_continue(self, request: Request) -> Response:
        """Continue (un-hang) the given sessions.

        Body: ``[{"session_id": ...}, ...]``.
        """
        payload = await self._read_control_ids(request)
        applied, missing = [], []
        for sid in payload:
            state = self.states.get(sid)
            if state is None:
                missing.append(sid)
                continue
            async with state.lock:
                state.marked_for_hang = False
                state.hang_state = SESSION_RUNNING
                state.continue_event.set()
                applied.append(sid)
        return JSONResponse(content={"continued": applied, "missing": missing})

    @staticmethod
    async def _read_control_ids(request: Request) -> list[str]:
        """Parse a control request body into a list of session ids.

        Accepts ``[{"session_id": ...}, ...]``, ``["sid", ...]``, or
        ``{"sessions": [...]}``.
        """
        try:
            body = await request.json()
        except Exception:
            return []
        if isinstance(body, dict):
            body = body.get("sessions", [])
        ids: list[str] = []
        for item in body or []:
            if isinstance(item, dict):
                sid = item.get("session_id")
            else:
                sid = item
            if sid is not None:
                ids.append(str(sid))
        return ids

    @staticmethod
    def _update_total_tokens(state: SessionState, result: HttpResponse) -> None:
        """Update state.total_tokens from a chat-completion response's usage block."""
        try:
            usage = result.json().get("usage") or {}
            prompt = int(usage.get("prompt_tokens", 0))
            completion = int(usage.get("completion_tokens", 0))
        except Exception:
            return
        footprint = prompt + completion
        if footprint > state.total_tokens:
            state.total_tokens = footprint

    async def _ensure_state(self, sid: str) -> SessionState:
        state = self.states.get(sid)
        if state is not None:
            return state
        async with self.states_lock:
            state = self.states.get(sid)
            if state is None:
                state = SessionState(headers={SESSION_ID_HEADER: sid})
                self.states[sid] = state
            return state

    async def _request_upstream(
        self,
        method: str,
        path: str,
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
    def build_response(result: HttpResponse) -> Response:
        content_type = result.headers.get("content-type", "")
        return Response(
            content=result.body,
            status_code=result.status,
            headers=result.headers,
            media_type=content_type or None,
        )

    @staticmethod
    def extract_session_id(result: HttpResponse) -> str:
        session_id = result.json()["session_id"]
        return session_id

    @staticmethod
    def session_headers(headers, sid: str) -> dict[str, str]:
        """Capture session defaults and bind them to the session identity."""
        allowed = {
            "x-prompt-id",
            "x-request-id",
            "x-is-validate",
            "x-is-sticky",
            "x-version-tag",
            "x-base-worker-id",
            "x-target-dp-rank",
        }
        session_headers = {key.lower(): value for key, value in headers.items() if key.lower() in allowed}
        session_headers[SESSION_ID_HEADER] = sid
        return session_headers

    def add_session_headers(
        self,
        request: Request,
        session_headers: dict[str, str],
    ) -> dict[str, str]:
        # Session headers are defaults and request-scoped values take precedence.
        # The session identity is reserved and can only come from SessionState.
        request_headers = filter_http_headers(request.headers)
        request_headers.pop(SESSION_ID_HEADER, None)
        if self.trajectory_id_strategy == "auto":
            request_headers.pop(TRAJECTORY_ID_HEADER, None)
        headers = dict(session_headers)
        headers.update(request_headers)
        if self.trajectory_id_strategy == "manual":
            headers.setdefault(TRAJECTORY_ID_HEADER, "0")
        return headers
