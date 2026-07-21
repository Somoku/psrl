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
from psrl.workers.gen.smg_adapter import TITO_SESSIONS_PATH

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

# Hang-state values (see SessionState.hang_state).
SESSION_RUNNING = "running"
SESSION_HUNG = "hung"

# Session status values, derived from inflight: a trajectory is either inferring
# on vLLM/SMG (generate) or between turns / calling the environment (env).
STATUS_GENERATE = "generate"
STATUS_ENV = "env"


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
    # One-shot pin target set by /control/continue: (base_worker_id, target_dp_rank).
    # When set, the session's NEXT turn is force-pinned to this instance (via the
    # x-force-pin-once header) and the field is cleared immediately after injection,
    # so only the first turn after continue is pinned. None means "let SMG route".
    pin_once_instance: tuple[str, str] | None = None

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

        # Extract trajectory_id from request headers (set in add_session_headers)
        trajectory_id = self._extract_trajectory_id(request)

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
            psrl_logger.debug(
                f"Session {sid!r} trajectory {trajectory_id} blocked at hang point, "
                f"awaiting continue."
            )
            await continue_event.wait()

        async with state.lock:
            if state.closing:
                return JSONResponse(status_code=409, content={"error": "session is closing"})
            expected_turn = state.get_trajectory_turn(trajectory_id)
            session_headers = state.headers
            base_worker_id = state.base_worker_id
            target_dp_rank = state.target_dp_rank
            version_tag = state.version_tag
            # Consume the one-shot pin (if any) so only this turn is force-pinned.
            pin_once_instance = state.pin_once_instance
            state.pin_once_instance = None
            state.inflight += 1
            state.drained.clear()

        headers = self.add_session_headers(request, sid, session_headers)
        if base_worker_id is not None:
            headers["x-base-worker-id"] = base_worker_id
        if target_dp_rank is not None:
            headers["x-target-dp-rank"] = target_dp_rank
        if version_tag is not None:
            headers["x-version-tag"] = version_tag
        # Force-pin this turn's first worker selection to the continue target.
        # SMG clears x-force-pin-once on the first loopback, so partial-rollout /
        # preemption re-dispatch within this turn falls back to free routing.
        if pin_once_instance is not None:
            headers["x-base-worker-id"] = pin_once_instance[0]
            headers["x-target-dp-rank"] = pin_once_instance[1]
            headers["x-force-pin-once"] = "true"
            psrl_logger.debug(
                f"Session {sid!r} turn force-pinned to instance {pin_once_instance!r} "
                f"(one-shot)."
            )
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

                # Deferred-hang conversion: a hang requested mid-turn takes effect
                # now that this trajectory has returned and the session is idle.
                if state.marked_for_hang and state.inflight == 0:
                    state.marked_for_hang = False
                    state.hang_state = SESSION_HUNG
                    state.continue_event.clear()
                    psrl_logger.debug(
                        f"Session {sid!r} deferred hang applied at turn boundary "
                        f"(trajectory {trajectory_id})."
                    )

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
        if applied or deferred or missing:
            psrl_logger.info(
                f"control_hang: hung={applied} deferred={deferred} missing={missing}."
            )
        return JSONResponse(
            content={"hung": applied, "deferred": deferred, "missing": missing}
        )

    async def control_continue(self, request: Request) -> Response:
        """Continue (un-hang) the given sessions.

        Body: ``[{"session_id": ..., "base_worker_id": ..., "target_dp_rank": ...}, ...]``.
        The two worker-id fields are optional: when present, the session's next
        turn is force-pinned to that instance (one-shot); when absent, the next
        turn is routed normally by SMG.
        """
        pins = await self._read_control_pins(request)
        applied, missing = [], []
        for sid, instance in pins.items():
            state = self.states.get(sid)
            if state is None:
                missing.append(sid)
                continue
            async with state.lock:
                state.marked_for_hang = False
                state.hang_state = SESSION_RUNNING
                if instance is not None:
                    # Update the routing hint and arm the one-shot pin for the
                    # next turn (consumed in session_chat_completions).
                    state.base_worker_id = instance[0]
                    state.target_dp_rank = instance[1]
                    state.pin_once_instance = instance
                state.continue_event.set()
                applied.append(sid)
        if applied or missing:
            pinned = {sid: inst for sid, inst in pins.items() if inst is not None}
            psrl_logger.info(
                f"control_continue: continued={applied} missing={missing} pinned={pinned}."
            )
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
    async def _read_control_pins(request: Request) -> dict[str, tuple[str, str] | None]:
        """Parse a continue request body into ``{session_id: instance | None}``.

        Accepts ``[{"session_id": ..., "base_worker_id": ..., "target_dp_rank": ...}, ...]``
        or ``{"sessions": [...]}``. ``instance`` is ``(base_worker_id, target_dp_rank)``
        when both are supplied, else ``None`` (route normally). Bare-string items
        (``["sid", ...]``) are accepted and map to ``None``.
        """
        try:
            body = await request.json()
        except Exception:
            return {}
        if isinstance(body, dict):
            body = body.get("sessions", [])
        pins: dict[str, tuple[str, str] | None] = {}
        for item in body or []:
            if isinstance(item, dict):
                sid = item.get("session_id")
                base_worker_id = item.get("base_worker_id")
                target_dp_rank = item.get("target_dp_rank")
                instance = None
                if base_worker_id is not None and target_dp_rank is not None:
                    instance = (str(base_worker_id), str(target_dp_rank))
            else:
                sid = item
                instance = None
            if sid is not None:
                pins[str(sid)] = instance
        return pins

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
