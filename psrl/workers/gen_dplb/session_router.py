"""Lightweight session router proxy for TITO-aware chat completions.

Responsibilities:
1. Session lifecycle (create/get/delete) → proxied to SMG /v1/tito/sessions
2. Inject x-smg-tito-session-id header on session-scoped requests
3. Force logprobs=true on /v1/chat/completions (required for vLLM backend)
4. Serialize concurrent chat completion requests per session using a split-lock
   pattern (Phase 1: lock + check → Phase 2: HTTP proxy unlocked → Phase 3:
   lock + increment turn counter).
"""

import asyncio
import json
import logging

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


class SessionRouter:
    def __init__(self, smg_url: str):
        self.smg_url = smg_url.rstrip("/")
        self.app = FastAPI()
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(None))
        # Per-session concurrency state
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._session_turn: dict[str, int] = {}
        self._session_closing: dict[str, bool] = {}
        self._session_inflight: dict[str, int] = {}
        self._session_drained: dict[str, asyncio.Event] = {}
        self._setup_routes()

    def _setup_routes(self):
        self.app.post("/sessions")(self.create_session)
        self.app.get("/sessions/{sid}")(self.get_session)
        self.app.delete("/sessions/{sid}")(self.delete_session)
        self.app.post("/sessions/{sid}/v1/chat/completions")(self.session_chat_completions)
        # Catch-all for all other session-scoped paths (including GET for sub-paths)
        self.app.api_route(
            "/sessions/{sid}/{path:path}",
            methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        )(self.session_proxy)

    def _get_or_create_lock(self, sid: str) -> asyncio.Lock:
        if sid not in self._session_locks:
            self._session_locks[sid] = asyncio.Lock()
        return self._session_locks[sid]

    def _get_or_create_drained_event(self, sid: str) -> asyncio.Event:
        event = self._session_drained.get(sid)
        if event is None:
            event = asyncio.Event()
            event.set()
            self._session_drained[sid] = event
        return event

    async def create_session(self):
        resp = await self.client.post(f"{self.smg_url}/v1/tito/sessions")
        data = resp.json()
        sid = data.get("session_id")
        if sid:
            self._session_locks[sid] = asyncio.Lock()
            self._session_turn[sid] = 0
            self._session_closing[sid] = False
            self._session_inflight[sid] = 0
            drained = asyncio.Event()
            drained.set()
            self._session_drained[sid] = drained
        return JSONResponse(status_code=resp.status_code, content=data)

    async def get_session(self, sid: str):
        resp = await self.client.get(f"{self.smg_url}/v1/tito/sessions/{sid}")
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )

    async def delete_session(self, sid: str):
        lock = self._get_or_create_lock(sid)
        async with lock:
            self._session_closing[sid] = True
            drained = self._get_or_create_drained_event(sid)

        await drained.wait()

        async with lock:
            await self.client.delete(f"{self.smg_url}/v1/tito/sessions/{sid}")
            # Clean up all per-session state while holding the lock.
            self._session_locks.pop(sid, None)
            self._session_turn.pop(sid, None)
            self._session_closing.pop(sid, None)
            self._session_inflight.pop(sid, None)
            self._session_drained.pop(sid, None)

        return Response(status_code=204)

    async def session_chat_completions(self, sid: str, request: Request):
        lock = self._get_or_create_lock(sid)

        # ── Phase 1: lock, check closing, capture expected_turn ──────────────
        async with lock:
            if self._session_closing.get(sid, False):
                return Response(
                    content=b'{"error":"session is closing"}',
                    status_code=409,
                    media_type="application/json",
                )
            expected_turn = self._session_turn.get(sid, 0)
            self._session_inflight[sid] = self._session_inflight.get(sid, 0) + 1
            self._get_or_create_drained_event(sid).clear()

        try:
            # ── Phase 2: proxy to SMG (lock NOT held — avoids blocking other ops) ─
            body = json.loads(await request.body())
            body["logprobs"] = True
            headers = self._extract_headers(request)
            headers["x-smg-tito-session-id"] = sid
            resp = await self.client.post(
                f"{self.smg_url}/v1/chat/completions",
                content=json.dumps(body),
                headers=headers,
            )

            # ── Phase 3: lock, handle stale write, increment turn ─────────────────
            async with lock:
                if sid not in self._session_turn:
                    # Session was fully deleted while request was in-flight. Pass the
                    # response through without touching any state — no zombie entries created.
                    return Response(
                        content=resp.content,
                        status_code=resp.status_code,
                        headers=dict(resp.headers),
                    )
                current_turn = self._session_turn[sid]
                if current_turn != expected_turn:
                    # Concurrent write detected: another request already incremented the
                    # counter. Pass the response through but do not double-increment.
                    logger.warning(
                        "SessionRouter: stale write detected for sid=%s "
                        "(expected_turn=%d, current=%d); skipping increment",
                        sid,
                        expected_turn,
                        current_turn,
                    )
                else:
                    self._session_turn[sid] = current_turn + 1

            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=dict(resp.headers),
            )
        finally:
            async with lock:
                inflight = self._session_inflight.get(sid, 0)
                if inflight <= 1:
                    self._session_inflight[sid] = 0
                    self._get_or_create_drained_event(sid).set()
                else:
                    self._session_inflight[sid] = inflight - 1

    async def session_proxy(self, sid: str, path: str, request: Request):
        headers = self._extract_headers(request)
        headers["x-smg-tito-session-id"] = sid
        resp = await self.client.request(
            method=request.method,
            url=f"{self.smg_url}/{path}",
            content=await request.body(),
            headers=headers,
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )

    @staticmethod
    def _extract_headers(request: Request) -> dict:
        return {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
