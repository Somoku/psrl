"""Lightweight session router proxy for TITO-aware chat completions.

Responsibilities:
1. Session lifecycle (create/get/delete) → proxied to SMG /v1/tito/sessions
2. Inject x-smg-tito-session-id header on session-scoped requests
3. Force logprobs=true on /v1/chat/completions (required for vLLM backend)
"""

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

    async def create_session(self):
        resp = await self.client.post(f"{self.smg_url}/v1/tito/sessions")
        return JSONResponse(status_code=resp.status_code, content=resp.json())

    async def get_session(self, sid: str):
        resp = await self.client.get(f"{self.smg_url}/v1/tito/sessions/{sid}")
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )

    async def delete_session(self, sid: str):
        await self.client.delete(f"{self.smg_url}/v1/tito/sessions/{sid}")
        return Response(status_code=204)

    async def session_chat_completions(self, sid: str, request: Request):
        body = json.loads(await request.body())
        body["logprobs"] = True
        headers = self._extract_headers(request)
        headers["x-smg-tito-session-id"] = sid
        resp = await self.client.post(
            f"{self.smg_url}/v1/chat/completions",
            content=json.dumps(body),
            headers=headers,
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )

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
