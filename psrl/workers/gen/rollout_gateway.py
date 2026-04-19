import argparse
import asyncio
import json
import logging
import os
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from psrl.utils.common.serialization import b64_dumps, b64_loads

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class GeneratePayload(BaseModel):
    dataproto_b64: str = Field(..., description="Pickle+base64 encoded DataProto")


class RolloutGateway:
    def __init__(self, host, port, concurrency: int, n_rollout_instances: int, rollout_router):
        self.rollout_router = rollout_router
        self.host = host
        self.port = port
        self.concurrency = concurrency
        self.n_rollout_instances = n_rollout_instances

        self.engine_urls: dict[int, str] = {}
        self.engine_lock = asyncio.Lock()

        self.app = None
        self._server: uvicorn.Server | None = None
        self._serve_task: asyncio.Task | None = None

        # Reuse a single HTTP client for proxying to rollout engine servers.
        self._proxy_client: httpx.AsyncClient | None = None

    def setup_routes(self):
        """Register all gateway routes onto the given FastAPI app.

        Contract:
        - /generate and /generate_async call into the RolloutRouter actor.
        - /add_worker registers an (instance_id -> engine base_url) mapping.
        - /remove_worker removes registered engine base_url mappings.
        - All other HTTP paths are proxied to the registered engine server selected by instance_id.
        """

        self.app.add_api_route("/health", self.health_check, methods=["GET"])
        self.app.add_api_route("/add_worker", self.add_worker, methods=["POST"])
        self.app.add_api_route("/remove_worker", self.remove_worker, methods=["POST"])
        self.app.add_api_route("/generate", self.generate, methods=["POST"])
        self.app.add_api_route("/generate_async", self.generate_async, methods=["POST"])

        # Catch-all route for proxying to rollout engine - must be registered LAST
        self.app.add_api_route(
            "/{path:path}",
            self.catch_all_proxy,
            methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        )

    def run_router(self):
        self.app = FastAPI()
        self.setup_routes()
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")

    async def start(self):
        """Initialize and start the RolloutGateway HTTP server."""
        if self._serve_task is not None and not self._serve_task.done():
            return

        self.app = FastAPI()
        self.setup_routes()

        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
        self._server = uvicorn.Server(config)

        loop = asyncio.get_running_loop()
        self._serve_task = loop.create_task(self._server.serve())
        self._serve_task.add_done_callback(lambda f: f.exception())

    async def stop(self):
        """Stop the RolloutGateway HTTP server."""
        self._server.should_exit = True
        self._serve_task.cancel()

        if self._proxy_client is not None:
            await self._proxy_client.aclose()

    def get_bind(self) -> dict[str, Any]:
        """Get the RolloutGateway server bind info."""
        return {"host": self.host, "port": self.port}

    async def health_check(self) -> dict[str, bool]:
        """Simple health check endpoint."""
        # TODO(linsh) implement thorough health checks
        return {"ok": True}

    async def add_worker(self, request: Request):
        """Register a rollout engine address."""
        body = await request.body()
        payload = json.loads(body) if body else {}
        worker_url = payload.get("worker_url")
        instance_id = payload.get("instance_id")

        async with self.engine_lock:
            self.engine_urls[int(instance_id)] = worker_url.rstrip("/")
        return {"ok": True, "instance_id": int(instance_id), "worker_url": worker_url.rstrip("/")}

    async def remove_worker(self, request: Request):
        """Remove a registered rollout engine address.

        Contract:
        - If remove_all=true: clear all mappings.
        - Else instance_id must be provided and that mapping will be removed if present.

        Returns a summary including list of removed instance ids.
        """
        body = await request.body()
        payload = json.loads(body) if body else {}
        remove_all = payload.get("remove_all", False)
        instance_id = payload.get("instance_id")

        removed: list[int] = []
        async with self.engine_lock:
            if remove_all:
                removed = sorted([int(k) for k in self.engine_urls])
                self.engine_urls.clear()
            else:
                if instance_id is None:
                    raise HTTPException(status_code=400, detail="Missing `instance_id` (or set remove_all=true)")
                iid = int(instance_id)
                if iid in self.engine_urls:
                    del self.engine_urls[iid]
                    removed = [iid]

        return {"ok": True, "removed": removed}

    async def generate(self, body: GeneratePayload):
        request = b64_loads(body.dataproto_b64)
        try:
            out_ref = self.rollout_router.generate.remote(request)
            output = await out_ref
        except Exception as e:
            psrl_logger.exception("/generate failed")
            raise HTTPException(status_code=500, detail=str(e)) from e
        if output is None:
            return {"result": None}
        return {"result": b64_dumps(output)}

    async def generate_async(self, body: GeneratePayload):
        request = b64_loads(body.dataproto_b64)
        try:
            out_ref = self.rollout_router.generate_async.remote(request)
            output = await out_ref
        except Exception as e:
            psrl_logger.exception("/generate_async failed")
            raise HTTPException(status_code=500, detail=str(e)) from e
        if output is None:
            return {"result": None}
        return {"result": b64_dumps(output)}

    async def _proxy_request_to_engine(self, request: Request, path: str):
        """Proxy an incoming HTTP request to the registered rollout engine server.

        Args:
            request: The original HTTP request.
            path: The path to forward the request to.
        """
        # Lazily init a shared proxy client. This client is only used
        # by the catch-all proxy route.
        if self._proxy_client is None:
            max_connections = max(8, self.concurrency * max(1, self.n_rollout_instances))
            self._proxy_client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=max_connections),
                timeout=httpx.Timeout(None),
            )

        body = await request.body()

        # 1) Prefer instance_id from query params
        instance_id = request.query_params.get("instance_id")
        # 2) Fallback to JSON body
        if instance_id is None:
            payload = json.loads(body) if body else {}
            instance_id = payload.get("instance_id")

        if instance_id is None:
            raise HTTPException(status_code=400, detail="Missing `instance_id`")

        base_url = await self._get_engine_url(instance_id)
        url = f"{base_url}/{path}" if path else base_url

        try:
            resp = await self._proxy_client.request(
                method=request.method,
                url=url,
                params=dict(request.query_params),
                content=body,
                headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text) from e
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e)) from e

        # Eagerly read content so we can return JSON (not streaming)
        content = await resp.aread()
        content_type = resp.headers.get("content-type", "")
        try:
            # Prefer parsing JSON if possible
            data = json.loads(content)
            return JSONResponse(
                content=data,
                status_code=resp.status_code,
                headers=dict(resp.headers),
            )
        except Exception:
            # Fall back to raw body with original content type
            return Response(
                content=content,
                status_code=resp.status_code,
                headers=dict(resp.headers),
                media_type=content_type or None,
            )

    async def _get_engine_url(self, instance_id: int) -> str:
        """Get the registered engine base_url for the given instance_id."""
        async with self.engine_lock:
            url = self.engine_urls.get(instance_id)
        if not url:
            raise HTTPException(status_code=404, detail=f"Engine url for instance_id={instance_id} not found")
        return url.rstrip("/")

    async def catch_all_proxy(self, path: str, request: Request):
        """Catch-all proxy route to forward requests to registered rollout engine servers.

        Args:
            path: The path to forward the request to.
            request: The original HTTP request.
        """
        return await self._proxy_request_to_engine(request=request, path=path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--n_rollout_instances", type=int, default=1)

    args = parser.parse_args()

    # Run the router
    gateway = RolloutGateway(
        host=args.host,
        port=args.port,
        concurrency=args.concurrency,
        n_rollout_instances=args.n_rollout_instances,
        rollout_router=None,
    )
    gateway.run_router()
