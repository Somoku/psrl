import argparse
import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastapi import FastAPI
from vllm.entrypoints.openai.api_server import build_app, init_app_state

logger = logging.getLogger(__name__)


async def build_openai_app(
    engine: Any,
    args: argparse.Namespace,
) -> FastAPI:
    """Build a vLLM OpenAI-compatible FastAPI app from an existing engine."""

    app = build_app(args)
    await init_app_state(engine, app.state, args)
    return app


@dataclass(frozen=True)
class EngineHttpBind:
    host: str
    port: int

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class EngineHttpServer:
    def __init__(
        self,
        host: str,
        port: int,
        args: argparse.Namespace,
        engine: Any,
    ):
        self.host = host
        self.port = int(port)
        self.args = args
        self.engine = engine

        self._server: uvicorn.Server | None = None
        self._serve_task: asyncio.Task | None = None

    async def start(self) -> EngineHttpBind:
        if self._serve_task is not None and not self._serve_task.done():
            return EngineHttpBind(host=self.host, port=self.port)

        app = await build_openai_app(engine=self.engine, args=self.args)
        cfg = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning")
        self._server = uvicorn.Server(cfg)

        loop = asyncio.get_running_loop()
        self._serve_task = loop.create_task(self._server.serve())
        self._serve_task.add_done_callback(lambda f: f.exception())

        logger.info("Started EngineHttpServer at %s:%s", self.host, self.port)
        return EngineHttpBind(host=self.host, port=self.port)

    async def stop(self) -> None:
        self._server.should_exit = True
        self._serve_task.cancel()
