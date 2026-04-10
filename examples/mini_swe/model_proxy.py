"""
Model Proxy Server for mini-SWE-agent.

This module provides a lightweight HTTP proxy server that intercepts OpenAI-compatible
API calls from mini-SWE-agent and forwards them to PSRL for processing.

The proxy implements an "anti-call" mechanism:
- mini-SWE-agent calls ``/v1/chat/completions`` -> proxy suspends the request.
- PSRL calls ``get_request()`` to retrieve the request.
- PSRL generates a response and calls ``send_response()``.
- Proxy returns the OpenAI-format response to mini-SWE-agent.
"""

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from aiohttp import web

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@dataclass
class ModelRequest:
    """
    Represents a model call request from mini-SWE-agent.
    """

    request_id: str
    messages: list[dict[str, Any]]  # OpenAI format messages
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    extra_params: dict[str, Any] | None = None


@dataclass
class ResponseState:
    """
    Tracks the lifecycle of a pending proxy response.
    """

    event: asyncio.Event
    state: Literal["pending", "completed", "failed"] = "pending"
    response_data: dict[str, Any] | None = None
    error_message: str | None = None


class ModelProxy:
    """
    Model call proxy server for intercepting mini-SWE-agent's OpenAI API calls.

    This proxy server:
    1. Listens on a configurable port for OpenAI-compatible requests.
    2. Suspends incoming requests and queues them for PSRL processing.
    3. Provides control interfaces for PSRL to retrieve requests and send responses.
    4. Returns responses to mini-SWE-agent in OpenAI format.

    Usage:
        .. code-block:: python

            proxy = ModelProxy()
            await proxy.start_server(port=0)

            # In PSRL loop:
            request = await proxy.get_request()
            response = await generate_response(request.messages)
            await proxy.send_response(response, request=request)

            await proxy.stop_server()
    """

    def __init__(self, port: int = 0, host: str = "127.0.0.1"):
        """
        Initialize the model proxy.

        Args:
            port (int): Port to bind the HTTP server to. Defaults to 0 (let OS assign).
            host (str): Host address to bind to. Defaults to "127.0.0.1" (localhost only).
        """
        self.port = port
        self.host = host

        # Request queue: stores `ModelRequest` objects waiting for PSRL processing.
        self.request_queue: asyncio.Queue[ModelRequest] = asyncio.Queue()

        # Response storage: maps request_id -> response state.
        self.response_storage: dict[str, ResponseState] = {}

        # Server components.
        self.app: web.Application | None = None
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None

        # Server state.
        self._server_started = False
        self._stopping = False
        self._lock = asyncio.Lock()

    async def start_server(self, port: int | None = None, max_retries: int = 1000) -> None:
        """
        Start the HTTP proxy server.

        If ``port == 0``, the OS picks an available ephemeral port atomically
        (recommended for high-concurrency startup). For fixed ports (``port > 0``),
        the server falls back to linear probing (port+1, port+2, ...).

        The default ``max_retries=1000`` covers large single-node deployments
        (e.g. hundreds of rollout workers).  Users can override this via
        ``proxy_config.max_port_retries`` in the YAML config.

        Args:
            port (int | None): Optional port override. If None, uses self.port.
            max_retries (int): Maximum number of consecutive ports to try when
                ``port > 0``. Defaults to 1000.

        Raises:
            RuntimeError: If server is already started or cannot find
                an available port within ``max_retries`` attempts.
        """
        async with self._lock:
            if self._server_started:
                raise RuntimeError("Server is already started.")

            if port is not None:
                self.port = port

            self._stopping = False

            # Try to bind to port.
            initial_port = self.port
            for attempt in range(max_retries):
                try:
                    # Create aiohttp application.
                    self.app = web.Application()
                    self.app.router.add_post("/v1/chat/completions", self._handle_chat_completion)

                    # Health check endpoint.
                    self.app.router.add_get("/health", self._handle_health)

                    # Setup runner and site.
                    self.runner = web.AppRunner(self.app)
                    await self.runner.setup()
                    self.site = web.TCPSite(self.runner, self.host, self.port)
                    await self.site.start()

                    # If binding to port 0, capture the actual assigned port.
                    self.port = self._resolve_bound_port()

                    self._server_started = True
                    psrl_logger.info(f"Model proxy server started on {self.host}:{self.port}.")
                    return

                except OSError as e:
                    if e.errno == 98 and self.port > 0:  # Address already in use.
                        psrl_logger.warning(f"Port {self.port} already in use, trying port {self.port + 1}.")
                        self.port += 1

                        # Cleanup failed attempt.
                        if self.runner:
                            await self.runner.cleanup()
                            self.runner = None
                        self.app = None
                        self.site = None
                    else:
                        raise

            # If we exhausted all retries.
            raise RuntimeError(
                f"Failed to start server after {max_retries} attempts. "
                f"Tried ports {initial_port} to {self.port - 1}."
            )

    def _resolve_bound_port(self) -> int:
        """
        Resolve actual bound port from aiohttp site after start().
        """
        if self.site is None:
            raise RuntimeError("Server site is not initialized.")

        server = getattr(self.site, "_server", None)
        sockets = getattr(server, "sockets", None)
        if not sockets:
            raise RuntimeError("Failed to resolve bound proxy port.")

        return int(sockets[0].getsockname()[1])

    async def stop_server(self) -> None:
        """
        Stop the HTTP proxy server.

        This method gracefully shuts down the server and cleans up resources.
        """
        async with self._lock:
            if not self._server_started:
                psrl_logger.warning("Server is not started, skipping stop.")
                return

            self._stopping = True
            self._fail_pending_requests("Model proxy server stopped")

            if self.site is not None:
                await self.site.stop()
                psrl_logger.info("Server site stopped.")

            if self.runner is not None:
                await self.runner.cleanup()
                psrl_logger.info("Server runner cleaned up.")

            # Clear pending state to avoid leaking across runs.
            self.request_queue = asyncio.Queue()
            self.response_storage.clear()

            self._server_started = False
            self._stopping = False
            psrl_logger.info("Model proxy server stopped.")

    def _fail_pending_requests(self, error_message: str) -> None:
        """
        Fail and wake all pending requests.
        """
        for response_state in self.response_storage.values():
            if response_state.state != "pending":
                continue
            response_state.state = "failed"
            response_state.error_message = error_message
            response_state.event.set()

    async def _handle_health(self, request: web.Request) -> web.Response:
        """
        Health check endpoint.
        """
        return web.json_response({"status": "ok", "service": "model_proxy"})

    async def _handle_chat_completion(self, request: web.Request) -> web.Response:
        """
        Handle OpenAI-compatible chat completion requests from mini-SWE-agent.

        This method:
        1. Parses the incoming request.
        2. Creates a `ModelRequest` and queues it for PSRL processing.
        3. Waits for PSRL to provide a response via ``send_response()``.
        4. Returns the response in OpenAI format.

        Args:
            request (web.Request): aiohttp request object containing the chat completion request.

        Returns:
            web.Response: JSON response in OpenAI format.
        """
        request_id: str | None = None
        try:
            if self._stopping:
                return web.json_response(
                    {"error": {"message": "Model proxy server is stopping", "type": "server_error"}}, status=503
                )

            # Parse request body.
            data = await request.json()

            # Extract messages (required).
            messages = data.get("messages", [])
            if not messages:
                return web.json_response(
                    {"error": {"message": "messages field is required", "type": "invalid_request_error"}}, status=400
                )

            # Generate unique request ID.
            request_id = str(uuid.uuid4())

            # Extract other parameters.
            model = data.get("model")
            temperature = data.get("temperature")
            max_tokens = data.get("max_tokens")
            stream = data.get("stream", False)

            if stream:
                return web.json_response(
                    {
                        "error": {
                            "message": "Streaming is not supported by ModelProxy",
                            "type": "invalid_request_error",
                        }
                    },
                    status=400,
                )

            # Store extra parameters.
            extra_params = {
                k: v
                for k, v in data.items()
                if k not in ["messages", "model", "temperature", "max_tokens", "stream"]
            }

            # Create `ModelRequest`.
            model_request = ModelRequest(
                request_id=request_id,
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                extra_params=extra_params,
            )

            psrl_logger.debug(f"Received request {request_id} with {len(messages)} messages.")

            # Create response event for this request.
            response_state = ResponseState(event=asyncio.Event())
            self.response_storage[request_id] = response_state

            # Queue the request for PSRL processing.
            await self.request_queue.put(model_request)

            # Wait for PSRL to provide response.
            await response_state.event.wait()

            # Retrieve response.
            final_state = self.response_storage.pop(request_id, None)

            if final_state is None:
                psrl_logger.error(f"No response state for request {request_id}.")
                return web.json_response(
                    {"error": {"message": "Internal server error: no response generated", "type": "server_error"}},
                    status=500,
                )

            if final_state.state == "failed":
                error_message = final_state.error_message or "Model proxy request failed"
                psrl_logger.warning(f"Request {request_id} failed: {error_message}.")
                return web.json_response(
                    {"error": {"message": error_message, "type": "server_error"}},
                    status=503 if self._stopping else 500,
                )

            if final_state.state != "completed" or final_state.response_data is None:
                psrl_logger.error(f"Invalid response state for request {request_id}: {final_state.state!r}.")
                return web.json_response(
                    {"error": {"message": "Internal server error: invalid response state", "type": "server_error"}},
                    status=500,
                )

            # Return OpenAI-format response.
            return web.json_response(final_state.response_data)

        except asyncio.CancelledError:
            if request_id is not None:
                self.response_storage.pop(request_id, None)
            psrl_logger.warning("Request cancelled.")
            raise
        except Exception as e:
            psrl_logger.exception(f"Error handling chat completion request: {e}.")
            return web.json_response(
                {"error": {"message": f"Internal server error: {e!s}", "type": "server_error"}}, status=500
            )

    async def get_request(self) -> ModelRequest:
        """
        Get the next model call request from the queue.

        This method is called by PSRL to retrieve the next request from mini-SWE-agent.
        It blocks until a request is available.

        Returns:
            ModelRequest: Object containing the request details.
        """
        request = await self.request_queue.get()
        psrl_logger.debug(f"Retrieved request {request.request_id} from queue.")
        return request

    async def send_response(
        self,
        response: str,
        request: ModelRequest | None = None,
        request_id: str | None = None,
        finish_reason: str = "stop",
    ) -> None:
        """
        Send a response back to mini-SWE-agent for a specific request.

        This method is called by PSRL after generating a response. It formats the
        response in OpenAI format and signals the waiting request handler.

        Args:
            response (str): The generated response text.
            request (ModelRequest | None): Optional `ModelRequest` object.
                If provided, uses its request_id.
            request_id (str | None): Optional request ID. Required if request is not provided.
            finish_reason (str): Finish reason for the response. Defaults to "stop".

        Raises:
            KeyError: If request_id is not found in response storage.
            ValueError: If neither request nor request_id is provided.
        """
        # Determine request_id.
        if request is not None:
            request_id = request.request_id
        elif request_id is None:
            raise ValueError("Either request or request_id must be provided.")

        if request_id not in self.response_storage:
            raise KeyError(f"Request ID {request_id} not found in response storage.")

        response_state = self.response_storage[request_id]
        if response_state.state != "pending":
            raise RuntimeError(f"Request ID {request_id} is already in state {response_state.state!r}.")

        # Format response in OpenAI format.
        response_data = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "swe-agent-proxy",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

        # Store response and signal event.
        response_state.state = "completed"
        response_state.response_data = response_data
        response_state.error_message = None
        response_state.event.set()

        psrl_logger.debug(f"Sent response for request {request_id}.")
