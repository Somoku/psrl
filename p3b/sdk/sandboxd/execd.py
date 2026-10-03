"""An execd runner for the SDK's data plane.

A sandbox created by the opensandbox backend in direct mode runs OpenSandbox's
`execd` agent, and the SDK reaches it directly: the create reply carries the
agent's address, and every command, file read, and file write goes there rather
than through the control plane. This module is what speaks that protocol.

It is injected rather than built in. `SandboxClient` takes an `agent_factory`
because a sandbox's agent is a property of the backend that made it: a bare
container has none and its node proxies instead, while a provider sandbox runs
one whose protocol the service does not need to know. Keeping execd's protocol
here means the SDK's core stays a transport, and a deployment whose agents speak
something else supplies its own factory.

Usage:

    from sandboxd import SandboxClient
    from sandboxd.execd import execd_agent_factory

    client = SandboxClient(endpoint, agent_factory=execd_agent_factory())

What execd returns is a newline-delimited event stream rather than one JSON
object, which shapes most of the code below.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Mapping
from urllib.parse import quote

import aiohttp

# execd's own routes. Commands and files are separate surfaces, and the file
# routes take their path as a query parameter rather than in the body.
_COMMAND_PATH = "/command"
_DOWNLOAD_PATH = "/files/download"
_UPLOAD_PATH = "/files/upload"

# The access token header execd requires when the platform configures one. The
# value travels with the endpoint rather than being named here, because a
# resumed sandbox can move and can require a different one.
_TOKEN_HEADER = "X-EXECD-ACCESS-TOKEN"


class ExecdError(RuntimeError):
    """Raised when the agent refused a request, rather than the workload failing."""


def execd_agent_factory(*, request_timeout_s: float = 300.0, connection_limit: int = 256):
    """Return a factory the SDK can use to reach execd.

    One session is shared across every sandbox, because a burst of sandboxes on
    one node is a burst of connections to one host and a per-sandbox pool would
    spend a handshake on each.

    Args:
        request_timeout_s: Ceiling for one request. A command carries its own
            deadline, which execd enforces in the guest; this only bounds a
            connection that stops answering entirely.
        connection_limit: Size of the shared connection pool. It has to be at
            least the peak number of concurrent sandboxes, or commands queue
            behind each other for no reason.

    Returns:
        A callable the SDK invokes with one agent endpoint. It returns a runner
        for an agent that has an address, and None for one that does not, which
        is how the SDK knows to let the owning node proxy instead.
    """
    state: dict[str, Any] = {"session": None}

    async def _session() -> aiohttp.ClientSession:
        session = state["session"]
        if session is None or session.closed:
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=request_timeout_s),
                connector=aiohttp.TCPConnector(limit=connection_limit),
            )
            state["session"] = session
        return session

    async def _close() -> None:
        session = state["session"]
        if session is not None and not session.closed:
            await session.close()
            state["session"] = None

    def factory(agent):
        address = getattr(agent, "address", "") or ""
        if not address:
            # No in-sandbox agent. The SDK falls back to the owning node, which
            # is correct for a bare container.
            return None
        headers = dict(getattr(agent, "headers", None) or {})
        return _ExecdRunner(address, headers, _session)

    # The pool outlives any one sandbox, so the caller closes it when it is done
    # with the client. Without this the loop shuts down holding open connections
    # and aiohttp reports them.
    factory.close = _close
    return factory


class _ExecdRunner:
    """Speaks execd's protocol for one sandbox."""

    def __init__(self, address: str, headers: Mapping[str, str], session_factory) -> None:
        self._base = address.rstrip("/")
        self._headers = dict(headers)
        self._session_factory = session_factory

    async def __call__(
        self, method: str, payload: Mapping[str, Any], *, timeout_s: float | None = None
    ) -> dict[str, Any]:
        """Dispatch one data-plane call.

        The SDK names the operation; this maps it onto execd's routes. An
        operation the agent has no route for is an error rather than a silent
        no-op, so a caller is never told something happened that did not.
        """
        if method == "exec":
            return await self._exec(payload, timeout_s)
        if method == "read_bytes":
            return await self._read_bytes(payload)
        if method == "write_bytes":
            return await self._write_bytes(payload)
        raise ExecdError(
            f"the sandbox agent has no route for {method!r}; it serves exec, "
            "read_bytes, and write_bytes"
        )

    async def _exec(self, payload: Mapping[str, Any], timeout_s: float | None) -> dict[str, Any]:
        """Run one command and fold the agent's event stream into one result."""
        body: dict[str, Any] = {"command": payload["command"]}
        if payload.get("cwd"):
            body["cwd"] = payload["cwd"]
        if payload.get("env"):
            body["envs"] = dict(payload["env"])
        deadline = timeout_s if timeout_s is not None else payload.get("timeout_s")
        if deadline is not None:
            # The agent's timeout is milliseconds and it is enforced in the
            # guest, which is the only deadline either side of this call can
            # honour, so it travels with the request rather than being capped
            # by the client.
            body["timeout"] = max(0, int(float(deadline) * 1000))

        session = await self._session_factory()
        stdout: list[str] = []
        stderr: list[str] = []
        exit_code = 0
        async with session.post(
            self._base + _COMMAND_PATH, json=body, headers=self._headers
        ) as response:
            if response.status >= 400:
                text = (await response.text()).strip()
                raise ExecdError(
                    f"the sandbox agent returned HTTP {response.status} for a command: {text}"
                )
            async for raw in response.content:
                event = _parse_event(raw)
                if event is None:
                    continue
                kind = str(event.get("type") or "")
                if kind == "stdout":
                    stdout.append(str(event.get("text") or ""))
                elif kind == "stderr":
                    stderr.append(str(event.get("text") or ""))
                elif kind == "result":
                    stdout.append(_result_text(event))
                elif kind == "error":
                    # The stream carries no exit code, so an error event is the
                    # only failure signal there is. A non-zero exit that emits
                    # no error event is not distinguishable from success here.
                    stderr.append(_error_text(event))
                    exit_code = 1
                elif kind == "execution_complete":
                    # Some builds report the status here. Where they do, it is
                    # better evidence than the presence of an error event.
                    reported = event.get("exit_code")
                    if reported is not None:
                        exit_code = int(reported)
        return {
            "exit_code": exit_code,
            "stdout": "".join(stdout),
            "stderr": "".join(stderr),
            "truncated": False,
        }

    async def _read_bytes(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Read one file out of the sandbox."""
        session = await self._session_factory()
        url = f"{self._base}{_DOWNLOAD_PATH}?path={quote(str(payload['path']), safe='')}"
        async with session.get(url, headers=self._headers) as response:
            if response.status >= 400:
                text = (await response.text()).strip()
                raise ExecdError(
                    f"the sandbox agent returned HTTP {response.status} reading "
                    f"{payload['path']!r}: {text}"
                )
            data = await response.read()
        # The SDK's transport carries bytes base64-encoded, so the shape matches
        # what a proxied read would have returned.
        return {"data": base64.b64encode(data).decode()}

    async def _write_bytes(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Write one file into the sandbox.

        The agent's upload is multipart: a JSON metadata part naming the
        destination, then the file itself. The path is inside the metadata rather
        than in the URL, and both parts are sent with a filename because the
        server reads metadata as a file part.
        """
        path = str(payload["path"])
        data = base64.b64decode(payload["data"])

        form = aiohttp.FormData()
        form.add_field(
            "metadata",
            json.dumps({"path": path}),
            filename="metadata.json",
            content_type="application/json",
        )
        form.add_field(
            "file",
            data,
            filename=path.rsplit("/", 1)[-1] or "file",
            content_type="application/octet-stream",
        )
        session = await self._session_factory()
        # aiohttp sets the multipart boundary, so the headers must not carry a
        # content type of their own.
        async with session.post(
            self._base + _UPLOAD_PATH, data=form, headers=self._headers
        ) as response:
            if response.status >= 400:
                text = (await response.text()).strip()
                raise ExecdError(
                    f"the sandbox agent returned HTTP {response.status} writing {path!r}: {text}"
                )
        return {}


def _parse_event(raw: bytes) -> dict[str, Any] | None:
    """Parse one frame of the agent's event stream, or return None.

    The stream is newline-delimited JSON, optionally with an SSE `data:` prefix.
    An unparsable frame is protocol noise rather than output: folding it into
    stdout would invent content the workload never wrote.
    """
    line = raw.decode(errors="replace").strip()
    if not line:
        return None
    if line.startswith("data:"):
        line = line[len("data:") :].strip()
    if not line:
        return None
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _result_text(event: Mapping[str, Any]) -> str:
    """Render a result event, whose body is a MIME map."""
    results = event.get("results")
    if isinstance(results, Mapping):
        return str(results.get("text/plain") or "")
    return str(event.get("text") or "")


def _error_text(event: Mapping[str, Any]) -> str:
    """Render an error event, which carries a name, a value, and a traceback."""
    error = event.get("error")
    if not isinstance(error, Mapping):
        return str(event.get("text") or error or "")
    parts = [str(error.get("ename") or ""), str(error.get("evalue") or "")]
    traceback = error.get("traceback")
    if isinstance(traceback, list):
        parts.append("\n".join(str(entry) for entry in traceback))
    return "\n".join(part for part in parts if part)
