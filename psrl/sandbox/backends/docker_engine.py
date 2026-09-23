"""Persistent asynchronous Docker Engine API transport."""

import asyncio
import io
import json
import os
import tarfile
import tempfile
from base64 import urlsafe_b64encode
from collections.abc import AsyncIterator, Mapping
from typing import Any, Protocol
from urllib.parse import quote

import aiohttp

from psrl.sandbox.core import SandboxTransportError

_STREAM_CHUNK_BYTES = 64 * 1024
# Archive bytes kept in memory before spilling to disk, so a routine patch read never
# touches the filesystem and a huge log never has to fit in the worker process.
_ARCHIVE_SPOOL_BYTES = 8 * 1024 * 1024


class DockerEngineError(SandboxTransportError):
    """
    Docker Engine returned an unexpected response.
    """

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"Docker Engine returned HTTP {status}: {message}.")
        self.status = status
        self.message = message


class DockerExecStream:
    """
    Decode Docker multiplexed frames while retaining a bounded output prefix.

    Payload beyond the budget is drained without allocating a full frame.
    Incomplete framing is a transport failure, never a successful truncation.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.stdout = bytearray()
        self.stderr = bytearray()
        self.truncated = False
        self._header = bytearray()
        self._remaining = 0
        self._stream = 1

    def feed(self, chunk: bytes) -> None:
        data = memoryview(chunk)
        while data:
            if self._remaining == 0:
                size = min(8 - len(self._header), len(data))
                self._header.extend(data[:size])
                data = data[size:]
                if len(self._header) < 8:
                    continue
                if self._header[0] not in (1, 2) or self._header[1:4] != b"\0\0\0":
                    raise DockerEngineError(200, "Invalid Docker exec stream header")
                self._stream = self._header[0]
                self._remaining = int.from_bytes(self._header[4:], "big")
                self._header.clear()
                continue
            size = min(self._remaining, len(data))
            room = max(0, self.limit - len(self.stdout) - len(self.stderr))
            kept = min(size, room)
            target = self.stdout if self._stream == 1 else self.stderr
            target.extend(data[:kept])
            self.truncated |= kept < size
            self._remaining -= size
            data = data[size:]

    def finish(self) -> tuple[bytes, bytes, bool]:
        if self._header or self._remaining:
            raise DockerEngineError(200, "Incomplete Docker exec stream frame")
        return bytes(self.stdout), bytes(self.stderr), self.truncated


class DockerEngine(Protocol):
    """
    Operations consumed by `DockerBackend`.
    """

    async def info(self) -> Mapping[str, Any]: ...

    async def pull_image(self, reference: str, auth: Mapping[str, str] | None = None) -> None: ...

    async def image_exists(self, reference: str) -> bool: ...

    async def create_container(self, name: str, config: Mapping[str, Any]) -> str: ...

    async def start_container(self, container_id: str) -> None: ...

    async def commit_container(self, container_id: str, repository: str, tag: str) -> str: ...

    async def inspect_container(self, container_id: str) -> Mapping[str, Any] | None: ...

    async def remove_container(self, container_id: str) -> None: ...

    async def remove_image(self, reference: str) -> None: ...

    async def pause_container(self, container_id: str) -> None: ...

    async def unpause_container(self, container_id: str) -> None: ...

    async def exec(
        self,
        container_id: str,
        command: list[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> tuple[int, bytes, bytes, bool]: ...

    async def read_file(self, container_id: str, path: str) -> bytes: ...

    async def write_file(self, container_id: str, path: str, data: bytes) -> None: ...

    async def stats(self, container_id: str) -> Mapping[str, Any]: ...

    def events(self, *, since: float | None = None) -> AsyncIterator[Mapping[str, Any]]:
        """Yield an empty marker once connected, then every container stop event."""

    async def close(self) -> None: ...


def _extract_single_file(archive_file, path: str) -> bytes:
    """
    Read the one regular file out of a Docker archive, off the event loop.
    """
    with tarfile.open(fileobj=archive_file, mode="r:*") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        if len(members) != 1:
            raise RuntimeError(f"Docker archive for {path!r} did not contain exactly one file.")
        extracted = archive.extractfile(members[0])
        if extracted is None:
            raise RuntimeError(f"Docker archive for {path!r} could not be read.")
        return extracted.read()


class DockerEngineClient:
    """
    Connection-pooled client for a local or remote Docker daemon.
    """

    def __init__(
        self,
        docker_host: str | None = None,
        *,
        request_timeout_s: float = 180.0,
        connection_limit: int = 128,
        max_exec_output_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        self.docker_host = docker_host or os.getenv("DOCKER_HOST", "unix:///var/run/docker.sock")
        self.request_timeout_s = request_timeout_s
        self.connection_limit = connection_limit
        if connection_limit < 1 or request_timeout_s <= 0 or max_exec_output_bytes < 1:
            raise ValueError("Docker connection, timeout, and output limits must be positive.")
        self.max_exec_output_bytes = max_exec_output_bytes
        self._session: aiohttp.ClientSession | None = None
        self._stream_session: aiohttp.ClientSession | None = None
        self._base_url = "http://docker"
        self._closed = False

    def _connector(self) -> aiohttp.BaseConnector:
        if self.docker_host.startswith("unix://"):
            return aiohttp.UnixConnector(path=self.docker_host.removeprefix("unix://"), limit=self.connection_limit)
        if self.docker_host.startswith("tcp://"):
            self._base_url = "http://" + self.docker_host.removeprefix("tcp://")
            return aiohttp.TCPConnector(limit=self.connection_limit)
        if self.docker_host.startswith(("http://", "https://")):
            self._base_url = self.docker_host.rstrip("/")
            return aiohttp.TCPConnector(limit=self.connection_limit)
        raise ValueError(f"Unsupported Docker host {self.docker_host!r}.")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._closed:
            raise RuntimeError("Docker Engine client is closed.")
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.request_timeout_s)
            self._session = aiohttp.ClientSession(connector=self._connector(), timeout=timeout)
        return self._session

    async def _get_stream_session(self) -> aiohttp.ClientSession:
        """
        Keep long command streams out of the lifecycle connection pool.
        """
        if self._closed:
            raise RuntimeError("Docker Engine client is closed.")
        if self._stream_session is None or self._stream_session.closed:
            self._stream_session = aiohttp.ClientSession(
                connector=self._connector(),
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=self.request_timeout_s),
            )
        return self._stream_session

    async def _request(
        self,
        method: str,
        path: str,
        *,
        expected: tuple[int, ...],
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> tuple[aiohttp.typedefs.LooseHeaders, bytes, int]:
        """
        Send a bounded-duration control request and validate its response.
        """
        session = await self._get_session()
        if timeout_s is not None:
            kwargs["timeout"] = aiohttp.ClientTimeout(total=timeout_s)
        async with session.request(method, f"{self._base_url}{path}", **kwargs) as response:
            body = await response.read()
            if response.status not in expected:
                message = body.decode(errors="replace")
                try:
                    message = str(json.loads(message).get("message", message))
                except (json.JSONDecodeError, AttributeError):
                    pass
                raise DockerEngineError(response.status, message.strip())
            return response.headers, body, response.status

    async def info(self) -> Mapping[str, Any]:
        _, body, _ = await self._request("GET", "/info", expected=(200,))
        return json.loads(body)

    async def pull_image(self, reference: str, auth: Mapping[str, str] | None = None) -> None:
        image = quote(reference, safe="")
        headers = None
        if auth:
            headers = {
                "X-Registry-Auth": urlsafe_b64encode(json.dumps(dict(auth), separators=(",", ":")).encode()).decode()
            }
        session = await self._get_session()
        async with session.post(f"{self._base_url}/images/create?fromImage={image}", headers=headers) as response:
            if response.status != 200:
                raise DockerEngineError(response.status, (await response.text()).strip())
            # Docker can report pull failures inside an HTTP 200 JSON stream.
            # Consume progress incrementally instead of retaining every layer update.
            async for line in response.content:
                if not line.strip():
                    continue
                progress = json.loads(line)
                error = progress.get("error") or (progress.get("errorDetail") or {}).get("message")
                if error:
                    raise DockerEngineError(response.status, str(error))

    async def image_exists(self, reference: str) -> bool:
        """
        Check the daemon cache without downloading or refreshing a mutable tag.
        """
        _, _, status = await self._request("GET", f"/images/{quote(reference, safe='')}/json", expected=(200, 404))
        return status == 200

    async def create_container(self, name: str, config: Mapping[str, Any]) -> str:
        path = f"/containers/create?name={quote(name, safe='')}"
        _, body, _ = await self._request("POST", path, expected=(201,), json=dict(config))
        return str(json.loads(body)["Id"])

    async def start_container(self, container_id: str) -> None:
        await self._request("POST", f"/containers/{container_id}/start", expected=(204, 304))

    async def commit_container(self, container_id: str, repository: str, tag: str) -> str:
        """Commit a running container's writable layer into a new image.

        Used by the per-image bake (git-purge derivative) and by the clean
        verifier snapshot, so a later sandbox starts from the committed state
        instead of repeating the one-time work. Returns the committed image ID.
        """
        path = (
            f"/commit?container={quote(container_id, safe='')}"
            f"&repo={quote(repository, safe='')}&tag={quote(tag, safe='')}"
        )
        # The Engine API rejects /commit unless Content-Type is application/json.
        _, body, _ = await self._request("POST", path, expected=(201,), json={})
        return str(json.loads(body)["Id"])

    async def inspect_container(self, container_id: str) -> Mapping[str, Any] | None:
        try:
            _, body, _ = await self._request("GET", f"/containers/{container_id}/json", expected=(200,))
        except DockerEngineError as exc:
            if exc.status == 404:
                return None
            raise
        return json.loads(body)

    async def remove_image(self, reference: str) -> None:
        """
        Remove an image by reference (used to clean up committed snapshots).
        """
        try:
            await self._request("DELETE", f"/images/{quote(reference, safe='')}", expected=(200,))
        except DockerEngineError as exc:
            if exc.status != 404:
                raise

    async def remove_container(self, container_id: str) -> None:
        try:
            await self._request("DELETE", f"/containers/{container_id}?force=1&v=1", expected=(204,))
        except DockerEngineError as exc:
            if exc.status != 404:
                raise

    async def pause_container(self, container_id: str) -> None:
        await self._request("POST", f"/containers/{container_id}/pause", expected=(204,))

    async def unpause_container(self, container_id: str) -> None:
        await self._request("POST", f"/containers/{container_id}/unpause", expected=(204,))

    async def _read_exec_output(self, exec_id: str, timeout_s: float | None) -> tuple[bytes, bytes, bool]:
        session = await self._get_stream_session()
        timeout = aiohttp.ClientTimeout(total=timeout_s, sock_connect=self.request_timeout_s)
        async with session.post(
            f"{self._base_url}/exec/{exec_id}/start",
            json={"Detach": False, "Tty": False},
            timeout=timeout,
        ) as response:
            if response.status != 200:
                raise DockerEngineError(response.status, (await response.text()).strip())
            decoder = DockerExecStream(self.max_exec_output_bytes)
            async for chunk in response.content.iter_chunked(64 * 1024):
                decoder.feed(chunk)
            return decoder.finish()

    async def exec(
        self,
        container_id: str,
        command: list[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> tuple[int, bytes, bytes, bool]:
        payload: dict[str, Any] = {
            "AttachStdout": True,
            "AttachStderr": True,
            "Tty": False,
            "Cmd": command,
        }
        if cwd:
            payload["WorkingDir"] = cwd
        if env:
            payload["Env"] = [f"{key}={value}" for key, value in env.items()]
        _, body, _ = await self._request(
            "POST",
            f"/containers/{container_id}/exec",
            expected=(201,),
            json=payload,
        )
        exec_id = str(json.loads(body)["Id"])
        stdout, stderr, truncated = await self._read_exec_output(exec_id, timeout_s)
        _, inspect_body, _ = await self._request("GET", f"/exec/{exec_id}/json", expected=(200,))
        inspection = json.loads(inspect_body)
        if inspection.get("Running") or inspection.get("ExitCode") is None:
            raise DockerEngineError(200, "Docker exec stream ended before a final exit status was available")
        exit_code = int(inspection["ExitCode"])
        return exit_code, stdout, stderr, truncated

    async def read_file(self, container_id: str, path: str) -> bytes:
        """Read one file out of a container without buffering the archive twice.

        The archive is streamed into a spooled file, so a multi-hundred-megabyte harness log
        costs one copy instead of the three that reading the whole response, wrapping it, and
        extracting from it used to cost.
        """
        session = await self._get_stream_session()
        url = f"{self._base_url}/containers/{container_id}/archive?path={quote(path, safe='')}"
        timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=self.request_timeout_s,
            sock_read=self.request_timeout_s,
        )
        with tempfile.SpooledTemporaryFile(max_size=_ARCHIVE_SPOOL_BYTES) as spool:
            async with session.get(url, timeout=timeout) as response:
                if response.status != 200:
                    raise DockerEngineError(response.status, (await response.text()).strip())
                async for chunk in response.content.iter_chunked(_STREAM_CHUNK_BYTES):
                    spool.write(chunk)
            spool.seek(0)
            return await asyncio.to_thread(_extract_single_file, spool, path)

    async def write_file(self, container_id: str, path: str, data: bytes) -> None:
        directory, _, filename = path.rpartition("/")
        directory = directory or "/"
        if not filename:
            raise ValueError("Sandbox file path must name a file.")
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            info = tarfile.TarInfo(filename)
            info.size = len(data)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(data))
        archive_path = quote(directory, safe="")
        await self._request(
            "PUT",
            f"/containers/{container_id}/archive?path={archive_path}",
            expected=(200,),
            data=buffer.getvalue(),
            headers={"Content-Type": "application/x-tar"},
        )

    async def stats(self, container_id: str) -> Mapping[str, Any]:
        _, body, _ = await self._request(
            "GET",
            f"/containers/{container_id}/stats?stream=false&one-shot=true",
            expected=(200,),
        )
        return json.loads(body)

    async def events(self, *, since: float | None = None) -> AsyncIterator[Mapping[str, Any]]:
        """Yield an empty marker once connected, then every container stop event.

        The marker exists because a healthy stream is silent until something stops, so a
        consumer would otherwise be unable to tell "connected and idle" from "still
        connecting" and would have to block until the node happened to produce an event.

        `since` replays what the daemon recorded while nothing was listening, which is how a
        reconnect recovers a stop from the gap. No read timeout is set, for the same reason
        the marker is needed.
        """
        filters = json.dumps({"type": ["container"], "event": ["die", "oom", "destroy"]}, separators=(",", ":"))
        path = f"/events?filters={quote(filters, safe='')}"
        if since is not None:
            path += f"&since={since:f}"
        session = await self._get_stream_session()
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=self.request_timeout_s)
        async with session.get(f"{self._base_url}{path}", timeout=timeout) as response:
            if response.status != 200:
                raise DockerEngineError(response.status, (await response.text()).strip())
            yield {}
            async for line in response.content:
                if not line.strip():
                    continue
                yield json.loads(line)

    async def close(self) -> None:
        self._closed = True
        if self._session is not None:
            await self._session.close()
            self._session = None
        if self._stream_session is not None:
            await self._stream_session.close()
            self._stream_session = None
