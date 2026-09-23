import io
import tarfile
import tracemalloc
from types import SimpleNamespace

import pytest
from psrl.sandbox.backends.docker_engine import DockerEngineClient, DockerEngineError, DockerExecStream


def _frame(stream: int, payload: bytes) -> bytes:
    return bytes([stream, 0, 0, 0]) + len(payload).to_bytes(4, "big") + payload


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 8, 31, 4096])
@pytest.mark.parametrize("budget", [1, 5, 10, 100])
def test_stream_decodes_arbitrary_boundaries_and_retains_payload_prefix(chunk_size, budget):
    body = _frame(1, b"out-1") + _frame(2, b"err") + _frame(1, b"out-2")
    decoder = DockerExecStream(budget)
    for offset in range(0, len(body), chunk_size):
        decoder.feed(body[offset : offset + chunk_size])
    stdout, stderr, truncated = decoder.finish()
    assert stdout == b"out-1"[:budget] + b"out-2"[: max(0, budget - 8)]
    assert stderr == b"err"[: max(0, budget - 5)]
    assert truncated == (budget < 13)
    assert len(stdout) + len(stderr) <= budget


@pytest.mark.parametrize("cut", [1, 4, 7, 9, 10])
def test_incomplete_frame_is_a_transport_failure(cut):
    decoder = DockerExecStream(1)
    decoder.feed(_frame(1, b"hello")[:cut])
    with pytest.raises(DockerEngineError, match="Incomplete"):
        decoder.finish()


def test_large_first_frame_does_not_leak_framing_bytes():
    decoder = DockerExecStream(3)
    decoder.feed(_frame(1, b"x" * 10000))
    assert decoder.finish() == (b"xxx", b"", True)


def test_invalid_stream_header_is_rejected():
    decoder = DockerExecStream(100)
    with pytest.raises(DockerEngineError, match="Invalid"):
        decoder.feed(b"raw text")


def _tar(name: str, payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


class _ArchiveResponse:
    """Model the archive endpoint's chunked response without a daemon."""

    def __init__(self, body: bytes, status: int = 200) -> None:
        self.status = status
        self._body = body
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def text(self) -> str:
        return "not found"

    async def iter_chunked(self, size: int):
        for offset in range(0, len(self._body), size):
            yield self._body[offset : offset + size]


def _engine_serving(body: bytes, status: int = 200) -> DockerEngineClient:
    engine = DockerEngineClient()

    async def stream_session():
        return SimpleNamespace(get=lambda *args, **kwargs: _ArchiveResponse(body, status))

    engine._get_stream_session = stream_session
    return engine


async def test_reading_a_file_costs_one_copy_of_it():
    # Reading the response, wrapping it, and extracting from it used to cost three copies,
    # which is what a several-hundred-megabyte harness log needs to not do in one worker.
    payload = b"y" * (40 * 1024 * 1024)
    engine = _engine_serving(_tar("big", payload))

    tracemalloc.start()
    try:
        content = await engine.read_file("container", "/big")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert content == payload
    assert peak / len(payload) < 1.6


@pytest.mark.parametrize("size", [0, 1, 1024, 3 * 1024 * 1024])
async def test_reading_a_file_round_trips_exactly(size):
    payload = b"p" * size
    engine = _engine_serving(_tar("f", payload))

    assert await engine.read_file("container", "/f") == payload


async def test_a_failed_archive_request_is_a_transport_failure():
    engine = _engine_serving(b"", status=404)

    with pytest.raises(DockerEngineError):
        await engine.read_file("container", "/missing")


async def test_an_archive_that_is_not_one_file_is_rejected():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in ("a", "b"):
            info = tarfile.TarInfo(name)
            info.size = 1
            archive.addfile(info, io.BytesIO(b"z"))
    engine = _engine_serving(buffer.getvalue())

    with pytest.raises(RuntimeError, match="exactly one file"):
        await engine.read_file("container", "/dir")
