from psrl.sandbox.backends.docker_engine import DockerEngineClient


def _frame(stream: int, payload: bytes) -> bytes:
    return bytes([stream, 0, 0, 0]) + len(payload).to_bytes(4, "big") + payload


def test_docker_exec_stream_demultiplexing() -> None:
    body = _frame(1, b"out-1") + _frame(2, b"err") + _frame(1, b"out-2")

    stdout, stderr, truncated = DockerEngineClient._demultiplex_exec(body)

    assert stdout == b"out-1out-2"
    assert stderr == b"err"
    assert truncated is False


def test_docker_exec_stream_drops_a_partial_frame() -> None:
    """A body cut mid-frame reports truncation instead of leaking framing bytes as output."""
    body = _frame(1, b"complete") + bytes([1, 0, 0, 0]) + (10).to_bytes(4, "big") + b"part"

    stdout, stderr, truncated = DockerEngineClient._demultiplex_exec(body)

    assert stdout == b"complete"
    assert stderr == b""
    assert truncated is True
