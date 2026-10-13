"""A source that does not hold a key is skipped, not an error (item D3).

Against a PS this never came up: every PS worker holds every key, so looking up the
remote's tensor info before consulting the plan was safe by accident. Under direct
supply the sources are train ranks, and pipeline parallelism gives each rank only its
own stage's keys -- so a rollout asking rank 0 for a key on rank 12's stage hit a bare
KeyError, which reads as a missing weight rather than as normal PP sharding.

These tests pin the ordering: the key check comes before the dereference, on both the
read and the write path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_test


def _client_source() -> str:
    import psrl.utils.nixl as nixl_pkg

    return (Path(nixl_pkg.__file__).parent / "client.py").read_text()


def _body_after(source: str, marker: str) -> str:
    """The function body starting at ``marker``, up to the next def at the same level.

    Sized by the next `    def ` rather than a character count: these functions carry a
    long docstring, and a fixed window silently cut the body short -- which made the
    ordering test fail against correct code.
    """
    start = source.index(marker)
    nxt = source.find("\n    def ", start + len(marker))
    return source[start : nxt if nxt != -1 else len(source)]


class TestReadPathSkipsUnheldKeys:
    def test_the_key_is_checked_before_the_lookup(self):
        body = _body_after(_client_source(), "    def client_read(")
        guard = body.index("if key not in remote_client_info.tensor_infos:")
        lookup = body.index("remote_info = remote_client_info.get_tensor_info(key)")

        assert guard < lookup, (
            "client_read dereferences the remote tensor info before checking the key, "
            "so a pipeline rank that lacks it raises KeyError instead of being skipped"
        )

    def test_an_unheld_key_returns_no_shards(self):
        body = _body_after(_client_source(), "    def client_read(")
        after_guard = body[body.index("if key not in remote_client_info.tensor_infos:") :]

        assert after_guard.lstrip().split("\n")[1].strip() == "return []", (
            "an unheld key must yield an empty shard list, which the caller already "
            "treats as 'nothing to transfer'"
        )


class TestWritePathIsSymmetric:
    def test_the_write_path_checks_too(self):
        """Unreachable today, but the asymmetry is how the read path got this wrong."""
        body = _body_after(_client_source(), "    def client_write(")

        assert "if key not in remote_client_info.tensor_infos:" in body


class TestCallersTreatEmptyAsSkip:
    def test_the_e2e_harness_skips_empty_shard_lists(self):
        """The guard is only safe because every caller already branches on this."""
        import tests.nixl as tests_pkg

        harness = (Path(tests_pkg.__file__).parent / "test_nixl_e2e.py").read_text()
        assert "if len(shards_to_transfer) > 0:" in harness
