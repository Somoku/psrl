"""The reader half of E1: a pull that does not land is re-pulled, not served.

`_pull_until_consistent` decides what a rollout does after a pull comes back torn or after
it read a version the publisher has since declared overwritten. Both mean the instance
holds bytes it must not generate from, and the only repair is another whole pull.

These tests run the method unbound against a stub, rather than importing its module: the
server module pulls in vLLM and verl, which cost more to import than the logic under test
needs. The source is extracted with `ast` so the test still exercises the shipped code
rather than a copy of it.
"""

from __future__ import annotations

import ast
import asyncio
import pathlib
import types

import pytest
from psrl.utils.nixl.pull_engine import PullVerdict, WeightRefreshFailed

pytestmark = pytest.mark.cpu_test

_SERVER = pathlib.Path(__file__).resolve().parents[2] / "psrl/workers/gen/vllm_async_server.py"


class _Logger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str) -> None:
        self.warnings.append(message)


_LOGGER = _Logger()


def _load_method():
    """Extract `_pull_until_consistent` from its class body and compile it standalone.

    `psrl_logger` is a module-level name the method reaches for, so it is supplied to the
    exec namespace; the stub reads its warnings out of `_LOGGER`.
    """
    tree = ast.parse(_SERVER.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_pull_until_consistent":
            module = ast.Module(body=[node], type_ignores=[])
            namespace: dict[str, object] = {
                "WeightRefreshFailed": WeightRefreshFailed,
                "psrl_logger": _LOGGER,
            }
            exec(compile(module, str(_SERVER), "exec"), namespace)  # noqa: S102 - own source
            return namespace["_pull_until_consistent"]
    raise AssertionError("_pull_until_consistent not found in the server module")


_PULL_UNTIL_CONSISTENT = _load_method()


class _Logger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str) -> None:
        self.warnings.append(message)


class _Engine:
    """Plays back one verdict list per attempt."""

    def __init__(self, attempts: list[list[PullVerdict]]) -> None:
        self._attempts = attempts
        self.calls: list[tuple] = []

    async def collective_rpc(self, method: str, args: tuple) -> list[PullVerdict]:
        assert method == "nixl_pull_model_core"
        self.calls.append(args)
        return self._attempts[len(self.calls) - 1]


class _PSManager:
    """Returns a fixed version state, raising if called more often than the engine."""

    def __init__(self, version: int, holders: frozenset[str], invalidated: frozenset[int]) -> None:
        self._state = (version, holders, invalidated)
        self.calls = 0

    def _get(self):
        return self._state

    def __getattr__(self, name):
        if name != "get_nixl_version_state":
            raise AttributeError(name)
        # Ray's `.remote()` is not available here; the stub is a plain attribute object.
        return _Remote(self)


class _Remote:
    def __init__(self, manager: _PSManager) -> None:
        self._manager = manager

    async def remote(self):
        self._manager.calls += 1
        return self._manager._state


class _Config:
    def __init__(self, max_attempts: int | None = None) -> None:
        self.nixl = {} if max_attempts is None else {"max_pull_attempts": max_attempts}


def _server(attempts, version=7, holders=frozenset({"t0"}), invalidated=frozenset(), max_attempts=None):
    stub = types.SimpleNamespace(
        base_worker_id="replica0",
        psrl_config=_Config(max_attempts),
        engine=_Engine(attempts),
        gen_interface=types.SimpleNamespace(
            ps_manager_handle=_PSManager(version, holders, invalidated),
        ),
        _solo_pulls=0,
        get_replica_idx=lambda: 2,
    )
    return stub


def _run(stub) -> None:
    _LOGGER.warnings.clear()
    asyncio.run(_PULL_UNTIL_CONSISTENT(stub, epoch=4, participants=[2, 3]))


def _landed(*versions: int) -> PullVerdict:
    return PullVerdict(landed=True, versions=frozenset(versions))


def _torn(*versions: int) -> PullVerdict:
    return PullVerdict(landed=False, versions=frozenset(versions), reason="torn")


def _unavailable() -> PullVerdict:
    return PullVerdict(landed=False, versions=frozenset(), reason="unavailable")


class TestASingleLandedPullIsNotRetried:
    def test_one_attempt_when_every_rank_agrees_on_a_valid_version(self):
        stub = _server([[_landed(7), _landed(7)]])
        _run(stub)
        assert len(stub.engine.calls) == 1

    def test_a_pull_that_read_no_keys_counts_as_landed(self):
        stub = _server([[_landed(), _landed()]])
        _run(stub)
        assert len(stub.engine.calls) == 1


class TestTornAndInvalidPullsAreRepaired:
    def test_a_torn_pull_is_retried_and_the_retry_is_accepted(self):
        stub = _server([[_torn(6, 7)], [_landed(7), _landed(7)]])
        _run(stub)
        assert len(stub.engine.calls) == 2
        assert "torn" in _LOGGER.warnings[0]

    def test_an_invalidated_version_is_retried(self):
        stub = _server(
            [[_landed(6), _landed(6)], [_landed(7), _landed(7)]],
            version=7,
            invalidated=frozenset({6}),
        )
        _run(stub)
        assert len(stub.engine.calls) == 2
        assert "invalidated" in _LOGGER.warnings[0]

    def test_ranks_landing_different_versions_is_torn(self):
        """One instance, so two ranks on two versions is the same mixture as one rank's."""
        stub = _server([[_landed(6), _landed(7)], [_landed(7), _landed(7)]])
        _run(stub)
        assert len(stub.engine.calls) == 2

    def test_one_torn_rank_among_unavailable_ones_still_retries(self):
        """A rank that wrote nothing does not excuse a rank that mixed bytes."""
        stub = _server([[_torn(6, 7), _unavailable()], [_landed(7), _landed(7)]])
        _run(stub)
        assert len(stub.engine.calls) == 2

    def test_the_retry_reads_the_version_state_again(self):
        """The holder set from the failed attempt is the one that was wrong."""
        stub = _server([[_torn(6, 7)], [_landed(7)]])
        _run(stub)
        assert stub.gen_interface.ps_manager_handle.calls == 2


class TestNoLiveSourceKeepsTheCurrentVersion:
    def test_unavailable_ranks_are_not_retried(self):
        stub = _server([[_unavailable(), _unavailable()]])
        _run(stub)
        assert len(stub.engine.calls) == 1
        assert "Keeping the current version" in _LOGGER.warnings[0]


class TestExhaustionFailsLoudly:
    def test_the_weights_mixing_versions_raise(self):
        stub = _server([[_torn(6, 7)] for _ in range(3)], max_attempts=3)
        with pytest.raises(WeightRefreshFailed, match="mix"):
            _run(stub)
        assert len(stub.engine.calls) == 3

    def test_every_attempt_uses_a_fresh_epoch_and_no_peers(self):
        """A reused epoch replays a stale chunk signal; an unmatched one waits out the
        peer timeout. Retries avoid both by reading everything themselves."""
        stub = _server([[_torn(6, 7)] for _ in range(3)], max_attempts=3)
        with pytest.raises(WeightRefreshFailed):
            _run(stub)
        epochs = [args[0] for args in stub.engine.calls]
        assert epochs[0] == 4, "the first attempt keeps the epoch the sync assigned"
        assert len(set(epochs)) == len(epochs), "epochs must never be reused"
        assert all(args[1] == [2] for args in stub.engine.calls[1:]), "retries read alone"

    def test_the_bound_is_configurable(self):
        stub = _server([[_torn(6, 7)] for _ in range(5)], max_attempts=5)
        with pytest.raises(WeightRefreshFailed):
            _run(stub)
        assert len(stub.engine.calls) == 5

    def test_a_bound_of_zero_still_attempts_once(self):
        stub = _server([[_landed(7)]], max_attempts=0)
        _run(stub)
        assert len(stub.engine.calls) == 1
