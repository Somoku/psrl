"""Control-plane round trips per weight sync (implementation item C2).

C2's criterion is that a steady-state sync costs zero rollout -> PS round trips. That is
not a property you can establish by reading the code -- the calls are spread across the
pull path, the shared-pull lock, and the version read -- so it has to be counted, and
the counter has to be trustworthy enough that a regression shows up as a failure rather
than as a slow cluster.
"""

from __future__ import annotations

import threading

import pytest
from psrl.workers.ps.control_plane_meter import ControlPlaneMeter

pytestmark = pytest.mark.cpu_test


class TestCounting:
    def test_calls_are_counted_by_name(self):
        meter = ControlPlaneMeter()
        with meter.measuring():
            meter.record("get_version")
            meter.record("get_version")
            meter.record("pull")

        assert meter.breakdown() == {"get_version": 2, "pull": 1}
        assert meter.total == 3

    def test_calls_outside_the_window_are_not_counted(self):
        """Handshake and retry traffic must not be charged to a steady-state sync."""
        meter = ControlPlaneMeter()
        meter.record("handshake")

        with meter.measuring():
            meter.record("steady_state")

        assert meter.breakdown() == {"steady_state": 1}

    def test_each_window_starts_from_zero(self):
        meter = ControlPlaneMeter()
        with meter.measuring():
            meter.record("first_sync")
        with meter.measuring():
            meter.record("second_sync")

        assert meter.breakdown() == {"second_sync": 1}

    def test_windows_do_not_nest(self):
        """Nesting would attribute an inner sync's calls to the outer one."""
        meter = ControlPlaneMeter()
        with meter.measuring():
            with pytest.raises(RuntimeError, match="does not nest"):
                with meter.measuring():
                    pass

    def test_the_window_closes_even_when_the_body_raises(self):
        meter = ControlPlaneMeter()
        with pytest.raises(ValueError):
            with meter.measuring():
                raise ValueError("sync failed")

        meter.record("after")
        assert meter.total == 0, "the meter stayed armed after an exception"


class TestBudgetAssertion:
    def test_a_free_sync_passes(self):
        meter = ControlPlaneMeter()
        with meter.measuring():
            pass
        meter.assert_steady_state_is_free()

    def test_a_round_trip_is_reported_with_its_name(self):
        """After a regression the useful question is which call came back."""
        meter = ControlPlaneMeter()
        with meter.measuring():
            meter.record("get_rollout_instance_model_version")

        with pytest.raises(AssertionError, match="get_rollout_instance_model_version"):
            meter.assert_steady_state_is_free()

    def test_a_budget_above_zero_is_respected(self):
        meter = ControlPlaneMeter()
        with meter.measuring():
            meter.record("lock")
            meter.record("unlock")

        meter.assert_steady_state_is_free(allowed=2)
        with pytest.raises(AssertionError):
            meter.assert_steady_state_is_free(allowed=1)


class TestThreadSafety:
    def test_concurrent_records_are_not_lost(self):
        """An undercount would make the criterion look satisfied when it is not."""
        meter = ControlPlaneMeter()
        per_thread = 500

        def hammer():
            for _ in range(per_thread):
                meter.record("call")

        with meter.measuring():
            threads = [threading.Thread(target=hammer) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        assert meter.total == 8 * per_thread


class TestVersionReadIsEliminated:
    """The measure C2 ranks highest: remove the call, not just speed it up."""

    def test_pull_returns_the_version_it_assigned(self):
        """Source-level check: the PS call must hand back the version.

        A round trip can only be removed if the caller can get the number another way,
        and the only honest source is the call that just set it.
        """
        from pathlib import Path

        import psrl.workers.ps as ps_pkg

        source = (Path(ps_pkg.__file__).parent / "ps_manager.py").read_text()
        marker = "def pull_model_state_dict_nixl("
        start = source.index(marker)
        body = source[start : start + 2000]
        assert "return self.model_store.version_tag" in body, (
            "pull_model_state_dict_nixl no longer returns the version, so the caller "
            "must read it back with a second round trip"
        )

    def test_caller_prefers_the_committed_version(self):
        """The sync must read the version from the commit, not fetch it again."""
        from pathlib import Path

        import psrl.workers.gen as gen_pkg

        source = (Path(gen_pkg.__file__).parent / "vllm_async_server.py").read_text()
        assert "lock.committed_version" in source, (
            "the sync no longer takes its version from the commit that rides on the "
            "lock release, so it must be fetching it with a round trip"
        )
        assert "self._last_pulled_model_version = None" in source, (
            "a version left over from the previous sync would be recorded as this one's"
        )


class TestCommitRidesOnTheLockRelease:
    """C2's second elimination: the commit and the release are one round trip.

    They are the two halves of "this pull finished" and always happen together, so
    charging two round trips for them is exactly the redundancy C2 targets.
    """

    class _FakeRemote:
        """Stands in for a Ray actor method: `.remote()` returns an awaitable."""

        def __init__(self, owner, name):
            self._owner = owner
            self._name = name

        def remote(self, *args, **kwargs):
            self._owner.calls.append(self._name)
            result = self._owner.results.get(self._name)

            async def _awaitable():
                return result

            return _awaitable()

    class FakePSManager:
        """Records which actor methods were called, in order."""

        def __init__(self, acquire_results=None):
            self.calls: list[str] = []
            self.results = {
                "_try_acquire_shared_pull_lock": True,
                "_release_shared_pull_lock": None,
                "finish_shared_pull": 7,
            }
            self._acquire_results = list(acquire_results or [])

        def __getattr__(self, name):
            return TestCommitRidesOnTheLockRelease._FakeRemote(self, name)

    @staticmethod
    async def _run(manager, commit_ids, body=None):
        from psrl.utils.ray.lock_context import shared_pull_model_context_async

        lock = shared_pull_model_context_async(manager, poll_interval=0, commit_instance_ids=commit_ids)
        async with lock:
            if body is not None:
                body()
        return lock

    def test_commit_and_release_are_one_round_trip(self):
        import asyncio

        manager = self.FakePSManager()
        lock = asyncio.run(self._run(manager, [(0, 0)]))

        assert manager.calls == ["_try_acquire_shared_pull_lock", "finish_shared_pull"]
        assert "_release_shared_pull_lock" not in manager.calls
        assert lock.committed_version == 7

    def test_without_commit_ids_the_exit_only_releases(self):
        """A caller that already committed must not commit twice."""
        import asyncio

        manager = self.FakePSManager()
        asyncio.run(self._run(manager, None))

        assert manager.calls == ["_try_acquire_shared_pull_lock", "_release_shared_pull_lock"]

    def test_a_failed_transfer_does_not_commit_its_version(self):
        """Committing after a failed pull would record weights never landed."""
        import asyncio

        manager = self.FakePSManager()

        def explode():
            raise RuntimeError("transfer failed")

        with pytest.raises(RuntimeError, match="transfer failed"):
            asyncio.run(self._run(manager, [(0, 0)], body=explode))

        assert "finish_shared_pull" not in manager.calls, "a failed pull committed its version"
        assert "_release_shared_pull_lock" in manager.calls, "the lock was not released"


class TestFinishSharedPullOrdering:
    """Commit before release, and release even if the commit fails."""

    class _Manager:
        """Minimal stand-in exercising PSManager.finish_shared_pull's real body."""

        def __init__(self, commit_raises=False):
            self._shared_pull_count = 1
            self.order: list[str] = []
            self._commit_raises = commit_raises

        def pull_model_state_dict_nixl(self, rollout_instance_ids):
            self.order.append("commit")
            if self._commit_raises:
                raise RuntimeError("commit failed")
            return 11

        def _release_shared_pull_lock(self):
            self.order.append("release")
            self._shared_pull_count -= 1

        finish_shared_pull = None  # bound below

    @classmethod
    def _bind(cls):
        from psrl.workers.ps.ps_manager import PSManager

        cls._Manager.finish_shared_pull = PSManager.finish_shared_pull
        return cls._Manager

    def test_version_is_committed_before_the_lock_is_released(self):
        """Releasing first lets a push advance the version mid-commit.

        The instance would then be recorded at a version newer than the bytes it pulled
        -- a staleness ledger that disagrees with what was generated.
        """
        manager = self._bind()()
        version = manager.finish_shared_pull([(0, 0)])

        assert manager.order == ["commit", "release"]
        assert version == 11

    def test_the_lock_is_released_even_when_the_commit_raises(self):
        """A lock never released stalls every later pull."""
        manager = self._bind()(commit_raises=True)

        with pytest.raises(RuntimeError, match="commit failed"):
            manager.finish_shared_pull([(0, 0)])

        assert manager.order == ["commit", "release"]
        assert manager._shared_pull_count == 0

    def test_no_instances_means_release_only(self):
        manager = self._bind()()
        assert manager.finish_shared_pull(None) is None
        assert manager.order == ["release"]
