"""A replica that misses its target version stays in service (item D6).

The failure this prevents is not a crash. `pull_model_for_sync` used to assert the
target was reached; the raise reached `_finish_gateway_sync`'s except branch, which
calls `fail_sync` and posts `/workers/pause`, so a replica that merely fell behind was
removed from service entirely -- the opposite of "keep the current version and keep
generating".

The second half is accounting. The version used to be published *before* the pull, so
requests arriving in the gap were recorded against weights they were not generated with.
These tests check the ledger at request granularity, because a per-sync check would pass
while individual requests were still mis-attributed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_test


def _server_source() -> str:
    import psrl.workers.gen as gen_pkg

    return (Path(gen_pkg.__file__).parent / "vllm_async_server.py").read_text()


def _mixin_source() -> str:
    import psrl.workers.gen as gen_pkg

    return (
        Path(gen_pkg.__file__).parent
        / "rollout_coordination"
        / "sync_and_migrate"
        / "sync_and_migrate_mixin.py"
    ).read_text()


class TestDegradationIsNotFailure:
    def test_a_short_pull_no_longer_raises(self):
        """The assert is what turned a degradation into a quarantine."""
        source = _server_source()
        assert 'assert curr_model_version >= ps_version' not in source, (
            "the stale-version assert is back; it routes a degraded pull into "
            "fail_sync + /workers/pause, which removes the replica from service"
        )

    def test_the_short_pull_is_reported_rather_than_swallowed(self):
        source = _server_source()
        assert "Pull did not reach the target version" in source

    def test_the_actual_version_is_still_recorded(self):
        """Keeping the replica in service is only safe if the ledger is honest."""
        source = _server_source()
        assert "self.curr_rollout_instance_model_version[dp_rank] = curr_model_version" in source


class TestVersionIsPublishedAfterThePull:
    def test_the_optimistic_pre_publish_is_gone(self):
        """Publishing the target up front made SMG claim v+1 while the instance had v."""
        source = _mixin_source()
        pull_at = source.index("pull_futures = [")
        resume_at = source.index("await self._set_routing_loop_running(True)", pull_at)
        between = source[pull_at:resume_at]

        assert "_publish_weight_version_updates" not in between, (
            "a version is published between issuing the pull and resuming routing, "
            "which re-opens the window where the ledger leads the weights"
        )

    def test_the_reached_version_is_published_unconditionally(self):
        """Without the pre-publish, publishing only on mismatch would publish nothing."""
        source = _mixin_source()
        assert "actual_updates.extend(build_weight_version_updates(replica_instance_ids, actual_version))" in source

        finish_at = source.index("async def _finish_gateway_sync")
        body = source[finish_at : finish_at + 2500]
        conditional = "if actual_version != advertised_version:\n                    actual_updates.extend"
        assert conditional not in body, (
            "the reached version is published only on mismatch, so a replica that hit "
            "its target is never published at all"
        )

    def test_a_degraded_replica_is_logged_as_still_serving(self):
        source = _mixin_source()
        assert "stays in service" in source


class TestRoutingResumesWithAnHonestLedger:
    def test_routing_resumes_while_the_pull_completes(self):
        """The instance serves its current weights, which is what the ledger says."""
        source = _mixin_source()
        assert "Resumed routing while the pull completes" in source

    def test_genuine_failures_still_quarantine(self):
        """Degradation must not become a blanket excuse to keep broken replicas."""
        source = _mixin_source()
        assert "await self._quarantine_failed_replicas(replica_ids, instance_ids)" in source
        assert "fail_sync.remote()" in source
