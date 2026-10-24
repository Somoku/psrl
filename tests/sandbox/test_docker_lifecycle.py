from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from psrl.sandbox.backends.docker.lifecycle import DockerLifecycle, DockerLifecycleConfig
from psrl.sandbox.capacity import SandboxCapacityConfig
from psrl.sandbox.reclaimer import ReclaimOutcome


@pytest.fixture(autouse=True)
def _no_real_startup_sweep(monkeypatch):
    """Keep the startup sweep off the real Docker daemon unless a test asks for it."""
    monkeypatch.setattr(
        "psrl.sandbox.backends.docker.lifecycle.NodeReclaimer",
        lambda *args, **kwargs: SimpleNamespace(sweep=lambda: ReclaimOutcome()),
    )


def test_lifecycle_restarts_an_idle_collector(monkeypatch, tmp_path) -> None:
    spawned = []
    removed = []

    def spawn(*args, **kwargs):
        process = SimpleNamespace(poll=lambda: 0)
        spawned.append(process)
        return process

    monkeypatch.setattr("psrl.sandbox.backends.docker.lifecycle.spawn_node_reclaimer", spawn)
    monkeypatch.setattr(
        "psrl.sandbox.backends.docker.lifecycle.force_remove_containers_by_label",
        lambda *args, **kwargs: removed.append((args, kwargs)),
    )
    lifecycle = DockerLifecycle(
        "worker-1",
        DockerLifecycleConfig(
            heartbeat_dir=str(tmp_path),
            heartbeat_interval_s=60,
            lease_ttl_s=480,
        ),
    )

    lifecycle.start()
    lifecycle.start()
    assert len(spawned) == 1
    lifecycle._next_gc_probe = 0.0
    lifecycle.start()
    lifecycle.close()

    assert len(spawned) == 2
    assert len(removed) == 1


def test_lifecycle_config_attaches_explicit_docker_host() -> None:
    config = DockerLifecycleConfig.from_value(None, docker_host="tcp://docker.example:2376")

    assert config.docker_command == ("docker", "--host", "tcp://docker.example:2376")


def test_typed_lifecycle_config_uses_the_engine_endpoint() -> None:
    config = DockerLifecycleConfig.from_value(DockerLifecycleConfig(), docker_host="tcp://docker.example:2376")
    assert config.docker_command == ("docker", "--host", "tcp://docker.example:2376")


def test_default_crash_recovery_precedes_capacity_expiry() -> None:
    lifecycle = DockerLifecycleConfig()
    capacity = SandboxCapacityConfig()

    assert lifecycle.lease_ttl_s + lifecycle.gc_interval_s < capacity.lease_ttl_s


def test_lifecycle_fails_when_crash_recovery_cannot_start(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("psrl.sandbox.backends.docker.lifecycle.spawn_node_reclaimer", lambda *args, **kwargs: None)
    lifecycle = DockerLifecycle(
        "worker-1",
        DockerLifecycleConfig(heartbeat_dir=str(tmp_path)),
    )

    with pytest.raises(RuntimeError, match="Could not start Docker sandbox crash recovery"):
        lifecycle.start()

    lifecycle.close()


def test_startup_reclaims_an_earlier_run_before_this_one_is_admitted(monkeypatch, tmp_path) -> None:
    # The collector would find these eventually, but not before this worker is admitted
    # against an envelope that does not know their memory is still spoken for.
    sweeps: list[dict] = []

    def build_reclaimer(heartbeat_dir, ttl_s, runtime, **kwargs):
        sweeps.append({"heartbeat_dir": heartbeat_dir, "ttl_s": ttl_s, **kwargs})
        return SimpleNamespace(sweep=lambda: ReclaimOutcome(removed=("leftover-a", "leftover-b"), remaining=0))

    monkeypatch.setattr("psrl.sandbox.backends.docker.lifecycle.NodeReclaimer", build_reclaimer)
    monkeypatch.setattr(
        "psrl.sandbox.backends.docker.lifecycle.spawn_node_reclaimer",
        lambda *args, **kwargs: SimpleNamespace(poll=lambda: None),
    )
    monkeypatch.setattr(
        "psrl.sandbox.backends.docker.lifecycle.force_remove_containers_by_label",
        lambda *args, **kwargs: [],
    )
    lifecycle = DockerLifecycle("worker-1", DockerLifecycleConfig(heartbeat_dir=str(tmp_path)))

    lifecycle.start()
    lifecycle._next_gc_probe = 0.0
    lifecycle.start()
    lifecycle.close()

    # Once per worker, not once per sandbox creation.
    assert len(sweeps) == 1
    assert sweeps[0]["heartbeat_dir"] == str(tmp_path)
    assert sweeps[0]["ttl_s"] == 120.0


def test_startup_sweep_failure_does_not_block_the_worker(monkeypatch, tmp_path) -> None:
    def build_reclaimer(*args, **kwargs):
        raise OSError("docker unreachable")

    monkeypatch.setattr("psrl.sandbox.backends.docker.lifecycle.NodeReclaimer", build_reclaimer)
    monkeypatch.setattr(
        "psrl.sandbox.backends.docker.lifecycle.spawn_node_reclaimer",
        lambda *args, **kwargs: SimpleNamespace(poll=lambda: None),
    )
    monkeypatch.setattr(
        "psrl.sandbox.backends.docker.lifecycle.force_remove_containers_by_label",
        lambda *args, **kwargs: [],
    )
    lifecycle = DockerLifecycle("worker-1", DockerLifecycleConfig(heartbeat_dir=str(tmp_path)))

    lifecycle.start()
    lifecycle.close()


def test_stopped_grace_must_outlast_one_sweep_interval() -> None:
    with pytest.raises(ValueError, match="stopped_grace_s"):
        DockerLifecycleConfig(gc_interval_s=300, stopped_grace_s=300)


def test_the_collector_receives_the_configured_grace_period(monkeypatch, tmp_path) -> None:
    spawned: list[dict] = []
    monkeypatch.setattr(
        "psrl.sandbox.backends.docker.lifecycle.spawn_node_reclaimer",
        lambda *args, **kwargs: spawned.append(kwargs) or SimpleNamespace(poll=lambda: None),
    )
    monkeypatch.setattr(
        "psrl.sandbox.backends.docker.lifecycle.force_remove_containers_by_label",
        lambda *args, **kwargs: [],
    )
    lifecycle = DockerLifecycle(
        "worker-1",
        DockerLifecycleConfig(heartbeat_dir=str(tmp_path), stopped_grace_s=600),
    )

    lifecycle.start()
    lifecycle.close()

    assert spawned[0]["stopped_grace_s"] == 600


class TestTheOwnerLeaseBudget:
    """A lease budget decides whether a running sandbox is reclaimed.

    An owner that misses its beats reads as dead, and the node then destroys every
    container that owner holds — so the budget has to absorb a heartbeat that was
    merely late (a starved writer, a filesystem hiccup, a host under pressure).
    """

    def test_the_default_budget_covers_many_beats(self):
        config = DockerLifecycleConfig(heartbeat_dir="/tmp/hb")

        assert config.lease_ttl_s >= 8 * config.heartbeat_interval_s

    def test_a_budget_too_small_to_survive_a_late_beat_is_refused(self):
        with pytest.raises(ValueError, match="must cover at least"):
            DockerLifecycleConfig(heartbeat_dir="/tmp/hb", heartbeat_interval_s=30, lease_ttl_s=120)

    def test_a_generous_budget_is_accepted(self):
        config = DockerLifecycleConfig(heartbeat_dir="/tmp/hb", heartbeat_interval_s=30, lease_ttl_s=300)

        assert config.lease_ttl_s == 300


class TestTheHeartbeatSurvivesWhatItIsFor:
    """The lease heartbeat is the worker's liveness, so nothing may stop it early."""

    def _lifecycle(self, tmp_path, monkeypatch, write):
        monkeypatch.setattr("psrl.sandbox.backends.docker.lifecycle.write_owner_heartbeat", write)
        monkeypatch.setattr(
            "psrl.sandbox.backends.docker.lifecycle.force_remove_containers_by_label",
            lambda *args, **kwargs: None,
        )
        return DockerLifecycle(
            "worker-1",
            DockerLifecycleConfig(
                heartbeat_dir=str(tmp_path),
                heartbeat_interval_s=0.01,
                lease_ttl_s=8 * 0.01,
                gc_enabled=False,
            ),
        )

    def test_a_failing_write_does_not_kill_the_heartbeat(self, monkeypatch, tmp_path):
        # A write that fails once must be retried, not end the lease: the collector
        # reads a stopped heartbeat as a dead owner and reclaims live sandboxes.
        calls = []

        def write(heartbeat_dir, owner_id):
            calls.append(owner_id)
            if len(calls) == 1:
                raise OSError("disk full")

        lifecycle = self._lifecycle(tmp_path, monkeypatch, write)
        lifecycle.start()

        deadline = time.time() + 2
        while len(calls) < 3 and time.time() < deadline:
            time.sleep(0.01)

        assert len(calls) >= 3, "the heartbeat stopped after a failed write"
        assert lifecycle._heartbeat_thread.is_alive()
        lifecycle.close()

    def test_the_lease_is_written_even_when_this_worker_collects_nothing(self, monkeypatch, tmp_path):
        # The lease says *this worker is alive*, for every other worker's collector.
        # Gating it on this process also running a collector would make a worker look
        # dead to the first collector that shares the directory.
        calls = []
        lifecycle = self._lifecycle(tmp_path, monkeypatch, lambda d, o: calls.append(o))

        assert calls == [], "the lease must not be written before start"
        lifecycle.start()

        assert calls == ["worker-1"]
        lifecycle.close()
