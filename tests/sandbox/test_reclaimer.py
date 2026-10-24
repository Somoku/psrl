"""Reclaimer behaviour that must hold without a container daemon.

The reclaimer is the part of the sandbox stack that outlives a worker, so its
sweep rules are tested against a fake runtime rather than through Docker.
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import pytest
from psrl.sandbox import reclaimer as reclaimer_module
from psrl.sandbox.reclaimer import (
    NodeReclaimer,
    OwnedContainer,
    ReclaimOutcome,
    owner_heartbeat_age_s,
    owner_heartbeat_path,
    remove_owner_heartbeat,
    write_owner_heartbeat,
)


class FakeRuntime:
    """A container runtime whose membership and removals the test controls."""

    def __init__(self, containers: list[OwnedContainer] | None) -> None:
        self.containers = containers
        self.removed: list[str] = []
        self.unremovable: set[str] = set()
        self.list_error: Exception | None = None
        self.orphan_tasks: list[str] = []
        self.orphan_task_error: Exception | None = None
        self.orphan_task_min_age_s: float | None = None

    def list_owned(self, lease_store: str):
        if self.list_error is not None:
            raise self.list_error
        return None if self.containers is None else list(self.containers)

    def reclaim_orphan_tasks(self, *, min_age_s: float):
        self.orphan_task_min_age_s = min_age_s
        if self.orphan_task_error is not None:
            raise self.orphan_task_error
        return list(self.orphan_tasks)

    def remove_containers(self, container_ids):
        ids = list(container_ids)
        self.removed.extend(ids)
        survivors = [container_id for container_id in ids if container_id in self.unremovable]
        if self.containers is not None:
            self.containers = [
                item
                for item in self.containers
                if item.container_id in self.unremovable or item.container_id not in ids
            ]
        return survivors


def _write_heartbeat(heartbeat_dir: str, owner_id: str, age_s: float) -> None:
    write_owner_heartbeat(heartbeat_dir, owner_id)
    when = time.time() - age_s
    os.utime(owner_heartbeat_path(heartbeat_dir, owner_id), (when, when))


def _reclaimer(tmp_path, containers, **kwargs) -> tuple[NodeReclaimer, FakeRuntime]:
    runtime = FakeRuntime(containers)
    instance = NodeReclaimer(str(tmp_path / "hb"), kwargs.pop("ttl_s", 900.0), runtime, **kwargs)
    return instance, runtime


def test_heartbeat_helpers_round_trip(tmp_path) -> None:
    heartbeat_dir = str(tmp_path / "hb")
    _write_heartbeat(heartbeat_dir, "owner", age_s=30)

    age = owner_heartbeat_age_s(heartbeat_dir, "owner")

    assert age is not None and age >= 29
    remove_owner_heartbeat(heartbeat_dir, "owner")
    assert owner_heartbeat_age_s(heartbeat_dir, "owner") is None
    # Removing an absent lease is not an error: two shutdown paths may both try.
    remove_owner_heartbeat(heartbeat_dir, "owner")


def test_sweep_reaps_only_containers_with_expired_owner_leases(tmp_path) -> None:
    heartbeat_dir = str(tmp_path / "hb")
    _write_heartbeat(heartbeat_dir, "stale-owner", age_s=10_000)
    _write_heartbeat(heartbeat_dir, "fresh-owner", age_s=1)
    instance, runtime = _reclaimer(
        tmp_path,
        [
            OwnedContainer("stale-sandbox", "stale-owner", "running"),
            OwnedContainer("fresh-sandbox", "fresh-owner", "running"),
        ],
    )

    outcome = instance.sweep()

    assert outcome.removed == ("stale-sandbox",)
    assert runtime.removed == ["stale-sandbox"]
    assert outcome.remaining == 1
    assert os.path.exists(owner_heartbeat_path(heartbeat_dir, "fresh-owner"))


def test_sweep_reaps_a_container_whose_owner_never_wrote_a_lease(tmp_path) -> None:
    instance, runtime = _reclaimer(tmp_path, [OwnedContainer("orphan", "dead-actor", "running")])

    outcome = instance.sweep()

    assert outcome.removed == ("orphan",)
    assert runtime.removed == ["orphan"]


def test_an_unreachable_runtime_is_not_reported_as_an_idle_node(tmp_path) -> None:
    # A transient daemon failure must never make crash recovery exit cleanly.
    instance, _ = _reclaimer(tmp_path, None)

    outcome = instance.sweep()

    assert outcome == ReclaimOutcome(removed=(), remaining=None)


def test_a_raising_runtime_is_contained_by_the_sweep(tmp_path) -> None:
    instance, runtime = _reclaimer(tmp_path, [])
    runtime.list_error = OSError("daemon down")

    with pytest.raises(OSError):
        instance.sweep()


def test_sweep_preserves_exit_diagnostics_within_the_grace_period(tmp_path) -> None:
    # The owning session reads OOMKilled off the stopped container before deleting it.
    heartbeat_dir = str(tmp_path / "hb")
    _write_heartbeat(heartbeat_dir, "live-owner", age_s=1)
    instance, runtime = _reclaimer(
        tmp_path,
        [
            OwnedContainer("oom-killed", "live-owner", "exited"),
            OwnedContainer("healthy", "live-owner", "running"),
        ],
        stopped_grace_s=300.0,
    )

    first = instance.sweep()

    assert first.removed == ()
    assert runtime.removed == []
    assert first.remaining == 2
    assert os.path.exists(owner_heartbeat_path(heartbeat_dir, "live-owner"))


def test_sweep_reaps_a_stopped_sandbox_once_its_grace_period_expires(tmp_path) -> None:
    # A stopped container can never serve another command, so a live owner that never
    # removed it must not keep its name and writable layer forever.
    _write_heartbeat(str(tmp_path / "hb"), "live-owner", age_s=1)
    instance, runtime = _reclaimer(
        tmp_path,
        [
            OwnedContainer("abandoned", "live-owner", "exited"),
            OwnedContainer("healthy", "live-owner", "running"),
        ],
        stopped_grace_s=300.0,
    )

    instance.sweep()
    outcome = instance.sweep(now=time.time() + 600)

    assert outcome.removed == ("abandoned",)
    assert runtime.removed == ["abandoned"]
    assert outcome.remaining == 1


def test_sweep_forgets_a_container_that_stopped_and_was_replaced(tmp_path) -> None:
    # A recycled id must not inherit the previous container's stopped clock.
    _write_heartbeat(str(tmp_path / "hb"), "live-owner", age_s=1)
    instance, _ = _reclaimer(tmp_path, [OwnedContainer("reused", "live-owner", "exited")])

    instance.sweep()
    instance.runtime.containers = [OwnedContainer("reused", "live-owner", "running")]
    outcome = instance.sweep(now=time.time() + 600)

    assert outcome.removed == ()


def test_a_container_the_runtime_refuses_to_delete_stays_counted(tmp_path) -> None:
    # Reporting it as gone would let the collector exit on a node that still holds memory.
    instance, runtime = _reclaimer(tmp_path, [OwnedContainer("wedged", "dead-actor", "running")])
    runtime.unremovable = {"wedged"}

    outcome = instance.sweep()

    assert outcome.removed == ()
    assert outcome.remaining == 1


def test_sweep_prunes_an_expired_heartbeat_after_its_containers_are_gone(tmp_path) -> None:
    heartbeat_dir = str(tmp_path / "hb")
    _write_heartbeat(heartbeat_dir, "stale-owner", age_s=10_000)
    instance, _ = _reclaimer(tmp_path, [])

    instance.sweep()

    assert not os.path.exists(owner_heartbeat_path(heartbeat_dir, "stale-owner"))


def test_reclaimer_rejects_a_non_positive_lease_ttl(tmp_path) -> None:
    with pytest.raises(ValueError, match="lease_ttl_s"):
        NodeReclaimer(str(tmp_path), 0, FakeRuntime([]))


def test_the_gc_lock_is_process_scoped_and_reusable(tmp_path) -> None:
    lock_path = str(tmp_path / "gc.lock")
    first = reclaimer_module._acquire_gc_lock(lock_path)
    assert first is not None
    assert reclaimer_module._acquire_gc_lock(lock_path) is None
    reclaimer_module._release_gc_lock(first)
    second = reclaimer_module._acquire_gc_lock(lock_path)
    assert second is not None
    reclaimer_module._release_gc_lock(second)


def test_the_loop_exits_when_the_node_is_idle(monkeypatch, tmp_path) -> None:
    released: list[int] = []
    lock_handle = object()
    monkeypatch.setattr(reclaimer_module, "_acquire_gc_lock", lambda lock_path: lock_handle)
    monkeypatch.setattr(reclaimer_module, "_release_gc_lock", lambda handle: released.append(1))
    instance, _ = _reclaimer(tmp_path, [])

    assert reclaimer_module.run_reclaimer_loop(instance, interval_s=1, idle_exit_cycles=1) == 0
    assert released == [1]


def test_an_unknown_node_state_never_counts_as_an_idle_cycle(monkeypatch, tmp_path) -> None:
    # `remaining is None` means "unknown", so three unreachable sweeps must not let an
    # idle_exit_cycles of one end the collector while containers may still exist.
    outcomes = [
        ReclaimOutcome(removed=(), remaining=None),
        ReclaimOutcome(removed=(), remaining=None),
        ReclaimOutcome(removed=(), remaining=None),
        ReclaimOutcome(removed=(), remaining=0),
    ]
    monkeypatch.setattr(reclaimer_module, "_acquire_gc_lock", lambda lock_path: object())
    monkeypatch.setattr(reclaimer_module, "_release_gc_lock", lambda handle: None)
    monkeypatch.setattr(reclaimer_module.time, "sleep", lambda seconds: None)
    instance, _ = _reclaimer(tmp_path, [])

    def sweep(**kwargs):
        return outcomes.pop(0)

    monkeypatch.setattr(instance, "sweep", sweep)

    assert reclaimer_module.run_reclaimer_loop(instance, interval_s=1, idle_exit_cycles=1) == 0
    assert outcomes == []


def test_the_loop_refuses_a_grace_period_shorter_than_one_sweep(tmp_path) -> None:
    instance, _ = _reclaimer(tmp_path, [], stopped_grace_s=30.0)

    with pytest.raises(ValueError, match="stopped_grace_s"):
        reclaimer_module.run_reclaimer_loop(instance, interval_s=30)


def test_spawn_validates_interval_and_ttl(tmp_path) -> None:
    with pytest.raises(ValueError):
        reclaimer_module.spawn_node_reclaimer(str(tmp_path), lease_ttl_s=900, interval_s=0)
    with pytest.raises(ValueError):
        reclaimer_module.spawn_node_reclaimer(str(tmp_path), lease_ttl_s=0, interval_s=60)


def test_the_spawned_collector_receives_every_argument_it_needs(monkeypatch, tmp_path) -> None:
    # The collector is detached with its output discarded, so a missing argument would
    # only surface as a silently idle node.
    popen_argv: list[str] = []
    monkeypatch.setattr(
        reclaimer_module.subprocess,
        "Popen",
        lambda args, **kwargs: popen_argv.extend(args) or SimpleNamespace(poll=lambda: None),
    )

    process = reclaimer_module.spawn_node_reclaimer(
        "/lease/store",
        lease_ttl_s=120,
        interval_s=30,
        idle_exit_cycles=7,
        stopped_grace_s=333,
        docker_command=("docker", "--host", "tcp://d:1"),
    )

    assert process is not None
    assert popen_argv[1:3] == ["-m", "psrl.sandbox.reclaimer"]

    def value_after(option: str) -> str:
        return popen_argv[popen_argv.index(option) + 1]

    assert value_after("--heartbeat-dir") == "/lease/store"
    assert value_after("--lease-ttl-s") == "120"
    assert value_after("--interval-s") == "30"
    assert value_after("--idle-exit-cycles") == "7"
    assert value_after("--stopped-grace-s") == "333"
    assert popen_argv[popen_argv.index("--docker-command") + 1 :] == ["docker", "--host", "tcp://d:1"]


def test_the_module_is_a_program_so_a_collector_needs_no_inline_source() -> None:
    # The entry point is what lets a detached collector be a real command.
    assert callable(reclaimer_module.main)


def test_a_container_the_runtime_will_not_destroy_is_reported_separately(tmp_path) -> None:
    # A node that cannot confirm cleanup still holds that memory, so its owner has to
    # be able to tell an unremovable container apart from a removed one.
    instance, runtime = _reclaimer(tmp_path, [OwnedContainer("wedged", "dead-actor", "running")])
    runtime.unremovable = {"wedged"}

    outcome = instance.sweep()

    assert outcome.removed == ()
    assert outcome.unremovable == ("wedged",)
    assert outcome.remaining == 1


def test_an_unreachable_runtime_reports_every_target_as_unremoved(tmp_path) -> None:
    # A raising runtime is not evidence that anything was destroyed.
    class RaisingRuntime(FakeRuntime):
        def remove_containers(self, container_ids):
            raise OSError("daemon down")

    runtime = RaisingRuntime([OwnedContainer("wedged", "dead-actor", "running")])
    instance = NodeReclaimer(str(tmp_path / "hb"), 900.0, runtime)

    outcome = instance.sweep()

    assert outcome.removed == ()
    assert outcome.unremovable == ("wedged",)


def test_a_container_with_no_owner_is_reclaimed(tmp_path) -> None:
    """Nobody claims it, so nobody can vouch for it.

    A container labelled with no owner used to be dropped from the listing entirely, and a
    reclaimer reading an empty list concludes the node is clean. That is how a previous
    run's containers accumulated until the node could admit nothing, with no error
    anywhere: the memory was spoken for and the envelope did not know it.
    """
    heartbeat_dir = str(tmp_path / "hb")
    runtime = FakeRuntime([OwnedContainer("unclaimed", reclaimer_module.UNOWNED_OWNER_ID, "running")])
    instance = NodeReclaimer(heartbeat_dir, 900.0, runtime)

    outcome = instance.sweep()

    assert outcome.removed == ("unclaimed",)
    assert outcome.remaining == 0


def test_an_unowned_container_cannot_be_rescued_by_a_live_lease(tmp_path) -> None:
    """The sentinel owner must never match a lease, however one is written.

    If it could, a live worker's heartbeat would vouch for containers that are not its
    own, and the leak would become invisible again while looking accounted for.
    """
    heartbeat_dir = str(tmp_path / "hb")
    os.makedirs(heartbeat_dir, exist_ok=True)
    write_owner_heartbeat(heartbeat_dir, "a-real-owner")
    runtime = FakeRuntime(
        [
            OwnedContainer("owned", "a-real-owner", "running"),
            OwnedContainer("unclaimed", reclaimer_module.UNOWNED_OWNER_ID, "running"),
        ]
    )
    instance = NodeReclaimer(heartbeat_dir, 900.0, runtime)

    outcome = instance.sweep()

    assert outcome.removed == ("unclaimed",), "A live lease must not vouch for an unowned container."
    assert outcome.remaining == 1


class TestARemovalSaysWhoAndWhy:
    """A reclaimer's removal is the only one no session observes.

    The container disappears and the episode that was using it dies at its next
    command. Nothing else in the run records what happened, so the reclaimer has to
    say which owner it judged gone, on what budget, and what it destroyed.
    """

    def test_a_stale_owner_is_reported_with_its_age_and_budget(self, tmp_path, caplog):
        owner = "worker-1"
        _write_heartbeat(str(tmp_path / "hb"), owner, age_s=5000)
        reclaimer, _ = _reclaimer(tmp_path, [OwnedContainer("c1", owner, "running")], ttl_s=120)

        with caplog.at_level("WARNING"):
            reclaimer.sweep()

        lines = [record.getMessage() for record in caplog.records]
        assert any("worker-1" in line and "stale" in line and "5000s" in line and "120s" in line for line in lines), (
            lines
        )

    def test_an_owner_with_no_lease_at_all_is_reported_as_missing(self, tmp_path, caplog):
        reclaimer, _ = _reclaimer(tmp_path, [OwnedContainer("c1", "never-wrote", "running")], ttl_s=120)

        with caplog.at_level("WARNING"):
            reclaimer.sweep()

        lines = [record.getMessage() for record in caplog.records]
        assert any("never-wrote" in line and "missing" in line for line in lines), lines

    def test_the_destruction_names_the_containers_and_the_owner(self, tmp_path, caplog):
        owner = "worker-2"
        _write_heartbeat(str(tmp_path / "hb"), owner, age_s=5000)
        reclaimer, runtime = _reclaimer(tmp_path, [OwnedContainer("c9", owner, "running")], ttl_s=120)

        with caplog.at_level("WARNING"):
            reclaimer.sweep()

        lines = [record.getMessage() for record in caplog.records]
        assert any("worker-2" in line and "c9" in line for line in lines), lines
        assert runtime.removed == ["c9"]

    def test_a_live_owner_is_not_reported_and_not_touched(self, tmp_path, caplog):
        owner = "worker-3"
        _write_heartbeat(str(tmp_path / "hb"), owner, age_s=1)
        reclaimer, runtime = _reclaimer(tmp_path, [OwnedContainer("c1", owner, "running")], ttl_s=120)

        with caplog.at_level("WARNING"):
            reclaimer.sweep()

        assert not [r for r in caplog.records if "stale" in r.getMessage()]
        assert runtime.removed == []


class TestThePreflightSweep:
    """One pass, on demand, for a run that wants to start from a clean node.

    A preflight asks whether a previous run left sandboxes behind, and it has to have its
    answer before the run binds anything, so it sweeps once instead of watching the node.
    """

    def test_it_reclaims_a_previous_runs_sandboxes(self, tmp_path) -> None:
        heartbeat_dir = str(tmp_path / "hb")
        _write_heartbeat(heartbeat_dir, "dead-owner", age_s=10_000)
        reclaimer, runtime = _reclaimer(tmp_path, [OwnedContainer("c1", "dead-owner", "running")])

        outcome = reclaimer_module.sweep_once(reclaimer)

        assert outcome is not None
        assert outcome.removed == ("c1",)
        assert runtime.removed == ["c1"]

    def test_it_leaves_a_live_runs_sandboxes_alone(self, tmp_path) -> None:
        # A preflight must be safe to run while another run is alive, so it reclaims on
        # evidence of a dead owner and never on the label alone.
        heartbeat_dir = str(tmp_path / "hb")
        _write_heartbeat(heartbeat_dir, "live-owner", age_s=1)
        reclaimer, runtime = _reclaimer(tmp_path, [OwnedContainer("c1", "live-owner", "running")])

        outcome = reclaimer_module.sweep_once(reclaimer)

        assert outcome is not None
        assert outcome.removed == ()
        assert runtime.removed == []

    def test_it_defers_to_a_collector_that_holds_the_store(self, tmp_path) -> None:
        heartbeat_dir = str(tmp_path / "hb")
        _write_heartbeat(heartbeat_dir, "dead-owner", age_s=10_000)
        reclaimer, runtime = _reclaimer(tmp_path, [OwnedContainer("c1", "dead-owner", "running")])
        held = reclaimer_module._acquire_gc_lock(reclaimer_module._gc_lock_path(heartbeat_dir))
        assert held is not None
        try:
            assert reclaimer_module.sweep_once(reclaimer) is None
        finally:
            reclaimer_module._release_gc_lock(held)

        assert runtime.removed == []

    def test_the_once_switch_sweeps_one_pass_and_exits(self, tmp_path, monkeypatch, caplog) -> None:
        heartbeat_dir = str(tmp_path / "hb")
        _write_heartbeat(heartbeat_dir, "dead-owner", age_s=10_000)
        runtime = FakeRuntime([OwnedContainer("c1", "dead-owner", "running")])
        monkeypatch.setattr(reclaimer_module, "_build_runtime", lambda *args, **kwargs: runtime)

        with caplog.at_level("INFO"):
            code = reclaimer_module.main(["--once", "--heartbeat-dir", heartbeat_dir, "--lease-ttl-s", "900"])

        assert code == 0
        assert runtime.removed == ["c1"]
        assert any("Reclaimed 1 sandbox container" in record.getMessage() for record in caplog.records)
