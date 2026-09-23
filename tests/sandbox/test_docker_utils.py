from __future__ import annotations

import os
import subprocess
import time
from types import SimpleNamespace

import pytest
from psrl.sandbox.utils import docker_utils


def test_force_remove_batches_all_matching_containers(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[1] == "ps":
            return SimpleNamespace(returncode=0, stdout=b"first\nsecond\n", stderr=b"")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    removed = docker_utils.force_remove_containers_by_label("psrl.actor_id", "actor")

    assert removed == ["first", "second"]
    assert calls[1] == ["docker", "rm", "-f", "-v", "first", "second"]


def _write_heartbeat(heartbeat_dir: str, owner_id: str, age_s: float) -> None:
    docker_utils.write_owner_heartbeat(heartbeat_dir, owner_id)
    when = time.time() - age_s
    os.utime(docker_utils.owner_heartbeat_path(heartbeat_dir, owner_id), (when, when))


def _install_docker_list(monkeypatch, containers: list[tuple[str, str, str]]) -> list[list[str]]:
    """Fake ``docker ps``/``docker rm -f`` and return the recorded argv calls."""
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[1] == "ps":
            stdout = "".join(f"{cid}\t{actor}\t{state}\n" for cid, actor, state in containers).encode()
            return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")
        return SimpleNamespace(returncode=0, stdout=("\n".join(args[3:]) + "\n").encode(), stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_sweep_reaps_only_containers_with_expired_owner_leases(monkeypatch, tmp_path) -> None:
    heartbeat_dir = str(tmp_path / "hb")
    _write_heartbeat(heartbeat_dir, "stale-owner", age_s=10_000)
    _write_heartbeat(heartbeat_dir, "fresh-owner", age_s=1)
    containers = [
        ("stale-sandbox", "stale-owner", "running"),
        ("fresh-sandbox", "fresh-owner", "running"),
    ]
    calls = _install_docker_list(monkeypatch, containers)

    reaped, _ = docker_utils.sweep_stale_sandboxes(heartbeat_dir, ttl_s=900)

    assert reaped == ["stale-sandbox"]
    assert ["docker", "rm", "-f", "-v", "stale-sandbox"] in calls
    assert os.path.exists(docker_utils.owner_heartbeat_path(heartbeat_dir, "fresh-owner"))


def test_sweep_reaps_missing_owner_heartbeat(monkeypatch, tmp_path) -> None:
    heartbeat_dir = str(tmp_path / "hb")
    calls = _install_docker_list(monkeypatch, [("orphan", "dead-actor", "running")])

    reaped, _ = docker_utils.sweep_stale_sandboxes(heartbeat_dir, ttl_s=900)

    assert reaped == ["orphan"]
    assert ["docker", "rm", "-f", "-v", "orphan"] in calls


def test_gc_lock_is_process_scoped_and_reusable(tmp_path) -> None:
    lock_path = str(tmp_path / "gc.lock")
    first = docker_utils._acquire_gc_lock(lock_path)
    assert first is not None
    assert docker_utils._acquire_gc_lock(lock_path) is None
    docker_utils._release_gc_lock(first)
    second = docker_utils._acquire_gc_lock(lock_path)
    assert second is not None
    docker_utils._release_gc_lock(second)


def test_run_gc_loop_exits_when_node_is_idle(monkeypatch, tmp_path) -> None:
    sweeps: list[int] = []
    lock_handle = object()
    monkeypatch.setattr(docker_utils, "_acquire_gc_lock", lambda lock_path: lock_handle)
    monkeypatch.setattr(docker_utils, "_release_gc_lock", lambda handle: sweeps.append(1))
    monkeypatch.setattr(docker_utils, "sweep_stale_sandboxes", lambda *a, **k: ([], 0))

    assert docker_utils.run_gc_loop(str(tmp_path), ttl_s=900, interval_s=1, idle_exit_cycles=1) == 0
    assert sweeps == [1]  # the lock is always released


def test_spawn_node_gc_validates_interval_and_ttl(tmp_path) -> None:
    with pytest.raises(ValueError):
        docker_utils.spawn_node_gc(str(tmp_path), ttl_s=900, interval_s=0)
    with pytest.raises(ValueError):
        docker_utils.spawn_node_gc(str(tmp_path), ttl_s=0, interval_s=60)


def test_failed_docker_query_is_not_reported_as_idle(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(docker_utils, "_list_sandbox_containers", lambda command, **kwargs: None)

    reaped, remaining = docker_utils.sweep_stale_sandboxes(str(tmp_path), ttl_s=900)

    assert reaped == []
    assert remaining is None


def test_sweep_preserves_exit_diagnostics_within_the_grace_period(monkeypatch, tmp_path) -> None:
    # The owning session reads OOMKilled off the stopped container before deleting it.
    heartbeat_dir = str(tmp_path / "hb")
    _write_heartbeat(heartbeat_dir, "live-owner", age_s=1)
    calls = _install_docker_list(
        monkeypatch,
        [
            ("oom-killed", "live-owner", "exited"),
            ("healthy", "live-owner", "running"),
        ],
    )
    stopped_since: dict[str, float] = {}

    reaped, remaining = docker_utils.sweep_stale_sandboxes(
        heartbeat_dir,
        ttl_s=900,
        stopped_since=stopped_since,
        stopped_grace_s=300,
    )

    assert reaped == []
    assert ["docker", "rm", "-f", "-v", "oom-killed"] not in calls
    assert remaining == 2
    assert set(stopped_since) == {"oom-killed"}
    assert os.path.exists(docker_utils.owner_heartbeat_path(heartbeat_dir, "live-owner"))


def test_sweep_reaps_a_stopped_sandbox_once_its_grace_period_expires(monkeypatch, tmp_path) -> None:
    # A stopped container can never serve another command, so a live owner that never
    # removed it must not keep its name and writable layer forever.
    heartbeat_dir = str(tmp_path / "hb")
    _write_heartbeat(heartbeat_dir, "live-owner", age_s=1)
    calls = _install_docker_list(
        monkeypatch,
        [
            ("abandoned", "live-owner", "exited"),
            ("healthy", "live-owner", "running"),
        ],
    )
    stopped_since = {"abandoned": time.time() - 600}

    reaped, remaining = docker_utils.sweep_stale_sandboxes(
        heartbeat_dir,
        ttl_s=900,
        stopped_since=stopped_since,
        stopped_grace_s=300,
    )

    assert reaped == ["abandoned"]
    assert ["docker", "rm", "-f", "-v", "abandoned"] in calls
    assert remaining == 1
    assert stopped_since == {}


def test_sweep_forgets_a_container_that_stopped_and_was_replaced(monkeypatch, tmp_path) -> None:
    # A recycled id must not inherit the previous container's stopped clock.
    heartbeat_dir = str(tmp_path / "hb")
    _write_heartbeat(heartbeat_dir, "live-owner", age_s=1)
    _install_docker_list(monkeypatch, [("reused", "live-owner", "running")])
    stopped_since = {"reused": time.time() - 600}

    reaped, _ = docker_utils.sweep_stale_sandboxes(
        heartbeat_dir,
        ttl_s=900,
        stopped_since=stopped_since,
        stopped_grace_s=300,
    )

    assert reaped == []
    assert stopped_since == {}


def test_grace_period_reaping_is_off_without_a_caller_owned_clock(monkeypatch, tmp_path) -> None:
    heartbeat_dir = str(tmp_path / "hb")
    _write_heartbeat(heartbeat_dir, "live-owner", age_s=1)
    _install_docker_list(monkeypatch, [("stopped", "live-owner", "exited")])

    reaped, _ = docker_utils.sweep_stale_sandboxes(heartbeat_dir, ttl_s=900)

    assert reaped == []


def test_run_gc_loop_refuses_a_grace_period_shorter_than_one_sweep(tmp_path) -> None:
    with pytest.raises(ValueError):
        docker_utils.run_gc_loop(str(tmp_path), ttl_s=900, interval_s=30, stopped_grace_s=30)


def test_force_remove_container_ids_reports_only_confirmed_deletions(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[1] == "inspect":
            # "wedged" survives the removal, "gone" does not.
            return SimpleNamespace(returncode=0 if args[-1] == "wedged" else 1, stdout=b"", stderr=b"")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    removed = docker_utils.force_remove_container_ids(["gone", "wedged"])

    assert removed == ["gone"]
    assert calls[0] == ["docker", "rm", "-f", "-v", "gone", "wedged"]


def test_the_collector_child_reads_every_argument_from_the_position_it_is_passed(monkeypatch) -> None:
    # The collector is detached with its output discarded, so an argv index that shifts
    # would corrupt the docker command silently instead of failing.
    popen_argv: list[str] = []
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda args, **kwargs: popen_argv.extend(args) or SimpleNamespace(poll=lambda: None),
    )
    docker_utils.spawn_node_gc(
        "/lease/store",
        ttl_s=120,
        interval_s=30,
        idle_exit_cycles=7,
        stopped_grace_s=333,
        docker_command=("docker", "--host", "tcp://d:1"),
    )

    received: dict = {}

    def capture(heartbeat_dir, ttl_s, interval_s, **kwargs):
        received.update({"heartbeat_dir": heartbeat_dir, "ttl_s": ttl_s, "interval_s": interval_s, **kwargs})
        return 0

    monkeypatch.setattr(docker_utils, "run_gc_loop", capture)
    # argv[0] is "-c" for `python -c CODE ...`, matching what the child sees.
    monkeypatch.setattr("sys.argv", ["-c", *popen_argv[3:]])
    # The child ends in `raise SystemExit(run_gc_loop(...))`, which is its exit status.
    with pytest.raises(SystemExit) as exit_info:
        exec(docker_utils._GC_CHILD_CODE, {})  # noqa: S102

    assert exit_info.value.code == 0
    assert received == {
        "heartbeat_dir": "/lease/store",
        "ttl_s": 120.0,
        "interval_s": 30.0,
        "idle_exit_cycles": 7,
        "stopped_grace_s": 333.0,
        "docker_command": ("docker", "--host", "tcp://d:1"),
    }
