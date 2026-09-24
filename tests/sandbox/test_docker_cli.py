"""Docker CLI helpers and the runtime adapter the reclaimer drives."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

from psrl.sandbox.backends.docker import cli
from psrl.sandbox.reclaimer import OwnedContainer


def test_force_remove_batches_all_matching_containers(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[1] == "ps":
            return SimpleNamespace(returncode=0, stdout=b"first\nsecond\n", stderr=b"")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    removed = cli.force_remove_containers_by_label("psrl.actor_id", "actor")

    assert removed == ["first", "second"]
    assert calls[1] == ["docker", "rm", "-f", "-v", "first", "second"]


def test_force_remove_container_ids_reports_only_confirmed_deletions(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[1] == "inspect":
            # "wedged" survives the removal, "gone" does not.
            return SimpleNamespace(returncode=0 if args[-1] == "wedged" else 1, stdout=b"", stderr=b"")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    removed = cli.force_remove_container_ids(["gone", "wedged"])

    assert removed == ["gone"]
    assert calls[0] == ["docker", "rm", "-f", "-v", "gone", "wedged"]


def _install_docker(monkeypatch, containers: list[tuple[str, str, str]]) -> list[list[str]]:
    """Fake ``docker ps``/``docker rm``/``docker inspect`` and record argv."""
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[1] == "ps":
            stdout = "".join(f"{cid}\t{owner}\t{state}\n" for cid, owner, state in containers).encode()
            return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")
        if args[1] == "inspect":
            return SimpleNamespace(returncode=1, stdout=b"", stderr=b"")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_the_runtime_lists_only_containers_in_its_own_lease_store(monkeypatch) -> None:
    calls = _install_docker(
        monkeypatch,
        [("sandbox-a", "owner-a", "Running"), ("sandbox-b", "owner-b", "Paused")],
    )
    runtime = cli.DockerContainerRuntime()

    containers = runtime.list_owned("store-1")

    assert containers == [
        OwnedContainer("sandbox-a", "owner-a", "running"),
        OwnedContainer("sandbox-b", "owner-b", "paused"),
    ]
    assert "label=psrl.sandbox=true" in calls[0]
    assert "label=psrl.lease_store=store-1" in calls[0]


def test_the_runtime_reports_an_unreachable_daemon_as_unknown(monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=b"", stderr=b"cannot connect"),
    )

    assert cli.DockerContainerRuntime().list_owned("store-1") is None


def test_the_runtime_reports_an_unremovable_container_as_still_present(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[1] == "inspect":
            # The container survives the removal, so it is still listed.
            return SimpleNamespace(returncode=0, stdout=b"id", stderr=b"")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert cli.DockerContainerRuntime().remove_containers(["wedged"]) == ["wedged"]


def test_the_dangling_image_prune_is_bounded_to_untagged_images(monkeypatch) -> None:
    calls: list[list[str]] = []
    listed: list[bytes] = [b"dead-image\n", b""]

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[1] == "images":
            return SimpleNamespace(returncode=0, stdout=listed.pop(0), stderr=b"")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    cli.prune_dangling_images(min_interval_secs=0.0)

    assert ["docker", "images", "-f", "dangling=true", "-q"] in calls
    assert ["docker", "rmi", "-f", "dead-image"] in calls
