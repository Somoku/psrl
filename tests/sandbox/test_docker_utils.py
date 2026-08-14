from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import mock_open

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
    assert calls[1] == ["docker", "rm", "-f", "first", "second"]


def test_reaper_closes_parent_log_descriptor(monkeypatch, tmp_path) -> None:
    log_file = mock_open()()
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: log_file)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace())

    docker_utils.spawn_actor_reaper("actor", log_dir=str(tmp_path))

    log_file.close.assert_called_once_with()


def test_reaper_rejects_busy_polling() -> None:
    with pytest.raises(ValueError, match="at least one second"):
        docker_utils.spawn_actor_reaper("actor", poll_interval=0)
