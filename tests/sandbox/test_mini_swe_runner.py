from __future__ import annotations

import threading
from typing import Any

import pytest
from examples.mini_swe.harness_adapter import MiniSWEAgentAdapter, MiniSWEAgentConfig, RunnerCancelled
from examples.mini_swe.runner import (
    _capture_resource_metrics,
    _snapshot_compatible,
    build_grader_spec,
    build_sandbox_spec,
    grade_patch,
    parse_duration_seconds,
    resolve_container_config,
)
from psrl.sandbox import ExecResult, SandboxRef, SandboxSource, SandboxSpec


def _payload() -> dict[str, Any]:
    return {
        "runtime_config": {
            "sandbox_config": {
                "backend": "fake",
                "policy_profile": "mini-swe",
                "environment": {
                    "image": "swe-image",
                    "cwd": "/testbed",
                    "forward_env": [],
                    "container_timeout": "2h",
                    "env": {"SHARED": "base", "OVERRIDE": "base"},
                },
                "rollout_environment": {
                    "memory": "8g",
                    "env": {"OVERRIDE": "rollout", "ROLLOUT_ONLY": "1"},
                },
                "grader_environment": {
                    "memory": "30g",
                    "timeout": 900,
                    "env": {"OVERRIDE": "grader", "GRADER_ONLY": "1"},
                },
            }
        },
        "observation": {
            "swe_task_id": "task-1",
            "use_preexisting_repo": False,
            "repo_path": "/tmp/repo",
        },
    }


def test_mini_swe_runner_maps_task_data_to_portable_specs() -> None:
    payload = _payload()

    rollout = build_sandbox_spec(payload)
    grader = build_sandbox_spec(payload, grading=True)

    assert rollout.resources.memory_mb == 8 * 1024
    assert rollout.mounts[0].target == "/testbed"
    assert rollout.policy_profile == "mini-swe"
    assert grader.resources.memory_mb == 30 * 1024
    assert not grader.mounts
    assert grader.metadata["psrl.grader_task_id"] == "task-1__eval"
    assert rollout.env["SHARED"] == grader.env["SHARED"] == "base"
    assert rollout.env["OVERRIDE"] == "rollout"
    assert grader.env["OVERRIDE"] == "grader"
    assert "GRADER_ONLY" not in rollout.env
    assert "ROLLOUT_ONLY" not in grader.env


def test_mini_swe_runner_preserves_forward_env_and_docker_memory_units() -> None:
    payload = _payload()
    payload["runtime_config"]["sandbox_config"]["environment"].update(
        {
            "forward_env": ["CUSTOM_TOKEN", "HTTP_PROXY"],
            "memory": 8 * 1024 * 1024 * 1024,
        }
    )

    settings = resolve_container_config(payload)
    spec = build_sandbox_spec(payload)

    assert settings["forward_env"].count("HTTP_PROXY") == 1
    assert "CUSTOM_TOKEN" in settings["forward_env"]
    assert spec.resources.memory_mb == 8 * 1024
    assert parse_duration_seconds("  ") is None


def test_mini_swe_runner_maps_provider_template_without_fixed_resource_override() -> None:
    payload = _payload()
    environment = payload["runtime_config"]["sandbox_config"]["environment"]
    environment["template"] = "microvm-template"
    payload["runtime_config"]["sandbox_config"]["rollout_environment"]["memory"] = None
    payload["runtime_config"]["sandbox_config"]["grader_environment"]["memory"] = None
    payload["runtime_config"]["sandbox_config"]["policy_profile"] = None

    rollout = build_sandbox_spec(payload)
    grader = build_sandbox_spec(payload, grading=True, template="microvm-template")

    assert rollout.source == SandboxSource.template("microvm-template")
    assert rollout.resources.memory_mb is None
    assert grader.source == SandboxSource.template("microvm-template")
    assert grader.resources.memory_mb is None


def test_verifier_snapshot_requires_matching_runtime_resources() -> None:
    payload = _payload()
    rollout = build_sandbox_spec(payload)
    grader = build_sandbox_spec(payload, grading=True)

    assert not _snapshot_compatible(rollout, grader)

    payload["runtime_config"]["sandbox_config"]["grader_environment"]["memory"] = "8g"
    grader = build_sandbox_spec(payload, grading=True)

    assert _snapshot_compatible(rollout, grader)


def test_ungraded_minisweagent_task_does_not_require_swebench_metadata() -> None:
    payload = _payload()

    assert build_grader_spec(payload) is None
    assert grade_patch(payload, "diff --git a/a.py b/a.py\n", object()) is None


def test_optional_resource_metrics_do_not_fail_rollout() -> None:
    class BrokenSession:
        def stats(self):
            raise RuntimeError("metrics unavailable")

    class FakeHarness:
        session = BrokenSession()

    timing: dict[str, float] = {}

    assert not _capture_resource_metrics(FakeHarness(), timing)
    assert timing == {}


def test_cancelled_harness_stops_before_starting_another_command() -> None:
    cancel_event = threading.Event()
    cancel_event.set()
    adapter = MiniSWEAgentAdapter(
        object(),
        MiniSWEAgentConfig(image="image"),
        cancel_event=cancel_event,
    )

    with pytest.raises(RunnerCancelled):
        adapter.execute({"command": "echo should-not-run"})


def test_fresh_grader_uses_generic_sandbox_data_plane(monkeypatch) -> None:
    from examples.mini_swe import swebench_grader

    class FakeSession:
        def __init__(self) -> None:
            self.ref = SandboxRef("fake", "grader-session")
            self.writes: dict[str, bytes] = {}
            self.commands: list[str] = []
            self.released = False

        def exec(self, command: str, **kwargs: Any) -> ExecResult:
            self.commands.append(command)
            return ExecResult(0, "tests passed", "")

        def write_bytes(self, path: str, data: bytes) -> None:
            self.writes[path] = data

        def close(self) -> None:
            self.released = True

    class FakeSyncSandbox:
        def __init__(self) -> None:
            self.session = FakeSession()
            self.specs: list[SandboxSpec] = []

        def create(self, spec: SandboxSpec) -> FakeSession:
            self.specs.append(spec)
            return self.session

    sandbox = FakeSyncSandbox()
    spec = SandboxSpec(source=SandboxSource.image("image"), workdir="/testbed")
    monkeypatch.setattr(swebench_grader, "_get_gym_eval_script", lambda problem: "pytest -q")
    monkeypatch.setattr(
        swebench_grader,
        "_grade_gym",
        lambda *args: {
            "resolved": True,
            "f2p_pass": 1,
            "f2p_total": 1,
            "p2p_pass": 0,
            "p2p_total": 0,
            "resolved_by": "test",
        },
    )

    result = swebench_grader.grade_fresh_container(
        {
            "instance_id": "task",
            "base_commit": "abc123",
            "FAIL_TO_PASS": ["test_a.py::test_a"],
            "eval_script": "pytest -q",
        },
        "diff --git a/a.py b/a.py\n",
        "gym",
        "image",
        sandbox=sandbox,
        sandbox_spec=spec,
    )

    assert result["resolved"]
    assert sandbox.specs == [spec]
    assert sandbox.session.writes["/tmp/psrl-model.patch"].startswith(b"diff --git")
    assert sandbox.session.writes["/tmp/psrl-eval.sh"] == b"pytest -q"
    assert "git reset --hard abc123 && git clean -fd" in sandbox.session.commands
    assert "git apply --binary /tmp/psrl-model.patch" in sandbox.session.commands
    assert sandbox.session.released
