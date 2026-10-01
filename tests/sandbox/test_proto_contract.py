"""The proto is the single source, so the Python contract may not drift from it.

Two implementations generate from `psrl/sandbox/api/v1/sandbox.proto` while they
coexist. Nothing stops them diverging except a check that reads the file, so this
parses the proto rather than restating it: a feature added to one side and not the
other fails here instead of at a backend that silently never advertises it.
"""

import re
from pathlib import Path

import pytest
from psrl.sandbox.core import ExecMode, PauseMode, ResumeLevel, SandboxExitReason, SandboxFeature, SnapshotKind

PROTO = Path(__file__).resolve().parents[2] / "psrl" / "sandbox" / "api" / "v1" / "sandbox.proto"


def enum_members(name: str) -> set[str]:
    """Return one proto enum's member names, without its unspecified zero value."""
    body = re.search(rf"^enum {name} \{{(.*?)^\}}", PROTO.read_text(), re.S | re.M)
    assert body is not None, f"proto has no enum {name}"
    members = set(re.findall(r"^\s+([A-Z][A-Z0-9_]*) = \d+;", body.group(1), re.M))
    return {member for member in members if not member.endswith("UNSPECIFIED")}


def test_the_proto_is_committed_beside_the_module_it_defines():
    assert PROTO.exists(), f"the contract must live at {PROTO}"


def test_every_sandbox_feature_crosses_the_wire():
    # A feature the wire cannot carry is one a remote caller can never require.
    assert enum_members("Feature") == {feature.name for feature in SandboxFeature}


def test_every_exit_reason_crosses_the_wire():
    # A post mortem reads these, so one that cannot cross arrives as "unknown" and
    # an operator loses the distinction the reason existed to draw.
    assert enum_members("ExitReason") == {
        reason.name for reason in SandboxExitReason if reason is not SandboxExitReason.UNKNOWN
    }


def test_every_exec_mode_crosses_the_wire():
    assert enum_members("ExecMode") == {mode.name for mode in ExecMode}


def test_every_resume_level_crosses_the_wire():
    assert enum_members("ResumeLevel") == {level.name for level in ResumeLevel}


def test_every_pause_mode_crosses_the_wire():
    assert enum_members("PauseMode") == {f"PAUSE_{mode.name}" for mode in PauseMode}


def test_every_snapshot_kind_crosses_the_wire():
    assert enum_members("SnapshotKind") == {f"SNAPSHOT_{kind.name}" for kind in SnapshotKind}


@pytest.mark.parametrize(
    "rpc",
    ["Create", "CreateGroup", "Restore", "Pause", "Resume", "Snapshot", "Release", "Fleet", "Quota"],
)
def test_the_control_surface_carries_every_caller_operation(rpc):
    assert re.search(rf"rpc {rpc}\(", PROTO.read_text()), f"the control service is missing {rpc}"


@pytest.mark.parametrize("rpc", ["Admit", "CreateOn", "ReleaseOn", "Report", "Sweep"])
def test_the_node_surface_carries_admission_lifecycle_and_reclamation(rpc):
    assert re.search(rf"rpc {rpc}\(", PROTO.read_text()), f"the node service is missing {rpc}"


def test_command_and_file_traffic_are_absent_from_the_control_plane():
    # The invariant the whole shape rests on: an episode issues dozens of commands,
    # so a control plane in that path adds a hop and a serialization per command
    # without making a decision. A create returns the agent endpoint instead.
    text = PROTO.read_text()
    for absent in ("rpc Exec(", "rpc ReadBytes(", "rpc WriteBytes("):
        assert absent not in text, f"{absent} belongs on the sandbox agent, not the control plane"
    assert "message AgentEndpoint" in text
    assert re.search(r"message CreateResponse \{[^}]*AgentEndpoint agent", text, re.S)
