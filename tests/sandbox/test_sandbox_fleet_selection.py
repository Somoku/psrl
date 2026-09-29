"""Which nodes the sandbox plane is built over.

`agent.node_ips` names the sandbox fleet. It used to be mandatory whenever placement was
on, which refused a cluster that simply wanted every node to run sandboxes. Leaving it
empty now means every alive node, and the fleet is whatever `alive_nodes` already holds —
the same set the workers are placed by, filtered by the same allow list at one call site.

Driven without Ray: the selection is a property of the method's arguments, and what needs
a real cluster is actor affinity, which `smoke_ray_plane.py` covers.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

pytestmark = pytest.mark.cpu_test

_NODES = [
    {"NodeID": "node-a", "NodeManagerAddress": "10.0.0.1"},
    {"NodeID": "node-b", "NodeManagerAddress": "10.0.0.2"},
]

_PLACEMENT = OmegaConf.create(
    {
        "enabled": True,
        "backend_name": "docker",
        "node_ttl_s": 120.0,
        "reservation_ttl_s": 60.0,
        "sweep_interval_s": 10.0,
        "heartbeat_interval_s": 30.0,
        "rpc_timeout_s": 60.0,
        "required_label": None,
        "callback_target": None,
    }
)


def _trainer(node_ips: list[str], monkeypatch: pytest.MonkeyPatch) -> tuple[object, list]:
    """Borrow the real method onto a stub, capturing the node ids the builder receives."""
    from psrl.trainer.ppo import ray_trainer

    captured: list = []

    def _fake_build_sandbox_plane(sandbox_config, node_ids, **kwargs):
        captured.append(list(node_ids))
        plane = MagicMock()
        plane.register_nodes.return_value = {}
        plane.capabilities = SimpleNamespace(features=())
        return plane

    monkeypatch.setattr(ray_trainer, "build_sandbox_plane", _fake_build_sandbox_plane)

    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {"gen_actor_rollout_ref": {"rollout": {"agent": {"node_ips": node_ips, "sandbox": {}}}}}
        )
    )
    trainer._build_sandbox_plane = ray_trainer.PSRL_RayPPOTrainer._build_sandbox_plane.__get__(trainer)
    return trainer, captured


def test_an_unnamed_fleet_is_every_alive_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """The empty list used to be refused; it now means the whole cluster."""
    trainer, captured = _trainer([], monkeypatch)

    trainer._build_sandbox_plane(_PLACEMENT, _NODES)

    assert captured == [["node-a", "node-b"]]


def test_an_unnamed_fleet_says_sandboxes_will_share_the_trainer_nodes(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Defaulting is allowed, but it co-locates sandboxes, so it must not be silent."""
    trainer, _ = _trainer([], monkeypatch)

    with caplog.at_level(logging.WARNING):
        trainer._build_sandbox_plane(_PLACEMENT, _NODES)

    message = "\n".join(record.getMessage() for record in caplog.records)
    assert "every alive node" in message
    assert "10.0.0.1" in message and "10.0.0.2" in message


def test_a_named_fleet_is_used_as_given(monkeypatch: pytest.MonkeyPatch) -> None:
    """`alive_nodes` is already filtered by the allow list, so the plane just uses it."""
    trainer, captured = _trainer(["10.0.0.2"], monkeypatch)

    trainer._build_sandbox_plane(_PLACEMENT, [_NODES[1]])

    assert captured == [["node-b"]]


def test_a_named_fleet_does_not_warn_about_sharing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    trainer, _ = _trainer(["10.0.0.2"], monkeypatch)

    with caplog.at_level(logging.WARNING):
        trainer._build_sandbox_plane(_PLACEMENT, [_NODES[1]])

    assert "every alive node" not in "\n".join(record.getMessage() for record in caplog.records)
