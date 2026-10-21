"""The shipped defaults are a p3b cluster over Docker, not a single-process dev shape.

The service's placement and admission only span a fleet when workers reach one
control plane; the in-process Docker backend and the Python/Ray placement plane
are development and ablation paths. These assertions keep the default
configuration honest without starting anything.
"""

import json
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
ROLLOUT = REPO / "psrl" / "trainer" / "config" / "rollout" / "psrl_rollout.yaml"
DEPLOYMENT = REPO / "psrl" / "trainer" / "config" / "psrl" / "deployment.yaml"
P3B = REPO / "p3b"


def test_the_rollout_defaults_to_the_p3b_control_plane():
    config = OmegaConf.load(ROLLOUT)
    sandbox = config.agent.sandbox

    assert sandbox.default_backend == "p3b"
    endpoint = str(sandbox.backends.p3b.endpoint)
    # A unix socket reaches a service co-located with the worker; the default
    # deployment places a control plane over node agents, so the default must
    # name a host:port a worker on another machine can reach.
    assert not endpoint.startswith("unix://"), endpoint
    assert ":" in endpoint, endpoint
    # The in-process Docker backend stays declared for ablations and for a
    # single-worker deployment, but it is not what a sandbox gets by default.
    assert "docker" in sandbox.backends


def test_the_p3b_endpoint_can_be_overridden_from_the_environment(monkeypatch):
    monkeypatch.setenv("PSRL_P3B_ENDPOINT", "control.example:9123")
    config = OmegaConf.load(ROLLOUT)

    assert str(config.agent.sandbox.backends.p3b.endpoint) == "control.example:9123"


def test_the_python_placement_plane_is_off_by_default():
    """p3b's control plane is the scheduler; a second one would split the fleet."""
    deployment = OmegaConf.load(DEPLOYMENT)

    assert deployment.sandbox_placement.enabled is False


def test_the_shipped_p3b_examples_are_a_docker_cluster():
    control = json.loads((P3B / "example_control.json").read_text())
    assert control["role"] == "control"
    assert control["default_backend"] == "docker"
    assert all(backend["type"] == "docker" for backend in control["backends"])
    assert len(control["fleet_nodes"]) >= 2
    assert not str(control["listen"]).startswith("unix://")

    fleet_ids = {node["node_id"] for node in control["fleet_nodes"]}
    cgroup_parents = set()
    for name in ("example_node.json", "example_node_b.json"):
        node = json.loads((P3B / name).read_text())
        assert node["role"] == "node"
        assert node["node_id"] in fleet_ids
        assert node["default_backend"] == "docker"
        assert all(backend["type"] == "docker" for backend in node["backends"])
        assert node["node_listen"]
        assert node["node"]["overcommit"] >= 1
        assert node["node"]["cgroup_parent"] not in cgroup_parents
        cgroup_parents.add(node["node"]["cgroup_parent"])
