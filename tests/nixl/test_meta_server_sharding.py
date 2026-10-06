"""Tests for the meta server's sharding bookkeeping (items A5d and A7 integration).

`NIXLMetaServer.__init__` constructs a real NIXL agent, so these tests build the
object via ``__new__`` and populate only the fields under test. That is deliberate:
`make_unified_sharding`, `log_finest_mesh_summary`, and `classify_incoming_client`
are pure bookkeeping over `client_sharding_dicts` and never touch ``self.agent``, so
exercising them without an agent keeps the test runnable on a single machine.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import pytest

from psrl.utils.nixl.join_admission import JoinVerdict
from psrl.utils.nixl.nixl_spec import NIXLSharding
from psrl.utils.nixl.server import NIXLMetaServer, _mesh_total


def _server(client_shardings: dict[str, dict[str, NIXLSharding]]) -> NIXLMetaServer:
    """A meta server with sharding state but no NIXL agent."""
    server = NIXLMetaServer.__new__(NIXLMetaServer)
    server.client_sharding_dicts = client_shardings
    server.client_unified_sharding_dicts = {}
    server.finest_shard_meshes = {}
    server._is_all_client_shardings_recved = True
    return server


def _sharding(shard_count: int, index: int, dim: int = 0) -> NIXLSharding:
    return NIXLSharding(
        shard_mesh=OrderedDict([(dim, shard_count)]),
        shard_indices=[(index,)],
    )


class TestFinestMeshRecording:
    def test_lcm_is_recorded_per_key(self):
        """finest_shard_meshes must hold the LCM actually used for refactoring."""
        server = _server(
            {
                "train": {"w": _sharding(8, 0)},
                "gen": {"w": _sharding(4, 0)},
            }
        )
        server.make_unified_sharding()
        assert server.finest_shard_meshes == {"w": {0: 8}}

    def test_coprime_degrees_multiply_the_mesh(self):
        """This is the amplification the log line exists to surface."""
        server = _server(
            {
                "train": {"w": _sharding(8, 0)},
                "gen": {"w": _sharding(3, 0)},
            }
        )
        server.make_unified_sharding()
        assert server.finest_shard_meshes["w"] == {0: 24}

    def test_summary_is_logged_with_shard_totals(self, caplog):
        server = _server(
            {
                "train": {"w": _sharding(8, 0), "b": _sharding(2, 0)},
                "gen": {"w": _sharding(4, 0), "b": _sharding(2, 0)},
            }
        )
        with caplog.at_level(logging.INFO):
            server.make_unified_sharding()
        line = next(r.getMessage() for r in caplog.records if "[finest mesh]" in r.getMessage())
        assert "n_keys=2" in line
        assert "total_shards=10" in line  # w=8 + b=2
        assert "w=" in line, "the largest-mesh key must be named so it is actionable"

    def test_summary_is_safe_when_nothing_registered(self):
        server = _server({})
        server.log_finest_mesh_summary()  # must not raise


class TestJoinAdmissionIntegration:
    @pytest.fixture
    def established(self) -> NIXLMetaServer:
        server = _server(
            {
                "train": {"w": _sharding(8, 0)},
                "gen": {"w": _sharding(4, 0)},
            }
        )
        server.make_unified_sharding()
        return server

    def test_divisor_joiner_reports_nothing(self, established):
        """FAST_PATH verdicts are omitted, so an empty dict means a free join."""
        assert established.classify_incoming_client({"w": _sharding(2, 0)}) == {}

    def test_inexpressible_joiner_is_reported_as_reject(self, established):
        """hidden=4096 with lcm(8,3)=24: 512 local rows cannot split into 3."""
        decisions = established.classify_incoming_client(
            {"w": _sharding(3, 0)},
            tensor_shapes={"w": (4096, 4096)},
        )
        assert decisions["w"].verdict is JoinVerdict.REJECT
        assert not decisions["w"].admitted

    def test_expressible_growth_is_reported_as_reshard(self, established):
        decisions = established.classify_incoming_client(
            {"w": _sharding(16, 0)},
            tensor_shapes={"w": (4096, 4096)},
        )
        assert decisions["w"].verdict is JoinVerdict.RESHARD
        assert decisions["w"].admitted
        assert decisions["w"].requires_reshard

    def test_unknown_key_is_ignored(self, established):
        """A joiner may carry keys this mesh has never seen; those are not a conflict."""
        assert established.classify_incoming_client({"brand_new": _sharding(3, 0)}) == {}

    def test_classification_does_not_mutate_established_state(self, established):
        before = dict(established.finest_shard_meshes["w"])
        established.classify_incoming_client({"w": _sharding(3, 0)}, tensor_shapes={"w": (4096, 4096)})
        assert established.finest_shard_meshes["w"] == before


def test_mesh_total_is_the_product_over_dimensions():
    assert _mesh_total({}) == 1
    assert _mesh_total({0: 8}) == 8
    assert _mesh_total({0: 8, 1: 2}) == 16
