"""Update-info broadcasts go only to clients that read the updated ones.

Every agent caches every client info, but only a reader of an updated client can
observe the change. Broadcasting to the whole fleet made each switch pay for a large
payload per destination and wait for every acknowledgement. A validation wake is the
sharpest case: the woken instances are pull destinations, so the direct-supply plan has
no reader for them and the broadcast is pure cost.

A peer-diffusion follower reads its designate without appearing in any plan direction,
so the reader set comes from `descriptor_readers_by_source`, not from the three plan
fields alone.
"""

from __future__ import annotations

import pytest
from psrl.utils.nixl.client import NIXLStorageClient
from psrl.utils.nixl.server import NIXLMetaServer
from psrl.workers.ps.ps_manager import PSManager

pytestmark = pytest.mark.cpu_test


def _server(readers: dict) -> NIXLMetaServer:
    server = NIXLMetaServer.__new__(NIXLMetaServer)
    server._descriptor_readers = readers
    return server


def test_readers_of_returns_only_plan_readers():
    server = _server({"gen0": {"w": {"gen1", "gen2"}}, "gen1": {"w": {"gen0"}}})

    assert server.readers_of(["gen0"]) == {"gen1", "gen2"}
    # gen1 is itself updated, but it still reads gen0, so it must receive gen0's new
    # address in the same round.
    assert server.readers_of(["gen0", "gen1"]) == {"gen0", "gen1", "gen2"}


def test_a_client_nobody_reads_has_no_destinations():
    """The validation-wake case: the woken instances are destinations."""
    server = _server({"validate0": {}, "validate1": {}})

    assert server.readers_of(["validate0", "validate1"]) == set()


def test_an_updated_client_is_not_its_own_reader():
    """A self-reference in the plan would make every source wait for a broadcast it
    does not need and cannot use."""
    server = _server({"train0": {"w": {"train0", "gen0"}}})

    assert server.readers_of(["train0"]) == {"gen0"}


class _Info:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name


class _ServerStub:
    def __init__(self, readers: set[str], clients: dict | None = None):
        self.client_infos = clients or {"gen0": _Info("agent0"), "gen1": _Info("agent1")}
        self._readers = readers
        self.calls: list[tuple[list[str], list[str]]] = []

    def readers_of(self, names):
        return set(self._readers)

    def broadcast_update_client_infos(self, dst_agents, update_client_names):
        self.calls.append((list(dst_agents), list(update_client_names)))


def test_the_manager_sends_once_per_reader_agent_and_returns_the_waiters():
    manager = PSManager.__new__(PSManager)
    manager.nixl_meta_server = stub = _ServerStub({"gen1"})

    delivered = manager.nixl_broadcast_update_client_infos_to_readers(["gen0"])

    assert delivered == ["gen1"]
    assert stub.calls == [(["agent1"], ["gen0"])]


def test_a_reader_without_metadata_is_not_reported_as_waiting():
    """It cannot receive the payload, so the caller must not wait on it: that would sit
    until the handshake budget expired."""
    manager = PSManager.__new__(PSManager)
    manager.nixl_meta_server = stub = _ServerStub({"missing"})

    assert manager.nixl_broadcast_update_client_infos_to_readers(["gen0"]) == []
    assert stub.calls == []


def test_every_rank_of_a_reader_instance_is_sent_to():
    """The acknowledgement is per instance: each rank calls `wait_for_update_infos`, so a
    rank that was left out would wait until the handshake budget expired."""
    clients = {
        "NIXLGenClient_I1_R0": _Info("agent-r0"),
        "NIXLGenClient_I1_R1": _Info("agent-r1"),
        "NIXLGenClient_I2_R0": _Info("agent-other"),
    }
    manager = PSManager.__new__(PSManager)
    manager.nixl_meta_server = stub = _ServerStub({"NIXLGenClient_I1_R0"}, clients)

    delivered = manager.nixl_broadcast_update_client_infos_to_readers(["NIXLTrainClient_0"])

    assert delivered == ["NIXLGenClient_I1_R0", "NIXLGenClient_I1_R1"]
    assert stub.calls == [(["agent-r0", "agent-r1"], ["NIXLTrainClient_0"])]


class TestDescriptorSerializationIsGated:
    """A direct-supply client's infos never need pairwise descriptor bytes.

    The pull engine builds its remote descriptors from `read_addrs` + strides + shard
    metadata, so shipping expanded strided descriptor lists only inflates the very
    payloads the switch broadcasts.
    """

    @staticmethod
    def _client(pairwise: bool):
        client = NIXLStorageClient.__new__(NIXLStorageClient)
        client.client_name = "gen0"
        client._pairwise_descriptors = pairwise
        client.calls = []
        client._serialize_pairwise_descriptors = lambda infos, cache: client.calls.append((infos, cache))
        return client

    def test_direct_supply_skips_them(self):
        client = self._client(pairwise=False)
        client._serialize_descriptors_if_pairwise({"k": object()}, {"k": {}})
        assert client.calls == []

    def test_ps_mode_keeps_them(self):
        client = self._client(pairwise=True)
        client._serialize_descriptors_if_pairwise({"k": object()}, {"k": {}})
        assert len(client.calls) == 1
