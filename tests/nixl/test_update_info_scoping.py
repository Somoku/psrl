"""Update-info broadcasts go only to readers, and only with what they read.

Every agent caches every client info, but only a reader of an updated client can
observe the change. Broadcasting to the whole fleet made each switch pay for a large
payload per destination and wait for every acknowledgement. A validation wake is the
sharpest case: the woken instances are pull destinations, so the direct-supply plan has
no reader for them and the broadcast is pure cost.

Two properties are pinned here:

* destinations come from the plan inversion (which includes peer-diffusion followers,
  who read their designate without appearing in any plan direction);
* each destination receives only the updated sources its readers actually read, and
  every rank of a reader instance is included because the acknowledgement is per
  instance.
"""

from __future__ import annotations

import pytest
from psrl.utils.nixl.handshake_notif import decode
from psrl.utils.nixl.server import NIXLMetaServer
from psrl.workers.ps.ps_manager import PSManager

pytestmark = pytest.mark.cpu_test


class _Info:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.conn_info_meta = b"conn:" + agent_name.encode()
        self.meta = b"meta:" + agent_name.encode()

    def serialize(self) -> bytes:
        return self.agent_name.encode()


class _AgentStub:
    def __init__(self):
        self.sent: list[tuple[str, bytes]] = []

    def send_notif(self, agent_name: str, payload: bytes):
        self.sent.append((agent_name, payload))


def _server(readers: dict, infos: dict | None = None) -> NIXLMetaServer:
    infos = infos or {}
    server = NIXLMetaServer.__new__(NIXLMetaServer)
    server.server_name = "meta"
    server._descriptor_readers = readers
    server.client_infos = infos
    server._client_temp_mappings = {name: {} for name in infos}
    server.agent = _AgentStub()
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
    server = _server({"train0": {"w": {"train0", "gen0"}}})

    assert server.readers_of(["train0"]) == {"gen0"}


def test_each_destination_gets_only_the_sources_its_readers_read():
    infos = {
        "NIXLTrainClient_0": _Info("agent-t0"),
        "NIXLTrainClient_1": _Info("agent-t1"),
        # One instance, two ranks, each reading a different trainer rank.
        "NIXLGenClient_I1_R0": _Info("agent-r0"),
        "NIXLGenClient_I1_R1": _Info("agent-r1"),
    }
    readers = {
        "NIXLTrainClient_0": {"w": {"NIXLGenClient_I1_R0"}},
        "NIXLTrainClient_1": {"w": {"NIXLGenClient_I1_R1"}},
    }
    server = _server(readers, infos)

    delivered = server.broadcast_update_client_infos_to_readers(["NIXLTrainClient_0", "NIXLTrainClient_1"])

    # Both ranks of the instance are acknowledged, even the one that reads only one source.
    assert delivered == ["NIXLGenClient_I1_R0", "NIXLGenClient_I1_R1"]
    payloads = {agent: decode(payload).entries for agent, payload in server.agent.sent}
    assert set(payloads) == {"agent-r0", "agent-r1"}
    assert set(payloads["agent-r0"]) == {"NIXLTrainClient_0"}
    assert set(payloads["agent-r1"]) == {"NIXLTrainClient_1"}


def test_the_plan_carries_only_each_destinations_sources():
    infos = {
        "NIXLTrainClient_0": _Info("agent-t0"),
        "NIXLTrainClient_1": _Info("agent-t1"),
        "NIXLGenClient_I1_R0": _Info("agent-r0"),
        "NIXLGenClient_I1_R1": _Info("agent-r1"),
    }
    readers = {
        "NIXLTrainClient_0": {"w": {"NIXLGenClient_I1_R0"}},
        "NIXLTrainClient_1": {"w": {"NIXLGenClient_I1_R1"}},
    }
    server = _server(readers, infos)

    delivered, plan = server.plan_update_broadcast(["NIXLTrainClient_0", "NIXLTrainClient_1"])

    assert delivered == ["NIXLGenClient_I1_R0", "NIXLGenClient_I1_R1"]
    assert plan["destinations"]["agent-r0"]["sources"] == ["NIXLTrainClient_0"]
    assert plan["destinations"]["agent-r1"]["sources"] == ["NIXLTrainClient_1"]
    assert plan["destinations"]["agent-r0"]["conn_info"] == b"conn:agent-r0"
    # Sources are serialized once, however many destinations receive them.
    assert set(plan["sources"]) == {"NIXLTrainClient_0", "NIXLTrainClient_1"}
    assert server.agent.sent == [], "planning must not send"


def test_a_rank_only_sends_its_slice_of_the_fan_out():
    """The fan-out is spread across the train ranks' agents: one agent sending every
    notification is O(readers) and does not scale."""
    from psrl.workers.train.engine_train_worker import PSRL_EngineTrainWorker

    class _Agent:
        def __init__(self):
            self.sent = []
            self.added = []
            self.removed = []

        def add_remote_agent(self, conn_info):
            self.added.append(conn_info)

        def remove_remote_agent(self, agent_name):
            self.removed.append(agent_name)

        def send_notif(self, agent_name, payload):
            self.sent.append(agent_name)

    class _Client:
        def __init__(self):
            self.agent = _Agent()

    class _Worker:
        nixl_send_update_infos = PSRL_EngineTrainWorker.nixl_send_update_infos

    worker = _Worker()
    worker.rank = 1
    worker.world_size = 4
    worker.log_prefix = "rank1"
    worker._update_dest_agents = set()
    worker.nixl_storage_client = _Client()
    plan = {
        "sources": {"src": {"info": b"info", "temp_mapping": {}}},
        "destinations": {
            f"agent{i}": {"sources": ["src"], "conn_info": f"conn{i}".encode()} for i in range(8)
        },
    }

    report = worker.nixl_send_update_infos(plan)

    assert report["rank"] == 1
    assert report["agents"] == 2
    assert worker.nixl_storage_client.agent.sent == ["agent1", "agent5"]
    assert worker.nixl_storage_client.agent.added == [b"conn1", b"conn5"]
    # Every fan-out refreshes the destination: it re-registers between switches, and a
    # cached entry can leave the notification queued nowhere.
    worker.nixl_storage_client.agent.added.clear()
    worker.nixl_storage_client.agent.removed.clear()
    worker.nixl_send_update_infos(plan)
    assert worker.nixl_storage_client.agent.removed == ["agent1", "agent5"]
    assert worker.nixl_storage_client.agent.added == [b"conn1", b"conn5"]


def test_no_reader_means_no_payload_and_no_wait():
    infos = {"NIXLGenClient_I1_R0": _Info("agent-r0")}
    server = _server({"NIXLGenClient_I1_R0": {}}, infos)

    assert server.broadcast_update_client_infos_to_readers(["NIXLGenClient_I1_R0"]) == []
    assert server.agent.sent == []


def test_a_reader_without_metadata_is_not_reported_as_waiting():
    """It cannot receive the payload, so the caller must not wait on it: that would sit
    until the handshake budget expired."""
    infos = {"NIXLGenClient_I1_R0": _Info("")}
    server = _server({"NIXLTrainClient_0": {"w": {"NIXLGenClient_I1_R0"}}}, infos)

    assert server.broadcast_update_client_infos_to_readers(["NIXLTrainClient_0"]) == []
    assert server.agent.sent == []


def test_the_manager_returns_the_delivered_clients():
    class _Server:
        def broadcast_update_client_infos_to_readers(self, names):
            return ["NIXLGenClient_I1_R0"]

    manager = PSManager.__new__(PSManager)
    manager.nixl_meta_server = _Server()

    assert manager.nixl_broadcast_update_client_infos_to_readers(["train0"]) == ["NIXLGenClient_I1_R0"]


class TestDescriptorSerializationIsGated:
    """A direct-supply client's infos never need pairwise descriptor bytes.

    The pull engine builds its remote descriptors from `read_addrs` + strides + shard
    metadata, so shipping expanded strided descriptor lists only inflates the very
    payloads the switch broadcasts.
    """

    @staticmethod
    def _client(pairwise: bool):
        from psrl.utils.nixl.client import NIXLStorageClient

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
