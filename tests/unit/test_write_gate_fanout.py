"""The write gate's fanout is O(readers), and its payload is bounded per node.

At 10k instances a step must not pay for the gate when nothing is reading, and a
preemption must not carry the fleet's holder list to every reader. Both are asserted as
operation counts and payload sizes rather than wall time, so the test is stable on a
shared CPU.
"""

from __future__ import annotations

import types

import pytest
from psrl.utils.common.worker_naming import gen_client_name
from psrl.utils.nixl.handshake_notif import NotifKind, decode
from psrl.workers.ps.ps_manager import PSManager
from psrl.workers.ps.write_state import WriteState

pytestmark = pytest.mark.cpu_test


class _CountingInfos(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lookups = 0

    def get(self, key, default=None):
        self.lookups += 1
        return super().get(key, default)


class _Agent:
    def __init__(self) -> None:
        self.sent: list[tuple[str, bytes]] = []

    def send_notif(self, dst: str, payload: bytes) -> None:
        self.sent.append((dst, payload))


class _Manager:
    def __init__(self, infos: _CountingInfos) -> None:
        self.nixl_meta_server = types.SimpleNamespace(
            client_infos=infos,
            agent=_Agent(),
        )


def _infos(instances: int, per_node: int = 8) -> _CountingInfos:
    infos = _CountingInfos()
    for i in range(instances):
        infos[gen_client_name(i, 0)] = types.SimpleNamespace(
            node_ip=f"node-{i // per_node}", agent_name=f"agent-{i}"
        )
    return infos


def _batch(instances: int, readers: int):
    state = WriteState(version=1)
    for i in range(instances):
        if i >= readers:
            state.record_published((i, 0), 1)
        else:
            assert state.begin_pull((i, 0), 1).admitted
    return state.begin_write()


def test_a_write_with_no_readers_sends_nothing_and_looks_nothing_up():
    infos = _infos(10_000)
    manager = _Manager(infos)
    batch = _batch(10_000, readers=0)
    assert batch.holders == ()
    PSManager._send_preempts(manager, batch)
    assert infos.lookups == 0
    assert manager.nixl_meta_server.agent.sent == []


def test_fanout_touches_each_holder_and_reader_once():
    instances, readers = 10_000, 1_000
    infos = _infos(instances)
    manager = _Manager(infos)
    batch = _batch(instances, readers)
    infos.lookups = 0

    PSManager._send_preempts(manager, batch)

    assert infos.lookups <= len(batch.holders) + len(batch.readers) + 2
    assert len(manager.nixl_meta_server.agent.sent) == readers


def test_a_preemption_payload_carries_only_one_node():
    instances = 200
    infos = _infos(instances, per_node=8)
    manager = _Manager(infos)
    batch = _batch(instances, readers=1)
    PSManager._send_preempts(manager, batch)

    _agent, payload = manager.nixl_meta_server.agent.sent[0]
    notif = decode(payload)
    assert notif is not None and notif.kind is NotifKind.PULL_PREEMPT
    entry = next(iter(notif.entries.values()))
    holders = entry["holders"]
    assert len(holders) <= 8, f"payload crossed a node boundary: {len(holders)} holders"
    assert all(infos[h].node_ip == infos[gen_client_name(0, 0)].node_ip for h in holders)
    assert len(payload) < 4096
