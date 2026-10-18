"""Fanout cost of the v2 write gate: O(holders) once, then O(readers) per write.

The writer preempts only pulls that are in flight, so the steady-state cost is zero. When
readers are active, each gets one notification carrying only its own node's holders; a
payload never grows with cluster size. This benchmark asserts the operation counts rather
than wall time, so it stays meaningful on a shared CPU.

Run: python tests/nixl/bench_write_gate.py [--instances 10000]
"""

from __future__ import annotations

import argparse
import time
import types

from psrl.utils.common.worker_naming import gen_client_name
from psrl.workers.ps.ps_manager import PSManager
from psrl.workers.ps.write_state import WriteState


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
        self.inbox: dict[str, list[bytes]] = {}

    def send_notif(self, dst: str, payload: bytes) -> None:
        self.sent.append((dst, payload))

    def get_new_notifs(self):
        taken, self.inbox = self.inbox, {}
        return taken


class _Server:
    def __init__(self, infos: _CountingInfos) -> None:
        self.client_infos = infos
        self.agent = _Agent()


class _Manager:
    """Just the attributes `_send_preempts` touches."""

    def __init__(self, infos: _CountingInfos) -> None:
        self.nixl_meta_server = _Server(infos)


def _infos(instances: int, per_node: int = 8) -> _CountingInfos:
    infos = _CountingInfos()
    for i in range(instances):
        infos[gen_client_name(i, 0)] = types.SimpleNamespace(
            node_ip=f"node-{i // per_node}", agent_name=f"agent-{i}"
        )
    return infos


def _state(instances: int, readers: int) -> tuple[WriteState, list[tuple[int, int]]]:
    state = WriteState(version=1)
    for i in range(instances):
        instance = (i, 0)
        if i >= readers:
            # Holders: committed at version 1 without a pull (resume / prior sync).
            state.record_published(instance, 1)
        else:
            admission = state.begin_pull(instance, 1)
            assert admission.admitted
    return state, [(i, 0) for i in range(readers)]


def bench(instances: int, readers: int) -> dict:
    state, _ = _state(instances, readers)
    infos = _infos(instances)
    manager = _Manager(infos)
    batch = state.begin_write()

    infos.lookups = 0
    started = time.perf_counter()
    PSManager._send_preempts(manager, batch)
    fanout_s = time.perf_counter() - started

    payload_sizes = [len(payload) for _, payload in manager.nixl_meta_server.agent.sent]
    state.start_writing()
    started = time.perf_counter()
    state.publish(2)
    publish_s = time.perf_counter() - started
    return {
        "instances": instances,
        "readers": readers,
        "holders": len(batch.holders),
        "lookups": infos.lookups,
        "notifs": len(payload_sizes),
        "max_payload_bytes": max(payload_sizes, default=0),
        "fanout_us": fanout_s * 1e6,
        "publish_us": publish_s * 1e6,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=10000)
    args = parser.parse_args()

    print(f"{'instances':>10} {'readers':>8} {'holders':>8} {'lookups':>8} {'notifs':>7} "
          f"{'max_payload':>11} {'fanout_us':>10} {'publish_us':>10}")
    for instances in (1000, args.instances):
        for readers in (0, 1, 10, 100, 1000):
            if readers > instances:
                continue
            row = bench(instances, readers)
            print(
                f"{row['instances']:>10} {row['readers']:>8} {row['holders']:>8} "
                f"{row['lookups']:>8} {row['notifs']:>7} {row['max_payload_bytes']:>11} "
                f"{row['fanout_us']:>10.1f} {row['publish_us']:>10.1f}"
            )
            # Complexity assertions: holders are looked up once, readers once each, and a
            # payload carries only one node's holders (<= per_node), not the whole fleet.
            assert row["lookups"] <= row["holders"] + row["readers"] + 2
            assert row["notifs"] == row["readers"]
            assert row["max_payload_bytes"] < 4096, row["max_payload_bytes"]


if __name__ == "__main__":
    main()
