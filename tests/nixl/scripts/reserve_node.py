"""Hold a node's GPUs in Ray so test runs schedule elsewhere.

Ray tracks what *it* has handed out, not what is actually free on the device. A node
whose memory is occupied by work outside this Ray cluster therefore still looks idle, and
placement groups land on it and fail with "Free memory on device cuda:0 ... is less than
desired GPU memory utilization" -- which reads as a configuration error rather than as the
node being busy.

This takes the node's GPUs out of Ray's pool with a detached placeholder actor, so the
scheduler simply does not consider it. It does nothing to whatever is running there; it
only stops *this* cluster from adding more.

    python reserve_node.py hold <node-address> --gpus 8
    python reserve_node.py release <node-address>

Reserving is explicit and so is releasing: a forgotten reservation silently shrinks the
cluster, which is the same class of confusion it exists to remove.
"""

from __future__ import annotations

import argparse
import sys

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


def _actor_name(ip: str) -> str:
    return f"psrl_reserved_{ip.replace('.', '_')}"


def _node_id(ip: str) -> str:
    for node in ray.nodes():
        if node["Alive"] and node["NodeManagerAddress"] == ip:
            return node["NodeID"]
    raise SystemExit(f"no alive node with address {ip}")


def hold(ip: str, gpus: int) -> None:
    @ray.remote(num_gpus=gpus, num_cpus=1)
    class _Reservation:
        def ping(self) -> str:
            return "held"

    handle = _Reservation.options(
        name=_actor_name(ip),
        namespace="psrl_reservations",
        lifetime="detached",
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=_node_id(ip), soft=False),
    ).remote()
    print(f"{ip}: {ray.get(handle.ping.remote())} {gpus} GPU(s)")


def release(ip: str) -> None:
    try:
        ray.kill(ray.get_actor(_actor_name(ip), namespace="psrl_reservations"))
        print(f"{ip}: released")
    except ValueError:
        print(f"{ip}: no reservation held")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("hold", "release"))
    parser.add_argument("ip")
    parser.add_argument("--gpus", type=int, default=8)
    args = parser.parse_args()

    ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False)
    if args.action == "hold":
        hold(args.ip, args.gpus)
    else:
        release(args.ip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
