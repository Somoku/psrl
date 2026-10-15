"""Which shards must be registered with RDMA, derived from the comm plan (design §3.7.3.0).

RDMA registration exists for exactly one reason: to let a remote NIC reach this process's
memory without going through the CPU. So the necessary and sufficient condition is that a
*reader exists that shares neither this address space nor this GPU*:

    register (key, shard)  <=>  some reader in the plan is on a different node

The tempting criterion is a deployment flag -- "colocate: don't register, disaggregate:
register" -- and it is wrong, because it misses a cell that really occurs: under
disaggregate, a trainer's own GPU is also switched colocate-style for eval, so its remote
rollout readers still read it over RDMA. The flag says "disaggregate", the truth is per
`(key, shard)`. Deriving the answer from the plan instead costs nothing and covers the
mixed case for free: one trainer can serve same-node readers over CUDA IPC and remote
readers over RDMA, and each shard gets the transport its readers actually need.

**Why it matters.** The client registers every local tensor on every
`register_local_tensors` call, and the colocate switch calls that on each cycle
(`nixl_sleep` -> `nixl_wake_up`). In a pure colocate run *none* of those registrations can
ever be used -- every reader is same-node -- so the whole cost is paid for nothing. It is
also not free: one `nixlRegDList` plus `get_xfer_descs`/`get_serialized_descs` per shard,
which at MoE scale is the same O(#shards) cost design §3.5.5 measures in the tens of
thousands.

**Conservatism is not optional here.** Being wrong in one direction costs performance;
being wrong in the other costs correctness -- a remote reader of an unregistered shard
fails, or worse, reads a stale descriptor. So every unknown answers "register":
a source with no recorded node, a reader missing from `client_infos`, a client reading
from itself. The only way to get `False` is to positively observe that every reader is
on this node.

**Deliberately ignored: peer-diffusion delegation.** A same-node follower reads its
designate's copy rather than the upstream source, so the plan over-counts readers. Not
modelling it can only add readers, i.e. only ever answer "register" -- which is the safe
direction, and the case it would matter for (a follower delegating a *remote* read) keeps
the remote source registered anyway.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .comm_plan import NIXLCommPlan
    from .nixl_spec import NIXLClientInfo

#: The three plan directions, all shaped `{reader: {key: {source: [shards]}}}`.
#: Read together rather than one at a time because a client can be a source in more than
#: one of them -- a trainer pushes to the PS *and* is read directly by rollouts.
_PLAN_FIELDS = ("push_to_ps_plan", "rollout_pull_from_ps_plan", "train_pull_from_ps_plan")


def readers_by_source(plan: "NIXLCommPlan") -> dict[str, dict[str, set[str]]]:
    """Invert the plan: ``source -> key -> readers``.

    Built once and shared across every source in a cluster, because inverting three
    nested dicts per client would repeat the same O(#clients x #keys) walk N times.
    """
    inverted: dict[str, dict[str, set[str]]] = {}
    for field_name in _PLAN_FIELDS:
        per_reader = getattr(plan, field_name, None) or {}
        for reader, key_plans in per_reader.items():
            for key, per_source in key_plans.items():
                for source, shards in per_source.items():
                    if not shards:
                        continue
                    inverted.setdefault(source, {}).setdefault(key, set()).add(reader)
    return inverted


def keys_read_across_nodes(
    plan: "NIXLCommPlan",
    client_infos: Mapping[str, "NIXLClientInfo"],
    source: str,
    readers: dict[str, dict[str, set[str]]] | None = None,
) -> set[str]:
    """Keys of ``source`` that at least one reader on another node will read.

    An empty set is the colocate case: nothing this client publishes can be reached by a
    remote NIC, so nothing needs registering.
    """
    readers = readers_by_source(plan) if readers is None else readers
    source_node = _node_of(client_infos, source)
    if source_node is None:
        # Without our own node there is nothing to compare against, so every key that
        # anyone reads must be assumed remote.
        return set(readers.get(source, {}))

    cross_node: set[str] = set()
    for key, key_readers in readers.get(source, {}).items():
        for reader in key_readers:
            if reader == source:
                continue
            reader_node = _node_of(client_infos, reader)
            if reader_node is None or reader_node != source_node:
                cross_node.add(key)
                break
    return cross_node


def needs_rdma_registration(
    plan: "NIXLCommPlan",
    client_infos: Mapping[str, "NIXLClientInfo"],
    source: str,
    readers: dict[str, dict[str, set[str]]] | None = None,
) -> bool:
    """Whether ``source`` must register anything at all.

    `client_infos` is required rather than optional: the node comparison is the whole
    criterion, and a caller that cannot supply node identities cannot answer the
    question. Making it optional would let a caller silently get "register everything"
    and mistake it for a measurement.
    """
    return bool(keys_read_across_nodes(plan, client_infos, source, readers))


def registration_report(
    plan: "NIXLCommPlan",
    client_infos: Mapping[str, "NIXLClientInfo"],
    source: str,
) -> dict[str, object]:
    """Human-readable account of one client's verdict, for logs and assertions."""
    readers = readers_by_source(plan)
    cross = keys_read_across_nodes(plan, client_infos, source, readers)
    all_keys = set(readers.get(source, {}))
    return {
        "source": source,
        "node": _node_of(client_infos, source),
        "published_keys": len(all_keys),
        "cross_node_keys": len(cross),
        "needs_registration": bool(cross),
        # The saving, stated as what would have been registered and cannot be read
        # remotely. Equal to `published_keys - cross_node_keys`.
        "skippable_keys": len(all_keys - cross),
    }


def _node_of(client_infos: Mapping[str, "NIXLClientInfo"], client: str) -> str | None:
    info = client_infos.get(client)
    if info is None:
        return None
    node_ip = getattr(info, "node_ip", None)
    return str(node_ip) if node_ip else None
