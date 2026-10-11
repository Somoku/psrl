"""Splitting agent metadata so join cost stops scaling with registered regions (item E4).

Every member loads every joining member's metadata, so a join costs
``members x md_bytes`` and `md_bytes` grows linearly with the joiner's registered region
count. Measured on H20 (`bench_partial_md.py`):

    regions     full MD      conn-info only
          1       2,139 B         1,640 B
      4,096   1,734,329 B         1,645 B      <- 1054x

Connection info is **constant**; the descriptors are the entire growth. So the exchange
splits in two: everyone gets connection info, and only the peers that will actually read
get descriptors. At 10k members that is 16 MB per join instead of 17 GB.

The split is safe because the two halves are needed at different moments. Connection info
is what `add_remote_agent` requires to establish a link; descriptors are what a *transfer*
requires, and a transfer only happens between pairs the comm plan already names. PSRL
compiles that plan at handshake (design 3.6, "plan once post many"), so the set of peers
needing stage two is known before any bytes move -- it does not have to be discovered.

**What this module does not do.** `invalidate_local_md` is absent from this NIXL build
(checked: `hasattr(nixl_agent, 'invalidate_local_md')` is False), so the leave path cannot
yet revoke descriptors a departing member handed out. That half of E4 is blocked upstream
rather than unimplemented, and `describe_leave_gap` states it in one place so it is not
rediscovered.
"""

from __future__ import annotations

from dataclasses import dataclass

import nixl._bindings as nixlBind


@dataclass(frozen=True)
class MetadataPlan:
    """Which peers need descriptors from this agent, and which need only a link.

    `descriptor_peers` is derived from the comm plan, not from membership: a client that
    never reads from this one needs no descriptors, however long it lives in the cluster.
    """

    conn_info_peers: frozenset[str]
    descriptor_peers: frozenset[str]

    def __post_init__(self) -> None:
        stray = self.descriptor_peers - self.conn_info_peers
        if stray:
            raise ValueError(
                f"{sorted(stray)} are to receive descriptors but not connection info; "
                f"they could not establish a link to use them"
            )

    @property
    def descriptor_fraction(self) -> float:
        """Share of peers taking the expensive half, for reporting."""
        if not self.conn_info_peers:
            return 0.0
        return len(self.descriptor_peers) / len(self.conn_info_peers)


def plan_metadata_fanout(
    client_name: str,
    all_clients: "list[str]",
    readers_of: "set[str] | frozenset[str]",
) -> MetadataPlan:
    """Split the recipients of this client's metadata into the two stages.

    ``readers_of`` is the set of clients the comm plan says will read from
    ``client_name`` -- including peer-diffusion followers, whose reads flow the opposite
    way from the notifications (see `chunk_notify`). Getting that direction wrong is what
    made a designate unable to address its own followers.
    """
    peers = frozenset(name for name in all_clients if name != client_name)
    readers = frozenset(readers_of) & peers
    return MetadataPlan(conn_info_peers=peers, descriptor_peers=readers)


def conn_info_only(agent, mem_type: str = "VRAM") -> bytes:
    """This agent's connection info, with no descriptors.

    The empty descriptor list is built directly rather than through
    `agent.get_reg_descs([])`, which infers its element type from `descs[0]` and so cannot
    express "none".
    """
    empty = nixlBind.nixlRegDList(getattr(nixlBind, f"{mem_type}_SEG"), [])
    return agent.get_partial_agent_metadata(empty, True)


def descriptors_for(agent, tensors: list) -> bytes:
    """Metadata covering only ``tensors``, without connection info.

    Connection info is omitted because the recipient already has it from stage one;
    including it again would add a constant ~1.6 KB per reader for nothing.
    """
    if not tensors:
        raise ValueError(
            "descriptors_for called with no tensors; use conn_info_only for the "
            "connection-info stage rather than sending an empty descriptor payload"
        )
    return agent.get_partial_agent_metadata(agent.get_reg_descs(tensors), False)


def describe_leave_gap() -> str:
    """Why the leave path cannot yet revoke descriptors, in one place.

    The design's four-step shrink is: drop holder status, drain in-flight readers,
    `invalidateLocalMD()`, then release memory. Step three has no binding in this NIXL
    build, so a departed member's descriptors stay loadable by whoever already fetched
    them. Steps one and two still bound the exposure -- a client that is not a holder is
    not in anyone's plan, so nothing *chooses* to read it -- but a stale descriptor used
    anyway would read freed memory and succeed, which is the failure mode the design
    wanted step three for.
    """
    return (
        "invalidate_local_md is not exposed by this NIXL build; the shrink path relies on "
        "holder removal plus in-flight drain, and cannot yet revoke descriptors already "
        "handed out"
    )
