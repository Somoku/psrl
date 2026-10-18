"""A running cluster releases a member, revoking its descriptors first (item E4).

`admit_joiners` made the cluster grow. Shrinking is not its mirror image, and the
difference is what these tests are about: a joiner nobody has heard of harms nobody, so
joining can be polled, whereas a leaver is about to **free memory its peers hold
descriptors for**. Until every holder has forgotten, a read from any of them lands on
those pages -- and on this NIXL build an un-revoked descriptor still works, so it lands
*successfully*. That is the silent corruption §3.3.1d catalogues, reached through
membership instead of through re-registration.

So `release_member` is driven rather than polled, and the three ways it can be wrong
while looking right are:

* the member is dropped from the roster but nobody was told, leaving live descriptors
  behind -- the failure looks like nothing at all until a stray read succeeds;
* the revocation is sent but not *confirmed*, which is the state a fire-and-forget
  `invalidateLocalMD` would leave the cluster in (see `leave_protocol` for why PSRL
  does not use it);
* the leaver is removed from one registry and not the others, so the next replan puts
  it back into a plan as a readable source.

Both sides run the shipped code -- `NIXLMetaServer.release_member` against
`NIXLStorageClient.handle_membership_notif` -- over an in-memory notification bus.
"""

from __future__ import annotations

from collections import Counter

import pytest
from psrl.utils.nixl.client import NIXLStorageClient
from psrl.utils.nixl.handshake_notif import NotifKind
from psrl.utils.nixl.leave_protocol import encode_member_leave
from psrl.utils.nixl.membership import Membership, MembershipEvent
from psrl.utils.nixl.server import LEAVE_RESEND_INTERVAL_S, NIXLMetaServer

pytestmark = pytest.mark.cpu_test


class _Bus:
    """Notification delivery between one server and several client agents.

    Delivery to a client is **synchronous**: sending a MEMBER_LEAVE runs that client's
    drain immediately. The test is single threaded, and the alternative -- queueing and
    hoping the server's loop pumps the clients -- would spin forever. What the shortcut
    gives up is timing, and timing is covered by registering a client as `deaf`, which
    is what an unreachable or busy peer looks like from the server's side.
    """

    def __init__(self):
        self.clients: dict[str, _Client] = {}
        self.deaf: set[str] = set()
        self.to_server: dict[str, list[bytes]] = {}
        self.server_agent = "server-agent"
        self.sent: list[tuple[str, str]] = []
        #: Exceptions raised by a peer's handler. `release_member` treats a failed send
        #: as transient and logs it, which in this harness would hide a crash in the
        #: handler behind an ordinary-looking ack timeout.
        self.handler_errors: list[BaseException] = []

    def register(self, client: _Client) -> None:
        self.clients[client.agent_name] = client

    def server_send(self, dst_agent: str, payload: bytes) -> None:
        self.sent.append((dst_agent, "leave"))
        if dst_agent in self.deaf:
            return
        client = self.clients[dst_agent]
        try:
            client.handle_membership_notif(self.server_agent, payload)
        except BaseException as exc:
            self.handler_errors.append(exc)
            raise

    def client_send(self, from_agent: str, dst_agent: str, payload: bytes) -> None:
        # Keyed by sender, as `get_new_notifs` is: the ack deliberately omits the
        # revoker's name because the delivery already carries it.
        assert dst_agent == self.server_agent, f"client addressed {dst_agent}"
        self.to_server.setdefault(from_agent, []).append(payload)

    def server_drain(self) -> dict[str, list[bytes]]:
        taken, self.to_server = self.to_server, {}
        return taken


class _ClientAgent:
    def __init__(self, bus: _Bus, agent_name: str, known: set[str]):
        self._bus = bus
        self.agent_name = agent_name
        self.known = set(known)
        self.removed: list[str] = []

    def remove_remote_agent(self, name: str) -> None:
        from nixl._bindings import nixlNotFoundError

        self.removed.append(name)
        if name not in self.known:
            raise nixlNotFoundError("NIXL_ERR_NOT_FOUND")
        self.known.discard(name)

    def send_notif(self, dst: str, payload: bytes) -> None:
        self._bus.client_send(self.agent_name, dst, payload)


class _Client:
    """A peer running the shipped revocation handler over stand-in NIXL state."""

    handle_membership_notif = NIXLStorageClient.handle_membership_notif
    _revoke_member = NIXLStorageClient._revoke_member
    revoked_clients = NIXLStorageClient.revoked_clients

    def __init__(self, bus: _Bus, client_name: str, agent_name: str, holds: dict[str, str]):
        self.client_name = client_name
        self.agent_name = agent_name
        # `holds` maps a held client to the agent serving it: the eviction is by agent,
        # the cached connection flag is by client, and conflating them is how a peer
        # ends up believing it is still connected to metadata it has dropped.
        self.agent = _ClientAgent(bus, agent_name, set(holds.values()))
        self._all_client_infos = {}
        self._target_client_connected = dict.fromkeys(holds, True)
        self._revoked_clients: list[str] = []
        # Empty means "this client is the only one on its agent", as in the real class.
        self._agent_siblings: list[_Client] = []
        self.pull_engines_dropped = 0
        bus.register(self)

    def _drop_pull_engine(self) -> None:
        self.pull_engines_dropped += 1


class _Info:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name


def _cluster(members: dict[str, str], *, holders: dict[str, set[str]] | None = None):
    """A server mid-flight plus one client per agent.

    ``members`` maps client name -> agent name; ``holders`` maps agent name -> the
    clients whose descriptors that agent has loaded, defaulting to "everyone holds
    everyone", which is the worst case the shrink has to clear.
    """
    bus = _Bus()
    agents = sorted(set(members.values()))
    holders = holders or {agent: set(members) for agent in agents}
    clients = {
        agent: _Client(
            bus,
            client_name=next(c for c, a in sorted(members.items()) if a == agent),
            agent_name=agent,
            holds={held: members[held] for held in holders.get(agent, set())},
        )
        for agent in agents
    }

    class _ServerAgent:
        @staticmethod
        def send_notif(dst, payload):
            bus.server_send(dst, payload)

        @staticmethod
        def get_new_notifs():
            return bus.server_drain()

    class _Server:
        release_member = NIXLMetaServer.release_member
        _holder_agents_of = NIXLMetaServer._holder_agents_of
        _forget_client = NIXLMetaServer._forget_client
        _route_notif = NIXLMetaServer._route_notif
        _drain_notifs = NIXLMetaServer._drain_notifs
        _add_client = NIXLMetaServer._add_client

        def __init__(self):
            self.agent = _ServerAgent()
            self.client_infos = {c: _Info(a) for c, a in members.items()}
            self.client_sharding_dicts = {c: {"w": object()} for c in members}
            self.client_unified_sharding_dicts = {c: {"w": object()} for c in members}
            self._client_temp_mappings = {c: {} for c in members}
            self.connected_clients = {}
            for client_name, agent_name in sorted(members.items()):
                self._add_client(agent_name, client_name)
            self._agents_reported = {kind: set(agents) for kind in NotifKind}
            self._handshake_timeout = 2.0
            self.calls: list[str] = []
            self._membership = Membership(admit=lambda _c: (True, False, ""))
            self._membership.members = set(members)

        def make_unified_sharding(self, *, rebuild=False):
            self.calls.append(f"unified(rebuild={rebuild})")

        def notify_all_client_shardings(self):
            self.calls.append("notify_shardings")

        def make_comm_plan(self, *, rebuild=False):
            self.calls.append(f"comm_plan(rebuild={rebuild})")

        def notify_all_client_infos_and_comm_plan(self):
            self.calls.append("notify_infos_and_plan")

    return _Server(), clients, bus


_THREE = {"train_0": "agent_t", "gen_0": "agent_g0", "gen_1": "agent_g1"}


class TestTheRevocationActuallyReachesHolders:
    def test_every_peer_agent_evicts_the_leaver(self):
        server, clients, bus = _cluster(_THREE)

        server.release_member("gen_1")

        assert not bus.handler_errors, f"a peer's handler raised: {bus.handler_errors[0]!r}"
        for agent in ("agent_t", "agent_g0"):
            assert clients[agent].agent.removed == ["agent_g1"], (
                f"{agent} was not told to forget the leaver's agent"
            )

    def test_the_leaver_is_not_asked_to_forget_itself(self):
        """NIXL refuses an agent as its own remote, and the leaver is going anyway."""
        server, clients, _ = _cluster(_THREE)

        server.release_member("gen_1")

        assert clients["agent_g1"].agent.removed == []

    def test_holders_drop_the_cached_connection_and_pull_engine(self):
        """Both are built from descriptors that are about to stop existing; a pull
        engine rebuilt from stale handles fails at post time with a NOT_FOUND that
        reads like a network fault."""
        server, clients, _ = _cluster(_THREE)

        server.release_member("gen_1")

        peer = clients["agent_g0"]
        assert "gen_1" not in peer._target_client_connected
        assert peer.pull_engines_dropped == 1
        assert peer.revoked_clients == ["gen_1"]

    def test_clients_sharing_an_agent_all_drop_their_caches(self):
        """`remove_remote_agent` is agent-level, so one client's eviction applies to
        its co-hosted siblings whether they know it or not. A sibling left believing it
        is still connected skips `_ensure_client_info_fetched` and posts a transfer
        against a remote the agent has already forgotten."""
        bus = _Bus()
        held = {"gen_1": "agent_g1"}
        primary = _Client(bus, client_name="actor", agent_name="agent_t", holds=held)
        sibling = _Client(bus, client_name="critic", agent_name="agent_t", holds=held)
        # Re-register the primary: both were built against the same agent name, and the
        # bus keys clients by agent, so the second construction displaced the first.
        bus.register(primary)
        primary._agent_siblings = [primary, sibling]
        sibling._agent_siblings = [primary, sibling]

        primary.handle_membership_notif("server-agent", encode_member_leave("gen_1", "agent_g1"))

        assert "gen_1" not in primary._target_client_connected
        assert "gen_1" not in sibling._target_client_connected, (
            "the sibling still believes it is connected to the departed member"
        )
        assert sibling.pull_engines_dropped == 1

    def test_an_agent_that_never_loaded_the_leaver_still_acks(self):
        """Under a comm plan most pairs never read from each other. Treating "nothing
        to forget" as silence would stall every shrink in such a cluster."""
        server, clients, _ = _cluster(_THREE, holders={"agent_t": set(), "agent_g0": {"gen_1"}})

        change = server.release_member("gen_1")

        assert change.event is MembershipEvent.LEFT

    def test_the_member_is_gone_from_the_roster(self):
        server, _, _ = _cluster(_THREE)

        server.release_member("gen_1")

        assert server._membership.members == {"train_0", "gen_0"}
        assert "gen_1" not in server._membership.draining


class TestConfirmationIsRequired:
    def test_an_unreachable_holder_blocks_the_release(self):
        """The whole reason this is not `invalidateLocalMD`: a fire-and-forget send
        cannot tell "forgotten" from "never arrived"."""
        server, _, bus = _cluster(_THREE)
        bus.deaf.add("agent_g0")

        with pytest.raises(TimeoutError, match="agent_g0"):
            server.release_member("gen_1", timeout=0.05)

    def test_a_blocked_release_leaves_the_member_drained_but_intact(self):
        """The safe degradation: no new reads are routed to it, and its memory stays.
        Freeing it is the unsafe half, and that is exactly what did not happen."""
        server, _, bus = _cluster(_THREE)
        bus.deaf.add("agent_g0")

        with pytest.raises(TimeoutError):
            server.release_member("gen_1", timeout=0.05)

        assert "gen_1" in server._membership.members
        assert "gen_1" not in server._membership.serving
        assert "gen_1" in server.client_infos, "its memory and metadata must survive"

    def test_a_blocked_release_does_not_replan(self):
        """A replan would publish a roster the cluster has not actually reached."""
        server, _, bus = _cluster(_THREE)
        bus.deaf.add("agent_g0")

        with pytest.raises(TimeoutError):
            server.release_member("gen_1", timeout=0.05)

        assert server.calls == []

    def test_a_retried_release_succeeds_once_the_holder_answers(self):
        server, _, bus = _cluster(_THREE)
        bus.deaf.add("agent_g0")
        with pytest.raises(TimeoutError):
            server.release_member("gen_1", timeout=0.05)

        bus.deaf.clear()
        assert server.release_member("gen_1").event is MembershipEvent.LEFT

    def test_readers_in_flight_still_block_the_release(self):
        """Revocation does not replace the drain; it is the step after it."""
        server, _, _ = _cluster(_THREE)

        with pytest.raises(RuntimeError, match="in flight"):
            server.release_member("gen_1", in_flight=3)

    def test_nothing_is_revoked_while_readers_are_in_flight(self):
        """The order is load-bearing, not cosmetic: NIXL fails an in-flight transfer
        whose remote section was invalidated (`getXferStatus` checks the same expired
        reference), so revoking before the drain breaks exactly the reads the drain
        step exists to protect."""
        server, clients, bus = _cluster(_THREE)

        with pytest.raises(RuntimeError, match="in flight"):
            server.release_member("gen_1", in_flight=1)

        assert bus.sent == [], "a revocation went out before the readers had drained"
        assert clients["agent_t"].agent.removed == []
        assert "gen_1" not in server._membership.serving, "it must still stop serving"

    def test_waiting_for_an_ack_does_not_flood_the_holder(self):
        """Announcing once per ack poll is the obvious loop and is a control-plane
        flood: a peer that is merely slow to drain would receive a hundred identical
        messages a second, and at 10k members that is what C2 exists to prevent. The
        message is idempotent, so re-sending only pays for losses.

        Found by a real-agent run, where an abandoned 1 s attempt left the peer over a
        hundred queued announcements to work through.
        """
        server, _, bus = _cluster(_THREE)
        bus.deaf.add("agent_g0")

        with pytest.raises(TimeoutError):
            server.release_member("gen_1", timeout=0.3)

        per_agent = Counter(agent for agent, _ in bus.sent)
        assert max(per_agent.values()) <= 2, (
            f"announced {dict(per_agent)} times in 0.3s; the resend interval is "
            f"{LEAVE_RESEND_INTERVAL_S}s, so at most one round should have gone out"
        )


class TestTheClusterForgetsTheMemberEverywhere:
    def test_every_registry_drops_it(self):
        """A leftover entry in any one of them puts the member back into the next plan
        as a readable source -- after its memory is gone."""
        server, _, _ = _cluster(_THREE)

        server.release_member("gen_1")

        for registry in (
            server.client_sharding_dicts,
            server.client_unified_sharding_dicts,
            server.client_infos,
            server._client_temp_mappings,
        ):
            assert "gen_1" not in registry

    def test_its_agent_stops_being_addressed(self):
        server, _, _ = _cluster(_THREE)

        server.release_member("gen_1")

        assert "agent_g1" not in server.connected_clients
        for reported in server._agents_reported.values():
            assert "agent_g1" not in reported

    def test_a_co_hosted_sibling_keeps_its_agent_registered(self):
        """One eviction covers an agent, but only the departing client leaves it."""
        members = {"train_0": "agent_t", "gen_0": "agent_g", "gen_1": "agent_g"}
        server, _, _ = _cluster(members)

        server.release_member("gen_1")

        assert server.connected_clients["agent_g"] == ["gen_0"]

    def test_the_cluster_replans_after_a_release(self):
        """Dropping the member without replanning leaves every peer's plan naming it."""
        server, _, _ = _cluster(_THREE)

        server.release_member("gen_1")

        assert server.calls == [
            "unified(rebuild=True)",
            "notify_shardings",
            "comm_plan(rebuild=True)",
            "notify_infos_and_plan",
        ]

    def test_releasing_the_last_member_does_not_replan_an_empty_cluster(self):
        server, _, _ = _cluster({"train_0": "agent_t"})

        server.release_member("train_0")

        assert server._membership.members == set()
        assert server.calls == []


class TestHolderSetIsConservative:
    def test_every_peer_agent_counts_not_just_the_planned_readers(self):
        """Descriptors are held by whoever loaded them, including an agent an earlier
        replan removed from the plan. Asking only current readers would skip it."""
        server, _, _ = _cluster(_THREE)

        assert server._holder_agents_of("gen_1") == {"agent_t", "agent_g0"}

    def test_agents_hosting_only_the_leaver_are_excluded(self):
        server, _, _ = _cluster(_THREE)

        assert "agent_g1" not in server._holder_agents_of("gen_1")

    def test_a_co_hosted_agent_is_still_required_to_ack(self):
        """It holds descriptors for its other clients' sake, so its eviction matters
        even though the leaver lives there too."""
        members = {"train_0": "agent_t", "gen_0": "agent_g", "gen_1": "agent_g"}
        server, _, _ = _cluster(members)

        assert server._holder_agents_of("train_0") == {"agent_g"}
