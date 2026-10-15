"""A client joins a cluster that is already running, over real NIXL (item E3, L3).

The CPU suite drives `admit_joiners` against a stand-in agent, so it pins the decision
logic and the replan ordering but says nothing about whether a late client's notification
*reaches* a server that has finished its handshake. That is the part the old code got
wrong, and it is a property of real agents and a real notification stream.

**Why this does not need multiple nodes, or GPUs.** The design files L3 under "multi-node",
but the claim here is about the handshake protocol, not about transport: a joiner's
notification has to arrive, be recognised as a sharding rather than filed as something
else, and drive a replan. None of that depends on the memory living on a GPU or on the
peer living on another host. Registering DRAM and running three agents in one process
exercises the same code and costs minutes instead of a cluster slot.

What it therefore does *not* establish is that a joining instance can then be read over
RDMA from another host. That is D1/D4 territory and is covered by the production runs.

**Cost.** Agent construction is dominated by UCX's per-interface `getifaddrs()` sweep and
has been measured from 60 s to over 300 s on this host (see `conftest_agents.py`). Three
agents are needed -- server, incumbent, joiner -- so this file is slow by construction and
is excluded from `-m cpu_test` by having no marker.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict

import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("nixl._api", reason="NIXL not installed")

from nixl._api import nixl_agent, nixl_agent_config  # noqa: E402
from psrl.utils.nixl.membership import MembershipEvent  # noqa: E402
from psrl.utils.nixl.nixl_spec import NIXLClientType, NIXLSharding  # noqa: E402

#: Same budget and rationale as `conftest_agents.AGENT_SETUP_BUDGET_S`.
AGENT_SETUP_BUDGET_S = float(os.environ.get("PSRL_NIXL_AGENT_BUDGET_S", "900"))

#: Divisible by every mesh used here, so admission turns on the meshes rather than on an
#: awkward shape. 24 = 4 x 6 = 8 x 3.
ROWS = 24
COLS = 8


def _config(port: int):
    return OmegaConf.create(
        {
            "server_ip": "127.0.0.1",
            "server_port": port,
            "max_pinned_temp_memory_slots": None,
            "enable_tms_for_temp_buffers": False,
            "peer_diffusion": False,
            "handshake_timeout_s": 60,
        }
    )


def _sharding(shards: int) -> dict[str, NIXLSharding]:
    """One key sharded `shards` ways along dim 0."""
    return {
        "w": NIXLSharding(
            shard_mesh=OrderedDict({0: shards}),
            shard_indices=[(i,) for i in range(shards)],
        )
    }


def _client(name: str, port: int, config):
    """A storage client on its own agent, bound explicitly.

    `binded_agent` is passed rather than letting the client pick a port, because the
    default path asks a Ray `PortScanner` actor for one -- a production dependency that
    has nothing to do with what is under test here.
    """
    from psrl.utils.nixl.client import NIXLStorageClient

    return NIXLStorageClient(
        client_name=name,
        server_name="meta_server",
        use_gpu=False,
        client_type=NIXLClientType.PUSH_SIDE,
        nixl_config=config,
        # The progress thread is on: without it the client's `send_notif` is only
        # flushed when that client itself polls, and nothing here polls between the
        # send and the server's wait -- the sharding would sit unsent and the server
        # would time out reporting zero agents, which reads as a connectivity fault.
        binded_agent=nixl_agent(name, nixl_agent_config(True, True, port)),
        enable_prog_thread=True,
    )


@pytest.fixture(scope="module")
def cluster():
    """A meta server with one incumbent client, handshake already complete.

    Module-scoped: three agents is the dominant cost of this file, and every test here
    is about what happens to an *already running* cluster, so they share one.
    """
    from psrl.utils.nixl.server import NIXLMetaServer

    port = int(os.environ.get("PSRL_ELASTIC_TEST_PORT", "28931"))
    config = _config(port)

    started = time.time()
    server = NIXLMetaServer("meta_server", config)
    incumbent = _client("incumbent", port + 1, config)
    if time.time() - started > AGENT_SETUP_BUDGET_S:
        pytest.skip(f"agent setup exceeded {AGENT_SETUP_BUDGET_S}s")

    state_dict = {"w": torch.ones(ROWS, COLS, dtype=torch.float32)}

    incumbent.connect_to_server()
    incumbent.send_local_sharding(_sharding(4))
    server.wait_for_client_shardings(expected_agents=1)
    server.make_unified_sharding()
    server.notify_all_client_shardings()
    unified = incumbent.wait_for_server_sharding()
    incumbent.register_local_tensors(state_dict, unified)
    incumbent.send_local_info()
    server.wait_for_client_infos(expected_agents=1)
    server.make_comm_plan()

    yield server, incumbent, config, state_dict

    server.shutdown()


class TestTheClusterIsRunningBeforeAnyoneJoins:
    def test_the_incumbent_is_planned_for(self, cluster):
        """The premise. Without it a later "the joiner was added" proves nothing."""
        server, _, _, _ = cluster

        assert server.comm_plan is not None
        assert "incumbent" in server.client_infos

    def test_a_finest_mesh_is_established(self, cluster):
        """Admission compares against this; an empty mesh admits anything."""
        server, _, _, _ = cluster

        assert server.finest_shard_meshes.get("w") == {0: 4}


class TestALateClientIsAdmittedOverRealNixl:
    def test_a_joiner_reaches_the_plan(self, cluster):
        """The whole item: a client that arrives after the handshake is served.

        Before this change its sharding either landed in the wrong registry or was
        dropped, and in neither case did anything report a problem -- the instance
        simply never appeared in a plan.
        """
        import threading

        server, incumbent, config, state_dict = cluster

        joiner = _client("joiner", config.server_port + 2, config)
        failures: list[BaseException] = []
        steps: list[str] = []

        # Connect and announce first, from this thread. The joiner's remaining steps
        # then run concurrently with the admission, because they are mutually dependent:
        # it cannot send its client info until the server republishes the sharding, and
        # the server cannot finish admitting until that info arrives. Driving both from
        # one thread deadlocks on whichever half is written second.
        joiner.connect_to_server()
        steps.append("connected")
        joiner.send_local_sharding(_sharding(4))
        steps.append("sent_sharding")

        def joiner_half():
            try:
                unified = joiner.wait_for_server_sharding()
                steps.append("got_server_sharding")
                joiner.register_local_tensors(
                    {"w": torch.ones(ROWS, COLS, dtype=torch.float32)}, unified
                )
                steps.append("registered")
                joiner.send_local_info()
                steps.append("sent_info")
            except BaseException as exc:  # surfaced below rather than lost in a thread
                failures.append(exc)

        thread = threading.Thread(target=joiner_half, daemon=True)
        thread.start()
        # Let the joiner reach its wait before the server republishes, so the test is
        # not racing delivery against a listener that has not started.
        time.sleep(1.0)

        admit_error: list[BaseException] = []
        try:
            changes = server.admit_joiners(timeout=60)
        except BaseException as exc:
            admit_error.append(exc)
            changes = []
        thread.join(timeout=30)

        assert not failures, (
            f"the joiner's own half failed after {steps}: {failures[0]!r}"
        )
        assert not admit_error, (
            f"admit_joiners raised; joiner reached {steps}: {admit_error[0]!r}"
        )
        assert steps == [
            "connected",
            "sent_sharding",
            "got_server_sharding",
            "registered",
            "sent_info",
        ], f"the joiner stalled; it got as far as {steps}"
        admitted = [c for c in changes if c.client == "joiner"]
        assert admitted, f"joiner was not seen; changes={changes}"
        assert admitted[0].event is MembershipEvent.JOINED
        assert "joiner" in server._membership.members
        assert "joiner" in server.client_sharding_dicts

    def test_the_incumbent_is_still_a_member_afterwards(self, cluster):
        """"Existing instances are not interrupted" is half the criterion, and a replan
        that drops them would still pass a joiner-only assertion."""
        server, _, _, _ = cluster

        assert "incumbent" in server.client_infos
        assert server.comm_plan is not None

    def test_the_plan_covers_both_clients(self, cluster):
        """A rebuild that silently planned for only the joiner would leave the incumbent
        registered and unreachable."""
        server, _, _, _ = cluster

        planned = set(server.client_infos)
        assert {"incumbent", "joiner"} <= planned


class TestAMemberLeavesOverRealNixl:
    """The shrink half of E3/E4, end to end on real agents.

    The CPU suite drives `release_member` against stand-ins, which pins the ordering
    and the refusals but cannot show that the MEMBER_LEAVE *arrives* -- and arrival is
    the half that was previously assumed rather than built. Here the server's
    notification has to cross a real agent stream, be recognised by a client that is
    not sitting in any waiter, and come back as an ack before the member is released.

    Runs last in the file: it removes the joiner, so the tests above would see a
    cluster of one.
    """

    def test_an_unconfirmed_leave_refuses_rather_than_releasing(self, cluster):
        """With nobody pumping, the ack never comes. The member must stay drained and
        intact -- freeing memory a peer can still address is the failure this exists
        to prevent, and a timeout is the safe answer to it.

        First, because it needs a peer that could hold the leaver's descriptors; once
        the joiner is gone the incumbent is alone and has nobody to be confirmed by.
        """
        server, _, _, _ = cluster
        assert "joiner" in server._membership.members, "the join test must run first"

        with pytest.raises(TimeoutError, match="confirm revocation"):
            server.release_member("joiner", timeout=1.0)

        assert "joiner" in server._membership.members
        assert "joiner" in server.client_infos
        assert "joiner" not in server._membership.serving, "it must stop being served"

    def test_the_leaver_is_released_only_after_the_peer_confirms(self, cluster):
        import threading

        server, incumbent, _, _ = cluster
        assert "joiner" in server._membership.members

        # The incumbent is doing something else -- which is the point. Nothing in
        # production parks a client in a drain loop waiting to be told about a leave,
        # so the message has to be picked up by a pump the client runs itself.
        stop = threading.Event()
        pumped: list[int] = []

        def pump():
            while not stop.is_set():
                pumped.append(incumbent.absorb_membership_notifs())
                time.sleep(0.01)

        thread = threading.Thread(target=pump, daemon=True)
        thread.start()
        try:
            change = server.release_member("joiner", timeout=30)
        finally:
            stop.set()
            thread.join(timeout=5)

        assert change.event is MembershipEvent.LEFT
        assert sum(pumped) >= 1, "the incumbent never saw the MEMBER_LEAVE"
        # Count, not equality: the previous test's abandoned attempt left announcements
        # queued, and the peer revokes once per message it sees. That duplication is
        # expected and harmless (eviction is idempotent); what matters is that it
        # revoked the leaver and nothing else.
        assert set(incumbent.revoked_clients) == {"joiner"}
        assert "joiner" not in server._membership.members
        assert "joiner" not in server.client_infos

    def test_the_incumbent_survives_the_shrink(self, cluster):
        """The mirror of the join criterion: shrinking must not disturb who remains."""
        server, _, _, _ = cluster

        assert "incumbent" in server.client_infos
        assert server.comm_plan is not None
