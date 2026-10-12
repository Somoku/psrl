"""A running cluster admits a new client, and replans for it (item E3).

`handshake_notif` made a late sharding *arrive* in the right registry. That alone changes
nothing: the unified sharding and the comm plan were each built once at startup and
guarded with "already made", so a joiner would sit in `client_sharding_dicts` and appear
in no plan. Nobody would error, and nobody would read from it.

So `admit_joiners` is the entry point the `TODO(lhy): Support elastic clients` named. The
tests here are about the three ways the wiring can be wrong while looking right:

* the joiner is recorded but no replan happens, so it is a member that serves nothing;
* a client A7 **rejected** is replanned in anyway, which is worse than refusing it --
  admitting an indivisible mesh makes every *existing* client raise on re-registration,
  so the cluster loses the members it had;
* admitting re-asks the admission question instead of reusing A7's, giving the cluster
  two places that can disagree about who is allowed in.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
from psrl.utils.nixl.handshake_notif import NotifKind
from psrl.utils.nixl.join_admission import JoinDecision, JoinVerdict
from psrl.utils.nixl.membership import Membership, MembershipEvent
from psrl.utils.nixl.server import NIXLMetaServer

pytestmark = pytest.mark.cpu_test


class _Sharding:
    """Only the attribute `classify_incoming_client` reads."""

    def __init__(self, mesh: dict[int, int]):
        self.shard_mesh = OrderedDict(mesh)


class _Info:
    """A client info carrying just the per-shard shapes `registered_tensor_shapes` reads.

    Shapes are given as the *global* extent; the stub divides by the mesh so the server's
    reconstruction (shard extent x mesh) returns what was asked for.
    """

    def __init__(self, global_shapes: dict[str, tuple[int, ...]], mesh: dict[int, int]):
        self.tensor_infos = {}
        for key, shape in global_shapes.items():
            shard = tuple(e // mesh.get(d, 1) for d, e in enumerate(shape))
            self.tensor_infos[key] = type(
                "_TI", (), {"shard_meta_infos": [type("_M", (), {"shape": shard})()]}
            )()


def _server(*, established: set[str], arriving: dict[str, dict[int, int]], verdict=None):
    """A server mid-flight: `established` already joined, `arriving` just showed up.

    The real `admit_joiners`, `_classify_joiner` and `_drain_notifs` run against a
    stand-in holding only what they touch, so this exercises the shipped code rather
    than a description of it. `verdict` overrides A7 when the test is about what the
    server does with an answer rather than about A7's own logic.
    """

    class _Server:
        def __init__(self):
            self.client_sharding_dicts = {
                name: {"w": _Sharding({0: 4})} for name in established
            }
            for name, mesh in arriving.items():
                self.client_sharding_dicts[name] = {"w": _Sharding(mesh)}
            # One registered member with a real shape, so A7 has something to test
            # divisibility against. Without shapes it can only ever answer RESHARD and
            # the rejection path below would be unreachable.
            self.client_infos = {"train_0": _Info({"w": (24,)}, mesh={0: 4})}
            self._client_temp_mappings = {}
            self.client_unified_sharding_dicts = {"built": True}
            self.finest_shard_meshes = {"w": {0: 4}}
            self.comm_plan = object()
            self.connected_clients = {}
            self._handshake_timeout = 5.0
            self._agents_reported = {kind: set() for kind in NotifKind}
            self._agents_reported[NotifKind.CLIENT_INFO] = {"agent_0"}
            self.calls: list[str] = []
            self._membership = Membership(admit=self._classify_joiner)
            self._membership.members = set(established)

        agent = type("_A", (), {"get_new_notifs": staticmethod(lambda: {})})()

        _classify_joiner = (
            NIXLMetaServer._classify_joiner if verdict is None else lambda self, c: verdict
        )
        classify_incoming_client = NIXLMetaServer.classify_incoming_client
        admit_joiners = NIXLMetaServer.admit_joiners
        _drain_notifs = NIXLMetaServer._drain_notifs
        _add_client = NIXLMetaServer._add_client
        _assert_reported = NIXLMetaServer._assert_reported
        registered_tensor_shapes = NIXLMetaServer.registered_tensor_shapes

        # The replan steps are recorded instead of run: each one needs real client infos
        # and a planner, and what this file is about is *whether and in what order* they
        # are driven after a join.
        def make_unified_sharding(self, *, rebuild=False):
            self.calls.append(f"unified(rebuild={rebuild})")

        def notify_all_client_shardings(self):
            self.calls.append("notify_shardings")

        def make_comm_plan(self, *, rebuild=False):
            self.calls.append(f"comm_plan(rebuild={rebuild})")

        def notify_all_client_infos_and_comm_plan(self):
            self.calls.append("notify_infos_and_plan")

        def _wait_for(self, kind, expected, timeout, what):
            self.calls.append(f"wait({kind.value})")
            self._agents_reported[kind].add("joiner_agent")

    return _Server()


class TestNothingHappensWhenNothingArrives:
    def test_no_joiners_is_an_empty_result_not_an_error(self):
        """Callers may poll this, so "nobody new" must be ordinary."""
        server = _server(established={"train_0"}, arriving={})

        assert server.admit_joiners() == []

    def test_no_replan_runs_when_nobody_joined(self):
        """A replan per poll would rebuild the comm plan continuously."""
        server = _server(established={"train_0"}, arriving={})

        server.admit_joiners()

        assert server.calls == []


class TestAnAdmittedJoinerIsActuallyPlannedFor:
    def test_the_joiner_becomes_a_member(self):
        server = _server(established={"train_0"}, arriving={"train_1": {0: 4}})

        changes = server.admit_joiners()

        assert [c.event for c in changes] == [MembershipEvent.JOINED]
        assert "train_1" in server._membership.members

    def test_both_plans_are_rebuilt_and_republished(self):
        """Recording the join without replanning leaves a member that serves nothing --
        no error anywhere, and the new instance simply never receives a pull."""
        server = _server(established={"train_0"}, arriving={"train_1": {0: 4}})

        server.admit_joiners()

        assert server.calls == [
            "unified(rebuild=True)",
            "notify_shardings",
            "wait(client_info)",
            "comm_plan(rebuild=True)",
            "notify_infos_and_plan",
        ]

    def test_the_joiners_info_is_waited_for_before_the_plan_is_built(self):
        """The comm plan is built from client infos. Building it before the joiner's
        info lands produces a plan that omits the very client being admitted."""
        server = _server(established={"train_0"}, arriving={"train_1": {0: 4}})

        server.admit_joiners()

        assert server.calls.index("wait(client_info)") < server.calls.index(
            "comm_plan(rebuild=True)"
        )

    def test_a_matching_mesh_joins_without_demanding_a_reshard(self):
        """A7's fast path: the finest mesh is unchanged, so existing descriptors stay
        valid and nobody else has to re-register."""
        server = _server(established={"train_0"}, arriving={"train_1": {0: 4}})

        change = server.admit_joiners()[0]

        assert not change.requires_epoch_bump

    def test_a_finer_mesh_is_admitted_but_demands_an_epoch_bump(self):
        """A7's RESHARD: admissible, but every existing member must re-fetch
        descriptors, which is a different obligation from a plain join."""
        server = _server(established={"train_0"}, arriving={"train_1": {0: 8}})

        change = server.admit_joiners()[0]

        assert change.event is MembershipEvent.JOINED
        assert change.requires_epoch_bump
        assert change.reason, "the obligation must come with A7's explanation"

    def test_two_joiners_are_admitted_in_one_replan(self):
        """Replanning per joiner would rebuild the plan N times for one scale-up."""
        server = _server(
            established={"train_0"}, arriving={"train_1": {0: 4}, "train_2": {0: 4}}
        )

        changes = server.admit_joiners()

        assert len(changes) == 2
        assert server.calls.count("comm_plan(rebuild=True)") == 1


class TestARefusedJoinerDoesNotReachThePlan:
    """The failure that costs the cluster its existing members, not just the joiner."""

    def _refusing_server(self):
        return _server(
            established={"train_0"},
            arriving={"train_9": {0: 5}},
            verdict=(False, False, "mesh 5 does not divide 4"),
        )

    def test_a_refused_client_is_reported_as_refused(self):
        server = self._refusing_server()

        changes = server.admit_joiners()

        assert [c.event for c in changes] == [MembershipEvent.REFUSED]
        assert "train_9" not in server._membership.members

    def test_no_replan_runs_for_a_refusal(self):
        """A7 said no, so the cluster is unchanged -- including its plan."""
        server = self._refusing_server()

        server.admit_joiners()

        assert server.calls == []

    def test_the_refused_sharding_is_dropped(self):
        """Left in `client_sharding_dicts` it would be folded into the next rebuild for
        any other joiner, admitting by the back door exactly what A7 rejected."""
        server = self._refusing_server()

        server.admit_joiners()

        assert "train_9" not in server.client_sharding_dicts

    def test_one_refusal_does_not_block_an_admissible_joiner(self):
        server = _server(
            established={"train_0"}, arriving={"train_1": {0: 4}, "train_9": {0: 5}}
        )

        changes = {c.client: c.event for c in server.admit_joiners()}

        assert changes["train_1"] is MembershipEvent.JOINED
        assert changes["train_9"] is MembershipEvent.REFUSED
        assert "train_9" not in server.client_sharding_dicts


class TestAdmissionIsA7sAnswerAndNotASecondOne:
    def test_a_reject_on_any_key_refuses_the_whole_client(self):
        """Partial admission is worse than refusal: the joiner gains nothing and every
        existing client raises when it re-registers against the new mesh."""
        server = _server(established={"train_0"}, arriving={})
        server.finest_shard_meshes = {"ok": {0: 4}, "bad": {0: 4}}
        server.client_infos = {"train_0": _Info({"ok": (24,), "bad": (24,)}, mesh={0: 4})}
        server.client_sharding_dicts["train_0"] = {
            "ok": _Sharding({0: 4}),
            "bad": _Sharding({0: 4}),
        }
        server.client_sharding_dicts["train_9"] = {
            "ok": _Sharding({0: 4}),
            "bad": _Sharding({0: 5}),
        }

        allowed, _, reason = server._classify_joiner("train_9")

        assert not allowed
        assert "bad" in reason

    def test_an_unknown_key_is_not_treated_as_a_rejection(self):
        """A joiner declaring a key no established client has is a FAST_PATH for that
        key -- there is no established mesh to conflict with."""
        server = _server(established={"train_0"}, arriving={})
        server.client_sharding_dicts["train_9"] = {"brand_new": _Sharding({0: 7})}

        allowed, needs_reshard, _ = server._classify_joiner("train_9")

        assert allowed
        assert not needs_reshard

    def test_the_verdict_comes_from_classify_incoming_client(self):
        """One place decides admissibility. A second copy of the rule is how the
        handshake and the elastic path drift into disagreeing."""
        import inspect

        body = inspect.getsource(NIXLMetaServer._classify_joiner)

        assert "classify_incoming_client" in body
        assert "merge_finest_mesh" not in body, "must not re-derive A7's judgement"

    def test_the_refusal_reason_carries_a7s_explanation(self):
        """"Refused" without the mesh arithmetic leaves the operator with no next step."""
        server = _server(established={"train_0"}, arriving={})
        server.client_sharding_dicts["train_9"] = {"w": _Sharding({0: 5})}

        decision = JoinDecision(JoinVerdict.REJECT, {0: 4}, {0: 12})
        assert decision.describe(), "A7 provides the text; this is what gets surfaced"

        allowed, _, reason = server._classify_joiner("train_9")
        if not allowed:
            assert "w" in reason and reason != "w: "


class TestTheBuildersRefuseAnAccidentalRebuild:
    """`rebuild` is explicit so an out-of-order handshake step still fails loudly.

    Deleting the guard outright would make elasticity work and silently allow a
    double-build during startup, which is a real bug and used to be caught.
    """

    def test_rebuilding_unified_sharding_without_asking_is_refused(self):
        server = _server(established={"train_0"}, arriving={})
        server._agents_reported[NotifKind.SHARDING] = {"agent_0"}

        with pytest.raises(AssertionError, match="rebuild=True"):
            NIXLMetaServer.make_unified_sharding(server)

    def test_rebuilding_the_comm_plan_without_asking_is_refused(self):
        server = _server(established={"train_0"}, arriving={})

        with pytest.raises(AssertionError, match="rebuild=True"):
            NIXLMetaServer.make_comm_plan(server)
