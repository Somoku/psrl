"""Node selection and the reservation protocol.

The protocol exists because a reservation is held in two places at once, and a slot
that is never released starves a cluster one sandbox at a time.
"""

from __future__ import annotations

import pytest
from psrl.sandbox.core import ResumeLevel, SandboxFeature
from psrl.sandbox.placement import (
    NodeCapabilities,
    NoPlacementCandidate,
    PlacementCapacityExhausted,
    PlacementRequest,
    PlacementService,
    fleet_capabilities,
)

pytestmark = pytest.mark.cpu_test

_DOCKER_FEATURES = frozenset(
    {
        SandboxFeature.FREEZE,
        SandboxFeature.FILESYSTEM_SNAPSHOT,
        SandboxFeature.RESTORE,
        SandboxFeature.RESUME_ANYWHERE,
        SandboxFeature.HOST_MOUNT,
    }
)


def _node(node_id: str = "node-a", **overrides) -> NodeCapabilities:
    payload = {
        "node_id": node_id,
        "backend": "docker",
        "features": sorted(feature.value for feature in _DOCKER_FEATURES),
        "resume_level": ResumeLevel.FILESYSTEM.value,
        "gpu_count": 0,
        "host_mounts": True,
        "labels": [],
        "reachable_session_server": True,
        "image_digests": [],
    }
    payload.update(overrides)
    return NodeCapabilities.from_advertisement(payload)


def _service(now: float = 1000.0) -> PlacementService:
    service = PlacementService(node_ttl_s=100.0, reservation_ttl_s=20.0)
    return service


def test_a_registered_node_is_a_candidate() -> None:
    service = _service()
    service.register(_node(), now=1000.0)

    decision = service.choose(PlacementRequest(backend="docker"), now=1000.0)

    assert decision.node_id == "node-a"
    assert decision.reservation_id


def test_a_node_that_missed_its_heartbeat_is_drained_not_trusted() -> None:
    # A cached load is what makes placement over-admit, so a stale node is refused
    # rather than used with a stale number.
    service = _service()
    service.register(_node(), now=1000.0)

    with pytest.raises(NoPlacementCandidate, match="drained"):
        service.choose(PlacementRequest(backend="docker"), now=1200.0)

    assert service.snapshot(now=1200.0).drained_nodes == 1


def test_a_heartbeat_returns_a_drained_node_to_service() -> None:
    service = _service()
    service.register(_node(), now=1000.0)
    service.heartbeat("node-a", now=1200.0)

    assert service.choose(PlacementRequest(backend="docker"), now=1205.0).node_id == "node-a"


def test_a_missing_feature_is_not_a_slower_candidate() -> None:
    service = _service()
    service.register(_node(features=[SandboxFeature.FREEZE.value]), now=1000.0)

    with pytest.raises(NoPlacementCandidate, match="required features"):
        service.choose(
            PlacementRequest(backend="docker", required_features=frozenset({SandboxFeature.RESTORE})),
            now=1000.0,
        )


def test_a_weaker_resume_level_cannot_satisfy_a_stronger_requirement() -> None:
    # This is the rule that keeps a full-state resume off a filesystem backend.
    service = _service()
    service.register(_node(resume_level=ResumeLevel.FILESYSTEM.value), now=1000.0)

    with pytest.raises(NoPlacementCandidate, match="resume level"):
        service.choose(
            PlacementRequest(backend="docker", required_resume_level=ResumeLevel.FULL_STATE),
            now=1000.0,
        )


def test_a_full_state_node_satisfies_a_weaker_requirement() -> None:
    service = _service()
    service.register(_node(resume_level=ResumeLevel.FULL_STATE.value), now=1000.0)

    assert service.choose(
        PlacementRequest(backend="docker", required_resume_level=ResumeLevel.FILESYSTEM),
        now=1000.0,
    )


@pytest.mark.parametrize(
    ("overrides", "needed"),
    [
        ({"host_mounts": False}, PlacementRequest(backend="docker", requires_host_mount=True)),
        ({"gpu_count": 0}, PlacementRequest(backend="docker", gpu_count=1)),
        ({"labels": []}, PlacementRequest(backend="docker", required_label="env")),
        ({"reachable_session_server": False}, PlacementRequest(backend="docker")),
        ({"backend": "e2b"}, PlacementRequest(backend="docker")),
    ],
)
def test_a_node_that_cannot_host_the_request_is_not_a_candidate(overrides, needed) -> None:
    # Each of these is a wrong node rather than a slow one: a sandbox that cannot
    # record TITO tokens, or that runs the wrong backend, is not a fallback.
    service = _service()
    service.register(_node(**overrides), now=1000.0)

    with pytest.raises(NoPlacementCandidate):
        service.choose(needed, now=1000.0)


def test_an_empty_registry_explains_itself() -> None:
    with pytest.raises(NoPlacementCandidate, match="No sandbox node has reported"):
        _service().choose(PlacementRequest(), now=1000.0)


def test_locality_outranks_load() -> None:
    service = _service()
    service.register(_node("cold", image_digests=[]), now=1000.0)
    service.register(_node("warm", image_digests=["sha256:task"]), now=1000.0)
    service.heartbeat("cold", load=0, now=1000.0)
    service.heartbeat("warm", load=5, now=1000.0)

    decision = service.choose(
        PlacementRequest(backend="docker", image_digests=frozenset({"sha256:task"})),
        now=1000.0,
    )

    assert decision.node_id == "warm"


def test_load_breaks_a_tie_between_equally_local_nodes() -> None:
    service = _service()
    service.register(_node("busy"), now=1000.0)
    service.register(_node("idle"), now=1000.0)
    service.heartbeat("busy", load=3, now=1000.0)
    service.heartbeat("idle", load=0, now=1000.0)

    assert service.choose(PlacementRequest(backend="docker"), now=1000.0).node_id == "idle"


def test_an_equal_choice_is_stable() -> None:
    # Two equally good nodes must not alternate, or a run's placement is impossible
    # to reproduce.
    service = _service()
    service.register(_node("node-b"), now=1000.0)
    service.register(_node("node-a"), now=1000.0)

    assert service.choose(PlacementRequest(backend="docker"), now=1000.0).node_id == "node-a"


def test_a_reservation_raises_the_node_load_until_it_is_returned() -> None:
    # Two equally good nodes start on the stable tie-break, and the one that took the
    # reservation is the one the next request avoids.
    service = _service()
    service.register(_node("node-a"), now=1000.0)
    service.register(_node("node-b"), now=1000.0)

    first = service.choose(PlacementRequest(backend="docker"), now=1000.0)
    second = service.choose(PlacementRequest(backend="docker"), now=1000.0)

    assert first.node_id == "node-a"
    assert second.node_id == "node-b"

    service.release(first.reservation_id)
    service.cancel(second.reservation_id)
    assert service.choose(PlacementRequest(backend="docker"), now=1000.0).node_id == "node-a"


def test_cancelling_a_reservation_returns_the_slot() -> None:
    service = _service()
    service.register(_node("idle"), now=1000.0)
    service.register(_node("other"), now=1000.0)
    reservation = service.choose(PlacementRequest(backend="docker"), now=1000.0)

    service.cancel(reservation.reservation_id)

    assert service.snapshot(now=1000.0).reservations_open == 0
    assert service.choose(PlacementRequest(backend="docker"), now=1000.0).node_id == "idle"


def test_releasing_a_reservation_returns_the_slot() -> None:
    service = _service()
    service.register(_node("idle"), now=1000.0)
    service.register(_node("other"), now=1000.0)
    reservation = service.choose(PlacementRequest(backend="docker"), now=1000.0)

    service.release(reservation.reservation_id)

    assert service.snapshot(now=1000.0).reservations_open == 0


def test_releasing_and_cancelling_an_unknown_reservation_is_not_an_error() -> None:
    service = _service()

    service.release("missing")
    service.cancel("missing")

    assert service.snapshot().reservations_open == 0


def test_a_renewed_reservation_survives_past_its_ttl() -> None:
    # Placement holds a cache, not a ledger: it cannot tell a live long-lived sandbox from an
    # abandoned reservation by age alone, so an owner that keeps renewing keeps the slot.
    service = _service()
    service.register(_node(), now=1000.0)
    reservation = service.choose(PlacementRequest(backend="docker"), now=1000.0)
    # The TTL is 20s, so a renewal at 1050 keeps the slot at 1060 and 1065.
    service.renew(reservation.reservation_id, now=1050.0)

    assert service.sweep(now=1060.0) == []
    assert service.snapshot(now=1060.0).reservations_open == 1
    # Only once the renewals actually stop, past the TTL from the last one, does the
    # slot come back.
    assert service.sweep(now=1075.0) == [reservation.reservation_id]
    assert service.snapshot(now=1075.0).reservations_open == 0


def test_a_reservation_nobody_renews_is_swept_once_its_ttl_elapses() -> None:
    service = _service()
    service.register(_node(), now=1000.0)
    reservation = service.choose(PlacementRequest(backend="docker"), now=1000.0)

    assert service.sweep(now=1005.0) == []
    assert service.sweep(now=1030.0) == [reservation.reservation_id]
    assert service.snapshot(now=1030.0).reservations_swept == 1


def test_sweeping_returns_the_slots_it_took() -> None:
    service = _service()
    service.register(_node("idle"), now=1000.0)
    service.register(_node("other"), now=1000.0)
    for _ in range(2):
        service.choose(PlacementRequest(backend="docker"), now=1000.0)
    service.sweep(now=1030.0)

    service.register(_node("idle"), now=1030.0)

    assert service.choose(PlacementRequest(backend="docker"), now=1030.0).node_id == "idle"


def test_a_reservation_ttl_must_expire_before_the_node_ttl() -> None:
    # Otherwise a drained node leaves reservations pointing at nothing.
    with pytest.raises(ValueError, match="shorter than the node TTL"):
        PlacementService(node_ttl_s=10.0, reservation_ttl_s=10.0)


def test_unregistering_a_node_drops_its_reservations() -> None:
    service = _service()
    service.register(_node(), now=1000.0)
    service.choose(PlacementRequest(backend="docker"), now=1000.0)

    service.unregister("node-a")

    assert service.nodes() == ()
    assert service.snapshot(now=1000.0).reservations_open == 0


def test_a_heartbeat_for_an_unknown_node_is_ignored() -> None:
    service = _service()

    service.heartbeat("ghost", load=3, now=1000.0)

    assert service.nodes() == ()


def test_the_snapshot_reports_what_an_operator_acts_on() -> None:
    service = _service()
    service.register(_node("live"), now=1000.0)
    service.register(_node("stale"), now=800.0)
    reservation = service.choose(PlacementRequest(backend="docker"), now=1000.0)

    snapshot = service.snapshot(now=1000.0)

    assert snapshot.nodes == 2
    assert snapshot.available_nodes == 1
    assert snapshot.drained_nodes == 1
    assert snapshot.reservations_open == 1
    assert snapshot.oldest_reservation_age_s >= 0
    assert "placement/reservations_open" in snapshot.as_dict()
    assert service.snapshot(now=1000.0).reservations_open == 1
    service.cancel(reservation.reservation_id)


def test_a_rejected_request_is_counted_as_a_rejection() -> None:
    service = _service()
    service.register(_node(gpu_count=0), now=1000.0)

    with pytest.raises(NoPlacementCandidate):
        service.choose(PlacementRequest(backend="docker", gpu_count=2), now=1000.0)

    assert service.snapshot(now=1000.0).rejections >= 1


async def test_shutdown_drops_the_registry_and_the_sweeper() -> None:
    service = _service()
    service.register(_node(), now=1000.0)
    service.choose(PlacementRequest(backend="docker"), now=1000.0)

    await service.shutdown()

    assert service.nodes() == ()
    assert service.snapshot().reservations_open == 0


def test_locality_accepts_a_reference_when_the_caller_has_no_digest() -> None:
    # A caller usually knows an image by its tag, so an index of references is what
    # makes locality usable without a registry round trip.
    service = _service()
    service.register(_node("cold"), now=1000.0)
    service.register(_node("warm", image_references=["ghcr.io/org/task:latest"]), now=1000.0)
    service.heartbeat("cold", load=0, now=1000.0)
    service.heartbeat("warm", load=3, now=1000.0)

    decision = service.choose(
        PlacementRequest(backend="docker", image_references=frozenset({"ghcr.io/org/task:latest"})),
        now=1000.0,
    )

    assert decision.node_id == "warm"


def test_an_exact_digest_outranks_a_matching_reference() -> None:
    service = _service()
    service.register(_node("by-reference", image_references=["image:latest"]), now=1000.0)
    service.register(_node("by-digest", image_digests=["registry/image@sha256:abc"]), now=1000.0)

    decision = service.choose(
        PlacementRequest(
            backend="docker",
            image_references=frozenset({"image:latest"}),
            image_digests=frozenset({"registry/image@sha256:abc"}),
        ),
        now=1000.0,
    )

    assert decision.node_id == "by-digest"


def test_a_draining_node_is_not_a_candidate() -> None:
    # It still holds containers whose memory its envelope already returned.
    service = _service()
    service.register(_node("leaking", draining=True), now=1000.0)
    service.register(_node("healthy"), now=1000.0)

    assert service.choose(PlacementRequest(backend="docker"), now=1000.0).node_id == "healthy"

    service.unregister("healthy")
    with pytest.raises(NoPlacementCandidate):
        service.choose(PlacementRequest(backend="docker"), now=1000.0)


def test_locality_credits_a_digest_the_chosen_node_already_holds() -> None:
    # The operator's question is whether tasks land where their image already is.
    service = _service()
    held = "registry.internal:5000/base@sha256:aaa"
    service.register(_node(image_digests=[held], now=1000.0), now=1000.0)

    service.choose(PlacementRequest(backend="docker", image_digests=[held]), now=1000.0)

    assert service.snapshot().as_dict()["image/locality_hit_ratio"] == 1.0


def test_a_moved_reference_cannot_claim_a_locality_hit() -> None:
    # A request that names a digest needs that digest. A node holding the same tag at a different
    # digest would still pull, so crediting the reference would flatter the ratio and hide the cost.
    service = _service()
    service.register(
        _node(image_digests=["registry.internal:5000/base@sha256:old"], image_references=["base:latest"]),
        now=1000.0,
    )

    service.choose(
        PlacementRequest(
            backend="docker",
            image_digests=["registry.internal:5000/base@sha256:new"],
            image_references=["base:latest"],
        ),
        now=1000.0,
    )

    assert service.snapshot().as_dict()["image/locality_hit_ratio"] == 0.0


def test_a_reference_only_request_is_answered_by_the_reference() -> None:
    # A caller usually knows only a tag, and a node holding that tag serves it from its
    # own daemon without pulling, which is exactly a hit.
    service = _service()
    service.register(_node(image_references=["base:latest"]), now=1000.0)

    service.choose(PlacementRequest(backend="docker", image_references=["base:latest"]), now=1000.0)

    assert service.snapshot().as_dict()["image/locality_hit_ratio"] == 1.0


def test_a_request_naming_no_image_is_not_counted() -> None:
    # A sandbox created from a snapshot or a template asks no locality question, so it
    # must not dilute the ratio in either direction.
    service = _service()
    service.register(_node(image_references=["base:latest"]), now=1000.0)

    service.choose(PlacementRequest(backend="docker"), now=1000.0)

    snapshot = service.snapshot()
    assert snapshot.locality_asked == 0
    assert snapshot.as_dict()["image/locality_hit_ratio"] == 0.0


def test_candidates_ranks_the_nodes_a_request_could_use() -> None:
    # A prefetch warms a run's working set on these, so the order matters as much as the
    # membership does.
    service = _service()
    service.register(_node("node-a", image_references=["base:latest"]), now=1000.0)
    service.register(_node("node-b"), now=1000.0)

    assert service.candidates(PlacementRequest(backend="docker", image_references=["base:latest"]), now=1000.0) == [
        "node-a",
        "node-b",
    ]
    assert service.candidates(PlacementRequest(backend="docker"), now=1000.0, limit=1) == ["node-a"]


def test_candidates_is_empty_when_nothing_can_serve_the_request() -> None:
    service = _service()
    service.register(_node(), now=1000.0)

    assert service.candidates(PlacementRequest(backend="nonexistent"), now=1000.0) == []


def test_a_fleet_declares_only_what_every_node_can_do() -> None:
    # A caller declares one capability set while the nodes underneath differ, so a feature
    # some node lacks is one the caller cannot rely on. Placement still filters per node.
    capabilities = fleet_capabilities(
        [
            {"backends": [{"features": ["restore", "resume_anywhere", "freeze"], "resume_level": "full_state"}]},
            {"backends": [{"features": ["restore", "resume_anywhere"], "resume_level": "filesystem"}]},
        ]
    )

    assert capabilities.supports(SandboxFeature.RESTORE)
    assert capabilities.supports(SandboxFeature.RESUME_ANYWHERE)
    assert not capabilities.supports(SandboxFeature.FREEZE)
    # The weakest resume wins, because a promise one node cannot keep is not a promise.
    assert capabilities.resume_level is ResumeLevel.FILESYSTEM


def test_a_fleet_that_cannot_resume_anywhere_declares_no_resume_level() -> None:
    # A level without RESUME_ANYWHERE would be a level for a resume nobody offers, which the
    # capability type refuses to be constructed with.
    capabilities = fleet_capabilities([{"backends": [{"features": ["restore"], "resume_level": "full_state"}]}])

    assert not capabilities.supports(SandboxFeature.RESUME_ANYWHERE)
    assert capabilities.resume_level is None


def test_a_node_that_declares_no_resume_level_weakens_the_fleet_to_none() -> None:
    capabilities = fleet_capabilities(
        [
            {"backends": [{"features": ["resume_anywhere"], "resume_level": "full_state"}]},
            {"backends": [{"features": ["resume_anywhere"]}]},
        ]
    )

    assert capabilities.resume_level is None


def test_a_fleet_with_no_nodes_declares_nothing() -> None:
    capabilities = fleet_capabilities([])

    assert capabilities.features == frozenset()
    assert capabilities.resume_level is None


def _sized(node_id: str, *, free_mb: int, total_mb: int = 1408 * 1024, **overrides) -> NodeCapabilities:
    """A node that reports an envelope, which is what makes it filterable on capacity."""
    return _node(
        node_id,
        available_memory_mb=free_mb,
        available_cpu_millis=free_mb // 8,
        envelope_memory_mb=total_mb,
        envelope_cpu_millis=total_mb // 8,
        **overrides,
    )


class TestCapacityIsAConstraintNotAPreference:
    """Placement must not send a sandbox where it cannot fit.

    Selection used to be capability, then image locality, then reservation count, and
    nothing in it asked whether the chosen node had room. A batch of same-image requests
    therefore all went to the one node holding that image, were all reserved there, and
    then queued at the node's own admission for an episode's length — with no error, since
    every layer believed it had done its job.
    """

    def test_a_node_without_room_is_not_a_candidate(self) -> None:
        service = _service()
        service.register(_sized("full", free_mb=8 * 1024), now=1000.0)
        service.register(_sized("roomy", free_mb=512 * 1024), now=1000.0)

        decision = service.choose(
            PlacementRequest(backend="docker", memory_mb=16 * 1024, cpu_millis=2000),
            now=1000.0,
        )

        assert decision.node_id == "roomy"

    def test_a_full_fleet_is_a_wait_not_a_dead_end(self) -> None:
        """The two failures need different recourse, so they are different exceptions."""
        service = _service()
        service.register(_sized("full", free_mb=1024), now=1000.0)

        with pytest.raises(PlacementCapacityExhausted, match="full"):
            service.choose(
                PlacementRequest(backend="docker", memory_mb=16 * 1024, cpu_millis=2000),
                now=1000.0,
            )
        # Not the other one: that says the fleet can never run this, which is untrue here
        # and would have the caller fail a rollout that only needed to wait.
        assert service.snapshot(now=1000.0).capacity_exhausted == 1
        assert service.snapshot(now=1000.0).no_candidate == 0

    def test_an_impossible_request_is_still_a_dead_end(self) -> None:
        service = _service()
        service.register(_sized("roomy", free_mb=512 * 1024), now=1000.0)

        with pytest.raises(NoPlacementCandidate):
            service.choose(
                PlacementRequest(backend="docker", gpu_count=8, memory_mb=1024, cpu_millis=1000),
                now=1000.0,
            )

    def test_a_node_that_declares_no_envelope_is_never_filtered_out(self) -> None:
        """A provider backend schedules its own capacity, so this service cannot judge it."""
        service = _service()
        service.register(_node("provider"), now=1000.0)

        decision = service.choose(
            PlacementRequest(backend="docker", memory_mb=1024 * 1024, cpu_millis=99000),
            now=1000.0,
        )

        assert decision.node_id == "provider"

    def test_a_request_with_no_declared_size_is_not_filtered(self) -> None:
        # Zero is the absence of a measurement, not a free sandbox.
        service = _service()
        service.register(_sized("full", free_mb=0), now=1000.0)

        assert service.choose(PlacementRequest(backend="docker"), now=1000.0).node_id == "full"


class TestBalanceOutranksLocality:
    """An image pull is an optimisation; a full node is a constraint."""

    def test_a_loaded_local_node_loses_to_an_idle_remote_one(self) -> None:
        service = _service()
        service.register(
            _sized("warm-but-loaded", free_mb=64 * 1024, image_digests=["sha256:task"]),
            now=1000.0,
        )
        service.register(_sized("cold-but-idle", free_mb=1408 * 1024), now=1000.0)

        decision = service.choose(
            PlacementRequest(
                backend="docker",
                image_digests=frozenset({"sha256:task"}),
                memory_mb=16 * 1024,
                cpu_millis=2000,
            ),
            now=1000.0,
        )

        assert decision.node_id == "cold-but-idle"

    def test_locality_still_decides_between_similarly_loaded_nodes(self) -> None:
        """Otherwise balance would have demoted locality to something that never fires."""
        service = _service()
        service.register(_sized("cold", free_mb=1408 * 1024), now=1000.0)
        service.register(
            _sized("warm", free_mb=1400 * 1024, image_digests=["sha256:task"]),
            now=1000.0,
        )

        decision = service.choose(
            PlacementRequest(
                backend="docker",
                image_digests=frozenset({"sha256:task"}),
                memory_mb=16 * 1024,
                cpu_millis=2000,
            ),
            now=1000.0,
        )

        assert decision.node_id == "warm"


class TestReservationsAreChargedBeforeTheGrantLands:
    """A promise has to cost something the moment it is made.

    A node cannot report a grant until it has made one, so between `choose` and the
    node's next report its headroom still counts that room as free. A batch of concurrent
    callers all read that same stale figure, all pick the same node, and the node ends up
    holding more sandboxes than its envelope admits.
    """

    def test_a_batch_spreads_across_the_fleet_without_any_report(self) -> None:
        service = _service()
        # Two identical nodes, each with room for exactly two of these sandboxes.
        service.register(_sized("node-a", free_mb=32 * 1024, total_mb=32 * 1024), now=1000.0)
        service.register(_sized("node-b", free_mb=32 * 1024, total_mb=32 * 1024), now=1000.0)
        request = PlacementRequest(backend="docker", memory_mb=16 * 1024, cpu_millis=2000)

        chosen = [service.choose(request, now=1000.0).node_id for _ in range(4)]

        assert sorted(chosen) == ["node-a", "node-a", "node-b", "node-b"]
        # And the fifth has nowhere to go, rather than being stacked onto a full node.
        with pytest.raises(PlacementCapacityExhausted):
            service.choose(request, now=1000.0)

    def test_a_retired_reservation_gives_its_room_back(self) -> None:
        service = _service()
        service.register(_sized("only", free_mb=16 * 1024, total_mb=16 * 1024), now=1000.0)
        request = PlacementRequest(backend="docker", memory_mb=16 * 1024, cpu_millis=2000)
        decision = service.choose(request, now=1000.0)

        service.cancel(decision.reservation_id)

        # Released on every path a reservation can end, or headroom leaks and the node is
        # avoided while it sits idle.
        assert service.choose(request, now=1000.0).node_id == "only"

    def test_a_fresh_report_supersedes_the_promises_it_already_includes(self) -> None:
        service = _service()
        service.register(_sized("only", free_mb=32 * 1024, total_mb=32 * 1024), now=1000.0)
        request = PlacementRequest(backend="docker", memory_mb=16 * 1024, cpu_millis=2000)
        service.choose(request, now=1000.0)

        # The node now reports the grant itself, so the promise must not be counted twice.
        service.heartbeat(
            "only",
            headroom={
                "available_memory_mb": 16 * 1024,
                "available_cpu_millis": 2000,
                "envelope_memory_mb": 32 * 1024,
                "envelope_cpu_millis": 4000,
            },
            now=1000.0,
        )

        assert service.choose(request, now=1000.0).node_id == "only"

    def test_a_reregistration_keeps_promises_its_advertisement_cannot_see(self) -> None:
        service = _service()
        service.register(_sized("only", free_mb=32 * 1024, total_mb=32 * 1024), now=1000.0)
        request = PlacementRequest(backend="docker", memory_mb=16 * 1024, cpu_millis=2000)
        service.choose(request, now=1000.0)

        # Re-advertised with the same headroom, because the grant has not landed yet.
        service.register(_sized("only", free_mb=32 * 1024, total_mb=32 * 1024), now=1000.0)

        # One slot left, not two: the open reservation still holds the other.
        assert service.choose(request, now=1000.0).node_id == "only"
        with pytest.raises(PlacementCapacityExhausted):
            service.choose(request, now=1000.0)


class TestHeadroomIsComparedPerClass:
    """A class cannot borrow another's unmet guarantee, so the remainder overstates it.

    The node reported `available_capacity` — the envelope's remainder — and placement
    compared every request against it. On a 1760 GiB node that read as room for 110
    sandboxes, while the coordinator would admit 88 at best and 39 once the grader class
    queued. Placement sent the work, the node queued it correctly, and the difference
    surfaced only as sandboxes that never started.
    """

    @staticmethod
    def _classed(node_id: str, *, rollout_mb: int, envelope_mb: int) -> NodeCapabilities:
        return _node(
            node_id,
            available_memory_mb=envelope_mb,
            available_cpu_millis=envelope_mb // 8,
            envelope_memory_mb=envelope_mb,
            envelope_cpu_millis=envelope_mb // 8,
            class_headroom={
                "rollout": {"memory_mb": rollout_mb, "cpu_millis": rollout_mb // 8},
                "grader": {"memory_mb": 0, "cpu_millis": 0},
            },
        )

    def test_a_request_is_held_to_its_own_class_not_the_envelope(self) -> None:
        service = _service()
        # The envelope has room for four of these, but rollout's own share has room for one.
        service.register(self._classed("only", rollout_mb=16 * 1024, envelope_mb=64 * 1024), now=1000.0)
        request = PlacementRequest(backend="docker", memory_mb=16 * 1024, cpu_millis=2000, resource_class="rollout")

        assert service.choose(request, now=1000.0).node_id == "only"
        # The second would fit the envelope and not the class, which is what admission
        # would have decided, so placement must decide the same way.
        with pytest.raises(PlacementCapacityExhausted, match="rollout"):
            service.choose(request, now=1000.0)

    def test_a_class_with_no_headroom_reported_falls_back_to_the_envelope(self) -> None:
        """A node that says nothing class-specific is bounded only by its remainder."""
        service = _service()
        service.register(self._classed("only", rollout_mb=16 * 1024, envelope_mb=64 * 1024), now=1000.0)

        # `default` is absent from this node's class_headroom, so the remainder applies.
        request = PlacementRequest(backend="docker", memory_mb=48 * 1024, cpu_millis=2000)

        assert service.choose(request, now=1000.0).node_id == "only"

    def test_a_starved_class_is_refused_even_on_an_empty_node(self) -> None:
        """Grader's share is fully reserved here, so no grader sandbox fits."""
        service = _service()
        service.register(self._classed("idle", rollout_mb=64 * 1024, envelope_mb=64 * 1024), now=1000.0)

        with pytest.raises(PlacementCapacityExhausted, match="grader"):
            service.choose(
                PlacementRequest(backend="docker", memory_mb=1024, cpu_millis=1000, resource_class="grader"),
                now=1000.0,
            )

    def test_balance_is_measured_within_the_requesting_class(self) -> None:
        service = _service()
        # Same envelope, but node-b's rollout share is nearly spent.
        service.register(self._classed("node-a", rollout_mb=64 * 1024, envelope_mb=64 * 1024), now=1000.0)
        service.register(self._classed("node-b", rollout_mb=16 * 1024, envelope_mb=64 * 1024), now=1000.0)

        decision = service.choose(
            PlacementRequest(backend="docker", memory_mb=16 * 1024, cpu_millis=2000, resource_class="rollout"),
            now=1000.0,
        )

        assert decision.node_id == "node-a"

    def test_a_heartbeat_refreshes_the_per_class_view(self) -> None:
        service = _service()
        service.register(self._classed("only", rollout_mb=16 * 1024, envelope_mb=64 * 1024), now=1000.0)
        request = PlacementRequest(backend="docker", memory_mb=16 * 1024, cpu_millis=2000, resource_class="rollout")
        service.choose(request, now=1000.0)

        # The grader drained, so rollout may now borrow more than its guarantee.
        service.heartbeat(
            "only",
            headroom={
                "available_memory_mb": 64 * 1024,
                "available_cpu_millis": 8000,
                "envelope_memory_mb": 64 * 1024,
                "envelope_cpu_millis": 8000,
                "class_headroom": {"rollout": {"memory_mb": 48 * 1024, "cpu_millis": 6000}},
            },
            now=1000.0,
        )

        assert service.choose(request, now=1000.0).node_id == "only"
