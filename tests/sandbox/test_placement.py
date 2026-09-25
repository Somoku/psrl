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
