"""Cluster-wide node selection and the two-sided reservation protocol.

A sandbox node advertises what it can do and how much room it has left, and this service
picks one. Selection is capability match, then room, then how loaded the node is, then
image locality — requirements before optimisations. Ranking locality first let a
per-instance image draw a whole batch onto the one node holding it, and with no room check
those requests were reserved onto a node that could admit a fraction of them; the rest
queued at that node for an episode's length with nothing anywhere reporting a fault.

Room has to be reported rather than inferred. A reservation count says where this service
has sent work, not where work fits, and the two diverge exactly when the cluster is full.

The reservation protocol exists because a reservation is held in two places at
once: here and on the node that will enforce it. Without a release path, every
rejected or lost provision leaks a slot and the cluster starves one sandbox at a
time. The rules are the same ones the single-node envelope already uses: a lease
with an id, an owner, a TTL, and a sweeper, and cancel kept separate from release
so an operator can tell a withdrawn request from a finished sandbox.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from psrl.sandbox.core import ResumeLevel, SandboxCapabilities, SandboxFeature

psrl_logger = logging.getLogger(__file__)

# A node that has missed this long is drained rather than trusted. A stale report is
# worse than no report, because a cached load is what makes placement over-admit.
DEFAULT_NODE_TTL_S = 120.0


# How finely utilisation is compared before locality is allowed to decide. Ten buckets
# means nodes within about a tenth of each other count as equally loaded, which leaves
# locality a real say without letting it override a node that is genuinely fuller.
_BALANCE_BUCKETS = 10


class NoPlacementCandidate(RuntimeError):
    """
    Raised when no registered node satisfies a request.

    This is a capacity planning fault rather than a task failure: the request never
    reached a node, so nothing about the workload was learned.
    """


class PlacementCapacityExhausted(RuntimeError):
    """Raised when every capable node is full right now.

    Deliberately not a `NoPlacementCandidate`. That one says the fleet can never run this
    request, which is a deployment fault and will not improve by waiting. This says the
    fleet can run it but has no room at this instant, which resolves as sandboxes are
    released — so the caller's recourse is to wait rather than to fail the task. Placing
    it anyway is what the old code did, and the request then waited at a node that could
    not admit it, with no record anywhere that it was queued.
    """


def _optional_int(value: Any) -> int | None:
    """Return an int, preserving `None` as the absence of a declared envelope.

    Zero and `None` must not collapse into each other here: zero means a node that is
    full right now, `None` means a node that never had an envelope to be full of.
    """
    return None if value is None else int(value)


@dataclass(frozen=True)
class NodeCapabilities:
    """
    What one sandbox node can do.
    """

    node_id: str
    backend: str
    features: frozenset[SandboxFeature] = frozenset()
    resume_level: ResumeLevel | None = None
    gpu_count: int = 0
    # Host bind mounts are a node property: a node without them cannot serve a spec
    # that names one, however capable its backend claims to be.
    host_mounts: bool = False
    # Node classes a request can demand, such as `env` for a dedicated environment node.
    labels: frozenset[str] = frozenset()
    # Whether this node can reach the session server that records TITO tokens.
    reachable_session_server: bool = True
    # Whether the node is refusing work because it cannot confirm earlier cleanup.
    draining: bool = False
    # Every image digest the node's daemon already holds, and the references it holds
    # them under. A caller knows only the reference, so a reference index makes locality usable.
    image_digests: frozenset[str] = frozenset()
    image_references: frozenset[str] = frozenset()
    # What this node's admission envelope can still grant, as the node last reported it.
    # Placement cannot decide where a sandbox fits without it: ranking by a reservation
    # count says which node this service has sent the most work to, not which node can
    # take more, and the two diverge exactly when the cluster is full. `None` means the
    # node declared no envelope, which is how a provider backend that schedules its own
    # capacity opts out of this filter rather than being refused by it.
    available_memory_mb: int | None = None
    available_cpu_millis: int | None = None
    # The envelope's total, kept alongside the remainder so a balance comparison is a
    # fraction rather than an absolute: two nodes of different sizes are otherwise ordered
    # by capacity rather than by how loaded they are.
    envelope_memory_mb: int | None = None
    envelope_cpu_millis: int | None = None

    @classmethod
    def from_advertisement(cls, payload: Mapping[str, object]) -> NodeCapabilities:
        """
        Build a capability record from a node's advertisement.
        """

        def _optional_int(key: str) -> int | None:
            value = payload.get(key)
            return None if value is None else int(value)

        return cls(
            node_id=str(payload["node_id"]),
            backend=str(payload["backend"]),
            features=frozenset(SandboxFeature(str(item)) for item in payload.get("features", ())),
            resume_level=(
                ResumeLevel(str(payload["resume_level"])) if payload.get("resume_level") is not None else None
            ),
            gpu_count=int(payload.get("gpu_count", 0) or 0),
            host_mounts=bool(payload.get("host_mounts", False)),
            labels=frozenset(str(item) for item in payload.get("labels", ())),
            reachable_session_server=bool(payload.get("reachable_session_server", True)),
            draining=bool(payload.get("draining", False)),
            image_digests=frozenset(str(item) for item in payload.get("image_digests", ())),
            image_references=frozenset(str(item) for item in payload.get("image_references", ())),
            available_memory_mb=_optional_int("available_memory_mb"),
            available_cpu_millis=_optional_int("available_cpu_millis"),
            envelope_memory_mb=_optional_int("envelope_memory_mb"),
            envelope_cpu_millis=_optional_int("envelope_cpu_millis"),
        )

    def as_dict(self) -> dict[str, object]:
        """
        Serialize for a control channel.
        """
        return {
            "node_id": self.node_id,
            "backend": self.backend,
            "features": sorted(feature.value for feature in self.features),
            "resume_level": self.resume_level.value if self.resume_level else None,
            "gpu_count": self.gpu_count,
            "host_mounts": self.host_mounts,
            "labels": sorted(self.labels),
            "reachable_session_server": self.reachable_session_server,
            "draining": self.draining,
            "image_digests": sorted(self.image_digests),
            "image_references": sorted(self.image_references),
            "available_memory_mb": self.available_memory_mb,
            "available_cpu_millis": self.available_cpu_millis,
            "envelope_memory_mb": self.envelope_memory_mb,
            "envelope_cpu_millis": self.envelope_cpu_millis,
        }


@dataclass(frozen=True)
class PlacementRequest:
    """
    What a caller needs from a node.
    """

    # The backend a spec asked for, or None to accept any capable one.
    backend: str | None = None
    required_features: frozenset[SandboxFeature] = frozenset()
    required_resume_level: ResumeLevel | None = None
    requires_host_mount: bool = False
    gpu_count: int = 0
    # A node label the request depends on, such as `env` for a dedicated node.
    required_label: str | None = None
    # Images the node is preferred to already hold. A digest is exact, a reference is
    # what a caller usually has.
    image_digests: frozenset[str] = frozenset()
    image_references: frozenset[str] = frozenset()
    # Owner of the reservation, so a dead caller's slots can be swept.
    owner_id: str = ""
    # What the sandbox will cost the node that takes it. Placement needs the size to
    # decide anything about fit: without it the only question it can answer is which node
    # it has sent the least work to, which is not the same question. Zero means the caller
    # declared no footprint, and such a request is not filtered on capacity.
    memory_mb: int = 0
    cpu_millis: int = 0

    def __post_init__(self) -> None:
        """Normalize a payload into the types this service compares.

        A request arrives over a control channel as lists of strings. Converting here
        means subset checks and explanations work on the enumerated types rather than
        on whatever the wire happened to carry.
        """
        features = frozenset(SandboxFeature(item) for item in self.required_features)
        object.__setattr__(self, "required_features", features)
        object.__setattr__(self, "image_digests", frozenset(self.image_digests))
        object.__setattr__(self, "image_references", frozenset(self.image_references))
        if self.required_resume_level is not None:
            object.__setattr__(self, "required_resume_level", ResumeLevel(self.required_resume_level))


@dataclass(frozen=True)
class PlacementDecision:
    """
    One chosen node and the reservation that protects it.
    """

    node_id: str
    backend: str
    reservation_id: str

    def as_dict(self) -> dict[str, str]:
        """
        Flatten for a control channel, which carries mappings rather than live objects.
        """
        return {"node_id": self.node_id, "backend": self.backend, "reservation_id": self.reservation_id}


@dataclass
class _NodeRecord:
    capabilities: NodeCapabilities
    # A count of reservations this node has taken that are not yet confirmed or
    # cancelled. Load is what breaks a tie, so it only has to be comparable.
    load: int = 0
    seen_at: float = 0.0
    # What this service has promised the node since its last report. A reservation is
    # made here and charged on the node later, so between those two moments the node's
    # reported headroom still counts the room as free. Without this, a batch of
    # concurrent requests all read the same stale headroom and all pick the same node,
    # which is how one node ends up holding more sandboxes than its envelope admits.
    pending_memory_mb: int = 0
    pending_cpu_millis: int = 0

    def free_memory_mb(self) -> int | None:
        """Return reported headroom minus what is already promised, or `None` if unbounded."""
        if self.capabilities.available_memory_mb is None:
            return None
        return self.capabilities.available_memory_mb - self.pending_memory_mb

    def free_cpu_millis(self) -> int | None:
        """Return reported headroom minus what is already promised, or `None` if unbounded."""
        if self.capabilities.available_cpu_millis is None:
            return None
        return self.capabilities.available_cpu_millis - self.pending_cpu_millis

    def can_hold(self, request: PlacementRequest) -> bool:
        """Return whether this node still has room for one request.

        A node that declared no envelope is never excluded: it schedules its own capacity,
        so this service has no basis to refuse it. A request that declares no footprint is
        likewise not filtered, because zero is the absence of a measurement here.
        """
        memory = self.free_memory_mb()
        if memory is not None and request.memory_mb > 0 and request.memory_mb > memory:
            return False
        cpu = self.free_cpu_millis()
        return not (cpu is not None and request.cpu_millis > 0 and request.cpu_millis > cpu)

    def utilisation(self) -> float:
        """Return how full this node is, as the worse of its two dimensions.

        A fraction rather than an absolute, so two nodes of different sizes are ordered by
        how loaded they are rather than by how large they are. A node with no envelope
        sorts as empty, since nothing here can claim otherwise.
        """
        worst = 0.0
        for free, total in (
            (self.free_memory_mb(), self.capabilities.envelope_memory_mb),
            (self.free_cpu_millis(), self.capabilities.envelope_cpu_millis),
        ):
            if free is None or not total:
                continue
            worst = max(worst, 1.0 - max(0, free) / total)
        return worst


@dataclass
class _Reservation:
    reservation_id: str
    node_id: str
    owner_id: str
    created_at: float
    # When the owner last said it still holds this reservation. The sweeper ages a
    # reservation from here, not from creation, so a live sandbox is only swept when it is not renewed.
    renewed_at: float = 0.0
    # What this reservation promised the node, so releasing it returns exactly what it took.
    memory_mb: int = 0
    cpu_millis: int = 0


def as_placement_request(request: PlacementRequest | Mapping[str, Any]) -> PlacementRequest:
    """Normalize a request that arrived as a payload into the type the service compares.

    A request crosses a control channel as a mapping, so the two entry points that take one
    normalize it here rather than depending on every transport remembering to. It is
    idempotent, because an in-process caller already has the real thing.
    """
    if isinstance(request, PlacementRequest):
        return request
    return PlacementRequest(**dict(request))


def fleet_capabilities(advertisements: Sequence[Mapping[str, Any]]) -> SandboxCapabilities:
    """Return what every node in a fleet can do, which is what a caller may promise.

    A caller declares one capability set while the nodes underneath it may differ, so the
    honest answer is the intersection: a feature some node lacks is one the caller cannot
    rely on. Placement still filters per request and per node, so this only decides which
    specs are admitted at all.

    The resume level follows the same rule, taking the weakest declared. A node that
    declares no level weakens the fleet to none, because a resume that keeps a workspace on
    one node and loses a live process on another is not a promise worth making.
    """
    features: frozenset[SandboxFeature] | None = None
    levels: list[ResumeLevel | None] = []
    for advertisement in advertisements:
        for payload in advertisement.get("backends", ()):  # one payload per backend on that node
            declared = frozenset(SandboxFeature(value) for value in payload.get("features", ()))
            features = declared if features is None else features & declared
            raw_level = payload.get("resume_level")
            levels.append(ResumeLevel(raw_level) if raw_level else None)
    if features is None:
        return SandboxCapabilities()
    if SandboxFeature.RESUME_ANYWHERE not in features:
        return SandboxCapabilities(features)
    if any(level is None for level in levels):
        return SandboxCapabilities(features, resume_level=None)
    weakest = min((level for level in levels if level is not None), key=lambda level: level.rank)
    return SandboxCapabilities(features, resume_level=weakest)


@dataclass
class PlacementSnapshot:
    """
    One point-in-time view, for the trainer's metric hook.
    """

    nodes: int = 0
    drained_nodes: int = 0
    available_nodes: int = 0
    reservations_open: int = 0
    reservations_swept: int = 0
    rejections: int = 0
    no_candidate: int = 0
    # Times every capable node was full. Apart from `no_candidate`, because this one
    # clears itself as sandboxes are released and that one never will.
    capacity_exhausted: int = 0
    oldest_reservation_age_s: float = 0.0
    mean_decision_s: float = 0.0
    # How often the chosen node already held what the request would have pulled. This
    # is the only plane that can answer "do tasks land where their image already is".
    locality_asked: int = 0
    locality_hits: int = 0
    counters: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        """
        Flatten for a metrics sink.
        """
        return {
            "placement/nodes": float(self.nodes),
            "placement/drained_nodes": float(self.drained_nodes),
            "placement/available_nodes": float(self.available_nodes),
            "placement/reservations_open": float(self.reservations_open),
            "placement/reservations_swept": float(self.reservations_swept),
            "placement/rejections": float(self.rejections),
            "placement/no_candidate": float(self.no_candidate),
            "placement/oldest_reservation_age_s": self.oldest_reservation_age_s,
            "placement/decision_s_mean": self.mean_decision_s,
            "image/locality_hit_ratio": (self.locality_hits / self.locality_asked if self.locality_asked else 0.0),
            **{f"placement/{key}": float(value) for key, value in self.counters.items()},
        }


class PlacementService:
    """One job's view of which sandbox nodes exist and which one to use.

    It holds a cache, not a ledger. On restart it rebuilds from the node agents
    rather than from persisted state, and a reservation no node knows about does
    not survive that rebuild.
    """

    def __init__(
        self,
        *,
        node_ttl_s: float = DEFAULT_NODE_TTL_S,
        reservation_ttl_s: float = 60.0,
        sweep_interval_s: float = 10.0,
    ) -> None:
        if node_ttl_s <= 0 or reservation_ttl_s <= 0 or sweep_interval_s <= 0:
            raise ValueError("Placement node TTL, reservation TTL, and sweep interval must be greater than zero.")
        if reservation_ttl_s >= node_ttl_s:
            # A reservation has to expire before the node it points at can be
            # drained, or a swept node leaves reservations pointing at nothing.
            raise ValueError("Placement reservation TTL must be shorter than the node TTL.")
        if sweep_interval_s >= reservation_ttl_s:
            # The sweep is what enforces the reservation TTL, so a slower sweep reports a
            # dead owner's slot long after the fact, and the node over-admits until then.
            raise ValueError(
                f"Placement sweep interval ({sweep_interval_s:g}s) must be shorter than the reservation TTL "
                f"({reservation_ttl_s:g}s), or a dead owner's slot is freed later than its TTL promises."
            )
        self.node_ttl_s = node_ttl_s
        self._reservation_ttl_s = reservation_ttl_s
        self.sweep_interval_s = sweep_interval_s
        self._nodes: dict[str, _NodeRecord] = {}
        self._reservations: dict[str, _Reservation] = {}
        # How reservations ended, counted apart. A cancel is a provision that never
        # happened and a release is a sandbox that ran, and an operator acts on each.
        self._outcomes: dict[str, int] = {}
        self._swept = 0
        self._rejections = 0
        self._no_candidate = 0
        # Counted apart from `_no_candidate`: a fleet with no room is a scheduling state
        # that clears itself, and a fleet with no capable node is a misconfiguration.
        # Reporting both as one number would hide which of the two a run is suffering.
        self._capacity_exhausted = 0
        self._locality_asked = 0
        self._locality_hits = 0
        self._decisions = 0
        self._total_decision_s = 0.0
        self._sweeper: asyncio.Task[None] | None = None

    def reservation_ttl_s(self) -> float:
        """Return how long a reservation survives without being renewed.

        It is a method rather than a plain attribute because a caller reaches this service
        over an actor boundary, and only methods cross one. The other settings stay
        attributes because nothing outside this process reads them.
        """
        return self._reservation_ttl_s

    def register(self, capabilities: NodeCapabilities, *, now: float | None = None) -> None:
        """Admit a node to the registry, or refresh one already known.

        A re-registration carries a fresh advertisement, which already accounts for every
        grant the node has made. The promises this service has not yet seen land there are
        carried over with the open reservations that made them, because dropping them would
        let the same room be promised twice while those reservations are still alive.
        """
        current = time.monotonic() if now is None else now
        existing = self._nodes.get(capabilities.node_id)
        self._nodes[capabilities.node_id] = _NodeRecord(
            capabilities=capabilities,
            load=existing.load if existing else 0,
            seen_at=current,
            pending_memory_mb=existing.pending_memory_mb if existing else 0,
            pending_cpu_millis=existing.pending_cpu_millis if existing else 0,
        )

    def unregister(self, node_id: str) -> None:
        """
        Drop a node, for example when its agent is shutting down.
        """
        self._nodes.pop(node_id, None)
        for reservation_id in [
            reservation.reservation_id for reservation in self._reservations.values() if reservation.node_id == node_id
        ]:
            self._reservations.pop(reservation_id, None)

    def heartbeat(
        self,
        node_id: str,
        *,
        load: int | None = None,
        headroom: Mapping[str, Any] | None = None,
        now: float | None = None,
    ) -> None:
        """Refresh a node's liveness and, when reported, its load and headroom.

        Headroom is refreshed here because it is the one input that goes stale on its own:
        every grant the node makes shrinks it, and placement has no way to observe that
        from its own side. A reported snapshot also clears the reservations this service
        was charging against the node, since the node's own accounting now includes
        whatever those reservations became.
        """
        record = self._nodes.get(node_id)
        if record is None:
            return
        record.seen_at = time.monotonic() if now is None else now
        if load is not None:
            record.load = max(0, load)
        if headroom is not None:
            record.capabilities = replace(
                record.capabilities,
                available_memory_mb=_optional_int(headroom.get("available_memory_mb")),
                available_cpu_millis=_optional_int(headroom.get("available_cpu_millis")),
                envelope_memory_mb=_optional_int(headroom.get("envelope_memory_mb")),
                envelope_cpu_millis=_optional_int(headroom.get("envelope_cpu_millis")),
            )
            record.pending_memory_mb = 0
            record.pending_cpu_millis = 0

    def has_node(self, node_id: str) -> bool:
        """Return whether this service still knows one node.

        A node asks, because registering and heartbeating are not the same request. A
        heartbeat for a node this service has forgotten is ignored rather than treated as
        an arrival, so a node that came back to an empty registry has to know to re-announce
        itself instead of quietly heartbeating into a cache it is not in.
        """
        return node_id in self._nodes

    def choose(self, request: PlacementRequest | Mapping[str, Any], *, now: float | None = None) -> PlacementDecision:
        """Pick a node that can actually hold the request, and reserve it there.

        Capability match, then room, then balance, then image locality. The order is the
        point: a capability is a requirement, room is a requirement, and locality is an
        optimisation. Ranking locality first let a per-instance image pin an entire batch
        onto whichever node happened to hold it, and because nothing then checked room,
        every one of those requests was reserved onto a node that could admit only a
        fraction of them. They queued at the node instead, invisibly, until the run's
        first buffer could never fill.

        Raises:
            NoPlacementCandidate: When no live node could ever satisfy the request.
            PlacementCapacityExhausted: When a node could satisfy it but none has room
                right now. Separate because the caller's recourse differs: the first is a
                deployment fault, the second resolves as sandboxes are released.
        """
        request = as_placement_request(request)
        started_at = time.monotonic()
        current = started_at if now is None else now
        candidates = [record for record in self._nodes.values() if self._is_live(record, current)]
        eligible = [record for record in candidates if self._satisfies(record.capabilities, request)]
        if not eligible:
            self._no_candidate += 1
            # Counted here rather than inside the check, because the same check ranks a
            # prefetch's targets and a read-only scan is not a rejection.
            self._rejections += len(candidates)
            raise NoPlacementCandidate(self._explain(request, candidates))
        roomy = [record for record in eligible if record.can_hold(request)]
        if not roomy:
            self._capacity_exhausted += 1
            raise PlacementCapacityExhausted(self._explain_capacity(request, eligible))
        chosen = self._rank(roomy, request)[0]
        reservation = _Reservation(
            reservation_id=uuid.uuid4().hex,
            node_id=chosen.capabilities.node_id,
            owner_id=request.owner_id,
            created_at=current,
            renewed_at=current,
            memory_mb=max(0, request.memory_mb),
            cpu_millis=max(0, request.cpu_millis),
        )
        chosen.load += 1
        # Charge the promise now, so the next caller in this same batch sees the room go
        # even though the node cannot report it until the grant lands.
        chosen.pending_memory_mb += reservation.memory_mb
        chosen.pending_cpu_millis += reservation.cpu_millis
        self._reservations[reservation.reservation_id] = reservation
        self._decisions += 1
        self._note_locality(chosen.capabilities, request)
        self._total_decision_s += time.monotonic() - started_at
        self.start_sweeper()
        return PlacementDecision(
            node_id=chosen.capabilities.node_id,
            backend=chosen.capabilities.backend,
            reservation_id=reservation.reservation_id,
        )

    def candidates(
        self,
        request: PlacementRequest | Mapping[str, Any],
        *,
        now: float | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """Return the node ids a request could be placed on, best first.

        A prefetch warms a run's working set on these, because they are the nodes a task
        is likely to land on. Warming every node instead would pay a pull per node for
        images most nodes will never serve, which is the cost the plan exists to avoid.
        """
        request = as_placement_request(request)
        current = time.monotonic() if now is None else now
        live = [record for record in self._nodes.values() if self._is_live(record, current)]
        eligible = [record for record in live if self._satisfies(record.capabilities, request)]
        ranked = [record.capabilities.node_id for record in self._rank(eligible, request)]
        return ranked if limit is None else ranked[: max(0, limit)]

    def renew(self, reservation_id: str, *, now: float | None = None) -> None:
        """Say this reservation's owner is still holding it.

        Placement cannot tell a live long-lived sandbox from an abandoned reservation
        by age alone, so the owner has to keep saying it is there. A worker that stops
        renewing, because it died or was preempted, loses its reservations to the
        sweeper instead of starving the node forever.
        """
        reservation = self._reservations.get(reservation_id)
        if reservation is not None:
            reservation.renewed_at = time.monotonic() if now is None else now

    def release(self, reservation_id: str) -> None:
        """
        Return a reservation whose sandbox ran and finished.
        """
        self._retire(reservation_id, "released")

    def cancel(self, reservation_id: str) -> None:
        """Withdraw a reservation whose provision never happened.

        Separate from release because the two mean different things to an operator: a
        cancel is a request the caller gave up on, and a release is a sandbox that ran.
        They are counted apart for that reason, so a cluster losing reservations to
        failed provisions is distinguishable from one doing ordinary work.
        """
        self._retire(reservation_id, "cancelled")

    def _retire(self, reservation_id: str, outcome: str) -> None:
        """Drop one reservation, record how it ended, and give back what it promised.

        Every way a reservation can end funnels through here, which is why the promised
        resources are returned here too: a charge released on only some of those paths
        would leak headroom, and a node whose headroom has leaked is one this service
        stops sending work to while it sits idle.
        """
        reservation = self._reservations.pop(reservation_id, None)
        if reservation is None:
            return
        self._outcomes[outcome] = self._outcomes.get(outcome, 0) + 1
        self._decrement(reservation.node_id)
        record = self._nodes.get(reservation.node_id)
        if record is not None:
            record.pending_memory_mb = max(0, record.pending_memory_mb - reservation.memory_mb)
            record.pending_cpu_millis = max(0, record.pending_cpu_millis - reservation.cpu_millis)

    def sweep(self, *, now: float | None = None) -> list[str]:
        """
        Drop reservations whose owner stopped renewing, and return their ids.
        """
        current = time.monotonic() if now is None else now
        expired = [
            reservation.reservation_id
            for reservation in self._reservations.values()
            if current - reservation.renewed_at > self._reservation_ttl_s
        ]
        for reservation_id in expired:
            self.cancel(reservation_id)
        self._swept += len(expired)
        if expired:
            psrl_logger.warning(
                f"Placement swept {len(expired)} reservation(s) not renewed within "
                f"{self._reservation_ttl_s:g}s. Their owners stopped working."
            )
        return expired

    def snapshot(self, *, now: float | None = None) -> PlacementSnapshot:
        """
        Return this service's metrics for the trainer's per-step hook.
        """
        current = time.monotonic() if now is None else now
        live = [record for record in self._nodes.values() if self._is_live(record, current)]
        return PlacementSnapshot(
            nodes=len(self._nodes),
            drained_nodes=len(self._nodes) - len(live),
            available_nodes=len(live),
            reservations_open=len(self._reservations),
            reservations_swept=self._swept,
            locality_asked=self._locality_asked,
            locality_hits=self._locality_hits,
            rejections=self._rejections,
            no_candidate=self._no_candidate,
            capacity_exhausted=self._capacity_exhausted,
            oldest_reservation_age_s=max(
                (current - reservation.created_at for reservation in self._reservations.values()),
                default=0.0,
            ),
            mean_decision_s=self._total_decision_s / self._decisions if self._decisions else 0.0,
            counters=dict(self._outcomes),
        )

    def nodes(self) -> tuple[NodeCapabilities, ...]:
        """
        Return every registered node's capabilities.
        """
        return tuple(record.capabilities for record in self._nodes.values())

    def _rank(self, eligible: Sequence[_NodeRecord], request: PlacementRequest) -> list[_NodeRecord]:
        """Order candidates by how full they are, then by locality, then by load and id.

        Balance leads because capacity is a constraint and locality is an optimisation:
        saving an image pull is worth nothing if the sandbox then waits behind a full
        node's queue for an episode's length. With locality first, a per-instance image
        drew every request in a batch to one node and the rest of the fleet stayed idle.

        Utilisation is bucketed rather than compared exactly, so locality still decides
        between nodes that are similarly loaded — which is what makes it an optimisation
        rather than a tie-breaker that never fires.

        The node id is the last key so two equally good nodes do not alternate, which
        would make a run's placement impossible to reproduce.
        """
        return sorted(
            eligible,
            key=lambda record: (
                round(record.utilisation() * _BALANCE_BUCKETS),
                # More matches first, so the score is negated rather than ascending.
                -self._locality_score(record.capabilities, request),
                record.load,
                record.capabilities.node_id,
            ),
        )

    @staticmethod
    def _locality_score(capabilities: NodeCapabilities, request: PlacementRequest) -> int:
        """Return how well this node already holds the images a request names.

        A digest match is worth more than a reference match, because a digest is exact
        and a reference can be moved.
        """
        digest_hits = len(capabilities.image_digests & request.image_digests)
        reference_hits = len(capabilities.image_references & request.image_references)
        return digest_hits * 2 + reference_hits

    def _note_locality(self, capabilities: NodeCapabilities, request: PlacementRequest) -> None:
        """Count whether the chosen node would have pulled the image at all.

        A request that names a digest needs that digest, because a moved tag cannot
        claim a hit it does not have. A request that names only a reference is answered
        by the reference: a node holding the tag serves it from its own daemon and does
        not pull, which is the question the ratio exists to answer.
        """
        if not (request.image_digests or request.image_references):
            return
        self._locality_asked += 1
        if request.image_digests:
            hit = bool(capabilities.image_digests & request.image_digests)
        else:
            hit = bool(capabilities.image_references & request.image_references)
        self._locality_hits += int(hit)

    def _is_live(self, record: _NodeRecord, now: float) -> bool:
        """
        Return whether a node's last heartbeat is recent enough to trust.
        """
        return now - record.seen_at <= self.node_ttl_s

    def _satisfies(self, capabilities: NodeCapabilities, request: PlacementRequest) -> bool:
        """Return whether one node can host a request at all.

        A satisfied check is deliberately strict. A node that cannot reach the
        session server, cannot bind a host path, or resumes at a weaker level is not
        a slower candidate, it is a wrong one.

        It is also pure, because `candidates` asks the same question to rank a prefetch's
        targets. Counting a rejection here would make a read-only scan inflate the metric
        an operator reads to decide the fleet is misconfigured.
        """
        if request.backend is not None and capabilities.backend != request.backend:
            return False
        if not request.required_features <= capabilities.features:
            return False
        if request.required_resume_level is not None:
            level = capabilities.resume_level
            if level is None or not level.satisfies(request.required_resume_level):
                return False
        if request.requires_host_mount and not capabilities.host_mounts:
            return False
        if request.gpu_count > capabilities.gpu_count:
            return False
        if request.required_label is not None and request.required_label not in capabilities.labels:
            return False
        if not capabilities.reachable_session_server:
            return False
        if capabilities.draining:
            # A node that cannot destroy what it holds still holds its memory, so
            # admitting against it would over-commit the host.
            return False
        return True

    @staticmethod
    def _explain(request: PlacementRequest, candidates: Sequence[_NodeRecord]) -> str:
        """
        Name why no node was chosen, which is what an operator has to act on.
        """
        if not candidates:
            return "No sandbox node has reported in recently, so every registered node is drained."
        missing = sorted(feature.value for feature in request.required_features)
        return (
            f"No live sandbox node satisfies this request (required features: {missing or 'none'}, "
            f"resume level: {request.required_resume_level.value if request.required_resume_level else 'none'}, "
            f"gpus: {request.gpu_count}, label: {request.required_label!r}, "
            f"candidate nodes: {[record.capabilities.node_id for record in candidates]})."
        )

    @staticmethod
    def _explain_capacity(request: PlacementRequest, eligible: Sequence[_NodeRecord]) -> str:
        """Say how full each capable node is, which is what decides whether to wait.

        The per-node remainder is the useful part: it separates "the fleet is busy" from
        "this request is larger than any node's envelope", and only the second is a
        planning fault.
        """
        rooms = ", ".join(
            f"{record.capabilities.node_id}={record.free_memory_mb()}MB/{record.free_cpu_millis()}mcpu free"
            for record in eligible
        )
        return (
            f"Every capable sandbox node is full: the request needs {request.memory_mb}MB and "
            f"{request.cpu_millis}mcpu, and the candidates have [{rooms}]. This clears as sandboxes "
            "are released; raise the envelope or lower concurrency if it does not."
        )

    def _decrement(self, node_id: str) -> None:
        record = self._nodes.get(node_id)
        if record is not None:
            record.load = max(0, record.load - 1)

    def start_sweeper(self) -> bool:
        """Start the periodic sweep, if there is an event loop to run it on.

        The TTL is enforced whenever `sweep` is called, so this is a convenience
        rather than the mechanism: a service used from synchronous code still
        expires its reservations, and one used from a loop does not have to be
        swept by hand.

        Returns:
            bool: Whether a sweeper is now running. It reports a boolean rather than the
                task, because this service is also reached over an actor boundary and a
                task cannot cross one.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        if self._sweeper is None or self._sweeper.done():
            self._sweeper = asyncio.create_task(self._sweep_loop())
        return True

    async def _sweep_loop(self) -> None:
        interval = max(1.0, self.sweep_interval_s)
        while True:
            await asyncio.sleep(interval)
            try:
                self.sweep()
            except asyncio.CancelledError:
                raise
            except Exception:
                psrl_logger.warning("Placement reservation sweep failed. Retrying next interval.", exc_info=True)

    async def shutdown(self) -> None:
        """
        Stop the sweeper and drop every reservation.
        """
        sweeper, self._sweeper = self._sweeper, None
        if sweeper is not None:
            sweeper.cancel()
            await asyncio.gather(sweeper, return_exceptions=True)
        self._reservations.clear()
        self._nodes.clear()

    @classmethod
    def from_capabilities(cls, capabilities: SandboxCapabilities, *, backend: str, **kwargs) -> NodeCapabilities:
        """
        Build a node advertisement from one backend's declared capabilities.

        In-process convenience, so a node agent does not have to restate what its
        backend already declares.
        """
        return NodeCapabilities(
            node_id=str(kwargs.pop("node_id")),
            backend=backend,
            features=frozenset(capabilities.features),
            resume_level=capabilities.resume_level,
            **kwargs,
        )
