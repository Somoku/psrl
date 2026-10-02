"""One timing contract for every level of the sandbox stack.

An operator can estimate one number: how long an episode takes. Every window the
module enforces follows from it, so a deployment states an intent and the code
derives the rest.

Two rules make that safe.

- **A derived value is never configured.** A ratio is a property of the mechanism,
  not of a deployment, so `pause_after_episodes` and its siblings are constants
  here rather than knobs. A deployment that genuinely needs a different window
  sets the window, not the ratio that produced it.
- **Every ordering is asserted in one place.** The orderings used to live in two
  files that did not know about each other, so the cross-level ones -- a node
  reporting less often than its own TTL, a sweep slower than the lease it
  enforces -- were comments rather than checks. `validate()` is now the only
  place an ordering exists, and a configuration that inverts one fails at
  startup instead of leaking a slot per episode.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

# Ratios. These are properties of the mechanism rather than of a deployment: a
# pause window shorter than an episode would pause a running episode, and a
# lifetime close to the reap window would fire before the reaper ever ran.
_PAUSE_EPISODES = 2.0
_REAP_PAUSE_WINDOWS = 3.0
_LIFETIME_REAP_WINDOWS = 4.0
# A report, a renewal, and a sweep each run several times inside the deadline they
# protect, so one lost round trip never costs the thing it was protecting.
_REPORTS_PER_TTL = 4.0
_RENEWALS_PER_TTL = 3.0
_SWEEPS_PER_TTL = 4.0
# A reservation is held by a caller and a node record by the fleet, so a reservation
# has to expire first or a drained node leaves reservations pointing at nothing.
_RESERVATION_TTL_FRACTION = 0.5
# Crash recovery is two phases, and both must finish inside the capacity lease.
_LIFECYCLE_LEASE_FRACTION = 0.5

DEFAULT_NODE_TTL_S = 120.0
DEFAULT_RPC_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class TimingContract:
    """Every deadline the stack enforces, derived from three declared values.

    Only `episode_deadline_s` has to be set. The other two have defaults that hold
    for any fleet, because they describe how fast a node answers rather than how
    long a workload runs.

    `overrides` names a derived value a deployment is replacing. It is validated
    exactly like a derived one, so an override that inverts an ordering is refused
    rather than honoured.
    """

    episode_deadline_s: float
    node_ttl_s: float = DEFAULT_NODE_TTL_S
    rpc_timeout_s: float = DEFAULT_RPC_TIMEOUT_S
    overrides: dict[str, float] | None = None

    def __post_init__(self) -> None:
        for name in ("episode_deadline_s", "node_ttl_s", "rpc_timeout_s"):
            value = getattr(self, name)
            if value is None or value <= 0:
                raise ValueError(f"Sandbox timing {name} must be greater than zero.")
        unknown = sorted(set(self.overrides or {}) - set(_DERIVED))
        if unknown:
            raise ValueError(
                f"Sandbox timing overrides {unknown} name no derived value. Valid names: {sorted(_DERIVED)}."
            )
        for name, value in (self.overrides or {}).items():
            if value is None or value <= 0:
                raise ValueError(f"Sandbox timing override {name!r} must be greater than zero.")

    # -- Idle policy, from the episode deadline ---------------------------------

    @property
    def pause_window_s(self) -> float:
        """Return the idle span after which a sandbox releases its compute."""
        return self._value("pause_window_s", self.episode_deadline_s * _PAUSE_EPISODES)

    @property
    def reap_window_s(self) -> float:
        """Return the idle span after which a sandbox is destroyed instead."""
        return self._value("reap_window_s", self.pause_window_s * _REAP_PAUSE_WINDOWS)

    @property
    def lifetime_s(self) -> float:
        """Return the absolute lifetime, which bounds a sandbox that is never idle."""
        return self._value("lifetime_s", self.reap_window_s * _LIFETIME_REAP_WINDOWS)

    @property
    def acquire_timeout_s(self) -> float:
        """Return how long admission may queue before the wait is itself the fault."""
        return self._value("acquire_timeout_s", self.episode_deadline_s)

    # -- Fleet liveness, from the node TTL --------------------------------------

    @property
    def load_report_interval_s(self) -> float:
        """Return how often a node publishes its view, several times per TTL."""
        return self._value("load_report_interval_s", self.node_ttl_s / _REPORTS_PER_TTL)

    @property
    def monitor_pull_interval_s(self) -> float:
        """Return how often the fleet view is collected, matching the report rate."""
        return self._value("monitor_pull_interval_s", self.node_ttl_s / _REPORTS_PER_TTL)

    @property
    def reservation_ttl_s(self) -> float:
        """Return how long a placement reservation survives without renewal."""
        return self._value("reservation_ttl_s", self.node_ttl_s * _RESERVATION_TTL_FRACTION)

    @property
    def reservation_renew_interval_s(self) -> float:
        """Return how often an owner says it still holds its reservations."""
        return self._value("reservation_renew_interval_s", self.reservation_ttl_s / _RENEWALS_PER_TTL)

    # -- Capacity and crash recovery --------------------------------------------

    @property
    def capacity_lease_ttl_s(self) -> float:
        """Return how long a capacity reservation outlives its owner's silence.

        It is at least the pause window, because a paused sandbox still holds the
        memory its reservation covers.
        """
        return self._value("capacity_lease_ttl_s", self.pause_window_s)

    @property
    def owner_heartbeat_interval_s(self) -> float:
        """Return how often a sandbox owner refreshes its capacity lease."""
        return self._value("owner_heartbeat_interval_s", self.capacity_lease_ttl_s / _RENEWALS_PER_TTL)

    @property
    def lifecycle_lease_ttl_s(self) -> float:
        """Return the node-local ownership lease used for crash recovery."""
        return self._value("lifecycle_lease_ttl_s", self.capacity_lease_ttl_s * _LIFECYCLE_LEASE_FRACTION)

    @property
    def lifecycle_gc_interval_s(self) -> float:
        """Return how often crash recovery sweeps for containers with no live owner."""
        return self._value("lifecycle_gc_interval_s", self.lifecycle_lease_ttl_s * _LIFECYCLE_LEASE_FRACTION)

    @property
    def sweep_interval_s(self) -> float:
        """Return the reclamation cadence, faster than the shortest span it enforces."""
        shortest = min(self.pause_window_s, self.reservation_ttl_s, self.capacity_lease_ttl_s)
        return self._value("sweep_interval_s", max(1.0, shortest / _SWEEPS_PER_TTL))

    # -- Validation --------------------------------------------------------------

    def validate(self) -> TimingContract:
        """Refuse a contract whose spans are in an order the code cannot honour.

        Each rule below is something the module relies on. A violation produces a
        slow leak or a sandbox reclaimed while in use, and neither failure points
        back at the configuration, so each is a check rather than a comment.

        Returns:
            TimingContract: This contract, so a caller can validate and bind in one
                expression.

        Raises:
            ValueError: When an ordering is inverted, naming both spans.
        """
        _require(
            self.pause_window_s < self.reap_window_s,
            f"pause window ({self.pause_window_s:g}s) must be shorter than the reap window "
            f"({self.reap_window_s:g}s), or a sandbox is destroyed before it is ever paused",
        )
        _require(
            self.reap_window_s < self.lifetime_s,
            f"reap window ({self.reap_window_s:g}s) must be shorter than the lifetime "
            f"({self.lifetime_s:g}s), or the backstop fires first and the reaper never runs",
        )
        _require(
            self.capacity_lease_ttl_s >= self.pause_window_s,
            f"capacity lease ({self.capacity_lease_ttl_s:g}s) must cover the pause window "
            f"({self.pause_window_s:g}s), or a paused sandbox loses the reservation holding its memory",
        )
        _require(
            self.owner_heartbeat_interval_s < self.capacity_lease_ttl_s,
            f"owner heartbeat ({self.owner_heartbeat_interval_s:g}s) must be shorter than the capacity "
            f"lease ({self.capacity_lease_ttl_s:g}s), or a live owner loses its reservation",
        )
        _require(
            self.load_report_interval_s * 3 < self.node_ttl_s,
            f"load report interval ({self.load_report_interval_s:g}s) must fit several times into the "
            f"node TTL ({self.node_ttl_s:g}s), or one missed report drains a live node",
        )
        _require(
            self.reservation_ttl_s < self.node_ttl_s,
            f"reservation TTL ({self.reservation_ttl_s:g}s) must be shorter than the node TTL "
            f"({self.node_ttl_s:g}s), or a drained node leaves reservations pointing at nothing",
        )
        _require(
            self.reservation_renew_interval_s < self.reservation_ttl_s,
            f"reservation renewal ({self.reservation_renew_interval_s:g}s) must be shorter than the "
            f"reservation TTL ({self.reservation_ttl_s:g}s), or every reservation expires under renewal",
        )
        _require(
            self.sweep_interval_s < min(self.pause_window_s, self.reservation_ttl_s, self.capacity_lease_ttl_s),
            f"sweep interval ({self.sweep_interval_s:g}s) must be shorter than the shortest span it "
            "enforces, or a window is reported long after it elapsed",
        )
        _require(
            self.lifecycle_lease_ttl_s + self.lifecycle_gc_interval_s < self.capacity_lease_ttl_s,
            f"crash recovery (lease {self.lifecycle_lease_ttl_s:g}s plus sweep "
            f"{self.lifecycle_gc_interval_s:g}s) must complete inside the capacity lease "
            f"({self.capacity_lease_ttl_s:g}s), or a dead owner's containers outlive the reservation "
            "that protected the node",
        )
        return self

    def with_overrides(self, **values: float) -> TimingContract:
        """Return a contract with derived values replaced, still unvalidated."""
        merged = {**(self.overrides or {}), **values}
        return replace(self, overrides=merged)

    def as_dict(self) -> dict[str, float]:
        """Return every resolved span, for a metric hook or an operator dump."""
        return {name: getattr(self, name) for name in sorted(_DERIVED)}

    def _value(self, name: str, derived: float) -> float:
        """Return an override when one is declared, and the derived span otherwise."""
        override = (self.overrides or {}).get(name)
        return derived if override is None else float(override)

    @classmethod
    def from_value(cls, value: Any) -> TimingContract:
        """Build a contract from a typed value or a resolved Hydra mapping.

        Raises:
            ValueError: When the mapping carries a key that is neither a declared
                value nor a derived span. A typo must be visible, because a dropped
                deadline is invisible until a run leaks.
        """
        if isinstance(value, cls):
            return value
        if value is None:
            raise ValueError("Sandbox timing requires episode_deadline_s.")
        fields = dict(value)
        declared = {name: fields.pop(name) for name in _DECLARED if name in fields}
        overrides = dict(fields.pop("overrides", None) or {})
        # A derived name written at the top level is an override, which is what an
        # operator means by it and what the old flat config accepted.
        for name in list(fields):
            if name in _DERIVED:
                overrides[name] = fields.pop(name)
        if fields:
            raise ValueError(
                f"Sandbox timing has unknown key(s) {sorted(fields)}. Declared: {sorted(_DECLARED)}; "
                f"overridable: {sorted(_DERIVED)}."
            )
        return cls(**declared, overrides=overrides or None)


def _require(condition: bool, explanation: str) -> None:
    """Raise a configuration error naming both spans in a broken ordering."""
    if not condition:
        raise ValueError(f"Sandbox timing: {explanation}.")


_DECLARED = ("episode_deadline_s", "node_ttl_s", "rpc_timeout_s")

# Every span a deployment may override. Listed explicitly, so a new derived value is
# made overridable deliberately rather than by existing.
_DERIVED = (
    "acquire_timeout_s",
    "capacity_lease_ttl_s",
    "lifecycle_gc_interval_s",
    "lifecycle_lease_ttl_s",
    "lifetime_s",
    "load_report_interval_s",
    "monitor_pull_interval_s",
    "owner_heartbeat_interval_s",
    "pause_window_s",
    "reap_window_s",
    "reservation_renew_interval_s",
    "reservation_ttl_s",
    "sweep_interval_s",
)
