"""The per-run prefetch plan over a task set's working set.

A cold cluster pays an image pull on every create, and the pull is the largest single
cost. The working set of a run is far smaller than the corpus, because a task set
rarely visits every image, so warming it before rollout is what makes a cold cluster
fast.

Two rules shape the plan. A digest and a reference naming the same image are one
entry, because warming both would pull twice. And a plan is an optimization, not a
requirement: a reference the plan missed still runs, it just pays the pull.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from psrl.sandbox.core import SandboxSourceKind, SandboxSpec


@dataclass(frozen=True)
class PrefetchPlan:
    """
    The images one run's task set will ask for.
    """

    references: tuple[str, ...] = ()

    @classmethod
    def from_specs(cls, specs: Iterable[SandboxSpec]) -> PrefetchPlan:
        """Build the working set from the specs a run will use.

        Only an image source has something to pull. A template or a snapshot is materialized
        by its own backend, so a plan that named one would warm the wrong thing.
        """
        references: list[str] = []
        seen: set[str] = set()
        for spec in specs:
            if spec.source.kind is not SandboxSourceKind.IMAGE:
                continue
            reference = spec.source.reference
            if reference in seen:
                continue
            seen.add(reference)
            references.append(reference)
        return cls(references=tuple(references))

    @property
    def size(self) -> int:
        """
        Return how many images the run will ask for.
        """
        return len(self.references)


@dataclass(frozen=True)
class PrefetchReport:
    """
    What a prefetch step actually achieved.
    """

    requested: int = 0
    warmed: int = 0

    @property
    def coverage(self) -> float:
        """
        Return the share of the working set that is now local.
        """
        return self.warmed / self.requested if self.requested else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "image/prefetch_requested": float(self.requested),
            "image/prefetch_warmed": float(self.warmed),
            "image/prefetch_coverage": self.coverage,
        }
