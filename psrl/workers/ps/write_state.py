"""Optimistic commit validation for weight reads against in-place optimizer writes.

The trainer's parameters are the bytes rollouts read (direct supply, matched precision),
and the optimizer overwrites them in place. NIXL READs are one-sided, so a read that
overlaps a write observes a mixture of two versions. No version stamp can detect that:
the buffer being overwritten is the same buffer the stamp describes.

The protocol here is optimistic concurrency control:

* A reader records ``start_epoch`` when it begins a pull. The epoch is bumped by the
  writer *before* its first parameter write.
* A pull may commit its version only if the epoch did not move during the pull, or if it
  was preempted, acknowledged before the write, and repaired every not-yet-complete key
  from a certified holder of the same version.
* The writer never waits for a pull to finish. It sends a preemption notice, waits a
  bounded time for acknowledgements (a control-plane round trip, not a data drain), then
  writes. A reader that misses the notice simply cannot commit and re-pulls.

The writer's wait is therefore independent of pull size and of slow readers; a reader
that cannot be repaired keeps its previous version (or waits for the next one) and never
serves mixed bytes.

This module is pure: no Ray, no NIXL. `PSManager` owns one instance and serialises
access; the unit tests drive it directly.
"""

from __future__ import annotations

from dataclasses import dataclass

RolloutInstanceId = tuple[int, int]

#: Write-cycle states. ``OPEN`` admits pulls; ``CLOSING`` has announced a write and is
#: collecting acknowledgements; ``WRITING`` is inside the parameter write.
OPEN = "open"
CLOSING = "closing"
WRITING = "writing"


@dataclass
class PullToken:
    """What one reader recorded when its pull began."""

    version: int
    start_epoch: int
    preempt_epoch: int | None = None
    acked: bool = False


@dataclass(frozen=True)
class Admission:
    """Whether a pull may start, and the token to present when it commits."""

    admitted: bool
    token: PullToken | None = None
    reason: str | None = None


@dataclass(frozen=True)
class Commit:
    """Whether a finished pull may be recorded as holding ``version``."""

    accepted: bool
    version: int | None = None
    reason: str | None = None


@dataclass(frozen=True)
class WriteBatch:
    """The readers a new write must preempt, and the holders it can repair from."""

    epoch: int
    version: int
    readers: tuple[RolloutInstanceId, ...]
    holders: tuple[RolloutInstanceId, ...]


class WriteState:
    """Version/epoch state shared by the writer and pull commits.

    Not thread-safe by design: `PSManager` runs under its actor lock, and the writer's
    acknowledgement loop drains notifications on the same thread.
    """

    def __init__(self, version: int = 0) -> None:
        self.version = int(version)
        self.epoch = 0
        self.state = OPEN
        self._inflight: dict[RolloutInstanceId, PullToken] = {}
        self._instance_version: dict[RolloutInstanceId, int] = {}
        self._acked: dict[int, set[RolloutInstanceId]] = {}
        self._pinned: set[RolloutInstanceId] = set()

    # ------------------------------------------------------------------ readers

    def begin_pull(self, instance: RolloutInstanceId, version: int) -> Admission:
        """Admit a pull of ``version`` if no write is open and the instance is not pinned.

        Pinned instances are certified holders being used as repair sources; letting one
        start its own pull would change the bytes another reader is repairing from.
        """
        if self.state is not OPEN:
            return Admission(False, reason=f"write_{self.state}")
        if instance in self._pinned:
            return Admission(False, reason="pinned")
        token = PullToken(version=int(version), start_epoch=self.epoch)
        self._inflight[instance] = token
        return Admission(True, token=token)

    def note_preempt(self, instance: RolloutInstanceId, epoch: int) -> bool:
        """Record that ``instance`` saw the preemption notice for ``epoch``.

        Returns True when the notice is current. The acknowledgement itself is recorded
        by `ack`; this only fixes the boundary for commit validation.
        """
        token = self._inflight.get(instance)
        if token is None or epoch != self.epoch or token.start_epoch > epoch:
            return False
        token.preempt_epoch = epoch
        return True

    def ack(self, instance: RolloutInstanceId, epoch: int) -> bool:
        """Record an acknowledgement for ``epoch``; False for a stale or unknown one."""
        token = self._inflight.get(instance)
        if token is None or epoch != self.epoch or token.start_epoch > epoch:
            return False
        if token.preempt_epoch is not None and token.preempt_epoch != epoch:
            return False
        token.preempt_epoch = epoch
        token.acked = True
        self._acked.setdefault(epoch, set()).add(instance)
        return True

    def finish_pull(
        self,
        instance: RolloutInstanceId,
        repaired: bool = False,
        pull_epoch: int | None = None,
        landed_version: int | None = None,
    ) -> Commit:
        """Decide whether the pull may be committed as holding ``landed_version``.

        Two accepted cases:

        * the reader actually read at the current epoch (``pull_epoch``, defaulting to the
          admission epoch), so every byte was read either before the writer's first write
          or after its version was published;
        * the reader was preempted in the current epoch, acknowledged before the write,
          and repaired every remaining key from certified holders. Pre-acknowledgement
          reads are safe because the writer waits for the acknowledgement, and repaired
          reads come from immutable ``version`` copies.

        ``pull_epoch`` may be newer than the admission epoch: after a write publishes, a
        retry inside the same admission legitimately reads the new version, and rejecting
        it because the admission predates the write would force an unnecessary lock cycle.
        ``landed_version`` is what the engine actually read; it can differ from the
        admission's target when the write published during the pull.

        A rejected pull is not recorded. The instance must not serve the bytes it wrote;
        the caller re-pulls the next version.
        """
        token = self._inflight.pop(instance, None)
        if token is None:
            return Commit(False, reason="not_inflight")
        effective_epoch = token.start_epoch if pull_epoch is None else int(pull_epoch)
        unchanged = effective_epoch == self.epoch
        repaired_ok = (
            repaired
            and token.acked
            and token.preempt_epoch == self.epoch
            and effective_epoch < self.epoch
        )
        if not (unchanged or repaired_ok):
            return Commit(False, reason="epoch_moved")
        if repaired_ok:
            # The repair completed the version that was being overwritten.
            version = token.version if landed_version is None else int(landed_version)
        else:
            # No write moved the epoch, so the published version is the one this pull was
            # admitted for and the only one the coordinator would have served (D1 filters
            # upstream sources by holder set). Do not compare with the engine's reported
            # shard stamps: rollout-side stamps are not refreshed on publish, so that
            # report lags the coordinator by design and would reject every steady pull.
            version = self.version
        self._instance_version[instance] = version
        return Commit(True, version=version)

    def abort_pull(self, instance: RolloutInstanceId) -> None:
        """Forget a pull that will not commit (reader died, sync cancelled)."""
        self._inflight.pop(instance, None)

    # ------------------------------------------------------------------ writer

    def begin_write(self) -> WriteBatch:
        """Announce the write that follows and return who must be preempted.

        The epoch is bumped here, before the caller performs any parameter write. Starting
        from ``WRITING`` is allowed: it is the recovery path for an update that raised
        after the gate opened, and the new attempt must invalidate further readers.
        """
        if self.state is OPEN:
            pass
        elif self.state is not WRITING:
            raise RuntimeError(f"begin_write called while {self.state!r}; a write is already closing.")
        self.epoch += 1
        self.state = CLOSING
        readers = tuple(sorted(self._inflight))
        # With no reader in flight there is nothing to preempt and nothing to repair, so
        # the steady-state write costs O(1) and does not enumerate the fleet's holders.
        holders = tuple(sorted(self.holders(self.version))) if readers else ()
        self._pinned = set(holders)
        return WriteBatch(self.epoch, self.version, readers, holders)

    def start_writing(self) -> None:
        """The caller is about to write parameters; new pulls stay closed."""
        self.state = WRITING

    def publish(self, version: int) -> None:
        """Open the new version and release the holders pinned for the write."""
        self.version = int(version)
        self.state = OPEN
        self._pinned.clear()
        # Acknowledgements for epochs older than the published one can never match again.
        self._acked = {epoch: who for epoch, who in self._acked.items() if epoch >= self.epoch}

    def abort_write(self) -> None:
        """Leave the version closed after a failed update.

        The buffer may hold a partial write, so opening it would let readers commit a
        mixture. The next successful `begin_write` recovers by bumping the epoch again;
        until then no pull is admitted.
        """
        self.state = WRITING

    # ------------------------------------------------------------------ queries

    def holders(self, version: int) -> frozenset[RolloutInstanceId]:
        """Instances whose committed version is exactly ``version``."""
        return frozenset(i for i, v in self._instance_version.items() if v == version)

    def pinned(self) -> frozenset[RolloutInstanceId]:
        return frozenset(self._pinned)

    def inflight(self) -> frozenset[RolloutInstanceId]:
        return frozenset(self._inflight)

    def token(self, instance: RolloutInstanceId) -> PullToken | None:
        return self._inflight.get(instance)

    def acked(self, epoch: int) -> frozenset[RolloutInstanceId]:
        return frozenset(self._acked.get(epoch, ()))

    def record_published(self, instance: RolloutInstanceId, version: int) -> None:
        """Record a version reached outside a tracked pull (resume, checkpoint load)."""
        self._instance_version[instance] = int(version)
