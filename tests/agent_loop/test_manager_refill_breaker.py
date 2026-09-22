"""Unit tests for the training refill breaker in `PSRL_AgentLoopManager`.

A train buffer entry is all-or-nothing, so a group that fails without producing data
is purged and replaced by one fresh prompt. Three things can go wrong and all used to
be silent:

- the replacement dispatch produces nothing, leaving the buffer permanently short,
- every replacement fails the same way, so refills churn forever at no progress,
- no sandbox can be admitted, so every refill pays the full acquisition deadline and the
  buffer still never fills.

All three end as a trainer that waits on a buffer that can never fill. These tests pin
that the manager latches a diagnosis instead, and that `wait_for_training_batch` surfaces
it, because that is the only call the driver makes with a bare `ray.get`.

Methods are borrowed from the real class onto a stub instance so the tests exercise
production code without a Ray cluster. All tests use `asyncio.run()` so they work
without `pytest-asyncio`.
"""

import asyncio
from collections import Counter
from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import OmegaConf
from psrl.workers.agent_loop.loops.utils import TerminateReason
from psrl.workers.agent_loop.manager import BoundedIdSet, PSRL_AgentLoopManager
from psrl.workers.agent_loop.timeouts import AgentLoopTimeouts

pytestmark = pytest.mark.cpu_test

# A ladder with an explicit admission deadline, so a capacity diagnosis can name it
# without the test building a whole trainer config tree.
_STUB_TIMEOUTS = AgentLoopTimeouts(
    episode_timeout_s=7200.0,
    setup_allowance_s=900.0,
    admission_timeout_s=1800.0,
    heartbeat_interval_s=120.0,
    entry_stall_timeout_s=360.0,
)


def _make_config(threshold: int, capacity_threshold: int) -> OmegaConf:
    """Build the smallest config tree the borrowed manager methods read."""
    return OmegaConf.create(
        {
            "psrl": {
                "agentic_rl": {
                    "refill_failure_threshold": threshold,
                    "capacity_failure_threshold": capacity_threshold,
                },
                "rollout_coordination": {
                    "redundant_rollout": {"enable": False},
                    "proactive_filter_strategy": {"method": None},
                },
            }
        }
    )


class FakeManager:
    """Minimal stub of `PSRL_AgentLoopManager` for testing the refill breaker.

    Borrows the failure-accounting, refill, and waiter methods from the real class so
    the assertions cover real behavior rather than a reimplementation.
    """

    def __init__(
        self,
        threshold: int = 3,
        rollout_n: int = 2,
        failed_id_capacity: int = 1024,
        capacity_threshold: int = 4,
    ) -> None:
        self.config = _make_config(threshold, capacity_threshold)
        self.rollout_n = rollout_n
        self.val_rollout_n = rollout_n
        self.refill_failure_threshold = threshold
        self.capacity_failure_threshold = capacity_threshold
        self.timeouts = _STUB_TIMEOUTS

        self._failed_train_group_ids: BoundedIdSet = BoundedIdSet(failed_id_capacity)
        self._failed_val_group_ids: set[int] = set()
        self.rollout_request_tracker: dict = {}
        self._consecutive_group_failures = 0
        self._capacity_failure_streak = 0
        self._group_failure_reasons: Counter = Counter()
        self._coordination_failures: Counter = Counter()
        self._shutting_down = False
        self._refill_breaker_diagnosis: str | None = None

        self.train_data_buffers: dict = {}
        self.train_accumulated_buffers: dict = {}
        self.train_accumulated_buffer_size: dict = {}
        self._train_buffer_waiters: dict = {}
        self._train_chunk_waiters: dict = {}
        self._resolved_train_chunks: dict = {}
        self.buffer_wait_log_interval_s = 60.0

        # Validation state, needed because a validation round is what used to wipe the
        # train-side failure record.
        self.val_buffer_size: int | None = None
        self.val_data_buffers: dict = {}
        self.val_accumulated_buffer_size: dict = {}
        self._val_buffer_waiters: dict = {}
        self._val_round_all_failed = False

        self.running_loop = None
        self._request_counter = 0

        # A real `AsyncBusyPollingRayLock` runs against this handle, so `acquire`
        # must resolve truthy or its polling loop never exits.
        self.ps_manager_handle = MagicMock()
        self.ps_manager_handle.acquire.remote = AsyncMock(return_value=True)
        self.ps_manager_handle.release.remote = AsyncMock(return_value=None)
        self.ps_manager_handle.abort_requests.remote = AsyncMock(return_value=None)
        self.ps_manager_handle.ensure_train_buffer_exists.remote = AsyncMock(return_value=None)

        # Refills return nothing by default, which is the broken-invariant case.
        self.data_processor = MagicMock()
        self.data_processor.sample_train_prompts.remote = AsyncMock(return_value=None)

        self.dispatched_batches: list = []

        self.notify_group_failed = PSRL_AgentLoopManager.notify_group_failed.__get__(self)
        self.set_val_buffer_size = PSRL_AgentLoopManager.set_val_buffer_size.__get__(self)
        self._retry_data = PSRL_AgentLoopManager._retry_data.__get__(self)
        self._record_group_failure = PSRL_AgentLoopManager._record_group_failure.__get__(self)
        self._record_capacity_failure = PSRL_AgentLoopManager._record_capacity_failure.__get__(self)
        self._report_coordination_failures = PSRL_AgentLoopManager._report_coordination_failures.__get__(self)
        self._reset_group_failure_streak = PSRL_AgentLoopManager._reset_group_failure_streak.__get__(self)
        self._trip_refill_breaker = PSRL_AgentLoopManager._trip_refill_breaker.__get__(self)
        self._refill_failed_group = PSRL_AgentLoopManager._refill_failed_group.__get__(self)
        self._refill_breaker_error = PSRL_AgentLoopManager._refill_breaker_error.__get__(self)
        self._raise_if_refill_breaker_tripped = PSRL_AgentLoopManager._raise_if_refill_breaker_tripped.__get__(self)
        self._await_buffer_future = PSRL_AgentLoopManager._await_buffer_future.__get__(self)
        self.wait_for_training_batch = PSRL_AgentLoopManager.wait_for_training_batch.__get__(self)
        self.wait_for_training_chunk = PSRL_AgentLoopManager.wait_for_training_chunk.__get__(self)

    async def _purge_tracker_group(self, parent_id, rollout_n, is_validate):
        """Stand in for the TQ payload purge, which needs a live TransferQueue."""
        return self.rollout_request_tracker.pop(parent_id, [])

    async def _inner_dispatch_data(self, data, is_validate):
        """Record dispatches instead of sending them to rollout workers."""
        self.dispatched_batches.append(data)


class TestZeroRefill:
    """A refill that dispatches nothing must not be reported as a recovery."""

    def test_group_failure_with_no_replacement_trips_the_breaker(self):
        """An empty refill leaves the buffer short forever, so it must fail loudly.

        By the time refill runs, the group is already purged and its siblings aborted.
        There is no state left to retry from, so a zero dispatch is unrecoverable.
        """

        async def _run():
            manager = FakeManager(threshold=1000)
            manager.running_loop = asyncio.get_running_loop()

            await manager.notify_group_failed(
                parent_id=7,
                failed_uid=14,
                is_validate=False,
                terminate_reason=TerminateReason.ROLLOUT_ERROR,
            )

            assert manager._refill_breaker_diagnosis is not None, (
                "A refill that dispatched nothing was accepted silently."
            )

        asyncio.run(_run())


class TestConsecutiveFailureBreaker:
    """Consecutive failures trip the breaker, and any success clears the count."""

    def test_failures_below_the_threshold_keep_refilling(self):
        """Sporadic failures in a long run must not end it."""

        async def _run():
            manager = FakeManager(threshold=3)
            manager.running_loop = asyncio.get_running_loop()
            manager.data_processor.sample_train_prompts.remote = AsyncMock(return_value=_stub_prompt_batch())

            for parent_id in range(2):
                await manager.notify_group_failed(
                    parent_id=parent_id,
                    failed_uid=parent_id * 2,
                    is_validate=False,
                    terminate_reason=TerminateReason.ROLLOUT_ERROR,
                )

            assert manager._consecutive_group_failures == 2
            assert manager._refill_breaker_diagnosis is None

        asyncio.run(_run())

    def test_reaching_the_threshold_trips_the_breaker_with_the_dominant_reason(self):
        """The diagnosis must name the cause, not just the count.

        A bare count tells the user the run stopped but not why, which is the whole
        point of failing instead of churning.
        """

        async def _run():
            manager = FakeManager(threshold=3)
            manager.running_loop = asyncio.get_running_loop()
            manager.data_processor.sample_train_prompts.remote = AsyncMock(return_value=_stub_prompt_batch())

            for parent_id in range(3):
                await manager.notify_group_failed(
                    parent_id=parent_id,
                    failed_uid=parent_id * 2,
                    is_validate=False,
                    terminate_reason=TerminateReason.ROLLOUT_ERROR,
                )

            assert manager._refill_breaker_diagnosis is not None
            assert TerminateReason.ROLLOUT_ERROR.value in manager._refill_breaker_diagnosis

        asyncio.run(_run())

    def test_an_occupied_group_resets_the_failure_count(self):
        """One group completing proves the pipeline works, so the count must clear."""

        async def _run():
            manager = FakeManager(threshold=3)
            manager.running_loop = asyncio.get_running_loop()
            manager.data_processor.sample_train_prompts.remote = AsyncMock(return_value=_stub_prompt_batch())

            for parent_id in range(2):
                await manager.notify_group_failed(
                    parent_id=parent_id,
                    failed_uid=parent_id * 2,
                    is_validate=False,
                    terminate_reason=TerminateReason.ROLLOUT_ERROR,
                )
            assert manager._consecutive_group_failures == 2

            PSRL_AgentLoopManager._reset_group_failure_streak.__get__(manager)()

            assert manager._consecutive_group_failures == 0
            assert not manager._group_failure_reasons

        asyncio.run(_run())


class TestWaiterSurfacesTheBreaker:
    """The latch is only useful if the driver's own call raises."""

    def test_wait_for_training_batch_raises_once_the_breaker_is_latched(self):
        """A waiter registering after the breaker trips must not block.

        `full_batch.py` calls this through a bare `ray.get`, so raising here is what
        actually ends the run. The worker cannot do it: its task done-callback
        swallows every exception into a log line.
        """

        async def _run():
            manager = FakeManager(threshold=1)
            manager._refill_breaker_diagnosis = "rollout_error x1"

            with pytest.raises(RuntimeError, match="rollout_error"):
                await manager.wait_for_training_batch(buffer_id=0)

        asyncio.run(_run())

    def test_tripping_the_breaker_wakes_an_already_blocked_waiter(self):
        """The common ordering is trainer-waits-first, so a latch alone is not enough."""

        async def _run():
            manager = FakeManager(threshold=1)
            manager.running_loop = asyncio.get_running_loop()

            waiter = asyncio.ensure_future(manager.wait_for_training_batch(buffer_id=0))
            await asyncio.sleep(0)  # Let the waiter register its future.
            assert manager._train_buffer_waiters

            await manager.notify_group_failed(
                parent_id=1,
                failed_uid=2,
                is_validate=False,
                terminate_reason=TerminateReason.ROLLOUT_ERROR,
            )

            with pytest.raises(RuntimeError):
                await asyncio.wait_for(waiter, timeout=5)

        asyncio.run(_run())

    def test_wait_for_training_chunk_raises_once_the_breaker_is_latched(self):
        """The chunk strategy blocks on a different call, which must fail too.

        Under `fine_grain_overlap` the driver never calls `wait_for_training_batch`,
        so guarding only that one leaves the livelock intact for that strategy.
        """

        async def _run():
            manager = FakeManager(threshold=1)
            manager._refill_breaker_diagnosis = "rollout_error x1"

            # Bounded so an unguarded waiter fails the test instead of hanging it.
            with pytest.raises(RuntimeError, match="rollout_error"):
                await asyncio.wait_for(
                    manager.wait_for_training_chunk(buffer_id=0, chunk_index=0),
                    timeout=5,
                )

        asyncio.run(_run())

    def test_tripping_the_breaker_wakes_an_already_blocked_chunk_waiter(self):
        """A chunk waiter registered before the trip must be woken as well."""

        async def _run():
            manager = FakeManager(threshold=1)
            manager.running_loop = asyncio.get_running_loop()

            waiter = asyncio.ensure_future(manager.wait_for_training_chunk(buffer_id=0, chunk_index=0))
            await asyncio.sleep(0)  # Let the waiter register its future.
            assert manager._train_chunk_waiters

            await manager.notify_group_failed(
                parent_id=1,
                failed_uid=2,
                is_validate=False,
                terminate_reason=TerminateReason.ROLLOUT_ERROR,
            )

            with pytest.raises(RuntimeError):
                await asyncio.wait_for(waiter, timeout=5)

        asyncio.run(_run())


class TestTrainValFailureIsolation:
    """Train and validation failure records must not share one set.

    `set_val_buffer_size` runs at the start of every validation round and clears the
    failure record. When train and val share it, a validation round makes the manager
    forget which train groups already failed. The dedupe guard in `notify_group_failed`
    and the late-arrival guard in `occupy_requests` both read that record, so forgetting
    it lets a straggler child of a dead train group be accepted into a buffer.
    """

    def test_a_validation_round_does_not_forget_train_failures(self):
        """Starting a validation round must leave the train failure record intact."""

        async def _run():
            manager = FakeManager(threshold=1000)
            manager.running_loop = asyncio.get_running_loop()
            manager.data_processor.sample_train_prompts.remote = AsyncMock(return_value=_stub_prompt_batch())

            await manager.notify_group_failed(
                parent_id=11,
                failed_uid=22,
                is_validate=False,
                terminate_reason=TerminateReason.ROLLOUT_ERROR,
            )
            assert 11 in manager._failed_train_group_ids

            # A validation round begins. It resets validation bookkeeping only.
            manager.set_val_buffer_size(8)

            assert 11 in manager._failed_train_group_ids, (
                "A validation round erased the train failure record, so a straggler from "
                "train group 11 can now be accepted into a buffer."
            )

        asyncio.run(_run())

    def test_validation_round_start_clears_only_validation_failures(self):
        """The val record is per-round, so `set_val_buffer_size` must still clear it."""

        async def _run():
            manager = FakeManager(threshold=1000)
            manager.running_loop = asyncio.get_running_loop()
            manager.val_buffer_size = 4

            await manager.notify_group_failed(
                parent_id=101,
                failed_uid=202,
                is_validate=True,
                terminate_reason=TerminateReason.ROLLOUT_ERROR,
            )
            assert 101 in manager._failed_val_group_ids

            manager.set_val_buffer_size(4)

            assert not manager._failed_val_group_ids, (
                "Validation failures must not leak across rounds, or a reused prompt id "
                "would be dropped as an already-failed group."
            )

        asyncio.run(_run())

    def test_a_train_failure_does_not_suppress_validation_recovery(self):
        """A shared set also aliases the other way, blocking val recovery.

        Train and val prompt ids come from disjoint namespaces today, so this pins the
        isolation itself rather than relying on that invariant holding.
        """

        async def _run():
            manager = FakeManager(threshold=1000)
            manager.running_loop = asyncio.get_running_loop()
            manager.data_processor.sample_train_prompts.remote = AsyncMock(return_value=_stub_prompt_batch())
            manager.val_buffer_size = 4

            await manager.notify_group_failed(
                parent_id=55,
                failed_uid=110,
                is_validate=False,
                terminate_reason=TerminateReason.ROLLOUT_ERROR,
            )

            # Same id arriving on the validation path must still be handled, not
            # skipped as a duplicate.
            await manager.notify_group_failed(
                parent_id=55,
                failed_uid=110,
                is_validate=True,
                terminate_reason=TerminateReason.ROLLOUT_ERROR,
            )

            assert manager.val_buffer_size == 3, (
                f"Validation recovery was skipped as a duplicate of a train failure, "
                f"val_buffer_size={manager.val_buffer_size}."
            )

        asyncio.run(_run())


class TestTrainFailureRecordIsBounded:
    """The train failure record has no clear-point, so it must be bounded by size.

    Train prompt ids never repeat within a realistic run, so there is no correctness
    reason to ever clear this record, and no per-step clear-point is safe: with
    `staleness > 0` an in-flight prompt outlives several buffers, and the chunk
    strategy skips `maybe_add_buffer` entirely. What is left is a slow memory leak
    over a long run, so the record evicts oldest-first instead.
    """

    def test_the_record_stays_capped_and_evicts_oldest_first(self):
        """Overflowing the cap must drop the oldest id, not the newest.

        Evicting the newest would readmit a straggler from the group that just died,
        which is the failure this record exists to prevent.
        """

        async def _run():
            manager = FakeManager(threshold=1000, failed_id_capacity=4)
            manager.running_loop = asyncio.get_running_loop()
            manager.data_processor.sample_train_prompts.remote = AsyncMock(return_value=_stub_prompt_batch())

            for parent_id in range(6):
                await manager.notify_group_failed(
                    parent_id=parent_id,
                    failed_uid=parent_id * 2,
                    is_validate=False,
                    terminate_reason=TerminateReason.ROLLOUT_ERROR,
                )

            assert len(manager._failed_train_group_ids) == 4, (
                f"Record grew past its cap to {len(manager._failed_train_group_ids)} entries."
            )
            # The two oldest are gone, the four most recent are retained.
            assert 0 not in manager._failed_train_group_ids
            assert 1 not in manager._failed_train_group_ids
            for parent_id in range(2, 6):
                assert parent_id in manager._failed_train_group_ids, (
                    f"Recent failure parent_id={parent_id} was evicted, so its stragglers "
                    "would be readmitted into a buffer."
                )

        asyncio.run(_run())

    def test_a_repeated_failure_does_not_refresh_eviction_order(self):
        """Order must follow insertion, not access.

        A late straggler probing the record must not extend the entry's lifetime,
        or eviction order would depend on straggler traffic rather than on age.
        """

        async def _run():
            manager = FakeManager(threshold=1000, failed_id_capacity=3)
            manager.running_loop = asyncio.get_running_loop()
            manager.data_processor.sample_train_prompts.remote = AsyncMock(return_value=_stub_prompt_batch())

            for parent_id in range(3):
                await manager.notify_group_failed(
                    parent_id=parent_id,
                    failed_uid=parent_id * 2,
                    is_validate=False,
                    terminate_reason=TerminateReason.ROLLOUT_ERROR,
                )

            # Probe the oldest entry the way `occupy_requests` does for a straggler.
            assert 0 in manager._failed_train_group_ids

            await manager.notify_group_failed(
                parent_id=99,
                failed_uid=198,
                is_validate=False,
                terminate_reason=TerminateReason.ROLLOUT_ERROR,
            )

            assert 0 not in manager._failed_train_group_ids, (
                "Reading the oldest entry refreshed it, so eviction follows access order."
            )

        asyncio.run(_run())


def _stub_prompt_batch():
    """Build a stand-in for the TensorDict `sample_train_prompts` returns.

    `_retry_data` only measures its length and stamps a `version_tag` onto it, so a
    dict of the right length is sufficient and avoids a tensordict dependency here.
    """
    import torch
    from tensordict import TensorDict

    return TensorDict({"input_ids": torch.zeros(2, 2)}, batch_size=[2])


class TestCoordinationFailuresDoNotFeedTheBreaker:
    """Scheduling and teardown failures must not end a run that is otherwise healthy.

    The failed end-to-end run tripped a threshold of 32 with exactly 16 genuine stall
    recoveries plus 16 cancellations that arrived while the run was being torn down.
    Counting cleanup as an environment fault turns a recoverable stall into a fatal abort.
    """

    def test_cancellations_are_tallied_but_never_trip_the_breaker(self):
        async def _run():
            manager = FakeManager(threshold=3)
            manager.running_loop = asyncio.get_running_loop()
            manager.data_processor.sample_train_prompts.remote = AsyncMock(return_value=_stub_prompt_batch())

            for parent_id in range(4):
                await manager.notify_group_failed(
                    parent_id=parent_id,
                    failed_uid=parent_id * 2,
                    is_validate=False,
                    terminate_reason=TerminateReason.ROLLOUT_CANCELLED,
                )

            assert manager._consecutive_group_failures == 0
            assert manager._refill_breaker_diagnosis is None

        asyncio.run(_run())

    def test_coordination_failures_are_reported_at_the_breaker_threshold(self, caplog):
        """Tallying without ever reporting would hide a real capacity or lifecycle fault."""

        async def _run():
            manager = FakeManager(threshold=3)
            manager.running_loop = asyncio.get_running_loop()
            manager.data_processor.sample_train_prompts.remote = AsyncMock(return_value=_stub_prompt_batch())

            # Exactly the threshold, because the periodic report fires on exact multiples.
            for parent_id in range(3):
                await manager.notify_group_failed(
                    parent_id=parent_id,
                    failed_uid=parent_id * 2,
                    is_validate=False,
                    terminate_reason=TerminateReason.ROLLOUT_CANCELLED,
                )

        with caplog.at_level("ERROR"):
            asyncio.run(_run())

        assert any("coordination reasons" in record.getMessage() for record in caplog.records), (
            "Coordination failures were counted without ever being reported."
        )

    def test_shutdown_swallows_every_failure(self):
        """Once teardown starts, no in-flight episode can say anything about health."""

        async def _run():
            manager = FakeManager(threshold=1)
            manager.running_loop = asyncio.get_running_loop()
            manager.data_processor.sample_train_prompts.remote = AsyncMock(return_value=_stub_prompt_batch())
            manager._shutting_down = True

            await manager.notify_group_failed(
                parent_id=7,
                failed_uid=14,
                is_validate=False,
                terminate_reason=TerminateReason.ROLLOUT_ERROR,
            )

            assert manager._consecutive_group_failures == 0
            assert manager._refill_breaker_diagnosis is None

        asyncio.run(_run())

    def test_the_failed_run_shape_no_longer_trips_a_threshold_of_32(self):
        """16 stalls plus the cancellations their own recovery caused must not abort."""

        async def _run():
            manager = FakeManager(threshold=32)
            manager.running_loop = asyncio.get_running_loop()
            manager.data_processor.sample_train_prompts.remote = AsyncMock(return_value=_stub_prompt_batch())

            for parent_id in range(16):
                await manager.notify_group_failed(
                    parent_id=parent_id,
                    failed_uid=parent_id * 2,
                    is_validate=False,
                    terminate_reason=TerminateReason.DOWNSTREAM_TIMEOUT,
                )
            for parent_id in range(16, 32):
                await manager.notify_group_failed(
                    parent_id=parent_id,
                    failed_uid=parent_id * 2,
                    is_validate=False,
                    terminate_reason=TerminateReason.ROLLOUT_CANCELLED,
                )

            assert manager._consecutive_group_failures == 16
            assert manager._refill_breaker_diagnosis is None

        asyncio.run(_run())


class TestCapacityFailuresHaveTheirOwnBound:
    """A node that admits nothing is not a broken harness, but it is not harmless either.

    Every capacity fault already cost a full acquisition deadline and its replacement pays
    it again, so an exhausted node refills forever while the buffer never fills. These
    tests pin both halves of the resolution: such failures stay out of the environment
    fault streak, which is the diagnosis the failed run needed, and they are bounded by
    their own streak so the run ends with a capacity diagnosis instead of hanging.
    """

    def test_capacity_timeouts_do_not_feed_the_environment_fault_streak(self):
        """Tallying is what keeps a capacity fault from being read as a broken harness.

        A stall and an unadmitted sandbox both leave a buffer slot empty, but only the
        first says anything about the task or the harness.
        """

        async def _run():
            manager = FakeManager(threshold=1000, capacity_threshold=1000)
            manager.running_loop = asyncio.get_running_loop()
            manager.data_processor.sample_train_prompts.remote = AsyncMock(return_value=_stub_prompt_batch())

            for parent_id in range(6):
                await manager.notify_group_failed(
                    parent_id=parent_id,
                    failed_uid=parent_id * 2,
                    is_validate=False,
                    terminate_reason=TerminateReason.SANDBOX_CAPACITY_TIMEOUT,
                )

            assert manager._consecutive_group_failures == 0, (
                "A capacity fault was counted as an environment fault."
            )
            assert manager._coordination_failures[TerminateReason.SANDBOX_CAPACITY_TIMEOUT.value] == 6
            assert manager._refill_breaker_diagnosis is None

        asyncio.run(_run())

    def test_consecutive_capacity_timeouts_latch_a_capacity_diagnosis(self):
        """An exhausted node must end the run with a capacity diagnosis, not hang.

        Without a bound the replacement pays the acquisition deadline again and again, so
        the buffer never fills and the trainer waits forever on a node that cannot admit
        anything. The diagnosis must also name the deadline that was burned, because that
        number is what makes the wait visible as the cost it is.
        """

        async def _run():
            manager = FakeManager(threshold=1000, capacity_threshold=3)
            manager.running_loop = asyncio.get_running_loop()
            manager.data_processor.sample_train_prompts.remote = AsyncMock(return_value=_stub_prompt_batch())

            for parent_id in range(3):
                await manager.notify_group_failed(
                    parent_id=parent_id,
                    failed_uid=parent_id * 2,
                    is_validate=False,
                    terminate_reason=TerminateReason.SANDBOX_CAPACITY_TIMEOUT,
                )

            diagnosis = manager._refill_breaker_diagnosis
            assert diagnosis is not None, (
                "Three consecutive capacity timeouts left the trainer waiting on a buffer "
                "that could never fill."
            )
            assert TerminateReason.SANDBOX_CAPACITY_TIMEOUT.value in diagnosis
            assert "capacity" in diagnosis
            assert f"{_STUB_TIMEOUTS.admission_timeout_s:.0f}s" in diagnosis, (
                "The diagnosis must name the admission deadline every attempt burned."
            )
            assert manager._consecutive_group_failures == 0, (
                "A capacity fault was reported as a broken harness."
            )

        asyncio.run(_run())

    def test_an_occupied_group_resets_the_capacity_streak(self):
        """A group that makes it through proves capacity is available again."""

        async def _run():
            manager = FakeManager(threshold=1000, capacity_threshold=3)
            manager.running_loop = asyncio.get_running_loop()
            manager.data_processor.sample_train_prompts.remote = AsyncMock(return_value=_stub_prompt_batch())

            for parent_id in range(2):
                await manager.notify_group_failed(
                    parent_id=parent_id,
                    failed_uid=parent_id * 2,
                    is_validate=False,
                    terminate_reason=TerminateReason.SANDBOX_CAPACITY_TIMEOUT,
                )
            assert manager._capacity_failure_streak == 2

            # One group reaches a buffer, so the next slot capacity timeout starts over.
            PSRL_AgentLoopManager._reset_group_failure_streak.__get__(manager)()
            await manager.notify_group_failed(
                parent_id=99,
                failed_uid=198,
                is_validate=False,
                terminate_reason=TerminateReason.SANDBOX_CAPACITY_TIMEOUT,
            )

            assert manager._capacity_failure_streak == 1
            assert manager._refill_breaker_diagnosis is None, (
                "A capacity streak survived a group that succeeded in completing."
            )

        asyncio.run(_run())

    def test_a_zero_threshold_waits_indefinitely_but_still_reports(self, caplog):
        """Waiting forever is a legitimate experiment setting, but not a silent one."""

        async def _run():
            manager = FakeManager(threshold=3, capacity_threshold=0)
            manager.running_loop = asyncio.get_running_loop()
            manager.data_processor.sample_train_prompts.remote = AsyncMock(return_value=_stub_prompt_batch())

            for parent_id in range(3):
                await manager.notify_group_failed(
                    parent_id=parent_id,
                    failed_uid=parent_id * 2,
                    is_validate=False,
                    terminate_reason=TerminateReason.SANDBOX_CAPACITY_TIMEOUT,
                )

            assert manager._refill_breaker_diagnosis is None

        with caplog.at_level("ERROR"):
            asyncio.run(_run())

        assert any("coordination reasons" in record.getMessage() for record in caplog.records), (
            "Capacity failures waited indefinitely without ever being reported."
        )
