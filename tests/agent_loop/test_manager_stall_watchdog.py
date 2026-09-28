"""Unit tests for the entry stall watchdog, validation recovery, and the buffer-wait heartbeat.

A buffer entry is all-or-nothing: it occupies its slot only once every child reports. A
rollout that never reports therefore blocks the whole step, and before the watchdog the
only way out was the 32-consecutive-failure breaker, which never sees a child that reports
nothing at all.

Validation gets the same watchdog and its own recovery. A validation prompt belongs to a
fixed evaluation set, so losing one silently changes what accuracy was measured over: the
round retries the same prompt, and only once its attempts are spent does it shrink the
target and release the slot.

Methods are borrowed from the real class onto a stub instance so the tests exercise
production code without a Ray cluster. Every test uses `asyncio.run()`, because
registration starts the watchdog task and needs a running loop.
"""

import asyncio
import logging
import time
from collections import Counter
from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import OmegaConf
from psrl.workers.agent_loop.loops.utils import TerminateReason
from psrl.workers.agent_loop.manager import BoundedIdSet, InflightGroup, PSRL_AgentLoopManager
from psrl.workers.agent_loop.timeouts import AgentLoopTimeouts
from verl.utils import tensordict_utils as tu

pytestmark = pytest.mark.cpu_test

# The capacity diagnosis names the admission deadline, so the stub ladder has to carry
# one. It is not otherwise read here; the watchdog uses its own stall threshold.
_STUB_TIMEOUTS = AgentLoopTimeouts(
    episode_timeout_s=7200.0,
    setup_allowance_s=900.0,
    admission_timeout_s=1800.0,
    heartbeat_interval_s=120.0,
    entry_stall_timeout_s=360.0,
)

_BORROWED = (
    "_register_inflight_groups",
    "_touch_inflight_group",
    "_unregister_inflight_group",
    "_ensure_stall_watchdog",
    "_stalled_entries",
    "_stall_watchdog",
    "_buffer_wait_progress",
    "touch_inflight_group",
    "_await_buffer_future",
    "notify_group_failed",
    "_retry_data",
    "_record_group_failure",
    "_record_capacity_failure",
    "_report_coordination_failures",
    "_reset_group_failure_streak",
    "_trip_refill_breaker",
    "_refill_failed_group",
    "_retry_validation_group",
    "_forget_validation_group",
    "_retain_val_round_prompts",
    "set_val_buffer_size",
    "_refill_breaker_error",
    "_raise_if_refill_breaker_tripped",
    "wait_for_training_batch",
)


def _make_config(stall_timeout: float = 9000.0, refill_threshold: int = 32) -> OmegaConf:
    """Build the smallest config tree the borrowed manager methods read."""
    return OmegaConf.create(
        {
            "psrl": {
                "agentic_rl": {
                    "entry_stall_timeout_s": stall_timeout,
                    "buffer_wait_log_interval_s": 0.05,
                    "refill_failure_threshold": refill_threshold,
                },
                "rollout_coordination": {
                    "redundant_rollout": {"enable": False},
                    "proactive_filter_strategy": {"method": None},
                },
            }
        }
    )


class FakeManager:
    """Minimal stub of `PSRL_AgentLoopManager` for the watchdog tests."""

    def __init__(
        self,
        *,
        stall_timeout: float = 9000.0,
        rollout_n: int = 4,
        refill_threshold: int = 32,
        capacity_threshold: int = 4,
        val_rollout_n: int = 1,
        val_retry_limit: int = 1,
    ) -> None:
        self.config = _make_config(stall_timeout, refill_threshold)
        self.rollout_n = rollout_n
        self.val_rollout_n = val_rollout_n
        self.val_retry_limit = val_retry_limit
        self._val_attempts: dict[int, int] = {}
        self._val_round_prompts: dict = {}
        self.curr_ps_version_tag = 0
        self.refill_failure_threshold = refill_threshold
        self.capacity_failure_threshold = capacity_threshold
        self.timeouts = _STUB_TIMEOUTS
        self.entry_stall_timeout_s = stall_timeout
        self.entry_stall_check_interval_s = max(0.05, min(60.0, stall_timeout / 20))
        self.buffer_wait_log_interval_s = 0.05
        self._stall_watchdog_task = None
        self._inflight_groups: dict[int, InflightGroup] = {}

        self._failed_train_group_ids: BoundedIdSet = BoundedIdSet(1024)
        self._failed_val_group_ids: set[int] = set()
        self.rollout_request_tracker: dict = {}
        self._consecutive_group_failures = 0
        self._capacity_failure_streak = 0
        self._group_failure_reasons: Counter = Counter()
        self._coordination_failures: Counter = Counter()
        self._grading_capacity_faults: set[int] = set()
        self._shutting_down = False
        self._refill_breaker_diagnosis: str | None = None

        self.train_data_buffers: dict = {}
        self.train_accumulated_buffers: dict = {}
        self.train_accumulated_buffer_size: dict = {}
        self.ready_entries_per_buffer = 8
        self._train_buffer_waiters: dict = {}
        self._train_chunk_waiters: dict = {}
        self._resolved_train_chunks: dict = {}

        self.val_buffer_size: int | None = None
        self.val_data_buffers: dict = {}
        self.val_accumulated_buffer_size: dict = {}
        self._val_buffer_waiters: dict = {}
        self._val_round_all_failed = False

        self.running_loop = None
        self._request_counter = 0
        self.dispatched: list = []
        self.flushed: list = []

        self.ps_manager_handle = MagicMock()
        self.ps_manager_handle.acquire.remote = AsyncMock(return_value=True)
        self.ps_manager_handle.release.remote = AsyncMock(return_value=None)
        self.ps_manager_handle.abort_requests.remote = AsyncMock(return_value=None)
        self.ps_manager_handle.ensure_train_buffer_exists.remote = AsyncMock(return_value=None)
        self.ps_manager_handle.readmit_requests.remote = AsyncMock(return_value=None)
        self.ps_manager_handle.update_request_status.remote = AsyncMock(return_value=True)

        # A refill that dispatches nothing trips the breaker, which would mask the
        # assertions, so every test refills successfully unless it says otherwise.
        self.data_processor = MagicMock()
        self.data_processor.sample_train_prompts.remote = AsyncMock(return_value=_stub_prompt_batch())

        for name in _BORROWED:
            setattr(self, name, getattr(PSRL_AgentLoopManager, name).__get__(self))

    def stop_watchdog(self) -> None:
        """Cancel the watchdog task so a finished test leaves nothing pending."""
        if self._stall_watchdog_task is not None:
            self._stall_watchdog_task.cancel()

    async def _purge_tracker_group(self, parent_id, rollout_n, is_validate):
        """Stand in for the TQ payload purge, which needs a live TransferQueue."""
        self._unregister_inflight_group(parent_id)
        return self.rollout_request_tracker.pop(parent_id, [])

    async def _inner_dispatch_data(self, data, is_validate: bool = False):
        """Record a dispatch instead of fanning out to Ray workers."""
        self.dispatched.append((tu.get(data, "uid"), is_validate))
        self._register_inflight_groups(tu.get(data, "uid"), is_validate)

    async def _flush_ready_buffer(self, buffer_id: int, is_validate: bool) -> bool:
        """Record a flush instead of assembling a batch from a live TransferQueue."""
        self.flushed.append((buffer_id, is_validate))
        return True


class TestRegistration:
    """Every dispatched entry that can block a buffer must be watched."""

    def test_a_dispatch_registers_its_entries(self):
        async def _run():
            manager = FakeManager(rollout_n=4)
            manager._register_inflight_groups([8, 9, 10, 11, 12, 13, 14, 15], is_validate=False)

            assert set(manager._inflight_groups) == {2, 3}
            assert manager._stall_watchdog_task is not None
            manager.stop_watchdog()

        asyncio.run(_run())

    def test_validation_entries_are_watched_too(self):
        # A val group that goes silent is never subtracted from the round's target, so the
        # round waits on a slot that will never arrive. Watching it is the only way out.
        async def _run():
            manager = FakeManager(val_rollout_n=1)
            manager._register_inflight_groups([5], is_validate=True)

            assert set(manager._inflight_groups) == {5}
            assert manager._inflight_groups[5].is_validate
            manager.stop_watchdog()

        asyncio.run(_run())

    def test_watching_can_be_disabled(self):
        async def _run():
            manager = FakeManager(stall_timeout=0)
            manager._register_inflight_groups([8, 9], is_validate=False)

            assert manager._inflight_groups == {}

        asyncio.run(_run())

    def test_progress_moves_the_stall_deadline(self):
        async def _run():
            manager = FakeManager(stall_timeout=10)
            manager._register_inflight_groups([8], is_validate=False)
            entry = manager._inflight_groups[2]

            manager._touch_inflight_group(2)
            second_progress = manager._inflight_groups[2].last_progress_at

            assert entry.dispatched_at == entry.last_progress_at
            assert second_progress > entry.last_progress_at
            manager.stop_watchdog()

        asyncio.run(_run())

    def test_a_completed_entry_stops_being_watched(self):
        async def _run():
            manager = FakeManager(stall_timeout=10)
            manager._register_inflight_groups([8], is_validate=False)
            manager._unregister_inflight_group(2)

            assert manager._stalled_entries(now=time.monotonic() + 1000) == []
            manager.stop_watchdog()

        asyncio.run(_run())


class TestStallSelection:
    """The watchdog fires on silence, not on slowness."""

    def test_a_silent_entry_is_selected(self):
        async def _run():
            manager = FakeManager(stall_timeout=100)
            manager._register_inflight_groups([8], is_validate=False)
            last_progress = manager._inflight_groups[2].last_progress_at

            assert manager._stalled_entries(now=last_progress + 99) == []
            assert [parent_id for parent_id, _ in manager._stalled_entries(now=last_progress + 100)] == [2]
            manager.stop_watchdog()

        asyncio.run(_run())

    def test_an_entry_that_keeps_reporting_is_never_selected(self):
        async def _run():
            manager = FakeManager(stall_timeout=100)
            manager._register_inflight_groups([8, 9, 10, 11], is_validate=False)
            now = 0.0
            for _ in range(3):
                manager._touch_inflight_group(2)
                # Each check lands 50s after the latest report, below the 100s timeout.
                now = time.monotonic() + 50

            assert manager._stalled_entries(now=now) == []
            manager.stop_watchdog()

        asyncio.run(_run())


class TestWatchdogRecovery:
    """A stalled entry must be refilled, not waited on."""

    def test_the_watchdog_recovers_a_silent_entry(self):
        async def _run():
            manager = FakeManager(stall_timeout=0.05)
            manager.running_loop = asyncio.get_running_loop()
            manager._register_inflight_groups([8, 9], is_validate=False)
            manager.stop_watchdog()

            watchdog = asyncio.ensure_future(manager._stall_watchdog())
            for _ in range(40):
                await asyncio.sleep(0.05)
                if 2 in manager._failed_train_group_ids:
                    break
            watchdog.cancel()

            assert 2 in manager._failed_train_group_ids, "A silent entry was never recovered."
            assert manager._consecutive_group_failures == 1
            assert manager._group_failure_reasons[TerminateReason.DOWNSTREAM_TIMEOUT.value] == 1
            manager.ps_manager_handle.abort_requests.remote.assert_awaited()

        asyncio.run(_run())

    def test_recovery_aborts_the_surviving_siblings(self):
        async def _run():
            manager = FakeManager(stall_timeout=0.05)
            manager.running_loop = asyncio.get_running_loop()
            manager._register_inflight_groups([8, 9, 10, 11], is_validate=False)
            manager.stop_watchdog()

            watchdog = asyncio.ensure_future(manager._stall_watchdog())
            for _ in range(40):
                await asyncio.sleep(0.05)
                if 2 in manager._failed_train_group_ids:
                    break
            watchdog.cancel()

            aborted = manager.ps_manager_handle.abort_requests.remote.await_args.args[0]
            assert aborted == [9, 10, 11], f"Expected the three survivors, got {aborted}."

        asyncio.run(_run())

    def test_recovery_refills_the_lost_slot(self):
        async def _run():
            manager = FakeManager(stall_timeout=0.05)
            manager.running_loop = asyncio.get_running_loop()
            manager._register_inflight_groups([8], is_validate=False)
            manager.stop_watchdog()

            watchdog = asyncio.ensure_future(manager._stall_watchdog())
            for _ in range(40):
                await asyncio.sleep(0.05)
                if 2 in manager._failed_train_group_ids:
                    break
            watchdog.cancel()

            assert manager._refill_breaker_diagnosis is None, "The refill path churned instead of recovering."

        asyncio.run(_run())


class TestBufferWaitHeartbeat:
    """An incomplete buffer must be visible in the log while the driver waits."""

    def test_progress_names_the_accumulated_and_inflight_entries(self):
        async def _run():
            manager = FakeManager()
            manager.train_accumulated_buffer_size[22] = 4
            manager._register_inflight_groups([8, 9], is_validate=False)

            message = manager._buffer_wait_progress(22, is_validate=False)

            assert "buffer_id=22" in message
            assert "accumulated=4/8" in message
            assert "in_flight_entries=1" in message
            assert f"capacity_failures=0/{manager.capacity_failure_threshold}" in message
            manager.stop_watchdog()

        asyncio.run(_run())

    def test_progress_reports_when_nothing_is_in_flight(self):
        async def _run():
            manager = FakeManager()
            manager.train_accumulated_buffer_size[22] = 4

            message = manager._buffer_wait_progress(22, is_validate=False)

            assert "in_flight_entries=0" in message

        asyncio.run(_run())

    def test_progress_shows_how_close_the_capacity_breaker_is(self):
        """A slow buffer wait must show whether capacity is the reason for it.

        The capacity streak is the only counter that turns an unadmittable node into an
        abort, so a user reading the progress line has to see it approaching.
        """

        async def _run():
            manager = FakeManager(capacity_threshold=4)
            manager._capacity_failure_streak = 3
            manager.train_accumulated_buffer_size[22] = 4

            message = manager._buffer_wait_progress(22, is_validate=False)

            assert "capacity_failures=3/4" in message

        asyncio.run(_run())

    def test_waiting_logs_progress_until_the_buffer_resolves(self, caplog):
        async def _run():
            manager = FakeManager()
            manager.train_accumulated_buffer_size[22] = 1
            fut: asyncio.Future = asyncio.get_running_loop().create_future()
            waiter = asyncio.ensure_future(manager._await_buffer_future(fut, buffer_id=22, is_validate=False))
            await asyncio.sleep(0.2)
            fut.set_result({"keys": [], "tags": [], "partition_id": "train"})
            return await waiter

        with caplog.at_level(logging.WARNING):
            result = asyncio.run(_run())

        assert result["partition_id"] == "train"
        assert any("Buffer wait: buffer_id=22" in record.getMessage() for record in caplog.records), (
            "The driver waited on an incomplete buffer without logging progress."
        )


def _stub_prompt_batch():
    """Build a stand-in for the TensorDict `sample_train_prompts` returns."""
    import torch
    from tensordict import TensorDict

    return TensorDict({"input_ids": torch.zeros(2, 2)}, batch_size=[2])


def _stub_val_batch(prompt_ids: list[int], val_rollout_n: int = 1):
    """Build a stand-in for the validation batch `generate_validate_sequences` dispatches."""
    import torch
    from tensordict import TensorDict

    rows = len(prompt_ids) * val_rollout_n
    batch = TensorDict({"input_ids": torch.zeros(rows, 2)}, batch_size=[rows])
    uids = [pid * val_rollout_n + i for pid in prompt_ids for i in range(val_rollout_n)]
    tu.assign_non_tensor_stack(batch, "uid", uids)
    tu.assign_non_tensor_stack(
        batch,
        "parent_id",
        [pid for pid in prompt_ids for _ in range(val_rollout_n)],
    )
    return batch


class TestValidationRecovery:
    """A validation round must retry a lost prompt, then really let its slot go."""

    def test_a_lost_prompt_is_retried_before_the_round_shrinks(self):
        """The evaluation set is fixed, so a coordination fault must not silently shrink it."""

        async def _run():
            manager = FakeManager(val_retry_limit=2)
            manager.set_val_buffer_size(3)
            manager._retain_val_round_prompts(_stub_val_batch([10, 11, 12]))

            await manager.notify_group_failed(
                10,
                failed_uid=10,
                is_validate=True,
                terminate_reason=TerminateReason.CONTAINER_LOST,
            )

            assert manager.val_buffer_size == 3, "The round shrank instead of retrying."
            assert manager.dispatched == [([10], True)]
            assert manager._val_attempts[10] == 2
            # The retry's own results arrive under the same ids, so the failure record must
            # not survive or `occupy_requests` would discard them as late arrivals.
            assert 10 not in manager._failed_val_group_ids
            manager.ps_manager_handle.readmit_requests.remote.assert_awaited_once()
            manager.stop_watchdog()

        asyncio.run(_run())

    def test_an_exhausted_prompt_is_removed_from_flight(self):
        """The deadlock: a shrink that leaves the slot in flight waits on it forever."""

        async def _run():
            manager = FakeManager(val_retry_limit=1, val_rollout_n=1)
            manager.set_val_buffer_size(3)
            manager._register_inflight_groups([10, 11, 12], is_validate=True)

            await manager.notify_group_failed(
                10,
                failed_uid=10,
                is_validate=True,
                terminate_reason=TerminateReason.CONTAINER_LOST,
            )

            assert manager.val_buffer_size == 2
            assert manager.dispatched == [], "A limit of 1 must not retry."
            assert 10 not in manager._inflight_groups, "The abandoned slot is still in flight."
            assert 10 not in manager._val_attempts
            assert set(manager._inflight_groups) == {11, 12}
            manager.stop_watchdog()

        asyncio.run(_run())

    def test_the_shrink_that_meets_the_waiting_count_fires_the_buffer(self):
        """The last group to give up is what completes the round, so it must publish it."""

        async def _run():
            manager = FakeManager(val_retry_limit=1)
            manager.set_val_buffer_size(16)
            # 15 arrived; the 16th is the group about to give up. Shrinking to 15 is the
            # moment the round becomes complete, and nothing else will arrive to notice.
            manager.val_accumulated_buffer_size[4] = 15

            await manager.notify_group_failed(
                10,
                failed_uid=10,
                is_validate=True,
                terminate_reason=TerminateReason.CONTAINER_LOST,
            )

            assert manager.val_buffer_size == 15
            assert manager.flushed == [(4, True)]
            manager.stop_watchdog()

        asyncio.run(_run())

    def test_the_target_never_steps_over_a_waiting_count(self):
        """Why equality suffices: each shrink moves the target by exactly one group.

        The comparison can only miss if the target can jump past `accumulated` while that
        count sits still. It cannot: every shrink is a single decrement taken under the same
        lock the accumulate path holds, so the target descends through every integer. A round
        left short is a group that was neither accumulated nor subtracted, which is the stall
        watchdog's job, not the comparison's.
        """

        async def _run():
            manager = FakeManager(val_retry_limit=1, val_rollout_n=1)
            manager.set_val_buffer_size(4)
            manager.val_accumulated_buffer_size[4] = 2

            for parent_id in (10, 11):
                await manager.notify_group_failed(
                    parent_id,
                    failed_uid=parent_id,
                    is_validate=True,
                    terminate_reason=TerminateReason.CONTAINER_LOST,
                )

            # 4 -> 3 -> 2: the target landed on the waiting count instead of passing it, so
            # the buffer fired exactly once, on the shrink that met it.
            assert manager.val_buffer_size == 2
            assert manager.flushed == [(4, True)]
            manager.stop_watchdog()

        asyncio.run(_run())

    def test_an_expired_episode_budget_is_not_retried(self):
        """An episode that used its whole budget would spend another one re-failing."""

        async def _run():
            manager = FakeManager(val_retry_limit=3)
            manager.set_val_buffer_size(3)
            manager._retain_val_round_prompts(_stub_val_batch([10, 11, 12]))

            await manager.notify_group_failed(
                10,
                failed_uid=10,
                is_validate=True,
                terminate_reason=TerminateReason.TRAJECTORY_TIMEOUT,
            )

            assert manager.dispatched == []
            assert manager.val_buffer_size == 2
            manager.stop_watchdog()

        asyncio.run(_run())

    def test_a_prompt_that_was_never_provisioned_is_retried(self):
        """Failed provisioning is a capacity fact about the node, not about the task."""

        async def _run():
            manager = FakeManager(val_retry_limit=2)
            manager.set_val_buffer_size(3)
            manager._retain_val_round_prompts(_stub_val_batch([10, 11, 12]))

            await manager.notify_group_failed(
                10,
                failed_uid=10,
                is_validate=True,
                terminate_reason=TerminateReason.ROLLOUT_DEADLINE_EXCEEDED,
            )

            assert manager.dispatched == [([10], True)]
            assert manager.val_buffer_size == 3
            manager.stop_watchdog()

        asyncio.run(_run())

    def test_a_new_round_restores_every_retry_budget(self):
        async def _run():
            manager = FakeManager(val_retry_limit=2)
            manager.set_val_buffer_size(3)
            manager._register_inflight_groups([10, 11], is_validate=True)
            manager._val_attempts[10] = 2

            manager.set_val_buffer_size(3)

            assert manager._val_attempts == {}
            assert manager._inflight_groups == {}, "Last round's entries outlived their round."
            manager.stop_watchdog()

        asyncio.run(_run())

    def test_a_silent_validation_entry_is_recovered(self):
        """Validation used to be unwatched, so a silent group was never subtracted."""

        async def _run():
            manager = FakeManager(stall_timeout=0.05, val_retry_limit=1, val_rollout_n=1)
            manager.running_loop = asyncio.get_running_loop()
            manager.set_val_buffer_size(3)
            manager._register_inflight_groups([10], is_validate=True)
            manager.stop_watchdog()

            watchdog = asyncio.ensure_future(manager._stall_watchdog())
            for _ in range(40):
                await asyncio.sleep(0.05)
                if 10 in manager._failed_val_group_ids:
                    break
            watchdog.cancel()

            assert 10 in manager._failed_val_group_ids, "A silent validation entry was never recovered."
            assert manager.val_buffer_size == 2
            assert 10 not in manager._inflight_groups

        asyncio.run(_run())


class TestLivenessKeepsASlowEntryAlive:
    """The watchdog is a silence check, so a working-but-slow episode must survive it."""

    def test_a_heartbeat_prevents_recovery(self):
        async def _run():
            manager = FakeManager(stall_timeout=0.2)
            manager.running_loop = asyncio.get_running_loop()
            manager._register_inflight_groups([8, 9], is_validate=False)
            manager.stop_watchdog()

            watchdog = asyncio.ensure_future(manager._stall_watchdog())
            for _ in range(12):
                await asyncio.sleep(0.05)
                await manager.touch_inflight_group(2)
            watchdog.cancel()

            assert 2 not in manager._failed_train_group_ids, "A working episode was abandoned because it was slow."
            assert 2 in manager._inflight_groups

        asyncio.run(_run())

    def test_silence_after_heartbeats_is_still_recovered(self):
        async def _run():
            manager = FakeManager(stall_timeout=0.2)
            manager.running_loop = asyncio.get_running_loop()
            manager._register_inflight_groups([8, 9], is_validate=False)
            manager.stop_watchdog()

            watchdog = asyncio.ensure_future(manager._stall_watchdog())
            for _ in range(4):
                await asyncio.sleep(0.05)
                await manager.touch_inflight_group(2)
            for _ in range(20):
                await asyncio.sleep(0.05)
                if 2 in manager._failed_train_group_ids:
                    break
            watchdog.cancel()

            assert 2 in manager._failed_train_group_ids, "A worker that stopped reporting must still be recovered."

        asyncio.run(_run())

    def test_progress_reports_silence_and_the_stall_countdown(self):
        async def _run():
            manager = FakeManager()
            manager.train_accumulated_buffer_size[22] = 4
            manager._register_inflight_groups([8, 9], is_validate=False)

            message = manager._buffer_wait_progress(22, is_validate=False)

            assert "silent_for_s=" in message
            assert "stall_in_s=" in message
            manager.stop_watchdog()

        asyncio.run(_run())
