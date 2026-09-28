"""Validation-round recovery and the buffer-wait heartbeat.

A validation prompt belongs to a fixed evaluation set, so losing one to an infrastructure
fault changes what the reported accuracy was measured over. The round retries the same
prompt, and only once its attempts are spent does it shrink the target and release the
slot.

Nothing here is on a timer. The manager used to run a stall watchdog that declared an
entry dead after a few minutes of silence, which killed healthy validation rollouts that
still had most of their episode budget left. Deadlines belong to the rollout: the episode
budget ends a rollout and reports it, and this layer only reacts to what it is told.

Methods are borrowed from the real class onto a stub instance so the tests exercise
production code without a Ray cluster. Every test uses `asyncio.run()`.
"""

import asyncio
import logging
from collections import Counter
from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import OmegaConf
from psrl.workers.agent_loop.loops.utils import TerminateReason
from psrl.workers.agent_loop.manager import BoundedIdSet, PSRL_AgentLoopManager
from psrl.workers.agent_loop.timeouts import AgentLoopTimeouts
from verl.utils import tensordict_utils as tu

pytestmark = pytest.mark.cpu_test

# The capacity diagnosis names the admission deadline, so the stub ladder has to carry one.
_STUB_TIMEOUTS = AgentLoopTimeouts(
    episode_timeout_s=7200.0,
    setup_allowance_s=900.0,
    admission_timeout_s=1800.0,
)

_BORROWED = (
    "_buffer_wait_progress",
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


def _make_config(refill_threshold: int = 32) -> OmegaConf:
    """Build the smallest config tree the borrowed manager methods read."""
    return OmegaConf.create(
        {
            "psrl": {
                "agentic_rl": {
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
    """Minimal stub of `PSRL_AgentLoopManager` for the recovery tests."""

    def __init__(
        self,
        *,
        rollout_n: int = 4,
        refill_threshold: int = 32,
        capacity_threshold: int = 4,
        val_rollout_n: int = 1,
        val_retry_limit: int = 1,
    ) -> None:
        self.config = _make_config(refill_threshold)
        self.rollout_n = rollout_n
        self.val_rollout_n = val_rollout_n
        self.val_retry_limit = val_retry_limit
        self._val_attempts: dict[int, int] = {}
        self._val_round_prompts: dict = {}
        self.curr_ps_version_tag = 0
        self.refill_failure_threshold = refill_threshold
        self.capacity_failure_threshold = capacity_threshold
        self.timeouts = _STUB_TIMEOUTS
        self.buffer_wait_log_interval_s = 0.05

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

    async def _purge_tracker_group(self, parent_id, rollout_n, is_validate):
        """Stand in for the TQ payload purge, which needs a live TransferQueue."""
        return self.rollout_request_tracker.pop(parent_id, [])

    async def _inner_dispatch_data(self, data, is_validate: bool = False):
        """Record a dispatch instead of fanning out to Ray workers."""
        self.dispatched.append((tu.get(data, "uid"), is_validate))

    async def _flush_ready_buffer(self, buffer_id: int, is_validate: bool) -> bool:
        """Record a flush instead of assembling a batch from a live TransferQueue."""
        self.flushed.append((buffer_id, is_validate))
        return True


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


class TestNothingExpiresOnACoordinationTimer:
    """This layer may not decide, on its own clock, that a rollout is finished."""

    def test_the_manager_owns_no_stall_timer(self):
        """A 360s silence threshold here once killed rollouts with 6800s of budget left."""
        for attribute in (
            "_stall_watchdog",
            "_stalled_entries",
            "_register_inflight_groups",
            "_touch_inflight_group",
            "touch_inflight_group",
            "entry_stall_timeout_s",
        ):
            assert not hasattr(PSRL_AgentLoopManager, attribute), (
                f"{attribute!r} reintroduces a coordination-side deadline; the rollout owns them."
            )

    def test_the_ladder_carries_no_deadline_shorter_than_the_episode(self):
        for field in ("entry_stall_timeout_s", "heartbeat_interval_s"):
            assert not hasattr(_STUB_TIMEOUTS, field)
        assert _STUB_TIMEOUTS.child_deadline_s > _STUB_TIMEOUTS.episode_timeout_s


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

        asyncio.run(_run())

    def test_an_exhausted_prompt_releases_its_slot(self):
        """A shrink that keeps the prompt's bookkeeping leaves the round waiting on it."""

        async def _run():
            manager = FakeManager(val_retry_limit=1)
            manager.set_val_buffer_size(3)
            manager._retain_val_round_prompts(_stub_val_batch([10, 11, 12]))

            await manager.notify_group_failed(
                10,
                failed_uid=10,
                is_validate=True,
                terminate_reason=TerminateReason.CONTAINER_LOST,
            )

            assert manager.val_buffer_size == 2
            assert manager.dispatched == [], "A limit of 1 must not retry."
            assert 10 not in manager._val_attempts
            assert 10 not in manager._val_round_prompts

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

        asyncio.run(_run())

    def test_the_target_never_steps_over_a_waiting_count(self):
        """Why equality suffices: each shrink moves the target by exactly one group.

        The comparison can only miss if the target can jump past `accumulated` while that
        count sits still. It cannot: every shrink is a single decrement taken under the same
        lock the accumulate path holds, so the target descends through every integer.
        """

        async def _run():
            manager = FakeManager(val_retry_limit=1)
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

        asyncio.run(_run())

    def test_a_new_round_restores_every_retry_budget(self):
        async def _run():
            manager = FakeManager(val_retry_limit=2)
            manager.set_val_buffer_size(3)
            manager._retain_val_round_prompts(_stub_val_batch([10, 11]))
            manager._val_attempts[10] = 2

            manager.set_val_buffer_size(3)

            assert manager._val_attempts == {}
            assert manager._val_round_prompts == {}, "Last round's prompts outlived their round."

        asyncio.run(_run())


class TestBufferWaitHeartbeat:
    """An incomplete buffer must be visible in the log while the driver waits."""

    def test_progress_names_what_the_buffer_is_still_owed(self):
        async def _run():
            manager = FakeManager()
            manager.train_accumulated_buffer_size[22] = 4

            message = manager._buffer_wait_progress(22, is_validate=False)

            assert "buffer_id=22" in message
            assert "accumulated=4/8" in message
            assert "pending_groups=4" in message
            assert f"capacity_failures=0/{manager.capacity_failure_threshold}" in message

        asyncio.run(_run())

    def test_progress_reports_no_age_for_a_running_rollout(self):
        """An age here would read as a deadline, and this layer enforces none."""

        async def _run():
            manager = FakeManager()
            manager.train_accumulated_buffer_size[22] = 4

            message = manager._buffer_wait_progress(22, is_validate=False)

            assert "silent_for_s" not in message
            assert "stall_in_s" not in message
            assert "oldest_dispatch_s" not in message

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

    def test_a_validation_wait_reports_its_own_retry_budget(self):
        async def _run():
            manager = FakeManager(val_retry_limit=2)
            manager.set_val_buffer_size(16)
            manager.val_accumulated_buffer_size[4] = 15
            manager._val_attempts[10] = 2

            message = manager._buffer_wait_progress(4, is_validate=True)

            assert "accumulated=15/16" in message
            assert "val_attempts=2" in message

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
