"""Unit tests for the root cause reported when a rollout group fails.

A group failure is the one event an operator has to be able to read: it is what ends a
run through the refill breaker, and it is the only place the reason appears. Two things
went wrong with it and both were silent:

- The cause was logged before the duplicate guard. A failed group aborts every sibling,
  and each aborted sibling reports its own failure with the same text, so one failure
  produced thousands of identical lines that buried the one line that mattered.
- The cause could be blank. A whitespace-only summary prints as empty lines, and empty
  lines read as "the logger is broken" rather than "no cause was recorded" -- and from a
  log there is no way to tell the two apart.

These tests pin that the cause is printed once per group, that it is never blank, and
that when nothing was captured the line still says which terminate reason it was.

Methods are borrowed from the real class onto a stub instance, so the assertions cover
production code without a Ray cluster, and all tests use `asyncio.run()`.
"""

import asyncio
import logging
from collections import Counter
from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import OmegaConf
from psrl.workers.agent_loop.loops.utils import TerminateReason
from psrl.workers.agent_loop.manager import PSRL_AgentLoopManager, _root_cause

pytestmark = pytest.mark.cpu_test


def _manager(rollout_n: int = 4):
    """Build a stub carrying the real `notify_group_failed`."""
    mgr = MagicMock(spec=[])
    mgr.rollout_n = rollout_n
    mgr.val_rollout_n = rollout_n
    mgr._failed_train_group_ids = set()
    mgr._failed_val_group_ids = set()
    mgr._grading_capacity_faults = set()
    mgr.config = OmegaConf.create({"psrl": {"agentic_rl": {}}})
    mgr.ps_manager_handle = MagicMock()
    mgr.ps_manager_handle.acquire.remote = AsyncMock(return_value=True)
    mgr.ps_manager_handle.release.remote = AsyncMock(return_value=None)
    mgr.ps_manager_handle.abort_requests.remote = AsyncMock(return_value=None)
    mgr.notify_group_failed = PSRL_AgentLoopManager.notify_group_failed.__get__(mgr)
    mgr._purge_tracker_group = AsyncMock(return_value=None)
    # The real accounting, so the group actually lands in the failed set and the second
    # child is recognised as a duplicate rather than recorded again.
    mgr._record_group_failure = PSRL_AgentLoopManager._record_group_failure.__get__(mgr)
    mgr._consecutive_group_failures = 0
    mgr._capacity_failure_streak = 0
    mgr._group_failure_reasons = Counter()
    mgr._coordination_failures = Counter()
    mgr._capacity_failure_threshold = 4
    mgr._refill_failure_threshold = 4
    mgr.capacity_failure_threshold = 4
    mgr.refill_failure_threshold = 4
    mgr._shutting_down = False
    # Refill is its own subsystem; here it would only dispatch into a void.
    mgr._refill_failed_group = AsyncMock(return_value=1)
    return mgr


def _records(caplog):
    return [r.getMessage() for r in caplog.records if "root cause" in r.getMessage()]


def test_a_group_reports_its_cause_once_not_once_per_child(caplog):
    # Four children in the group, all reporting the same failure. Only the first carries
    # the diagnosis: the rest are the same group and add nothing.
    mgr = _manager(rollout_n=4)
    with caplog.at_level(logging.ERROR):
        for uid in range(4):
            asyncio.run(
                mgr.notify_group_failed(
                    parent_id=7,
                    failed_uid=uid,
                    is_validate=False,
                    terminate_reason=TerminateReason.ROLLOUT_ERROR,
                    failure_summary="Traceback: SandboxTransportError: connection refused",
                )
            )

    assert len(_records(caplog)) == 1, _records(caplog)
    assert "connection refused" in _records(caplog)[0]


def test_a_blank_cause_says_so_and_names_the_reason(caplog):
    # The failure this run actually hit: 9102 lines of "root cause:" followed by nothing.
    mgr = _manager()
    with caplog.at_level(logging.ERROR):
        asyncio.run(
            mgr.notify_group_failed(
                parent_id=11,
                failed_uid=0,
                is_validate=False,
                terminate_reason=TerminateReason.ROLLOUT_ERROR,
                failure_summary="   \n\n  ",
            )
        )

    body = _records(caplog)[0]
    assert "terminate_reason=rollout_error" in body
    assert "no underlying exception" in body
    # The point: the line has content, so it cannot be mistaken for a broken logger.
    assert body.splitlines()[-1].strip()


def test_a_cause_that_is_only_whitespace_never_reaches_the_log():
    assert _root_cause("   \n\t\n ", TerminateReason.CONTAINER_LOST).startswith("<")
    assert "container_lost" in _root_cause("  ", TerminateReason.CONTAINER_LOST)


def test_a_real_cause_is_passed_through_untouched():
    text = "Traceback (most recent call last):\n  File ...\nSandboxOomError: boom"
    assert _root_cause(text, TerminateReason.ROLLOUT_ERROR) == text


def test_a_cause_is_useful_even_without_a_terminate_reason():
    assert _root_cause(None, None) == ("<the worker captured no underlying exception; terminate_reason=<unset>>")
