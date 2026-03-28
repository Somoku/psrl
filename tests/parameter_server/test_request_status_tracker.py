# tests/parameter_server/test_request_status_tracker.py
import pytest
from psrl.workers.ps.request_status_tracker import (
    PSRL_RequestStatus,
)

pytestmark = pytest.mark.cpu_test


# ── Happy path: full lifecycle ────────────────────────────────────────────────


class TestRequestLifecycleHappyPath:
    def test_add_request_creates_pending_status(self, tracker):
        tracker.add_request(request_id=1)
        status = tracker.get_request_status(1)
        assert status == [PSRL_RequestStatus.PENDING]

    def test_update_to_rollout_routing(self, tracker):
        tracker.add_request(1)
        tracker.update_request_status(1, PSRL_RequestStatus.ROLLOUT_ROUTING)
        assert tracker.get_request_status(1) == [PSRL_RequestStatus.ROLLOUT_ROUTING]

    def test_full_happy_path_to_completed(self, tracker):
        """Walk PENDING → ROLLOUT_ROUTING → DISPATCHED → RUNNING → COMPLETED → REWARD → COMPLETED."""
        tracker.add_request(100)
        for status in [
            PSRL_RequestStatus.ROLLOUT_ROUTING,
            PSRL_RequestStatus.ROLLOUT_DISPATCHED,
            PSRL_RequestStatus.ROLLOUT_RUNNING,
            PSRL_RequestStatus.ROLLOUT_COMPLETED,
            PSRL_RequestStatus.REWARD_RUNNING,
            PSRL_RequestStatus.REWARD_COMPLETED,
            PSRL_RequestStatus.COMPLETED,
        ]:
            tracker.update_request_status(100, status)
        assert tracker.get_request_status(100) == [PSRL_RequestStatus.COMPLETED]

    def test_generic_running_state_reachable(self, tracker):
        tracker.add_request(2)
        tracker.update_request_status(2, PSRL_RequestStatus.RUNNING)
        assert tracker.get_request_status(2) == [PSRL_RequestStatus.RUNNING]


# ── Interruption paths ────────────────────────────────────────────────────────


class TestInterruptionPaths:
    def test_rollout_interrupted(self, tracker):
        tracker.add_request(10)
        tracker.update_request_status(10, PSRL_RequestStatus.ROLLOUT_RUNNING)
        tracker.update_request_status(10, PSRL_RequestStatus.ROLLOUT_INTERRUPTED)
        assert tracker.get_request_status(10) == [PSRL_RequestStatus.ROLLOUT_INTERRUPTED]

    def test_all_interrupt_states_covered(self, tracker):
        """ROLLOUT_INTERRUPTED is the single interruption state on this branch.

        Note: ROLLOUT_INTERRUPTED_BY_SCHEDULER was removed in base_agentic_rl;
        ROLLOUT_INTERRUPTED now covers both coordinator-driven and scheduler-driven
        interruptions.
        """
        tracker.add_request(11)
        tracker.update_request_status(11, PSRL_RequestStatus.ROLLOUT_RUNNING)
        tracker.update_request_status(11, PSRL_RequestStatus.ROLLOUT_INTERRUPTED)
        assert tracker.get_request_status(11) == [PSRL_RequestStatus.ROLLOUT_INTERRUPTED]
        # Verify ROLLOUT_INTERRUPTED_BY_SCHEDULER does not exist on this branch
        assert not hasattr(PSRL_RequestStatus, "ROLLOUT_INTERRUPTED_BY_SCHEDULER")


# ── Error / edge cases ────────────────────────────────────────────────────────


class TestErrorCases:
    def test_get_unknown_request_raises_key_error(self, tracker):
        with pytest.raises(KeyError):
            tracker.get_request_status(99999)

    def test_duplicate_add_request_overwrites(self, tracker):
        """add_request silently overwrites on duplicate — no exception raised.

        Note: update_request_status does NOT raise on any transition; it returns
        True (success) or False (aborted/stale). There is no invalid-transition
        guard in the implementation — any PSRL_RequestStatus value is accepted.
        """
        tracker.add_request(20, status=PSRL_RequestStatus.PENDING)
        # overwrite: status resets to PENDING (the new call's default)
        tracker.add_request(20, status=PSRL_RequestStatus.PENDING)
        assert tracker.get_request_status(20) == [PSRL_RequestStatus.PENDING]

    def test_update_returns_true_on_success(self, tracker):
        """update_request_status returns True (plain bool) for a single request_id."""
        tracker.add_request(30)
        result = tracker.update_request_status(30, PSRL_RequestStatus.ROLLOUT_ROUTING)
        # When a single int is passed, the implementation unwraps and returns a plain bool
        assert result is True

    def test_update_aborted_request_returns_false(self, tracker):
        """update_request_status returns False when a request has been marked for abortion."""
        tracker.add_request(40)
        tracker.update_request_status(40, PSRL_RequestStatus.ROLLOUT_RUNNING)
        tracker._abort_request_ids.add(40)  # simulate abort signal
        result = tracker.update_request_status(40, PSRL_RequestStatus.ROLLOUT_COMPLETED)
        assert result is False

    def test_cross_status_overwrite_leaves_stale_status_set(self, tracker):
        """Known limitation: add_request overwrite with a different status leaves stale
        entry in _status_to_request_ids. Documents the behaviour for future reference.
        """
        tracker.add_request(50, status=PSRL_RequestStatus.PENDING)
        # Overwrite with a different status
        tracker.add_request(50, status=PSRL_RequestStatus.ROLLOUT_RUNNING)
        # The primary map reflects the new status
        assert tracker.get_request_status(50) == [PSRL_RequestStatus.ROLLOUT_RUNNING]
        # Known limitation: PENDING set still contains 50 (stale due to missing cleanup in add_request)
        assert 50 in tracker._status_to_request_ids[PSRL_RequestStatus.PENDING]


# ── Multi-request independence ────────────────────────────────────────────────


class TestMultiRequestIndependence:
    def test_three_requests_tracked_independently(self, tracker):
        for rid in [1, 2, 3]:
            tracker.add_request(rid)
        tracker.update_request_status(1, PSRL_RequestStatus.ROLLOUT_ROUTING)
        tracker.update_request_status(2, PSRL_RequestStatus.ROLLOUT_DISPATCHED)
        # Request 3 stays PENDING
        assert tracker.get_request_status(1) == [PSRL_RequestStatus.ROLLOUT_ROUTING]
        assert tracker.get_request_status(2) == [PSRL_RequestStatus.ROLLOUT_DISPATCHED]
        assert tracker.get_request_status(3) == [PSRL_RequestStatus.PENDING]

    def test_bulk_add_request(self, tracker):
        """add_request accepts a list of IDs."""
        tracker.add_request([10, 11, 12])
        for rid in [10, 11, 12]:
            assert tracker.get_request_status(rid) == [PSRL_RequestStatus.PENDING]
