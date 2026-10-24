"""The status tracker primitive a member-level refill depends on.

A member refill keeps the failed child's id, so the manager must be able to tell
"this attempt failed and its worker aborted it" from "a refill is already in
flight", then release the abort record and dispatch the same id again.
"""

from omegaconf import OmegaConf
from psrl.workers.ps.request_status_tracker import PSRL_RequestStatus, RequestStatusTracker


def _tracker() -> RequestStatusTracker:
    return RequestStatusTracker(
        OmegaConf.create(
            {
                "rollout_n": 4,
                "val_rollout_n": 1,
                "logging_path": "/tmp",
                "rollout_coordination": {"redundant_rollout": {"enable": False}},
            }
        )
    )


def test_a_member_can_be_readmitted_after_its_worker_aborted_it() -> None:
    tracker = _tracker()
    tracker.add_request([14])

    assert tracker.is_request_aborted(14) is False
    tracker._abort_requests([14])
    assert tracker.is_request_aborted(14) is True

    tracker.readmit_requests([14])

    assert tracker.is_request_aborted(14) is False
    assert tracker.update_request_status([14], PSRL_RequestStatus.RUNNING) is True
