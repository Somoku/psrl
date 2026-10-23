"""The PSManager abort scope a member-level refill depends on.

Aborting a single failed child with the default group scope clears the group's
buffer reservation. A refill then has no entry to rejoin, while the buffer that
already counted the group still expects the missing payload. The worker must
abort the child without the group scope; only the manager's exhausted-member
fallback aborts the whole group.
"""

import threading
from types import SimpleNamespace

from psrl.workers.ps.ps_manager import PSManager
from psrl.workers.ps.request_status_tracker import PSRL_RequestStatus


class _Entry:
    def __init__(self) -> None:
        self.prompt_id = 3
        self.request_idx = [0, 1, 2, 3]
        self.rollout_instance_id = [(-1, -1)] * 4
        self.model_version = [0] * 4
        self.n_trajectory = [3] * 4

    def get_all_request_relative_ids(self) -> list[int]:
        return list(self.request_idx)


class _Inventory:
    def __init__(self) -> None:
        self.entry = _Entry()
        self.data_tracker = {3: (0, 5)}
        self.buffers = {0: SimpleNamespace(entries={5: SimpleNamespace(entry_info=self.entry)})}
        self.cleared: list[int] = []

    def clear_reserved_entries(self, prompt_ids: list[int]) -> None:
        self.cleared.extend(prompt_ids)


def _manager() -> PSManager:
    manager = PSManager.__new__(PSManager)
    manager._state_lock = threading.RLock()
    manager.rollout_n = 4
    manager.alg_rollout_n = 4
    manager.staleness_inventory = _Inventory()
    manager._abort_request_ids = set()
    manager._request_id_to_status = {
        uid: PSRL_RequestStatus.RUNNING for uid in range(12, 16)
    }
    return manager


def test_a_member_abort_keeps_the_group_reservation() -> None:
    manager = _manager()
    inventory = manager.staleness_inventory

    manager.abort_requests([15], abort_group=False)

    assert inventory.cleared == []
    assert inventory.data_tracker == {3: (0, 5)}
    assert inventory.entry.request_idx == [0, 1, 2]
    assert inventory.entry.n_trajectory == [3, 3, 3]
    assert manager._abort_request_ids == {15}


def test_the_exhausted_member_fallback_aborts_the_whole_group() -> None:
    manager = _manager()
    inventory = manager.staleness_inventory

    manager.abort_requests([15], abort_group=True)

    assert inventory.cleared == [3]
    assert manager._abort_request_ids == {12, 13, 14, 15}
