"""Every coordinated sync gets a fresh epoch and names its participants by NIXL replica index."""

from __future__ import annotations

import pytest

from psrl.utils.common.worker_naming import gen_client_instance, gen_client_name
from psrl.workers.gen.rollout_coordination.coordinator import RolloutCoordinator

pytestmark = pytest.mark.cpu_test


def test_epochs_never_repeat_and_participants_are_replica_indices():
    coordinator = RolloutCoordinator.__new__(RolloutCoordinator)
    coordinator._sync_epoch = 0
    coordinator.replica_id_to_replica_idx = {"a": 3, "b": 0, "c": 7}
    first = coordinator._sync_round(["c", "a"])
    second = coordinator._sync_round(["b"])
    assert first == (1, [3, 7])
    assert second == (2, [0])


@pytest.mark.parametrize("instance,rank", [(0, 0), (12, 3), (105, 15)])
def test_gen_client_names_round_trip(instance, rank):
    assert gen_client_instance(gen_client_name(instance, rank)) == instance
