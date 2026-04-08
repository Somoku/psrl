"""CPU-only structural tests for PSRL_EngineTrainWorker."""

import pytest

pytestmark = pytest.mark.cpu_test


def test_engine_train_worker_is_importable():
    """The module must import without GPU or Ray initialised."""
    from psrl.workers.train.engine_train_worker import PSRL_EngineTrainWorker

    assert PSRL_EngineTrainWorker is not None


def test_engine_train_worker_inherits_both_bases():
    from psrl.workers.train.base_train_worker import PSRL_BaseTrainWorker
    from psrl.workers.train.engine_train_worker import PSRL_EngineTrainWorker
    from verl.workers.engine_workers import ActorRolloutRefWorker

    assert issubclass(PSRL_EngineTrainWorker, ActorRolloutRefWorker)
    assert issubclass(PSRL_EngineTrainWorker, PSRL_BaseTrainWorker)


def test_engine_train_worker_has_required_methods():
    from psrl.workers.train.engine_train_worker import PSRL_EngineTrainWorker

    for method in [
        "init_model",
        "compute_log_prob",
        "update_actor",
        "init_nixl_client",
        "nixl_convert_params",
        "nixl_protocol",
        "nixl_sleep",
        "nixl_wake_up",
        "sleep_model",
        "wake_up_model",
        "ray_push_model",
        "get_replica_id",
        "is_train_representative_rank",
    ]:
        assert hasattr(PSRL_EngineTrainWorker, method), f"Missing: {method}"


def test_engine_train_worker_importable_from_package():
    from psrl.workers.train import PSRL_EngineTrainWorker

    assert PSRL_EngineTrainWorker is not None
