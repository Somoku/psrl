# tests/conftest.py
import pytest
from psrl.workers.gen_dplb.utils import RolloutInstanceId  # = tuple[str, int]
from psrl.workers.ps.staleness_controller import EntryInfo

# ── Ray cluster fixtures (used by integration tests only) ────────────────────


@pytest.fixture(scope="session")
def ray_cluster():
    """Start a single-node Ray cluster once for the whole test session."""
    import ray

    already_running = ray.is_initialized()
    if not already_running:
        ray.init(
            num_cpus=4,
            ignore_reinit_error=True,
            include_dashboard=False,
            log_to_driver=False,
        )
    yield
    if not already_running:
        ray.shutdown()


@pytest.fixture(scope="function")
def ray_cluster_fn(ray_cluster):
    """Function-scoped Ray fixture — reuses session cluster, kills named actors after each test."""
    import warnings

    import ray

    yield
    for actor_info in ray.util.list_named_actors(all_namespaces=True):
        try:
            ray.kill(ray.get_actor(actor_info["name"], namespace=actor_info.get("namespace")))
        except Exception as e:
            warnings.warn(f"Failed to kill Ray actor {actor_info.get('name', '?')}: {e}", stacklevel=2)


# ── Shared data helpers (cpu_test safe) ─────────────────────────────────────


@pytest.fixture
def dummy_rollout_instance_id() -> RolloutInstanceId:
    """RolloutInstanceId is tuple[str, int] — a type alias, NOT a class."""
    return ("worker", 0)


@pytest.fixture
def dummy_entry_info(dummy_rollout_instance_id) -> EntryInfo:
    return EntryInfo(
        rollout_instance_id=dummy_rollout_instance_id,
        prompt_id=42,
        request_idx=0,
        model_version=1,
    )
