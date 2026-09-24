import pytest
import yaml
from examples.airs_bench.config import build_runtime_config
from psrl.environments.mlgym_env import (
    assert_default_history_processor,
    build_sandbox_spec,
)


@pytest.mark.cpu_test
def test_runtime_config_defaults_are_sane():
    config = build_runtime_config(None)

    assert config.image, "A default sandbox image must be set."
    assert config.per_action_timeout_s >= 3600.0, "Per-action timeout must cover MLGym's 3600s training_timeout."
    assert config.max_observation_chars > 0, "Observation truncation must be on by default."


@pytest.mark.cpu_test
def test_runtime_config_merges_overrides():
    config = build_runtime_config({"sandbox_memory": "64g", "sandbox_cpus": 16.0})

    assert config.sandbox_memory == "64g"
    assert config.sandbox_cpus == 16.0


@pytest.mark.cpu_test
def test_build_sandbox_spec_mounts_prepared_task_data_read_only():
    config = build_runtime_config({"sandbox_memory": "32g"})
    spec = build_sandbox_spec(
        config,
        dataset_data_path="/shared/airs_prepared/GraphRegressionZincMae",
        labels={"psrl.airs_task_id": "t1"},
        episode_id="ep-1",
    )

    assert spec.resources.memory_mb == 32768
    assert spec.resources.cpu_count == config.sandbox_cpus
    assert spec.resources.gpu_count is None, "AIRS-Bench tasks are CPU graded."
    assert spec.mounts[0].read_only, "Task data must be mounted read only so labels cannot be edited."
    assert "airs_prepared" in spec.mounts[0].source
    assert spec.metadata["psrl.airs_task_id"] == "t1"


@pytest.mark.cpu_test
def test_build_sandbox_spec_scopes_the_lease_to_one_episode():
    """One sandbox per episode, so the episode is the phase the reservation protects."""
    config = build_runtime_config(None)
    spec = build_sandbox_spec(
        config,
        dataset_data_path="/shared/airs_prepared/SomeTask",
        labels={},
        episode_id="ep-7",
    )

    assert spec.workflow_id == "ep-7"
    assert spec.idempotency_key == "ep-7:airs"
    assert spec.policy_profile == "airs_bench"
    assert spec.lifetime_timeout_s == config.episode_timeout_s


@pytest.mark.cpu_test
def test_assert_default_history_processor_accepts_the_correct_value(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump({"history_processor": "DefaultHistoryProcessor"}))

    assert_default_history_processor(str(path))  # must not raise


@pytest.mark.cpu_test
def test_assert_default_history_processor_rejects_last5(tmp_path):
    """Last5Observations rewrites history in place, which corrupts TITO data."""
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump({"history_processor": "Last5Observations"}))

    with pytest.raises(ValueError, match="DefaultHistoryProcessor"):
        assert_default_history_processor(str(path))


@pytest.mark.cpu_test
def test_assert_default_history_processor_rejects_a_missing_key(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump({"system_template": "hello"}))

    with pytest.raises(ValueError, match="history_processor"):
        assert_default_history_processor(str(path))


@pytest.mark.cpu_test
def test_shipped_agent_config_pins_default_history_processor():
    """Guard the real config file, not just the checker."""
    assert_default_history_processor("examples/airs_bench/agent_config.yaml")
