"""The timing contract derives every span and asserts every ordering in one place."""

import pytest
from psrl.sandbox.timing import DEFAULT_NODE_TTL_S, TimingContract


def test_one_declared_value_resolves_every_span():
    timing = TimingContract(episode_deadline_s=1800).validate()

    resolved = timing.as_dict()
    assert resolved, "a contract resolves at least one span"
    assert all(value > 0 for value in resolved.values())
    assert timing.pause_window_s == 3600
    assert timing.reap_window_s == 10800
    assert timing.lifetime_s == 43200


def test_the_idle_windows_stay_ordered_for_every_episode_length():
    # A window order that holds only for one deadline is an accident, so the
    # property is checked across the range an operator might configure.
    for deadline in (1, 10, 60, 600, 1800, 7200, 86400):
        timing = TimingContract(episode_deadline_s=deadline).validate()
        assert timing.pause_window_s < timing.reap_window_s < timing.lifetime_s


def test_liveness_spans_stay_ordered_for_every_node_ttl():
    for ttl in (5, 30, 120, 600, 3600):
        timing = TimingContract(episode_deadline_s=1800, node_ttl_s=ttl).validate()
        assert timing.reservation_renew_interval_s < timing.reservation_ttl_s < timing.node_ttl_s
        assert timing.load_report_interval_s * 3 < timing.node_ttl_s


def test_crash_recovery_completes_inside_the_capacity_lease():
    timing = TimingContract(episode_deadline_s=1800).validate()

    assert timing.lifecycle_lease_ttl_s + timing.lifecycle_gc_interval_s < timing.capacity_lease_ttl_s


def test_a_paused_sandbox_keeps_the_reservation_holding_its_memory():
    timing = TimingContract(episode_deadline_s=1800).validate()

    assert timing.capacity_lease_ttl_s >= timing.pause_window_s


def test_the_sweep_is_faster_than_the_shortest_span_it_enforces():
    timing = TimingContract(episode_deadline_s=1800).validate()

    assert timing.sweep_interval_s < min(timing.pause_window_s, timing.reservation_ttl_s, timing.capacity_lease_ttl_s)


def test_an_override_replaces_a_derived_span():
    timing = TimingContract(episode_deadline_s=60).with_overrides(pause_window_s=30, reap_window_s=120)

    assert timing.validate().pause_window_s == 30
    assert timing.reap_window_s == 120
    # Still derived from the override rather than from the deadline.
    assert timing.lifetime_s == 480


def test_an_override_that_inverts_an_ordering_is_refused():
    timing = TimingContract(episode_deadline_s=1800, overrides={"reap_window_s": 60})

    with pytest.raises(ValueError, match="shorter than the reap window"):
        timing.validate()


def test_an_override_naming_no_derived_span_is_refused():
    # A typo in a deadline is invisible until a run leaks, so it fails at startup.
    with pytest.raises(ValueError, match="name no derived value"):
        TimingContract(episode_deadline_s=1800, overrides={"reap_windows": 60})


def test_a_declared_value_must_be_positive():
    for field in ("episode_deadline_s", "node_ttl_s", "rpc_timeout_s"):
        with pytest.raises(ValueError, match=field):
            TimingContract(**{"episode_deadline_s": 1800, field: 0})


def test_a_mapping_resolves_declared_values_and_overrides():
    timing = TimingContract.from_value({"episode_deadline_s": 900, "node_ttl_s": 60})

    assert timing.episode_deadline_s == 900
    assert timing.node_ttl_s == 60
    assert timing.rpc_timeout_s > 0


def test_a_derived_name_at_the_top_level_reads_as_an_override():
    # An operator writes the span, not the nesting, so a flat key is honoured.
    timing = TimingContract.from_value({"episode_deadline_s": 1800, "pause_window_s": 120})

    assert timing.pause_window_s == 120


def test_an_unknown_timing_key_is_refused_rather_than_dropped():
    with pytest.raises(ValueError, match="unknown key"):
        TimingContract.from_value({"episode_deadline_s": 1800, "pause_windows": 120})


def test_a_typed_contract_passes_through_from_value():
    timing = TimingContract(episode_deadline_s=300)

    assert TimingContract.from_value(timing) is timing


def test_a_contract_without_an_episode_deadline_is_refused():
    with pytest.raises(ValueError, match="episode_deadline_s"):
        TimingContract.from_value(None)


def test_the_node_ttl_default_is_used_when_a_deployment_omits_it():
    assert TimingContract(episode_deadline_s=1800).node_ttl_s == DEFAULT_NODE_TTL_S
