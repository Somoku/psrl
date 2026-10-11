"""The steady-state switch-back must not re-read storage (items D4, T3).

Colocate eval parks the trainer, runs rollout on its GPUs, then switches back. Where the
weights come from on that switch-back decides whether training continues or silently
restarts from the last save:

* **first arrival** (resuming a run) -- the checkpoint is the only source, and the
  existing code defers it until NIXL is reconnected;
* **steady state** (every eval after the first) -- the weights must be restored, because
  the checkpoint is older than the current version by however many steps have run.

The second case fails *silently*: the loaded weights are valid, merely stale, so nothing
raises and the loss curve simply flattens. There is no runtime signal, which is why the
guard is an assertion at the branch rather than a test of the outcome.

These tests read `ray_trainer.py` rather than driving it: the switch path needs a live
cluster with vLLM instances, and the property under test is a control-flow invariant that
source inspection pins exactly. The cluster-level check belongs to the colocate e2e
configuration (T3), which these tests complement rather than replace.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_test


def _trainer_source() -> str:
    """Read `ray_trainer.py` from disk rather than importing it.

    Importing pulls in `verl.trainer.ppo.metric_utils`, which fails in this environment
    on a pre-existing version skew (`RolloutMoELoadBalanceMetricsAccumulator`). The
    property under test is a control-flow invariant in the source, so reading the file
    tests exactly what importing would -- and keeps an unrelated dependency break from
    masking a real regression here.
    """
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    return (repo_root / "psrl" / "trainer" / "ppo" / "ray_trainer.py").read_text()


class TestTheMarkerIsOneWay:
    def test_the_marker_exists_and_starts_false(self):
        source = _trainer_source()

        assert "self._trainer_has_held_real_weights = False" in source

    def test_the_marker_is_not_derived_from_the_colocate_mode(self):
        """`is_rollout_mode_in_actor` flips every switch, so it cannot carry this.

        Deriving the marker from it would make every switch-back look like a first
        arrival -- which is exactly the silent rollback this guards.
        """
        source = _trainer_source()
        assignments = re.findall(
            r"self\._trainer_has_held_real_weights\s*=\s*(.+)", source
        )

        assert assignments, "the marker is never assigned"
        for value in assignments:
            assert "is_rollout_mode_in_actor" not in value, (
                f"the marker is derived from the colocate mode ({value.strip()}), which "
                f"flips on every switch"
            )

    def test_the_marker_is_only_ever_set_true_after_the_initial_false(self):
        """One-way: nothing clears it, or a later switch-back becomes a first arrival."""
        source = _trainer_source()
        assignments = [
            value.strip().rstrip(")").strip()
            for value in re.findall(r"self\._trainer_has_held_real_weights\s*=\s*(\w+)", source)
        ]

        assert assignments.count("False") == 1, (
            f"expected exactly one initialisation to False, found {assignments}"
        )
        assert all(value == "True" for value in assignments[1:]), (
            f"the marker is cleared somewhere: {assignments}"
        )


class TestSwitchBackGuardsTheCheckpointRead:
    def test_an_assertion_guards_the_deferred_load(self):
        source = _trainer_source()

        assert "_trainer_has_held_real_weights" in source
        guard = source[source.index("deferred_path = getattr") :]
        assert "assert not (" in guard[:600], (
            "the deferred checkpoint load is not guarded; a steady-state switch-back "
            "could read storage"
        )

    def test_the_guard_names_the_consequence_not_just_the_condition(self):
        """An assertion message that only restates the condition is unactionable here.

        The failure is silent in production, so whoever hits this needs to know what it
        would have cost, not merely which flag was set.
        """
        source = _trainer_source()
        start = source.index("assert not (deferred_path is not None")
        message = source[start : start + 400]

        assert "discard" in message or "roll" in message

    def test_the_marker_is_set_after_the_weights_are_back(self):
        """Set on the way out, so a failure mid-switch leaves it a first arrival still."""
        source = _trainer_source()
        switch = source[source.index("def switch_to_trainer_mode") :]
        switch = switch[: switch.index("def _make_broadcast_plan")]

        pull_at = switch.index('execute_all_async("pull_model")')
        marker_at = switch.index("self._trainer_has_held_real_weights = True")

        assert pull_at < marker_at, (
            "the marker is set before the weights are pulled; an exception in between "
            "would leave the trainer marked as holding weights it does not have"
        )


class TestDeferredLoadIsResumeOnly:
    def test_the_deferred_path_is_set_only_while_resuming(self):
        """It must not be reachable from the eval switch, or the guard would fire legitimately."""
        source = _trainer_source()
        setters = [
            line
            for line in source.splitlines()
            if "self._deferred_actor_ckpt_path = " in line and "None" not in line
        ]

        assert len(setters) == 1, f"expected one setter, found {setters}"

    def test_the_deferred_path_is_cleared_after_use(self):
        """Leaving it set would make the *next* switch-back read storage too."""
        source = _trainer_source()

        assert "self._deferred_actor_ckpt_path = None" in source
