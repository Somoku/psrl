"""Train-step boundary and version publication (implementation item C3).

The design asks publication to be driven by a train-step-end callback so it lands once
the weights are in place and the NIC is idle. The callback it names --
Megatron-Bridge's `on_train_step_end` -- is real but unreachable from PSRL, which uses
Bridge for model conversion only and runs its optimizer step through verl. The boundary
is therefore defined on PSRL's own loop, and these tests pin the properties the design's
criterion actually depends on.
"""

from __future__ import annotations

import pytest

from psrl.workers.train.step_boundary import StepBoundary, StepBoundaryRecorder

pytestmark = pytest.mark.cpu_test


class TestBoundaryRecording:
    def test_a_boundary_is_recorded_per_step(self):
        recorder = StepBoundaryRecorder()
        for step in range(3):
            recorder.on_train_step_end(step=step, version=step + 1)

        assert [b.step for b in recorder.history] == [0, 1, 2]
        assert recorder.versions() == [1, 2, 3]

    def test_boundaries_are_ordered_in_time(self):
        recorder = StepBoundaryRecorder()
        first = recorder.on_train_step_end(step=0)
        second = recorder.on_train_step_end(step=1)

        assert first.ordered_before(second)

    def test_history_is_bounded(self):
        """This lives in a long-running trainer; an unbounded list is a slow leak."""
        recorder = StepBoundaryRecorder(capacity=4)
        for step in range(50):
            recorder.on_train_step_end(step=step)

        assert len(recorder.history) == 4
        assert [b.step for b in recorder.history] == [46, 47, 48, 49]

    def test_last_reports_the_most_recent_boundary(self):
        recorder = StepBoundaryRecorder()
        assert recorder.last is None
        recorder.on_train_step_end(step=7, version=8)
        assert recorder.last.step == 7


class TestHooks:
    def test_hooks_receive_the_boundary(self):
        recorder = StepBoundaryRecorder()
        seen: list[StepBoundary] = []
        recorder.register(seen.append)

        boundary = recorder.on_train_step_end(step=1, version=2)

        assert seen == [boundary]

    def test_every_hook_runs_in_registration_order(self):
        recorder = StepBoundaryRecorder()
        order: list[str] = []
        recorder.register(lambda _b: order.append("first"))
        recorder.register(lambda _b: order.append("second"))

        recorder.on_train_step_end()

        assert order == ["first", "second"]

    def test_a_failing_hook_does_not_take_down_the_step(self):
        """Publication observability must not be able to fail the training step."""
        recorder = StepBoundaryRecorder()
        ran: list[str] = []

        def explode(_boundary):
            raise RuntimeError("hook failed")

        recorder.register(explode)
        recorder.register(lambda _b: ran.append("later hook still ran"))

        recorder.on_train_step_end(step=1)

        assert ran == ["later hook still ran"]
        assert len(recorder.hook_errors) == 1
        assert isinstance(recorder.hook_errors[0], RuntimeError)


class TestVersionsMustAdvance:
    """Two weight states under one version is undetectable by readers."""

    def test_strictly_increasing_versions_pass(self):
        recorder = StepBoundaryRecorder()
        for version in (1, 2, 3):
            recorder.on_train_step_end(version=version)

        recorder.assert_versions_advance()

    def test_a_repeated_version_is_rejected(self):
        recorder = StepBoundaryRecorder()
        recorder.on_train_step_end(version=4)
        recorder.on_train_step_end(version=4)

        with pytest.raises(AssertionError, match="strictly increase"):
            recorder.assert_versions_advance()

    def test_a_regressing_version_is_rejected(self):
        recorder = StepBoundaryRecorder()
        recorder.on_train_step_end(version=4)
        recorder.on_train_step_end(version=3)

        with pytest.raises(AssertionError, match="strictly increase"):
            recorder.assert_versions_advance()

    def test_boundaries_without_a_version_are_skipped(self):
        """Not every boundary publishes; those must not look like a regression."""
        recorder = StepBoundaryRecorder()
        recorder.on_train_step_end(version=1)
        recorder.on_train_step_end(version=None)
        recorder.on_train_step_end(version=2)

        assert recorder.versions() == [1, 2]
        recorder.assert_versions_advance()


class TestPublicationPointIsWired:
    """The boundary must be recorded where publication happens, not merely exist."""

    def test_nixl_push_records_the_published_version(self):
        """Pin the call site: a refactor that drops it leaves the trace silently empty."""
        from pathlib import Path

        import psrl.workers.train as train_pkg

        source = (Path(train_pkg.__file__).parent / "base_train_worker.py").read_text()
        assert "self.step_boundaries.on_train_step_end(version=next_ps_model_version)" in source, (
            "the NIXL publish path no longer records a step boundary"
        )

    def test_ray_push_records_the_published_version(self):
        from pathlib import Path

        import psrl.workers.train as train_pkg

        source = (Path(train_pkg.__file__).parent / "engine_train_worker.py").read_text()
        assert "self.step_boundaries.on_train_step_end(version=next_version)" in source, (
            "the Ray publish path no longer records a step boundary"
        )
