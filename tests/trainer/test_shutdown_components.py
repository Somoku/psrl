"""Teardown must run on every exit path, not only on a clean finish.

The agent loop workers own this job's sandboxes, so the only thing that can hand
them back is their own shutdown. A run that ended with an exception or a signal
used to skip it and leave containers holding capacity until their own lifetime
expired, which is long enough for the next run to inherit them as live clients.
"""

from __future__ import annotations

# The order the trainer stops them in. Every one owns something a later run would
# otherwise find still running.
_STEPS = (
    "stop_data_processor",
    "stop_agent_loop_manager",
    "stop_rollout_coordinator",
    "stop_reward_manager",
    "_stop_elastic_executor",
    "stop_ps_manager",
    "stop_sandbox_plane",
    "_shutdown_dump_executor",
)


class _Trainer:
    """Stands in for the trainer: each teardown step is recorded, never run."""

    _components_stopped = False
    port_scanner_handles: dict = {}

    def __init__(self, failing: str | None = None) -> None:
        self.ran: list[str] = []
        self.failing = failing

    def __getattr__(self, name: str):
        if name not in _STEPS:
            raise AttributeError(name)

        def step() -> None:
            self.ran.append(name)
            if name == self.failing:
                raise RuntimeError(f"{name} failed")

        return step


def _shutdown(**kwargs):
    """Borrow the real method onto the stub, the way the trainer calls it."""
    from psrl.trainer.ppo import ray_trainer

    trainer = _Trainer(**kwargs)
    ray_trainer.PSRL_RayPPOTrainer.shutdown_components.__get__(trainer)()
    return trainer


def test_every_component_is_stopped():
    assert _shutdown().ran == list(_STEPS)


def test_a_failing_step_does_not_strand_the_rest():
    # The reason to run the steps at all is that the ones after a failure still
    # run: stopping at the first error would skip exactly the releases that free
    # the sandboxes.
    trainer = _shutdown(failing="stop_rollout_coordinator")

    assert trainer.ran == list(_STEPS)


def test_it_runs_at_most_once():
    # A normal finish and the interrupt that follows it both call this, and the
    # second pass would touch actors the first one already shut down.
    from psrl.trainer.ppo import ray_trainer

    trainer = _Trainer()
    ray_trainer.PSRL_RayPPOTrainer.shutdown_components.__get__(trainer)()
    ray_trainer.PSRL_RayPPOTrainer.shutdown_components.__get__(trainer)()

    assert trainer.ran == list(_STEPS)
