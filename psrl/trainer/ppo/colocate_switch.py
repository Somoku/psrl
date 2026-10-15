"""The order of a colocate switch, as data (design §3.7.3).

A colocate switch hands the same GPUs from the trainer to the validate instances. With a
parameter server the order barely matters: the validate instances read the PS, so the
trainer can leave first. **Under direct supply the trainer is the only copy of the
weights, so it must still be serving when the validate instances read.** The old order --
trainer sleeps, then validate wakes and reads -- read a source that had already gone, and
the sleep itself discarded the weights.

The direct-supply order follows verl's `update_weights` (transfer, then offload, then KV
cache), with one refinement it does not need: the validate instance wakes its *weights*
first and its KV cache only after the trainer has left. vLLM sizes the KV cache against
the whole card at init, so waking it while the trainer is still resident would ask for
memory that is not free.

Kept as a pure function so the order is testable without a cluster: each step is a name
the trainer dispatches on, and a test can assert, for example, that no step reads the
trainer after the step that offloads it.
"""

from __future__ import annotations

from enum import Enum


class Step(str, Enum):
    #: Trainer deregisters and moves off the GPU (weights preserved iff direct supply).
    TRAINER_OFFLOAD = "trainer_offload"
    #: Validate instances allocate weights (and KV cache when `VALIDATE_WAKE_ALL`).
    VALIDATE_WAKE_WEIGHTS = "validate_wake_weights"
    VALIDATE_WAKE_ALL = "validate_wake_all"
    #: Validate instances register their new addresses and the meta server rebroadcasts.
    VALIDATE_PUBLISH = "validate_publish"
    #: Validate instances pull the current version from whoever holds it.
    VALIDATE_PULL = "validate_pull"
    #: Validate instances allocate their KV cache.
    VALIDATE_WAKE_KV = "validate_wake_kv"
    #: Gateway starts routing to the validate instances.
    VALIDATE_RESUME = "validate_resume"


#: Steps that read the trainer's weights when it is the source.
_READS_TRAINER = frozenset({Step.VALIDATE_PULL})


def to_rollout_steps(*, has_ps_data_plane: bool) -> list[Step]:
    """Ordered steps for trainer -> rollout.

    With a parameter server: free the GPU first, then wake everything and pull from the
    PS -- the existing behaviour, kept because the validate instance gets its full KV
    cache budget from the start.

    Under direct supply: wake only weights, pull from the still-resident trainer, *then*
    offload it, *then* allocate the KV cache. The pull must also have **completed**, not
    merely been issued, before the offload -- the trainer's memory is deregistered there.
    """
    if has_ps_data_plane:
        return [
            Step.TRAINER_OFFLOAD,
            Step.VALIDATE_WAKE_ALL,
            Step.VALIDATE_PUBLISH,
            Step.VALIDATE_PULL,
            Step.VALIDATE_RESUME,
        ]
    return [
        Step.VALIDATE_WAKE_WEIGHTS,
        Step.VALIDATE_PUBLISH,
        Step.VALIDATE_PULL,
        Step.TRAINER_OFFLOAD,
        Step.VALIDATE_WAKE_KV,
        Step.VALIDATE_RESUME,
    ]


def check_order(steps: list[Step], *, trainer_is_source: bool) -> None:
    """Raise if ``steps`` would read the trainer after it left, or serve without a cache.

    The invariants, stated once so the plan and any future edit are checked against the
    same rules rather than against a remembered sequence.
    """
    seen: set[Step] = set()
    for step in steps:
        if trainer_is_source and step in _READS_TRAINER and Step.TRAINER_OFFLOAD in seen:
            raise ValueError(f"{step.value} reads the trainer after {Step.TRAINER_OFFLOAD.value}")
        if step is Step.VALIDATE_PULL and not ({Step.VALIDATE_WAKE_WEIGHTS, Step.VALIDATE_WAKE_ALL} & seen):
            raise ValueError("validate pulls into weights it has not allocated")
        if step is Step.VALIDATE_PULL and Step.VALIDATE_PUBLISH not in seen:
            raise ValueError("validate pulls before its new addresses are published")
        if step is Step.VALIDATE_WAKE_KV and Step.TRAINER_OFFLOAD not in seen:
            raise ValueError("KV cache allocated while the trainer still holds the GPU")
        if step is Step.VALIDATE_RESUME:
            has_kv = Step.VALIDATE_WAKE_ALL in seen or Step.VALIDATE_WAKE_KV in seen
            if not has_kv or Step.VALIDATE_PULL not in seen:
                raise ValueError("validate resumes serving before it has weights and a KV cache")
        seen.add(step)
    if Step.TRAINER_OFFLOAD not in seen:
        raise ValueError("the trainer never leaves the GPU")
