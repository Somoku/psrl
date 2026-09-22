from __future__ import annotations

import asyncio
import time

import pytest
from psrl.sandbox.capacity import (
    ResourceQuantity,
    SandboxCapacityConfig,
    SandboxCapacityCoordinator,
    parse_memory_mb,
    resolve_sandbox_capacity,
)
from psrl.sandbox.core import ResourceSpec, SandboxCapacityTimeout


@pytest.mark.parametrize(
    ("value", "expected"),
    [("8g", 8192), ("512m", 512), (1073741824, 1024), (None, None)],
)
def test_parse_memory_mb(value, expected) -> None:
    assert parse_memory_mb(value) == expected


def test_resource_quantity_uses_actual_container_request() -> None:
    quantity = ResourceQuantity.from_spec(ResourceSpec(cpu_count=1.25, memory_mb=3072))

    assert quantity == ResourceQuantity(memory_mb=3072, cpu_millis=1250)


def test_resolve_capacity_has_one_shared_utilization_margin() -> None:
    resolved = resolve_sandbox_capacity(
        SandboxCapacityConfig(utilization=0.75),
        detected_memory_mb=10000,
        detected_cpu_cores=8,
    )

    assert resolved.resources == ResourceQuantity(memory_mb=7500, cpu_millis=6000)
    assert resolved.memory_source == "detected"
    assert resolved.cpu_source == "detected"


def test_explicit_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError, match="memory_mb"):
        SandboxCapacityConfig(memory_mb=0)
    with pytest.raises(ValueError, match="cpu_cores"):
        SandboxCapacityConfig(cpu_cores=0)


def test_lease_max_age_must_outlive_the_lease_ttl() -> None:
    """The age cap may not reclaim a lease the heartbeat would still have renewed."""

    with pytest.raises(ValueError, match="lease_max_age_s"):
        SandboxCapacityConfig(lease_ttl_s=180, heartbeat_interval_s=30, lease_max_age_s=180)

    # Null disables the age cap, which is a deliberate choice rather than a contradiction.
    assert SandboxCapacityConfig(lease_ttl_s=180, heartbeat_interval_s=30, lease_max_age_s=None).lease_max_age_s is None


@pytest.mark.asyncio
async def test_multi_resource_admission_blocks_until_complete_request_fits() -> None:
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(memory_mb=100, cpu_cores=4, utilization=1, lease_ttl_s=10, heartbeat_interval_s=1)
    )
    await coordinator.acquire("large", "worker-a", 80, 1)
    waiting = asyncio.create_task(coordinator.acquire("cpu-heavy", "worker-b", 10, 4))
    await asyncio.sleep(0)

    assert not waiting.done()
    snapshot = await coordinator.snapshot()
    assert snapshot["available_capacity"] == {"memory_mb": 20, "cpu_millis": 3000}
    assert snapshot["waiters"] == 1

    await coordinator.release("large")
    await asyncio.wait_for(waiting, timeout=1)
    assert (await coordinator.snapshot())["allocations"] == 1
    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_different_request_sizes_share_capacity_without_static_pools() -> None:
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(memory_mb=100, cpu_cores=10, utilization=1, lease_ttl_s=10, heartbeat_interval_s=1)
    )
    requests = [("rollout-a", 20, 2), ("grader", 50, 4), ("rollout-b", 30, 3)]

    await asyncio.gather(*(coordinator.acquire(lease_id, "worker", memory, cpu) for lease_id, memory, cpu in requests))

    snapshot = await coordinator.snapshot()
    assert snapshot["available_capacity"] == {"memory_mb": 0, "cpu_millis": 1000}
    assert snapshot["allocations"] == 3
    await coordinator.release_owner("worker")
    assert (await coordinator.snapshot())["available_capacity"] == {"memory_mb": 100, "cpu_millis": 10000}
    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_grader_class_is_admitted_while_rollouts_wait_to_borrow() -> None:
    """The failing run's shape: a large class must not wait behind smaller requests.

    Rollouts filled the whole envelope by borrowing the elastic pool, and the freed
    slots kept going to more rollouts, so a 30g grader could never fit its 16g
    fragments. A class guarantee makes the grader's slice reclaimable.
    """
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(
            memory_mb=1000,
            cpu_cores=10,
            utilization=1,
            lease_ttl_s=10,
            heartbeat_interval_s=1,
            classes={"rollout": {"guaranteed_share": 0.4}, "grader": {"guaranteed_share": 0.3}},
        )
    )
    admitted_rollouts = [
        asyncio.create_task(coordinator.acquire(f"rollout-{index}", "owner", 100, 1, "rollout"))
        for index in range(10)
    ]
    await asyncio.gather(*admitted_rollouts)
    extra_rollouts = [
        asyncio.create_task(coordinator.acquire(f"waiting-{index}", "owner", 100, 1, "rollout"))
        for index in range(4)
    ]
    grader = asyncio.create_task(coordinator.acquire("grader", "owner", 300, 1, "grader"))
    await asyncio.sleep(0)

    # Rollouts hold the elastic pool, so the grader waits for the envelope to free up,
    # but the freed capacity is reserved for it instead of being borrowed again.
    for index in range(3):
        assert not grader.done()
        await coordinator.release(f"rollout-{index}")
        await asyncio.sleep(0)
        assert not any(task.done() for task in extra_rollouts)

    await asyncio.wait_for(grader, timeout=1)
    snapshot = await coordinator.snapshot()
    assert snapshot["per_class"]["grader"]["used"] == {"memory_mb": 300, "cpu_millis": 1000}

    await coordinator.release_owner("owner")
    for task in extra_rollouts:
        task.cancel()
    await asyncio.gather(*extra_rollouts, return_exceptions=True)
    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_class_guarantees_are_satisfiable_together() -> None:
    """Both classes fit at once, whatever order their requests arrive in.

    Guarantees leave headroom precisely so a burst of one class cannot make the other
    class's share impossible to satisfy without preempting a running sandbox.
    """
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(
            memory_mb=1000,
            cpu_cores=10,
            utilization=1,
            lease_ttl_s=10,
            heartbeat_interval_s=1,
            classes={"rollout": {"guaranteed_share": 0.5}, "grader": {"guaranteed_share": 0.3}},
        )
    )
    requests = [
        coordinator.acquire("grader", "owner", 300, 1, "grader"),
        *(coordinator.acquire(f"rollout-{index}", "owner", 100, 1, "rollout") for index in range(5)),
    ]

    await asyncio.wait_for(asyncio.gather(*requests), timeout=0.5)

    snapshot = await coordinator.snapshot()
    assert snapshot["per_class"]["rollout"]["used"] == {"memory_mb": 500, "cpu_millis": 5000}
    assert snapshot["per_class"]["grader"]["used"] == {"memory_mb": 300, "cpu_millis": 1000}
    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_elastic_pool_is_borrowable_while_no_other_class_waits() -> None:
    """Guarantees are floors, not partitions: spare capacity still gets used."""
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(
            memory_mb=1000,
            cpu_cores=10,
            utilization=1,
            lease_ttl_s=10,
            heartbeat_interval_s=1,
            classes={"rollout": {"guaranteed_share": 0.4}, "grader": {"guaranteed_share": 0.3}},
        )
    )

    await asyncio.gather(*(coordinator.acquire(f"rollout-{index}", "owner", 100, 1, "rollout") for index in range(9)))

    snapshot = await coordinator.snapshot()
    assert snapshot["available_capacity"] == {"memory_mb": 100, "cpu_millis": 1000}
    assert snapshot["per_class"]["rollout"]["used"] == {"memory_mb": 900, "cpu_millis": 9000}
    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_class_ceiling_bounds_borrowing() -> None:
    """A class may not take the pool beyond its ceiling even when nothing else waits."""
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(
            memory_mb=1000,
            cpu_cores=10,
            utilization=1,
            lease_ttl_s=10,
            heartbeat_interval_s=1,
            classes={"rollout": {"guaranteed_share": 0.1, "max_share": 0.3}},
        )
    )

    for index in range(3):
        await coordinator.acquire(f"rollout-{index}", "owner", 100, 1, "rollout")
    with pytest.raises(SandboxCapacityTimeout):
        await coordinator.acquire("over-ceiling", "owner", 100, 1, "rollout", timeout_s=0.05)

    snapshot = await coordinator.snapshot()
    assert snapshot["per_class"]["rollout"]["used"] == {"memory_mb": 300, "cpu_millis": 3000}
    assert snapshot["per_class"]["rollout"]["ceiling"] == {"memory_mb": 300, "cpu_millis": 3000}
    await coordinator.shutdown()


def test_class_guarantees_cannot_exceed_the_envelope() -> None:
    with pytest.raises(ValueError, match="exceeds the node envelope"):
        SandboxCapacityConfig(
            classes={"rollout": {"guaranteed_share": 0.7}, "grader": {"guaranteed_share": 0.4}},
        )
    with pytest.raises(ValueError, match="guaranteed_share"):
        SandboxCapacityConfig(classes={"rollout": {"guaranteed_share": 0}})


@pytest.mark.asyncio
async def test_undeclared_class_has_no_guarantee_and_warns_once(caplog) -> None:
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(
            memory_mb=100,
            cpu_cores=4,
            utilization=1,
            lease_ttl_s=10,
            heartbeat_interval_s=1,
            classes={"rollout": {"guaranteed_share": 0.5}},
        )
    )

    with caplog.at_level("WARNING"):
        await coordinator.acquire("first", "owner", 10, 1, "mystery")
        await coordinator.release("first")
        await coordinator.acquire("second", "owner", 10, 1, "mystery")

    warnings = [record for record in caplog.records if "mystery" in record.getMessage()]
    assert len(warnings) == 1
    snapshot = await coordinator.snapshot()
    assert snapshot["per_class"]["mystery"]["guaranteed"] == {"memory_mb": 0, "cpu_millis": 0}
    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_single_undeclared_class_does_not_warn(caplog) -> None:
    """One shared class is a valid setup, not a missing declaration."""
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(memory_mb=100, cpu_cores=4, utilization=1, lease_ttl_s=10, heartbeat_interval_s=1)
    )

    with caplog.at_level("WARNING"):
        await coordinator.acquire("only", "owner", 10, 1)

    assert not [record for record in caplog.records if "guaranteed share" in record.getMessage()]
    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_acquire_deadline_raises_capacity_timeout() -> None:
    """An unserved request reports a capacity fault instead of hanging forever."""
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(memory_mb=100, cpu_cores=4, utilization=1, lease_ttl_s=10, heartbeat_interval_s=1)
    )
    await coordinator.acquire("holder", "owner", 100, 4)

    with pytest.raises(SandboxCapacityTimeout, match="was not admitted within 0.05s"):
        await coordinator.acquire("queued", "owner", 100, 4, timeout_s=0.05)

    snapshot = await coordinator.snapshot()
    assert snapshot["waiters"] == 0
    assert snapshot["capacity_wait_timeouts"] == 1
    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_expired_owner_reports_capacity_timeout_not_cancellation() -> None:
    """Lease expiry is a capacity fault, so it must not look like a caller cancellation."""
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(
            memory_mb=100,
            cpu_cores=4,
            utilization=1,
            lease_ttl_s=0.06,
            heartbeat_interval_s=0.02,
            acquire_timeout_s=None,
        )
    )
    await coordinator.acquire("holder", "live-owner", 100, 4)

    with pytest.raises(SandboxCapacityTimeout, match="owner lease"):
        await coordinator.acquire("queued", "dead-owner", 100, 4)

    snapshot = await coordinator.snapshot()
    assert snapshot["expired_waiters"] == 1
    assert snapshot["capacity_wait_timeouts"] == 0
    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_class_guarantee_admits_the_larger_class_first() -> None:
    """The class with room goes before a class that would have to borrow."""
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(
            memory_mb=100,
            cpu_cores=10,
            utilization=1,
            lease_ttl_s=10,
            heartbeat_interval_s=1,
            classes={"small": {"guaranteed_share": 0.9}, "large": {"guaranteed_share": 0.1}},
        )
    )

    await coordinator.acquire("large-first", "owner", 10, 1, "large")
    small = asyncio.create_task(coordinator.acquire("small", "owner", 10, 1, "small"))
    await asyncio.sleep(0)

    assert small.done()
    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_oversized_request_fails_immediately() -> None:
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(memory_mb=100, cpu_cores=4, utilization=1, lease_ttl_s=10, heartbeat_interval_s=1)
    )

    with pytest.raises(ValueError, match="exceeds node envelope"):
        await coordinator.acquire("too-large", "worker", 101, 1)

    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_cancel_removes_queued_request() -> None:
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(memory_mb=100, cpu_cores=4, utilization=1, lease_ttl_s=10, heartbeat_interval_s=1)
    )
    await coordinator.acquire("active", "worker-a", 100, 4)
    waiting = asyncio.create_task(coordinator.acquire("waiting", "worker-b", 100, 4))
    await asyncio.sleep(0)

    await coordinator.cancel("waiting")

    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert (await coordinator.snapshot())["waiters"] == 0
    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_expired_worker_lease_returns_capacity() -> None:
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(
            memory_mb=100,
            cpu_cores=4,
            utilization=1,
            lease_ttl_s=0.08,
            heartbeat_interval_s=0.02,
        )
    )
    await coordinator.acquire("abandoned", "dead-worker", 100, 4)
    waiting = asyncio.create_task(coordinator.acquire("next", "live-worker", 100, 4))

    async def heartbeat() -> None:
        while not waiting.done():
            await asyncio.sleep(0.02)
            await coordinator.renew_owner("live-worker")

    heartbeat_task = asyncio.create_task(heartbeat())

    await asyncio.wait_for(waiting, timeout=0.5)
    await heartbeat_task

    snapshot = await coordinator.snapshot()
    assert snapshot["expired"] == 1
    assert snapshot["allocations"] == 1
    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_a_live_owner_lease_is_reclaimed_once_it_outlives_every_sandbox(caplog) -> None:
    """The only recovery path for a lease whose release was missed.

    A lease belonging to a worker that is still running is renewed forever, so owner
    expiry can never reclaim it. Without an age cap the node stays charged for a sandbox
    that no longer exists, every later request times out, and the trainer waits on a buffer
    that cannot fill: that is the hang this cap exists to break.
    """
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(
            memory_mb=100,
            cpu_cores=4,
            utilization=1,
            lease_ttl_s=0.05,
            heartbeat_interval_s=0.01,
            lease_max_age_s=0.15,
        )
    )
    await coordinator.acquire("leaked", "live-worker", 100, 4)

    async def heartbeat() -> None:
        # Keep the owner alive, so the ordinary TTL path is never what frees the capacity.
        while True:
            await asyncio.sleep(0.01)
            await coordinator.renew_owner("live-worker")

    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        # The root level, not the dashed path loggers use: `psrl_logger` is named after
        # its source file, so targeting a dotted name would leave the emitter at whatever
        # level an earlier test left the root logger on.
        with caplog.at_level("WARNING"):
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if (await coordinator.snapshot())["stale_leases_reclaimed"]:
                    break
                await asyncio.sleep(0.02)
    finally:
        heartbeat_task.cancel()
        await asyncio.gather(heartbeat_task, return_exceptions=True)

    snapshot = await coordinator.snapshot()
    assert snapshot["stale_leases_reclaimed"] == 1, (
        "A lease held by a live owner was never reclaimed, so the node stays full forever."
    )
    assert snapshot["allocations"] == 0
    assert snapshot["available_capacity"] == {"memory_mb": 100, "cpu_millis": 4000}
    assert any("never released" in record.getMessage() for record in caplog.records), (
        "A silent capacity leak must be reported with the owner that leaked it."
    )
    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_the_oldest_lease_age_exposes_a_leak_as_it_grows() -> None:
    """The snapshot has to show the leak before the cap reclaims it, not only after."""
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(
            memory_mb=100,
            cpu_cores=4,
            utilization=1,
            lease_ttl_s=10,
            heartbeat_interval_s=1,
        )
    )

    assert (await coordinator.snapshot())["oldest_lease_age_s"] == 0.0

    await coordinator.acquire("held", "worker-a", 10, 1)
    await asyncio.sleep(0.05)

    assert (await coordinator.snapshot())["oldest_lease_age_s"] >= 0.05
    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_nothing_is_reclaimed_while_the_age_cap_is_disabled() -> None:
    """Null is the documented way to keep the old unbounded behavior."""
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(
            memory_mb=100,
            cpu_cores=4,
            utilization=1,
            lease_ttl_s=0.05,
            heartbeat_interval_s=0.01,
            lease_max_age_s=None,
        )
    )
    await coordinator.acquire("held", "live-worker", 100, 4)

    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(0.01)
            await coordinator.renew_owner("live-worker")

    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        await asyncio.sleep(0.3)
    finally:
        heartbeat_task.cancel()
        await asyncio.gather(heartbeat_task, return_exceptions=True)

    snapshot = await coordinator.snapshot()
    assert snapshot["stale_leases_reclaimed"] == 0
    assert snapshot["allocations"] == 1
    await coordinator.shutdown()
