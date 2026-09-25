from __future__ import annotations

import asyncio

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
    assert snapshot["available_capacity"] == {"memory_mb": 20, "cpu_millis": 3000, "gpu_count": 0, "disk_mb": 0}
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
    assert snapshot["available_capacity"] == {"memory_mb": 0, "cpu_millis": 1000, "gpu_count": 0, "disk_mb": 0}
    assert snapshot["allocations"] == 3
    await coordinator.release_owner("worker")
    assert (await coordinator.snapshot())["available_capacity"] == {
        "memory_mb": 100,
        "cpu_millis": 10000,
        "gpu_count": 0,
        "disk_mb": 0,
    }
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
        asyncio.create_task(coordinator.acquire(f"rollout-{index}", "owner", 100, 1, "rollout")) for index in range(10)
    ]
    await asyncio.gather(*admitted_rollouts)
    extra_rollouts = [
        asyncio.create_task(coordinator.acquire(f"waiting-{index}", "owner", 100, 1, "rollout")) for index in range(4)
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
    assert snapshot["per_class"]["grader"]["used"] == {
        "memory_mb": 300,
        "cpu_millis": 1000,
        "gpu_count": 0,
        "disk_mb": 0,
    }

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
    assert snapshot["per_class"]["rollout"]["used"] == {
        "memory_mb": 500,
        "cpu_millis": 5000,
        "gpu_count": 0,
        "disk_mb": 0,
    }
    assert snapshot["per_class"]["grader"]["used"] == {
        "memory_mb": 300,
        "cpu_millis": 1000,
        "gpu_count": 0,
        "disk_mb": 0,
    }
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
    assert snapshot["available_capacity"] == {"memory_mb": 100, "cpu_millis": 1000, "gpu_count": 0, "disk_mb": 0}
    assert snapshot["per_class"]["rollout"]["used"] == {
        "memory_mb": 900,
        "cpu_millis": 9000,
        "gpu_count": 0,
        "disk_mb": 0,
    }
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
    assert snapshot["per_class"]["rollout"]["used"] == {
        "memory_mb": 300,
        "cpu_millis": 3000,
        "gpu_count": 0,
        "disk_mb": 0,
    }
    assert snapshot["per_class"]["rollout"]["ceiling"] == {
        "memory_mb": 300,
        "cpu_millis": 3000,
        "gpu_count": 0,
        "disk_mb": 0,
    }
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
    assert snapshot["per_class"]["mystery"]["guaranteed"] == {
        "memory_mb": 0,
        "cpu_millis": 0,
        "gpu_count": 0,
        "disk_mb": 0,
    }
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

    await asyncio.wait_for(small, timeout=0.5)
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
async def test_live_owner_capacity_is_not_reclaimed_by_age() -> None:
    """Live allocations remain charged regardless of age."""
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(
            memory_mb=100,
            cpu_cores=4,
            utilization=1,
            # Ten renewals per lease lifetime, so event loop jitter cannot expire a live owner
            # and make this flake.
            lease_ttl_s=0.5,
            heartbeat_interval_s=0.05,
        )
    )
    await coordinator.acquire("held", "live-worker", 100, 4)

    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(0.05)
            await coordinator.renew_owner("live-worker")

    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        await asyncio.sleep(0.3)
    finally:
        heartbeat_task.cancel()
        await asyncio.gather(heartbeat_task, return_exceptions=True)

    snapshot = await coordinator.snapshot()
    assert snapshot["allocations"] == 1
    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_two_borrowing_classes_make_progress_without_mutual_blocking():
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(
            memory_mb=100,
            cpu_cores=10,
            utilization=1,
            classes={"a": {"guaranteed_share": 0.1}, "b": {"guaranteed_share": 0.1}},
        )
    )
    await coordinator.acquire("holder", "owner", 100, 10, "a")
    first = asyncio.create_task(coordinator.acquire("a", "owner", 60, 6, "a"))
    second = asyncio.create_task(coordinator.acquire("b", "owner", 60, 6, "b"))
    await asyncio.sleep(0)
    await coordinator.release("holder")
    await asyncio.wait_for(first, timeout=0.5)
    assert not second.done()
    await coordinator.release("a")
    await asyncio.wait_for(second, timeout=0.5)
    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_a_guaranteed_request_is_never_blocked_by_a_waiting_borrower():
    """The guarantee is a contract, so a starved borrower cannot hold the node hostage.

    Reserving released capacity for an oversized borrower would stall every class that fits
    its own share, for as long as the longest running episode. That is a node-wide stall, so
    the borrower is counted and left to the admission deadline instead.
    """
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(
            memory_mb=100,
            cpu_cores=10,
            utilization=1,
            classes={"small": {"guaranteed_share": 0.5}, "large": {"guaranteed_share": 0.1}},
        )
    )
    await coordinator.acquire("held", "owner", 60, 6, "blocker")
    # 90 exceeds the large class guarantee of 10, so it can only borrow, and it does not fit.
    large = asyncio.create_task(coordinator.acquire("large", "owner", 90, 9, "large"))
    await asyncio.sleep(0)
    for index in range(8):
        await coordinator.acquire(f"small-{index}", "owner", 10, 1, "small")
        await coordinator.release(f"small-{index}")

    await asyncio.wait_for(coordinator.acquire("refill", "owner", 10, 1, "small"), timeout=0.5)

    assert not large.done()
    assert (await coordinator.snapshot())["max_borrow_bypasses"] >= 8
    await coordinator.release("held")
    await coordinator.release("refill")
    await asyncio.wait_for(large, timeout=0.5)
    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_a_borrower_that_does_not_fit_does_not_block_one_that_does():
    """Best-effort FIFO among borrowers, so head-of-line blocking cannot idle the envelope."""
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(
            memory_mb=100,
            cpu_cores=10,
            utilization=1,
            classes={"big": {"guaranteed_share": 0.1}, "small": {"guaranteed_share": 0.1}},
        )
    )
    await coordinator.acquire("held", "owner", 70, 7, "blocker")
    # Both exceed their guarantee, so both are borrowers. The older one cannot fit.
    older = asyncio.create_task(coordinator.acquire("older", "owner", 60, 6, "big"))
    await asyncio.sleep(0)
    younger = asyncio.create_task(coordinator.acquire("younger", "owner", 20, 2, "small"))

    await asyncio.wait_for(younger, timeout=0.5)

    assert not older.done()
    await coordinator.release("held")
    await coordinator.release("younger")
    await asyncio.wait_for(older, timeout=0.5)
    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_slack_is_reserved_for_a_guaranteed_request_that_does_not_fit():
    """A borrower must not take capacity a queued guarantee is still short of."""
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(
            memory_mb=100,
            cpu_cores=10,
            utilization=1,
            classes={"rollout": {"guaranteed_share": 0.6}, "grader": {"guaranteed_share": 0.1}},
        )
    )
    await coordinator.acquire("held", "owner", 80, 8, "blocker")
    # Inside the rollout guarantee of 60, but only 20 is free.
    guaranteed = asyncio.create_task(coordinator.acquire("guaranteed", "owner", 50, 5, "rollout"))
    await asyncio.sleep(0)
    # Exceeds the grader guarantee of 10, so it borrows, and it would fit the free 20.
    borrower = asyncio.create_task(coordinator.acquire("borrower", "owner", 20, 2, "grader"))
    await asyncio.sleep(0)

    assert not guaranteed.done()
    assert not borrower.done(), "The slack belongs to the queued guarantee."

    await coordinator.release("held")
    await asyncio.wait_for(guaranteed, timeout=0.5)
    await asyncio.wait_for(borrower, timeout=0.5)
    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_shutdown_wakes_queued_requests_and_refuses_new_admission():
    coordinator = SandboxCapacityCoordinator(SandboxCapacityConfig(memory_mb=10, cpu_cores=1, utilization=1))
    await coordinator.acquire("held", "owner", 10, 1)
    waiting = asyncio.create_task(coordinator.acquire("waiting", "owner", 10, 1))
    await asyncio.sleep(0)
    await coordinator.shutdown()
    with pytest.raises(RuntimeError, match="closed"):
        await waiting
    with pytest.raises(RuntimeError, match="closed"):
        await coordinator.acquire("new", "owner", 1, 1)
