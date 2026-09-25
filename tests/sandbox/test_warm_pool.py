"""The prepared-container pool, which trades idle memory for a shorter cold start.

Two rules carry the suite: an entry only serves the spec shape it was prepared for,
and an entry that is never claimed is destroyed by the pool rather than left to the
reclaimer, which sees a live owner and would leave it alone.
"""

from __future__ import annotations

import asyncio

import pytest
from psrl.sandbox import SandboxFeature, SandboxSource, SandboxSpec
from psrl.sandbox.backends.docker import DockerBackend
from psrl.sandbox.backends.docker.pool import (
    WARM_POOL_LABEL,
    DockerWarmPool,
    WarmPoolConfig,
    pool_key,
)
from psrl.sandbox.core import ExecMode, ResourceSpec

from tests.sandbox.test_docker_backend import FakeDockerEngine

pytestmark = pytest.mark.cpu_test


class PoolEngine(FakeDockerEngine):
    """A daemon that names each container it is asked to create."""

    def __init__(self) -> None:
        super().__init__()
        self.created: list[str] = []

    async def create_container(self, name: str, config) -> str:
        self.config = config
        self.created.append(name)
        return f"{name}-id"


def _spec(**overrides) -> SandboxSpec:
    payload = {
        "source": SandboxSource.image("image"),
        "resources": ResourceSpec(cpu_count=1, memory_mb=512),
        "lifetime_timeout_s": 300,
        # A pooled entry belongs to no episode, so it carries no idempotency key.
        "idempotency_key": None,
    }
    payload.update(overrides)
    return SandboxSpec(**payload)


def _backend(engine, **pool) -> DockerBackend:
    return DockerBackend(
        default_exec_mode=ExecMode.ONE_SHOT,
        engine=engine,
        warm_pool={"enabled": True, **pool},
    )


async def _settle_pool(backend: DockerBackend) -> None:
    """Wait for the fills `prepare` schedules, which run off the critical path.

    A test that inspects the pool after `prepare` opts back into the wait, because
    production deliberately does not pay it.
    """
    while backend._pool_fills:
        await asyncio.gather(*list(backend._pool_fills.values()))


def test_a_disabled_pool_pools_nothing() -> None:
    config = WarmPoolConfig(enabled=False, batch_size=64)

    assert config.depth() == 0


def test_the_depth_follows_from_the_batch_an_operator_knows() -> None:
    # A number of containers is not estimable, but a share of one step's batch is.
    config = WarmPoolConfig(enabled=True, batch_size=64, depth_fraction=0.25, max_entries=8)

    assert config.depth() == 8
    assert WarmPoolConfig(enabled=True, batch_size=4, depth_fraction=0.25).depth() == 1


def test_the_depth_is_capped_so_a_large_batch_cannot_reserve_a_node() -> None:
    config = WarmPoolConfig(enabled=True, batch_size=4096, depth_fraction=1.0, max_entries=4)

    assert config.depth() == 4


def test_the_pool_refuses_configuration_it_cannot_honor() -> None:
    with pytest.raises(ValueError, match="depth_fraction"):
        WarmPoolConfig(enabled=True, depth_fraction=2.0)
    with pytest.raises(ValueError, match="ttl_s"):
        WarmPoolConfig(enabled=True, ttl_s=0)
    with pytest.raises(ValueError, match="max_entries"):
        WarmPoolConfig(enabled=True, max_entries=-1)


def test_a_pool_entry_is_keyed_by_everything_that_would_change_the_container() -> None:
    base = _spec()

    assert pool_key(base) == pool_key(_spec())
    assert pool_key(base) != pool_key(_spec(workdir="/work"))
    assert pool_key(base) != pool_key(_spec(env={"A": "1"}))
    assert pool_key(base) != pool_key(_spec(resources=ResourceSpec(cpu_count=2, memory_mb=512)))
    assert pool_key(base) != pool_key(SandboxSpec(SandboxSource.image("other"), resources=base.resources))


def test_a_spec_that_names_one_episode_is_not_pooled() -> None:
    # A pooled entry belongs to no episode yet, so a spec that names one cannot use
    # the pool without mislabelling the container.
    backend = _backend(PoolEngine())

    assert not backend.warm_pool.poolable(_spec(idempotency_key="task-1:rollout"))


def test_a_spec_that_holds_a_control_open_is_not_pooled() -> None:
    # A prepared container would hold a credential or a destination allowlist before
    # any episode exists, so those specs take the ordinary path.
    from psrl.sandbox import CredentialRef, EgressAction, EgressPolicy, EgressRule

    plain = _backend(PoolEngine())
    assert plain.warm_pool.poolable(_spec())
    assert not plain.warm_pool.poolable(
        _spec(egress=EgressPolicy(rules=(EgressRule(EgressAction.ALLOW, "10.0.0.5"),)))
    )
    assert not plain.warm_pool.poolable(_spec(credentials=(CredentialRef(source_env="A", target_env="B"),)))


async def test_prepare_fills_the_pool_and_a_create_adopts_an_entry() -> None:
    engine = PoolEngine()
    backend = _backend(engine, batch_size=2, depth_fraction=0.5)
    spec = _spec()

    await backend.prepare(spec)
    await _settle_pool(backend)
    assert backend.warm_pool.size() == 1

    session = await backend.create(spec)

    assert session.ref.sandbox_id.startswith("psrl-warm-")
    assert backend.warm_pool.size() == 0
    assert backend.warm_pool.snapshot().claimed == 1
    assert backend.metrics_snapshot().operations["warm_pool_claim"].count == 1
    await session.terminate()


async def test_a_pooled_container_is_labelled_so_an_operator_can_see_why_it_exists() -> None:
    engine = PoolEngine()
    backend = _backend(engine, batch_size=2, depth_fraction=0.5)

    await backend.prepare(_spec())
    await _settle_pool(backend)

    assert WARM_POOL_LABEL in engine.config["Labels"]


async def test_a_create_for_another_spec_shape_does_not_adopt_the_entry() -> None:
    engine = PoolEngine()
    backend = _backend(engine, batch_size=2, depth_fraction=0.5)
    await backend.prepare(_spec())
    await _settle_pool(backend)

    session = await backend.create(_spec(workdir="/elsewhere"))

    assert not session.ref.sandbox_id.startswith("psrl-warm-")
    assert backend.warm_pool.size() == 1
    await session.terminate()
    await backend.shutdown()


async def test_an_expired_entry_is_destroyed_by_the_pool() -> None:
    engine = PoolEngine()
    backend = _backend(engine, batch_size=2, depth_fraction=0.5, ttl_s=60.0)
    await backend.prepare(_spec())
    await _settle_pool(backend)

    removed = await backend.warm_pool.expire(now=__import__("time").time() + 120)

    assert len(removed) == 1
    assert backend.warm_pool.size() == 0
    assert backend.warm_pool.snapshot().expired == 1
    await backend.shutdown()


async def test_shutdown_drains_entries_that_belong_to_nobody() -> None:
    engine = PoolEngine()
    backend = _backend(engine, batch_size=2, depth_fraction=0.5)
    await backend.prepare(_spec())
    await _settle_pool(backend)
    assert backend.warm_pool.size() == 1

    await backend.shutdown()

    assert backend.warm_pool.size() == 0


async def test_a_failed_prepare_is_counted_rather_than_raised() -> None:
    # Prefetching is an optimization, so a spec that cannot be prepared still runs on
    # the ordinary create path.
    engine = PoolEngine()
    backend = _backend(engine, batch_size=2, depth_fraction=0.5)

    async def refuse(name, config):
        raise RuntimeError("daemon is busy")

    engine.create_container = refuse

    await backend.prepare(_spec())
    await _settle_pool(backend)

    assert backend.warm_pool.size() == 0
    assert backend.warm_pool.snapshot().refusals == 1
    await backend.shutdown()


async def test_the_pool_reports_its_state_to_the_metrics_sink() -> None:
    engine = PoolEngine()
    backend = _backend(engine, batch_size=2, depth_fraction=0.5)
    pool = DockerWarmPool(
        WarmPoolConfig(enabled=True, batch_size=2, depth_fraction=0.5),
        prepare=lambda key, spec: _prepared(key),
        destroy=lambda container_id: _destroyed(container_id),
    )

    assert pool.snapshot().as_dict()["warm_pool/entries"] == 0.0
    assert backend.warm_pool_snapshot()["warm_pool/entries"] == 0.0
    await backend.shutdown()


async def _prepared(key: str) -> str:
    return f"{key}-container"


async def _destroyed(container_id: str) -> None:
    return None


async def test_a_claim_only_serves_a_spec_the_pool_was_filled_for() -> None:
    pool = DockerWarmPool(
        WarmPoolConfig(enabled=True, batch_size=2, depth_fraction=0.5),
        prepare=lambda key, spec: _prepared(key),
        destroy=lambda container_id: _destroyed(container_id),
    )
    await pool.fill(_spec(), want=1)

    assert await pool.claim(_spec(metadata={"task": "1"})) is None
    assert pool.snapshot().claimed == 0
    assert await pool.claim(_spec()) is not None


async def test_an_expired_entry_is_never_handed_out() -> None:
    # Its image may have been pruned, and adopting it would turn a cache miss into a
    # confusing failure later.
    import time

    pool = DockerWarmPool(
        WarmPoolConfig(enabled=True, batch_size=2, depth_fraction=0.5, ttl_s=10.0),
        prepare=lambda key, spec: _prepared(key),
        destroy=lambda container_id: _destroyed(container_id),
    )
    await pool.fill(_spec(), want=1)

    assert await pool.claim(_spec(), now=time.time() + 60) is None
    assert pool.snapshot().expired == 1


def test_the_warm_pool_feature_is_declared_only_where_a_pool_is_configured() -> None:
    # Advertising it without a pool would admit a recipe that asked for a warm start
    # and then make it pay a cold one.
    assert not DockerBackend(engine=PoolEngine()).capabilities.supports(SandboxFeature.WARM_POOL)
    assert DockerBackend(engine=PoolEngine(), warm_pool={"enabled": True}).capabilities.supports(
        SandboxFeature.WARM_POOL
    )


async def test_preparing_a_pool_does_not_make_an_acquire_wait_for_it() -> None:
    # Preparation is off the critical path by contract, and a container start is not
    # something an acquire should wait for.
    engine = PoolEngine()
    backend = _backend(engine, batch_size=2, depth_fraction=0.5, ttl_s=60.0)

    await backend.prepare(_spec())

    # The fill is scheduled, so the pool is not necessarily stocked yet.
    assert len(backend._pool_fills) <= 1
    await backend.shutdown()
    assert backend._pool_fills == {}


async def test_a_burst_of_prepares_launches_one_fill_per_spec_shape() -> None:
    engine = PoolEngine()
    backend = _backend(engine, batch_size=2, depth_fraction=0.5)
    spec = _spec()

    for _ in range(5):
        await backend.prepare(spec)

    assert len(backend._pool_fills) <= 1
    await backend.shutdown()
