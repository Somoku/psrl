"""Captured task environments, which is what stops the group being a scheduling unit.

A task's setup belongs to the task, not to the group that runs it, so capturing it
once serves every later step and every later group. The rule the suite pins down is
that a capture which only one node can restore is not served to another.
"""

from __future__ import annotations

import pytest
from psrl.sandbox import (
    SandboxFeature,
    SandboxSource,
    SandboxSpec,
    SandboxStatePolicy,
    SnapshotKind,
    SnapshotRef,
)
from psrl.sandbox.manager import SandboxManager
from psrl.sandbox.task_snapshot import TaskSnapshotCache, resource_fingerprint, task_key

from tests.sandbox.test_manager import FakeBackend

pytestmark = pytest.mark.cpu_test

_POLICY = SandboxStatePolicy(enabled=True, allow_external_side_effects=True)


def _cache(tmp_path, **overrides) -> TaskSnapshotCache:
    return TaskSnapshotCache(index_path=str(tmp_path / "tasks.json"), **overrides)


def _ref(snapshot_id: str = "psrl/snapshot/x:1", *, published: bool = False) -> SnapshotRef:
    metadata = {"psrl.snapshot.published": True} if published else {}
    return SnapshotRef("fake", snapshot_id, SnapshotKind.FILESYSTEM, metadata=metadata)


def _spec(**overrides) -> SandboxSpec:
    payload = {
        "source": SandboxSource.image("image"),
        "state_policy": _POLICY,
        "workflow_id": "task-1#0",
    }
    payload.update(overrides)
    return SandboxSpec(**payload)


def test_the_key_covers_everything_a_capture_depends_on() -> None:
    base = dict(task_id="task-1", image="image", setup="pip install x")

    assert task_key(**base) == task_key(**base)
    assert task_key(**base) != task_key(task_id="task-2", image="image", setup="pip install x")
    assert task_key(**base) != task_key(task_id="task-1", image="other", setup="pip install x")
    assert task_key(**base) != task_key(task_id="task-1", image="image", setup=None)


def test_a_miss_reports_nothing_to_restore(tmp_path) -> None:
    cache = _cache(tmp_path)

    assert cache.get(task_id="task-1", image="image", setup=None) is None
    assert cache.snapshot()["task_snapshot/misses"] == 1.0


def test_a_recorded_capture_comes_back_to_a_later_caller(tmp_path) -> None:
    cache = _cache(tmp_path)

    cache.put(task_id="task-1", image="image", setup=None, snapshot=_ref())
    captured = cache.get(task_id="task-1", image="image", setup=None)

    assert captured is not None
    assert captured.snapshot_id == "psrl/snapshot/x:1"
    assert cache.snapshot()["task_snapshot/hits"] == 1.0


def test_a_node_local_capture_is_not_served_to_another_node(tmp_path) -> None:
    # Serving it would be a lie: the other node cannot see the image.
    cache = _cache(tmp_path)
    cache.put(task_id="task-1", image="image", setup=None, snapshot=_ref(), node_id="node-a")

    assert cache.get(task_id="task-1", image="image", setup=None, node_id="node-a") is not None
    assert cache.get(task_id="task-1", image="image", setup=None, node_id="node-b") is None
    assert cache.snapshot()["task_snapshot/unusable_on_this_node"] == 1.0


def test_a_published_capture_is_served_anywhere(tmp_path) -> None:
    cache = _cache(tmp_path)
    cache.put(
        task_id="task-1",
        image="image",
        setup=None,
        snapshot=_ref("registry/repo@sha256:abc", published=True),
        node_id="node-a",
    )

    assert cache.get(task_id="task-1", image="image", setup=None, node_id="node-b") is not None


def test_a_capture_is_not_served_without_a_node_to_check_it_against(tmp_path) -> None:
    # A caller that does not say where it is cannot be told the capture is reachable.
    cache = _cache(tmp_path)
    cache.put(task_id="task-1", image="image", setup=None, snapshot=_ref(), node_id="node-a")

    assert cache.get(task_id="task-1", image="image", setup=None) is None


def test_the_cache_survives_a_restart(tmp_path) -> None:
    _cache(tmp_path).put(task_id="task-1", image="image", setup=None, snapshot=_ref())

    reopened = _cache(tmp_path)

    assert reopened.get(task_id="task-1", image="image", setup=None) is not None


def test_collection_expires_only_what_is_past_its_window(tmp_path) -> None:
    cache = _cache(tmp_path, ttl_s=3600.0)
    cache.put(task_id="task-1", image="image", setup=None, snapshot=_ref())

    record = cache.records[0]
    assert cache.collect(now=record.created_at + 10) == []
    assert cache.collect(now=record.created_at + 4000) == [record.key]
    assert cache.records == ()


def test_a_capture_can_be_forgotten_explicitly(tmp_path) -> None:
    cache = _cache(tmp_path)
    record = cache.put(task_id="task-1", image="image", setup=None, snapshot=_ref())

    cache.forget(record.key)

    assert cache.records == ()


def test_the_cache_refuses_a_window_it_cannot_honor(tmp_path) -> None:
    with pytest.raises(ValueError, match="ttl_s"):
        _cache(tmp_path, ttl_s=0)


def test_a_resource_fingerprint_is_part_of_the_key() -> None:
    base = dict(task_id="task-1", image="image", setup=None)

    assert task_key(**base, resources=resource_fingerprint({"memory_mb": 512})) != task_key(
        **base, resources=resource_fingerprint({"memory_mb": 1024})
    )


async def test_the_first_caller_pays_the_setup_and_later_callers_reuse_it(tmp_path) -> None:
    setup_calls: list[str] = []
    backend = FakeBackend({SandboxFeature.FILESYSTEM_SNAPSHOT, SandboxFeature.RESTORE})
    original = backend.create

    async def create(spec):
        session = await original(spec)

        async def counting_setup(command, **kwargs):
            from psrl.sandbox import ExecResult

            setup_calls.append(command)
            return ExecResult(0, "", "")

        session.exec = counting_setup
        return session

    backend.create = create
    manager = SandboxManager({"fake": backend}, "fake")
    cache = _cache(tmp_path)
    spec = _spec()

    first = await manager.acquire_prepared(
        spec, task_id="task-1", cache=cache, setup="prepare", state_policy=_POLICY
    )

    assert len(cache.records) == 1
    assert setup_calls == ["prepare"]

    # The workflow holds one sandbox phase at a time, so the capture is what a later
    # phase restores rather than something two phases hold at once.
    await first.release()
    second = await manager.acquire_prepared(
        spec, task_id="task-1", cache=cache, setup="prepare", state_policy=_POLICY
    )

    # The second caller restored the capture instead of paying the setup again, and the sandbox
    # came from the restore path, which is what proves the capture was used.
    assert second.ref.sandbox_id.startswith("restored-")
    assert setup_calls == ["prepare"]
    await manager.shutdown()


async def test_a_different_task_does_not_reuse_the_capture(tmp_path) -> None:
    backend = FakeBackend({SandboxFeature.FILESYSTEM_SNAPSHOT, SandboxFeature.RESTORE})
    manager = SandboxManager({"fake": backend}, "fake")
    cache = _cache(tmp_path)

    await manager.acquire_prepared(
        _spec(), task_id="task-1", cache=cache, setup="prepare", state_policy=_POLICY
    )
    await manager.acquire_prepared(
        _spec(workflow_id="task-2#0"), task_id="task-2", cache=cache, setup="prepare", state_policy=_POLICY
    )

    assert len(cache.records) == 2
    await manager.shutdown()


async def test_a_failed_setup_captures_nothing_and_leaves_no_sandbox(tmp_path) -> None:
    # A captured environment that is not what the task needs would be wrong for every
    # later reuse, so a failed setup must not be recorded.
    from psrl.sandbox import ExecResult, SandboxSetupError

    backend = FakeBackend({SandboxFeature.FILESYSTEM_SNAPSHOT, SandboxFeature.RESTORE})

    async def failing_setup(command, **kwargs):
        return ExecResult(1, "", "no network")

    original = backend.create

    async def create(spec):
        session = await original(spec)
        session.exec = failing_setup
        return session

    backend.create = create
    manager = SandboxManager({"fake": backend}, "fake")
    cache = _cache(tmp_path)

    with pytest.raises(SandboxSetupError, match="no network"):
        await manager.acquire_prepared(
            _spec(), task_id="task-1", cache=cache, setup="prepare", state_policy=_POLICY
        )

    assert cache.records == ()
    assert manager.ownership_snapshot()["leases"] == 0
    await manager.shutdown()


async def test_a_capture_without_an_enabled_policy_is_refused(tmp_path) -> None:
    backend = FakeBackend({SandboxFeature.FILESYSTEM_SNAPSHOT, SandboxFeature.RESTORE})
    manager = SandboxManager({"fake": backend}, "fake")

    with pytest.raises(RuntimeError, match="explicitly enabled SandboxStatePolicy"):
        await manager.acquire_prepared(
            _spec(state_policy=SandboxStatePolicy()),
            task_id="task-1",
            cache=_cache(tmp_path),
            state_policy=SandboxStatePolicy(),
        )
    await manager.shutdown()
