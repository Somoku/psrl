"""The shared snapshot store and the node-local cache budget.

Two rules carry the suite. A checkpoint that cannot publish must fail rather than
return a reference only one node can use, and the digest rather than the tag is
what a restore names.
"""

from __future__ import annotations

import pytest
from psrl.sandbox import (
    SandboxFeature,
    SandboxSource,
    SandboxSpec,
    SandboxStatePolicy,
    SnapshotKind,
)
from psrl.sandbox.backends.docker import DockerBackend
from psrl.sandbox.core import ExecMode
from psrl.sandbox.snapshot_store import (
    LocalSnapshotBudget,
    RegistrySnapshotStore,
    SnapshotPublishError,
    SnapshotStoreConfig,
    build_snapshot_store,
)

from tests.sandbox.test_docker_backend import FakeDockerEngine

pytestmark = pytest.mark.cpu_test


class StoreEngine(FakeDockerEngine):
    """A daemon that records tags and pushes and returns a configurable digest."""

    def __init__(self, digest: str | None = "sha256:abc123") -> None:
        super().__init__()
        self.tagged: list[tuple[str, str, str]] = []
        self.pushed: list[tuple[str, str]] = []
        self.digest = digest
        self.pushes_fail = False
        self.removed_images: list[str] = []
        self.images: list[dict] = []

    async def remove_image(self, reference: str) -> None:
        self.removed_images.append(reference)

    async def list_images(self) -> list[dict]:
        return list(self.images)

    async def commit_container(self, container_id: str, repository: str, tag: str) -> str:
        self.committed = f"{repository}:{tag}"
        return "sha256:localcommit"

    async def tag_image(self, reference: str, repository: str, tag: str) -> None:
        self.tagged.append((reference, repository, tag))

    async def push_image(self, reference: str, tag: str, auth=None) -> None:
        if self.pushes_fail:
            raise OSError("registry unreachable")
        self.pushed.append((reference, tag))

    async def image_digests(self, reference: str) -> list[str]:
        if self.digest is None:
            return []
        # A repository can carry a port, so the tag is stripped from the right.
        repository = reference.rsplit(":", 1)[0]
        return [f"{repository}@{self.digest}"]


def _store(tmp_path, engine, **overrides) -> RegistrySnapshotStore:
    config = SnapshotStoreConfig(
        registry="registry.internal:5000",
        index_path=str(tmp_path / "index.json"),
        **overrides,
    )
    return RegistrySnapshotStore(config, engine)


def test_a_retention_intent_becomes_a_window() -> None:
    assert SnapshotStoreConfig(registry="r:5000", retention="one_day").ttl_seconds == 86400.0
    assert SnapshotStoreConfig(registry="r:5000", retention="one_run").ttl_seconds is None
    assert SnapshotStoreConfig(registry="r:5000", ttl_s=42).ttl_seconds == 42.0


def test_an_unknown_retention_is_refused() -> None:
    with pytest.raises(ValueError, match="retention"):
        SnapshotStoreConfig(registry="r:5000", retention="forever")


def test_a_store_without_a_registry_is_disabled() -> None:
    assert not SnapshotStoreConfig().enabled
    assert build_snapshot_store(None, StoreEngine()) is None


def test_snapshots_are_namespaced_per_run_and_workflow() -> None:
    config = SnapshotStoreConfig(registry="r:5000", snapshot_namespace="psrl-snapshots")

    repository = config.repository_for("Run 1", "task/7")

    assert repository == "psrl-snapshots/run-1/task-7"


async def test_publishing_returns_the_digest_reference_and_records_it(tmp_path) -> None:
    engine = StoreEngine()
    store = _store(tmp_path, engine)

    reference = await store.publish("psrl/snapshot/x:tag", run_id="run-1", workflow_id="task-1")

    assert reference == "registry.internal:5000/psrl-snapshots/run-1/task-1@sha256:abc123"
    assert engine.tagged and engine.pushed
    assert [record.digest_ref for record in store.records] == [reference]


async def test_a_push_that_fails_returns_no_reference_and_leaves_a_collectable_intent(tmp_path) -> None:
    # A reference that only this node can use looks like success and fails later, on a node that
    # never had the image. The intent recorded before the push stays, so collection can still expire it.
    engine = StoreEngine()
    engine.pushes_fail = True
    store = _store(tmp_path, engine, retention="one_hour")

    with pytest.raises(SnapshotPublishError, match="could not be published"):
        await store.publish("psrl/snapshot/x:tag", run_id="run-1", workflow_id="task-1")

    assert [record.provisional for record in store.records] == [True]
    intent = store.records[0]
    assert store.collect(now=intent.published_at + 3601) == [intent.digest_ref]
    assert store.records == ()


async def test_a_push_without_a_digest_returns_no_reference(tmp_path) -> None:
    engine = StoreEngine(digest=None)
    store = _store(tmp_path, engine)

    with pytest.raises(SnapshotPublishError, match="reported no digest"):
        await store.publish("psrl/snapshot/x:tag", run_id="run-1", workflow_id="task-1")

    assert [record.provisional for record in store.records] == [True]


async def test_a_digest_from_another_repository_is_ignored(tmp_path) -> None:
    engine = StoreEngine()
    store = _store(tmp_path, engine)

    async def other_registry(_reference: str) -> list[str]:
        return ["elsewhere.invalid/repo@sha256:deadbeef"]

    engine.image_digests = other_registry  # type: ignore[assignment]

    with pytest.raises(SnapshotPublishError, match="reported no digest"):
        await store.publish("psrl/snapshot/x:tag", run_id="run-1", workflow_id="task-1")


async def test_the_record_survives_a_restart(tmp_path) -> None:
    engine = StoreEngine()
    store = _store(tmp_path, engine)
    reference = await store.publish("psrl/snapshot/x:tag", run_id="run-1", workflow_id="task-1")

    reopened = _store(tmp_path, engine)

    assert [record.digest_ref for record in reopened.records] == [reference]


async def test_collection_expires_only_what_is_past_its_retention(tmp_path) -> None:
    engine = StoreEngine()
    store = _store(tmp_path, engine, retention="one_hour")
    reference = await store.publish("psrl/snapshot/x:tag", run_id="run-1", workflow_id="task-1")

    assert store.collect(now=store.records[0].published_at + 10) == []
    assert store.collect(now=store.records[0].published_at + 3601) == [reference]
    assert store.records == ()


async def test_a_run_scoped_retention_can_forget_one_run(tmp_path) -> None:
    engine = StoreEngine()
    store = _store(tmp_path, engine)
    await store.publish("psrl/snapshot/a:tag", run_id="run-1", workflow_id="task-1")
    await store.publish("psrl/snapshot/b:tag", run_id="run-2", workflow_id="task-1")

    dropped = store.forget_run("run-1")

    assert len(dropped) == 1
    assert [record.run_id for record in store.records] == ["run-2"]


async def test_deleting_a_snapshot_drops_its_record(tmp_path) -> None:
    engine = StoreEngine()
    store = _store(tmp_path, engine)
    reference = await store.publish("psrl/snapshot/x:tag", run_id="run-1", workflow_id="task-1")

    await store.delete(reference)

    assert store.records == ()
    # The drop survives a restart, so a deleted snapshot is not reclaimed as live.
    assert _store(tmp_path, engine).records == ()


async def test_a_delete_removes_the_manifest_where_the_transport_can(tmp_path) -> None:
    # The Docker Engine API cannot delete a registry manifest, so reclamation falls back to the
    # store's TTL plus the object store's lifecycle rule. Where the transport can delete one, it must ask.
    class ManifestEngine(StoreEngine):
        def __init__(self) -> None:
            super().__init__()
            self.manifests: list[tuple[str, str]] = []

        async def delete_manifest(self, repository: str, digest: str) -> None:
            self.manifests.append((repository, digest))

    engine = ManifestEngine()
    store = _store(tmp_path, engine)
    reference = await store.publish("psrl/snapshot/x:tag", run_id="run-1", workflow_id="task-1")

    await store.delete(reference)

    assert engine.manifests == [("registry.internal:5000/psrl-snapshots/run-1/task-1", "sha256:abc123")]


async def test_a_manifest_the_registry_refuses_to_delete_is_still_forgotten(tmp_path) -> None:
    # Keeping the record would make every later collect retry a manifest nobody can
    # remove, so the record goes and the backstop is the store's lifecycle rule.
    class StubbornEngine(StoreEngine):
        async def delete_manifest(self, repository: str, digest: str) -> None:
            raise OSError("registry refused")

    engine = StubbornEngine()
    store = _store(tmp_path, engine)
    reference = await store.publish("psrl/snapshot/x:tag", run_id="run-1", workflow_id="task-1")

    await store.delete(reference)

    assert store.records == ()


def test_the_local_cache_budget_is_a_share_of_the_disk_allowance() -> None:
    assert LocalSnapshotBudget(total_mb=1000, fraction=0.25).budget_mb == 250
    assert LocalSnapshotBudget(total_mb=0, fraction=0.5).budget_mb == 0


def test_the_oldest_snapshot_leaves_the_cache_first() -> None:
    # A snapshot a restore keeps reading is the one worth keeping, and its age says
    # nothing about that, which is why the ranking uses use rather than age.
    budget = LocalSnapshotBudget(
        total_mb=100,
        fraction=0.5,
        usage={"fresh": 100.0, "stale": 1.0},
    )

    evictions = budget.select_evictions({"stale": 30.0, "fresh": 30.0})

    assert evictions == ["stale"]


def test_a_cache_inside_its_budget_evicts_nothing() -> None:
    budget = LocalSnapshotBudget(total_mb=1000, fraction=0.5)

    assert budget.select_evictions({"a": 10.0, "b": 10.0}) == []


def test_eviction_stops_as_soon_as_the_budget_fits() -> None:
    # A 50 MiB budget over 60 MiB of snapshots: the oldest one brings it inside.
    budget = LocalSnapshotBudget(total_mb=100, fraction=0.5, usage={"a": 1.0, "b": 2.0, "c": 3.0})

    evictions = budget.select_evictions({"a": 20.0, "b": 20.0, "c": 20.0})

    assert evictions == ["a"]


def _snapshot_backend(engine, tmp_path, **overrides) -> DockerBackend:
    return DockerBackend(
        default_exec_mode=ExecMode.ONE_SHOT,
        engine=engine,
        snapshot_store=SnapshotStoreConfig(
            registry="registry.internal:5000",
            index_path=str(tmp_path / "index.json"),
            **overrides,
        ),
    )


async def test_on_demand_publishing_leaves_a_plain_checkpoint_node_local(tmp_path) -> None:
    # The common case pays no push cost, which is the whole point of on-demand.
    engine = StoreEngine()
    backend = _snapshot_backend(engine, tmp_path)
    session = await backend.create(
        SandboxSpec(SandboxSource.image("image"), state_policy=SandboxStatePolicy(enabled=True))
    )

    snapshot = await session.snapshot(SnapshotKind.FILESYSTEM)

    assert snapshot.metadata.get("psrl.snapshot.published") is None
    assert engine.pushed == []
    assert snapshot.snapshot_id.startswith("psrl/snapshot/")


async def test_a_resume_anywhere_requirement_publishes_the_checkpoint(tmp_path) -> None:
    engine = StoreEngine()
    backend = _snapshot_backend(engine, tmp_path)
    session = await backend.create(
        SandboxSpec(
            SandboxSource.image("image"),
            required_features=frozenset({SandboxFeature.RESUME_ANYWHERE}),
            state_policy=SandboxStatePolicy(enabled=True),
        )
    )

    snapshot = await session.snapshot(SnapshotKind.FILESYSTEM)

    assert snapshot.metadata["psrl.snapshot.published"] is True
    assert snapshot.snapshot_id.endswith("@sha256:abc123")
    # The image the restore path will use is the published reference, so a restore on
    # another node is the ordinary create that pulls a missing image.
    assert snapshot.metadata["psrl.docker.image"] == snapshot.snapshot_id
    assert engine.pushed


async def test_an_always_publishing_deployment_publishes_every_checkpoint(tmp_path) -> None:
    engine = StoreEngine()
    backend = _snapshot_backend(engine, tmp_path, publish="always")
    session = await backend.create(
        SandboxSpec(SandboxSource.image("image"), state_policy=SandboxStatePolicy(enabled=True))
    )

    snapshot = await session.snapshot(SnapshotKind.FILESYSTEM)

    assert snapshot.metadata["psrl.snapshot.published"] is True


async def test_a_resume_anywhere_requirement_without_a_store_is_refused(tmp_path) -> None:
    engine = StoreEngine()
    backend = DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, engine=engine)
    session = await backend.create(
        SandboxSpec(
            SandboxSource.image("image"),
            required_features=frozenset({SandboxFeature.RESUME_ANYWHERE}),
            state_policy=SandboxStatePolicy(enabled=True),
        )
    )

    with pytest.raises(RuntimeError, match="has no snapshot store"):
        await session.snapshot(SnapshotKind.FILESYSTEM)


async def test_deleting_a_published_snapshot_also_drops_its_record(tmp_path) -> None:
    engine = StoreEngine()
    backend = _snapshot_backend(engine, tmp_path, publish="always")
    session = await backend.create(
        SandboxSpec(SandboxSource.image("image"), state_policy=SandboxStatePolicy(enabled=True))
    )
    snapshot = await session.snapshot(SnapshotKind.FILESYSTEM)

    await backend.delete_snapshot(snapshot)

    assert backend.snapshot_store.records == ()


async def test_a_restore_creates_from_the_published_reference(tmp_path) -> None:
    engine = StoreEngine()
    backend = _snapshot_backend(engine, tmp_path, publish="always")
    session = await backend.create(
        SandboxSpec(SandboxSource.image("image"), state_policy=SandboxStatePolicy(enabled=True))
    )
    snapshot = await session.snapshot(SnapshotKind.FILESYSTEM)

    restored = await backend.restore(snapshot, SandboxSpec(SandboxSource.image("image")))

    # The digest reference became the container's image, which is what makes a restore
    # on another node the ordinary create that pulls a missing image.
    assert engine.config["Image"] == snapshot.snapshot_id
    assert restored.ref.backend == backend.name


def test_collecting_without_a_store_reports_nothing() -> None:
    assert DockerBackend(engine=StoreEngine()).collect_snapshots() == []


async def test_the_cache_counts_snapshots_restored_from_the_store_not_only_local_commits(tmp_path) -> None:
    # A node-local commit and a snapshot pulled back from the shared store both land in Docker
    # storage, so a budget counting only the first lets the second grow unbounded and fill the disk.
    engine = StoreEngine()
    engine.images = [
        {"RepoTags": ["psrl/snapshot/task-1:latest"], "Size": 100 * 1024 * 1024},
        {"RepoTags": ["registry.internal:5000/psrl-snapshots/run-1/task-1:abc"], "Size": 300 * 1024 * 1024},
        {"RepoTags": ["python:3.11"], "Size": 900 * 1024 * 1024},
    ]
    backend = _snapshot_backend(engine, tmp_path)

    usage = await backend.snapshot_cache_usage()

    # The base image is outside the cache figure on purpose.
    assert usage["snapshot_cache/entries"] == 2
    assert usage["snapshot_cache/size_mb"] == 400.0


async def test_a_restored_snapshot_is_ranked_by_use_not_by_commit_age(tmp_path) -> None:
    engine = StoreEngine()
    local = "psrl/snapshot/task-1:latest"
    restored = "registry.internal:5000/psrl-snapshots/run-1/task-1:abc"
    engine.images = [
        {"RepoTags": [local], "Size": 100 * 1024 * 1024},
        {"RepoTags": [restored], "Size": 100 * 1024 * 1024},
    ]
    backend = _snapshot_backend(engine, tmp_path)
    backend.note_snapshot_use(local)
    backend.note_snapshot_use(restored)

    removed = await backend.evict_local_snapshots(total_mb=400)

    assert removed == [local]
    assert engine.removed_images == [local]


async def test_the_journal_tolerates_a_truncated_last_line(tmp_path) -> None:
    # A crash can cut the final append in half. Everything before it is intact, and the
    # missing entry is at worst one snapshot the store re-publishes.
    engine = StoreEngine()
    store = _store(tmp_path, engine)
    reference = await store.publish("psrl/snapshot/x:tag", run_id="run-1", workflow_id="task-1")
    journal = tmp_path / "index.json"
    journal.write_text(journal.read_text() + '{"op": "put", "record": {"digest_ref": "tor')

    reopened = _store(tmp_path, engine)

    assert [record.digest_ref for record in reopened.records] == [reference]


async def test_the_journal_compacts_once_it_outgrows_the_live_records(tmp_path) -> None:
    # A whole-file rewrite per publish made publishing n snapshots cost O(n^2) bytes on
    # the checkpoint path. Appending is O(1) and compaction keeps the file bounded.
    engine = StoreEngine()
    store = _store(tmp_path, engine)
    journal = tmp_path / "index.json"
    for index in range(70):
        await store.publish("psrl/snapshot/x:tag", run_id=f"run-{index}", workflow_id="task-1")

    assert len(store.records) == 70
    assert len(journal.read_text().splitlines()) <= 2 * len(store.records) + 64
    assert len(_store(tmp_path, engine).records) == 70


async def test_a_run_scoped_retention_is_forgotten_when_the_run_ends(tmp_path) -> None:
    engine = StoreEngine()
    store = _store(tmp_path, engine)
    await store.publish("psrl/snapshot/a:tag", run_id="run-1", workflow_id="task-1")
    await store.publish("psrl/snapshot/b:tag", run_id="run-2", workflow_id="task-1")

    assert len(store.forget_run("run-1")) == 1
    assert [record.run_id for record in store.records] == ["run-2"]
    # No run id means every run this store holds, which is what a worker shutting down
    # has to say.
    assert len(store.forget_run()) == 1
    assert store.records == ()


async def test_a_time_based_retention_ignores_a_run_end(tmp_path) -> None:
    # A one-hour snapshot keeps its record for the hour, whatever the run does.
    engine = StoreEngine()
    store = _store(tmp_path, engine, retention="one_hour")
    await store.publish("psrl/snapshot/a:tag", run_id="run-1", workflow_id="task-1")

    assert store.forget_run("run-1") == []
    assert len(store.records) == 1
