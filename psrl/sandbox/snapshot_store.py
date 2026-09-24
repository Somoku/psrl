"""The shared snapshot store behind the internal Docker backend's state path.

A filesystem snapshot is a committed image, and a committed image lives on one
daemon. The store is what makes it reachable from any node.

The design in one sentence: **the store is a pull source, not a new protocol.** A
snapshot is pushed to a registry and the registry's digest reference becomes the
sandbox source, so a restore on another node reuses the create path that already
pulls a missing image. There is no separate materialize step, and no second way
to make a container.

Two rules shape the failure behavior.

- A checkpoint that cannot publish fails. Returning a node-local reference would
  look like success and only fail later, on the node that cannot see it.
- The digest, never the tag, is the durable key, so a moved tag cannot change what
  a restore gets.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

psrl_logger = logging.getLogger(__file__)

# What a published snapshot is retained for. An intent rather than a number, because
# an operator knows how long it should outlive its run but not the right seconds.
_RETENTION_SECONDS = {
    "one_run": None,
    "one_hour": 3600.0,
    "one_day": 86400.0,
    "one_week": 604800.0,
}

_DEFAULT_INDEX_NAME = "psrl-snapshot-store.json"

# A journal holding at most this many entries is left alone even when every entry is
# live, so a short run does not compact on every write.
_JOURNAL_FLOOR = 64


class SnapshotPublishError(RuntimeError):
    """
    Raised when a snapshot cannot be made restorable from another node.

    The checkpoint it belongs to must fail: a node-local reference returned as a
    published one only fails later, on a node that never had the image.
    """


@dataclass(frozen=True)
class SnapshotStoreConfig:
    """
    Where published snapshots live and how long they are kept.
    """

    # Registry host and optional port, for example `registry.internal:5000`.
    registry: str | None = None
    # Repository namespace all snapshots live under.
    snapshot_namespace: str = "psrl-snapshots"
    # `on_demand` publishes only when a spec requires RESUME_ANYWHERE, so the common
    # node-local checkpoint pays no push cost.
    publish: str = "on_demand"
    # Retention intent, or an explicit ttl_s.
    retention: str = "one_run"
    ttl_s: float | None = None
    # Share of the node's sandbox disk allowance the local snapshot cache may hold.
    # A fraction, because an operator can estimate a split and not a byte count.
    local_cache_fraction: float = 0.3
    # Where the publish record lives, so a collector can expire what it owns.
    index_path: str | None = None

    def __post_init__(self) -> None:
        if self.publish not in {"on_demand", "always"}:
            raise ValueError("Snapshot store publish must be 'on_demand' or 'always'.")
        if self.retention not in _RETENTION_SECONDS and self.ttl_s is None:
            raise ValueError(
                f"Snapshot store retention {self.retention!r} is not a known intent "
                f"({sorted(_RETENTION_SECONDS)}) and no ttl_s was given."
            )
        if self.ttl_s is not None and self.ttl_s <= 0:
            raise ValueError("Snapshot store ttl_s must be greater than zero when set.")
        if not 0 < self.local_cache_fraction <= 1:
            raise ValueError("Snapshot store local_cache_fraction must be in (0, 1].")

    @property
    def ttl_seconds(self) -> float | None:
        """
        Return the effective retention window, or None for the lifetime of the run.
        """
        if self.ttl_s is not None:
            return self.ttl_s
        return _RETENTION_SECONDS[self.retention]

    @property
    def enabled(self) -> bool:
        """
        Return whether a registry was configured.
        """
        return bool(self.registry)

    def repository_for(self, run_id: str, workflow_id: str) -> str:
        """
        Return the repository one workflow's snapshots live under.

        Namespaced per run so a credential, a collector, or a restore cannot reach
        across runs.
        """
        namespace = self.snapshot_namespace.strip("/")
        return f"{namespace}/{_slug(run_id)}/{_slug(workflow_id)}"

    @classmethod
    def from_value(cls, value: SnapshotStoreConfig | Mapping[str, object] | None) -> SnapshotStoreConfig:
        """
        Normalize a Hydra mapping into immutable store configuration.
        """
        if isinstance(value, cls):
            return value
        return cls(**dict(value or {}))


@dataclass
class SnapshotRecord:
    """
    One published snapshot, as the store remembers it.
    """

    digest_ref: str
    repository: str
    tag: str
    run_id: str
    published_at: float
    local_image: str = ""
    # Written before the push and replaced by the digest record on success. A lost
    # push response leaves a provisional record that collection can still expire.
    provisional: bool = False


def _slug(value: str) -> str:
    """
    Render one path segment as a legal repository component.
    """
    cleaned = "".join(character if character.isalnum() or character in "-_." else "-" for character in value.lower())
    return cleaned.strip("-.") or "unnamed"


class SnapshotImageTransport(Protocol):
    """The image operations a registry-backed store needs, and nothing more.

    Declared here rather than imported from the Docker engine, because the store
    depends on four operations and not on a runtime. Keeping the dependency narrow is
    also what stops the backend package and this module importing each other.
    """

    async def tag_image(self, reference: str, repository: str, tag: str) -> None:
        """
        Tag a local image so it can be pushed under its own repository.
        """

    async def push_image(self, reference: str, tag: str, auth: Mapping[str, str] | None = None) -> None:
        """
        Push a tagged image to its registry, failing on a reported error.
        """

    async def image_digests(self, reference: str) -> list[str]:
        """
        Return the registry digests a local image is known by.
        """

    async def delete_manifest(self, repository: str, digest: str) -> None:
        """Remove one manifest from the registry, when the transport can.

        Optional. The Docker Engine API cannot delete a registry manifest, so an
        engine that does not implement this leaves reclamation to the store's TTL
        plus the object store's own lifecycle rule, which is the documented backstop.
        The store still forgets the record either way.
        """


class SnapshotStore(Protocol):
    """
    What a backend needs from a durable snapshot store.
    """

    @property
    def config(self) -> SnapshotStoreConfig: ...

    async def publish(self, local_image: str, *, run_id: str, workflow_id: str) -> str:
        """
        Publish a committed image and return the reference that restores it anywhere.
        """

    async def delete(self, digest_ref: str) -> None:
        """
        Delete one published snapshot.
        """

    def collect(self, *, now: float | None = None) -> list[str]:
        """
        Expire the snapshots this store owns that are past their retention.
        """

    def forget_run(self, run_id: str | None = None) -> list[str]:
        """
        Drop the records of a run that ended, when retention is run scoped.
        """


class RegistrySnapshotStore:
    """A `docker push` store in front of an OCI registry.

    The registry needs no extra tool on a sandbox node, because Docker already
    pushes and pulls it. Content addressing gives dedup of the shared base layers
    and a digest that a restore can name exactly.
    """

    def __init__(
        self,
        config: SnapshotStoreConfig,
        engine: SnapshotImageTransport,
        *,
        registry_auth: Mapping[str, str] | None = None,
        index_path: str | None = None,
    ) -> None:
        if not config.enabled:
            raise ValueError("RegistrySnapshotStore requires a configured registry.")
        self._config = config
        self.engine = engine
        self.registry_auth = dict(registry_auth or {})
        self.index_path = Path(index_path or config.index_path or _default_index_path())
        self._records: dict[str, SnapshotRecord] = {}
        self._lines = 0
        self._load()

    @property
    def config(self) -> SnapshotStoreConfig:
        return self._config

    @property
    def records(self) -> tuple[SnapshotRecord, ...]:
        """
        Return the snapshots this store believes it owns.
        """
        return tuple(self._records.values())

    async def publish(self, local_image: str, *, run_id: str, workflow_id: str) -> str:
        """Push one committed image and return its digest reference.

        The intent is recorded before the push, so a response that never arrives still
        leaves something collection can expire. A record written only after a
        confirmed digest would let an interrupted push leave an invisible manifest.
        """
        repository = f"{self._config.registry}/{self._config.repository_for(run_id, workflow_id)}"
        tag = uuid.uuid4().hex
        tag_ref = f"{repository}:{tag}"
        self._record(
            SnapshotRecord(
                digest_ref=tag_ref,
                repository=repository,
                tag=tag,
                run_id=run_id,
                published_at=time.time(),
                local_image=local_image,
                provisional=True,
            )
        )
        try:
            await self.engine.tag_image(local_image, repository, tag)
            await self.engine.push_image(repository, tag, self.registry_auth or None)
            digests = await self.engine.image_digests(tag_ref)
        except Exception as exc:
            # Any failure on the push path means the snapshot is not restorable from
            # another node, which is this call's only promise. The transport error is preserved.
            raise SnapshotPublishError(
                f"Snapshot could not be published to {repository}: {exc!r}. A checkpoint must not return a "
                "reference that only this node can use."
            ) from exc
        digest = _select_digest(digests, repository)
        if digest is None:
            raise SnapshotPublishError(
                f"Snapshot push to {tag_ref} reported no digest, so the store cannot name what it "
                "published."
            )
        digest_ref = f"{repository}@{digest}"
        self._forget(tag_ref)
        self._record(
            SnapshotRecord(
                digest_ref=digest_ref,
                repository=repository,
                tag=tag,
                run_id=run_id,
                published_at=time.time(),
                local_image=local_image,
            )
        )
        return digest_ref

    async def delete(self, digest_ref: str) -> None:
        """
        Delete a snapshot's manifest and drop its record.

        A manifest the registry refuses to delete is reported and then forgotten,
        because keeping the record would make every later collect retry it forever.
        """
        repository, _, digest = digest_ref.partition("@")
        if repository and digest and hasattr(self.engine, "delete_manifest"):
            try:
                await self.engine.delete_manifest(repository, digest)
            except Exception:
                psrl_logger.warning(f"Registry refused to delete snapshot {digest_ref!r}.", exc_info=True)
        self._forget(digest_ref)

    def collect(self, *, now: float | None = None) -> list[str]:
        """Expire this store's snapshots whose retention has elapsed.

        The record is the clock. A registry exposes no publish time, and the only
        other way to age a manifest is to guess from an unrelated timestamp.
        """
        current = time.time() if now is None else now
        ttl = self._config.ttl_seconds
        expired = [
            record.digest_ref
            for record in self._records.values()
            if ttl is not None and current - record.published_at > ttl
        ]
        for digest_ref in expired:
            self._forget(digest_ref)
        if expired:
            psrl_logger.info(f"Snapshot store expired {len(expired)} snapshot(s) past their retention.")
        return expired

    def forget_run(self, run_id: str | None = None) -> list[str]:
        """Drop the records of a run that ended, for a run-scoped retention.

        A time-based retention keeps its records until the window elapses, so this is
        a no-op there. `run_id` of None forgets every run this store holds, which is
        what a worker shutting down has to say: its runs end with it.
        """
        if self._config.ttl_seconds is not None:
            return []
        dropped = [
            record.digest_ref
            for record in self._records.values()
            if run_id is None or record.run_id == run_id
        ]
        for digest_ref in dropped:
            self._forget(digest_ref)
        return dropped

    def _record(self, record: SnapshotRecord) -> None:
        self._records[record.digest_ref] = record
        self._append({"op": "put", "record": asdict(record)})

    def _forget(self, digest_ref: str) -> None:
        if self._records.pop(digest_ref, None) is not None:
            self._append({"op": "drop", "digest_ref": digest_ref})

    def _load(self) -> None:
        """Replay the journal into the record table.

        A journal rather than a document, because a publish must not rewrite every
        other record: that made publishing n snapshots cost O(n^2) bytes written, on
        the event loop, for work that is already on the checkpoint path.
        """
        try:
            lines = self.index_path.read_text().splitlines()
        except FileNotFoundError:
            return
        except OSError:
            psrl_logger.warning(f"Could not read the snapshot store journal at {self.index_path!r}.", exc_info=True)
            return
        self._lines = len(lines)
        for line in lines:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                # A crash can truncate the last line. Everything before it is intact,
                # and the missing entry is at worst one snapshot the store re-publishes.
                psrl_logger.warning(f"Ignoring a truncated snapshot store journal entry at {self.index_path!r}.")
                continue
            if entry.get("op") == "drop":
                self._records.pop(entry.get("digest_ref"), None)
            elif entry.get("op") == "put":
                try:
                    record = SnapshotRecord(**entry["record"])
                except (KeyError, TypeError):
                    psrl_logger.warning(f"Ignoring a malformed snapshot store record at {self.index_path!r}.")
                    continue
                self._records[record.digest_ref] = record

    def _append(self, entry: Mapping[str, object]) -> None:
        """
        Append one entry, compacting the journal once it outgrows the live records.
        """
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            with self.index_path.open("a") as handle:
                handle.write(json.dumps(entry) + "\n")
            self._lines += 1
        except OSError:
            psrl_logger.warning(f"Could not write the snapshot store journal at {self.index_path!r}.", exc_info=True)
            return
        if self._lines > 2 * len(self._records) + _JOURNAL_FLOOR:
            self._compact()

    def _compact(self) -> None:
        """
        Rewrite the journal as one `put` per live record.
        """
        lines = [json.dumps({"op": "put", "record": asdict(record)}) for record in self._records.values()]
        temporary = self.index_path.with_suffix(".tmp")
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text("".join(f"{line}\n" for line in lines))
            os.replace(temporary, self.index_path)
            self._lines = len(lines)
        except OSError:
            psrl_logger.warning(f"Could not compact the snapshot store journal at {self.index_path!r}.", exc_info=True)


@dataclass(frozen=True)
class LocalSnapshotBudget:
    """The node-local snapshot cache's share of the sandbox disk allowance.

    A snapshot image is larger than a base image because it carries the workspace,
    and both land in the same Docker storage. Without a budget they compete for one
    disk with no policy, and the loser is whichever prune reaches first, so a node
    can evict its base images to hold snapshots and then pay a pull on every create.
    """

    total_mb: int = 0
    fraction: float = 0.3
    # Least recently used entries are evicted to stay inside the budget. A local miss
    # pulls from the store, so eviction is never a correctness event.
    usage: Mapping[str, float] = field(default_factory=dict)

    @property
    def budget_mb(self) -> int:
        return int(self.total_mb * self.fraction)

    def select_evictions(self, image_sizes: Mapping[str, float], *, now: float | None = None) -> list[str]:
        """Return the snapshot images to drop so the cache fits its budget.

        Oldest use first, because a snapshot a restore keeps reading is the one
        worth keeping and its age says nothing about that.

        Args:
            image_sizes (Mapping[str, float]): Local snapshot reference to its size in MiB.
            now (float | None): Unused clock, kept so a caller can pass one.

        Returns:
            list[str]: References to remove, in the order they should go.
        """
        budget = self.budget_mb
        if budget <= 0:
            return []
        ordered = sorted(image_sizes.items(), key=lambda item: self.usage.get(item[0], 0.0))
        total = sum(image_sizes.values())
        evictions: list[str] = []
        for reference, size in ordered:
            if total <= budget:
                break
            evictions.append(reference)
            total -= size
        return evictions


def _select_digest(digests: list[str], repository: str) -> str | None:
    """
    Pick the digest that belongs to a repository, ignoring another registry's.
    """
    for digest in digests:
        name, _, value = digest.partition("@")
        if name == repository and value:
            return value
    return None


def _default_index_path() -> str:
    """
    Return the default record location for the local deployment.
    """
    return os.path.join(os.path.expanduser("~"), ".psrl", _DEFAULT_INDEX_NAME)


def build_snapshot_store(
    config: SnapshotStoreConfig | Mapping[str, object] | None,
    engine: SnapshotImageTransport,
    *,
    registry_auth: Mapping[str, str] | None = None,
) -> SnapshotStore | None:
    """
    Build the configured store, or None when a deployment has no registry.
    """
    resolved = SnapshotStoreConfig.from_value(config)
    if not resolved.enabled:
        return None
    return RegistrySnapshotStore(resolved, engine, registry_auth=registry_auth)
