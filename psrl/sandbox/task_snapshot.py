"""Task environments captured once and reused across steps and groups.

A task's setup is a property of the task, not of the group that happens to run it,
and an RL run visits the same task in many steps. Capturing the prepared
environment once turns that cost from once per group per step into once per task
per run, which is the largest win the snapshot path enables.

The reuse is also not time coupled. A fork needs its parent alive while the group
fans out, and a snapshot is already durable, so a member arriving in any later step
reuses it just as well. That is why the group stops being a scheduling unit once
this cache exists.

One rule keeps the cache honest: a snapshot that was never published is only
restorable on the node that holds the image, and a cache hit from another node
would be a lie.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from psrl.sandbox.core import SnapshotKind, SnapshotRef

psrl_logger = logging.getLogger(__file__)

_DEFAULT_INDEX_NAME = "psrl-task-snapshots.json"


def task_key(*, task_id: str, image: str, setup: str | None, resources: str = "") -> str:
    """Return the identity of one captured task environment.

    A cached environment is only valid for the same task, the same image, and the
    same setup, because any of those changes what the environment has to contain.
    """
    identity = {
        "task_id": task_id,
        "image": image,
        "setup": setup or "",
        "resources": resources,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:32]


@dataclass(frozen=True)
class TaskSnapshotRecord:
    """
    One captured task environment, as the cache remembers it.
    """

    key: str
    task_id: str
    image: str
    snapshot_id: str
    # The node that holds the local image. Empty when the snapshot was published, in
    # which case any node can restore it.
    node_id: str = ""
    published: bool = False
    created_at: float = 0.0

    def usable_on(self, node_id: str | None) -> bool:
        """Return whether this record can be restored from the asking node.

        A published capture is reachable from anywhere. An unpublished one lives on
        exactly one node, and an asking caller that did not name a node means this
        process's own node, so it matches a record that also named none and no other.
        """
        if self.published:
            return True
        return self.node_id == (node_id or "")


class TaskSnapshotCache:
    """A durable index of captured task environments.

    It holds references, not images. The image lives in the snapshot store or on the
    node that made it, and this is what remembers which one and where.
    """

    def __init__(
        self,
        *,
        index_path: str | Path | None = None,
        ttl_s: float | None = None,
    ) -> None:
        if ttl_s is not None and ttl_s <= 0:
            raise ValueError("Task snapshot cache ttl_s must be greater than zero when set.")
        self.index_path = Path(index_path or _default_index_path())
        self.ttl_s = ttl_s
        self._records: dict[str, TaskSnapshotRecord] = {}
        self._hits = 0
        self._misses = 0
        self._unusable = 0
        self._load()

    @property
    def records(self) -> tuple[TaskSnapshotRecord, ...]:
        """
        Return every captured environment this cache knows about.
        """
        return tuple(self._records.values())

    def get(
        self,
        *,
        task_id: str,
        image: str,
        setup: str | None,
        node_id: str | None = None,
        resources: str = "",
        now: float | None = None,
    ) -> SnapshotRef | None:
        """Return a captured environment for this task, or None to build one.

        A record that only one node can restore is a miss for any other node, and the
        miss is counted apart from an ordinary one so an operator can tell a cache
        that is cold from a cache that is node bound.
        """
        key = task_key(task_id=task_id, image=image, setup=setup, resources=resources)
        record = self._records.get(key)
        if record is None:
            self._misses += 1
            return None
        if not record.usable_on(node_id):
            self._unusable += 1
            return None
        if self.ttl_s is not None and (time.time() if now is None else now) - record.created_at > self.ttl_s:
            self._misses += 1
            return None
        self._hits += 1
        return SnapshotRef(
            backend="",
            snapshot_id=record.snapshot_id,
            kind=SnapshotKind.FILESYSTEM,
            metadata={"psrl.task_snapshot.key": key, "psrl.task_snapshot.image": record.image},
        )

    def put(
        self,
        *,
        task_id: str,
        image: str,
        setup: str | None,
        snapshot: SnapshotRef,
        node_id: str = "",
        resources: str = "",
        now: float | None = None,
    ) -> TaskSnapshotRecord:
        """
        Record one captured environment.
        """
        key = task_key(task_id=task_id, image=image, setup=setup, resources=resources)
        record = TaskSnapshotRecord(
            key=key,
            task_id=task_id,
            image=image,
            snapshot_id=snapshot.snapshot_id,
            node_id=node_id,
            published=bool(snapshot.metadata.get("psrl.snapshot.published")),
            created_at=time.time() if now is None else now,
        )
        self._records[key] = record
        self._save()
        return record

    def forget(self, key: str) -> None:
        """
        Drop one record, for example after its image is deleted.
        """
        if self._records.pop(key, None) is not None:
            self._save()

    def collect(self, *, now: float | None = None) -> list[str]:
        """
        Drop the records past their TTL and return their keys.
        """
        if self.ttl_s is None:
            return []
        current = time.time() if now is None else now
        expired = [key for key, record in self._records.items() if current - record.created_at > self.ttl_s]
        for key in expired:
            self._records.pop(key, None)
        if expired:
            self._save()
        return expired

    def snapshot(self) -> dict[str, float]:
        """
        Return this cache's counters for the metrics sink.
        """
        return {
            "task_snapshot/records": float(len(self._records)),
            "task_snapshot/hits": float(self._hits),
            "task_snapshot/misses": float(self._misses),
            "task_snapshot/unusable_on_this_node": float(self._unusable),
        }

    def _load(self) -> None:
        try:
            payload = json.loads(self.index_path.read_text())
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError):
            psrl_logger.warning(f"Could not read the task snapshot cache at {self.index_path!r}.", exc_info=True)
            return
        for entry in payload.get("records", []):
            record = TaskSnapshotRecord(**entry)
            self._records[record.key] = record

    def _save(self) -> None:
        """
        Write the index atomically, so a crash cannot leave it half written.
        """
        payload = {"records": [asdict(record) for record in self._records.values()]}
        temporary = self.index_path.with_suffix(".tmp")
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(payload, indent=2) + "\n")
            os.replace(temporary, self.index_path)
        except OSError:
            psrl_logger.warning(f"Could not write the task snapshot cache at {self.index_path!r}.", exc_info=True)


def _default_index_path() -> str:
    """
    Return the default cache location for a local deployment.
    """
    return os.path.join(os.path.expanduser("~"), ".psrl", _DEFAULT_INDEX_NAME)


def resource_fingerprint(resources: Mapping[str, object]) -> str:
    """
    Render a resource request as part of a cache key.
    """
    return json.dumps(dict(resources), sort_keys=True, separators=(",", ":"))
