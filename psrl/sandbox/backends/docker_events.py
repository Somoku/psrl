"""Container stop observation through the Docker event stream."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from psrl.sandbox.backends.docker_engine import DockerEngine

psrl_logger = logging.getLogger(__file__)

_RECONNECT_BASE_S = 1.0
_RECONNECT_MAX_S = 30.0
# Docker reports an OOM kill as a separate `oom` action before the `die` that follows it.
_OOM_ACTIONS = frozenset({"oom"})
_STOP_ACTIONS = frozenset({"die", "destroy"})


class ContainerEventWatcher:
    """Translate one shared Docker event stream into per-container stop notifications.

    Polling every session costs one inspect per interval per running command and detects a
    stop no sooner than that interval, which is long enough for an OOM-killed container to
    leave its command hanging. One stream reports every stop as it happens.

    A stop that happens while the stream is down is recovered two ways: the reconnect asks
    the daemon to replay from the last event it saw, and every container still being waited
    on is inspected directly. Neither alone is sufficient, because a daemon restart loses
    its event history.
    """

    def __init__(
        self,
        engine: DockerEngine,
        inspect: Callable[[str], Awaitable[Mapping[str, Any] | None]],
    ) -> None:
        self._engine = engine
        self._inspect = inspect
        self._task: asyncio.Task[None] | None = None
        self._stopped: dict[str, str] = {}
        self._signals: dict[str, asyncio.Event] = {}
        self._oom: set[str] = set()
        self._last_event_at: float | None = None
        self._closed = False
        # None until the first connection attempt settles. False disables event observation
        # for the rest of the backend's life, so callers fall back rather than hang.
        self.supported: bool | None = None
        self._ready = asyncio.Event()

    async def wait_for_stop(self, container_id: str) -> str | None:
        """Wait until one container stops, or report that events are unavailable.

        Returns:
            str | None: The stop reason, or None when the daemon does not serve events and
                the caller must fall back to polling.
        """
        if self._closed:
            return None
        self._ensure_running()
        signal = self._signals.setdefault(container_id, asyncio.Event())
        await self._ready.wait()
        if self.supported is False:
            return None
        try:
            reason = self._stopped.get(container_id)
            if reason is None:
                await signal.wait()
                reason = self._stopped.get(container_id)
            return reason
        finally:
            if self._stopped.get(container_id) is not None:
                self._signals.pop(container_id, None)

    def forget(self, container_id: str) -> None:
        """
        Drop a destroyed container's observation state.
        """
        self._stopped.pop(container_id, None)
        self._signals.pop(container_id, None)
        self._oom.discard(container_id)

    def _ensure_running(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._observe())

    def _record(self, container_id: str, reason: str) -> None:
        self._stopped.setdefault(container_id, reason)
        signal = self._signals.get(container_id)
        if signal is not None:
            signal.set()

    async def _observe(self) -> None:
        delay = _RECONNECT_BASE_S
        while not self._closed:
            try:
                async for event in self._engine.events(since=self._last_event_at):
                    if self.supported is None:
                        self.supported = True
                        self._ready.set()
                    delay = _RECONNECT_BASE_S
                    self._handle(event)
            except asyncio.CancelledError:
                raise
            except Exception:
                if self.supported is None:
                    # The daemon never served a single event, so nothing will wait on this.
                    self.supported = False
                    self._ready.set()
                    psrl_logger.warning(
                        "Docker event stream is unavailable. Falling back to inspect polling for "
                        "container stop detection.",
                        exc_info=True,
                    )
                    return
                psrl_logger.warning(f"Docker event stream dropped. Reconnecting in {delay:g}s.", exc_info=True)
            if self._closed:
                return
            await asyncio.sleep(delay)
            delay = min(delay * 2, _RECONNECT_MAX_S)
            await self._resync()

    def _handle(self, event: Mapping[str, Any]) -> None:
        """
        Record one container event, preferring a proven OOM over the die that follows it.
        """
        container_id = str(event.get("id") or (event.get("Actor") or {}).get("ID") or "")
        action = str(event.get("Action") or event.get("status") or "")
        timestamp = event.get("timeNano")
        if timestamp is not None:
            self._last_event_at = int(timestamp) / 1e9
        elif event.get("time") is not None:
            self._last_event_at = float(event["time"])
        if not container_id:
            return
        if action in _OOM_ACTIONS:
            self._oom.add(container_id)
            return
        if action not in _STOP_ACTIONS:
            return
        if action == "destroy":
            self._record(container_id, "removed")
            return
        self._record(container_id, "oom_killed" if container_id in self._oom else "exited")

    async def _resync(self) -> None:
        """Inspect every container still being waited on after a stream gap.

        A daemon that restarted has no history to replay, so the replay request alone can
        miss a stop. Only containers with a live waiter are inspected, so this costs nothing
        on an idle node.
        """
        for container_id in [key for key, signal in self._signals.items() if not signal.is_set()]:
            try:
                inspection = await self._inspect(container_id)
            except Exception:
                continue
            if inspection is None:
                self._record(container_id, "removed")
                continue
            state = inspection.get("State") or {}
            if state.get("Running") is False:
                self._record(container_id, "oom_killed" if state.get("OOMKilled") else "exited")

    async def close(self) -> None:
        """
        Stop observing and release every waiter.
        """
        self._closed = True
        self._ready.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        for signal in list(self._signals.values()):
            signal.set()

    def snapshot(self) -> dict[str, Any]:
        """
        Report observation state for diagnostics.
        """
        return {
            "supported": self.supported,
            "watched": len(self._signals),
            "observed_stops": len(self._stopped),
            "last_event_at": self._last_event_at,
        }
