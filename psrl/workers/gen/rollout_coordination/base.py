"""CoordinatorBase: shared low-level helpers used by all RolloutCoordinator mixins."""

from __future__ import annotations

import json
import logging
import os

from psrl.workers.gen.smg_adapter import (
    ROUTING_LOOP_STATUS_PATH,
    WORKERS_STATS_PATH,
    WORKERS_UPDATE_WEIGHT_VERSION_PATH,
)

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class CoordinatorBase:
    """Shared HTTP gateway helpers and routing-loop control.

    All ``self.*`` attributes are initialized by ``RolloutCoordinator.__init__``;
    this class has no ``__init__`` of its own.
    """

    async def _gateway_post_json(self, path: str, payload, params: dict | None = None):
        if self.gateway_base_url is None:
            raise RuntimeError("Rust gateway base url is not initialized")
        url = f"{self.gateway_base_url}{path}"
        async with self.gateway_client.post(url, json=payload, params=params) as resp:
            resp.raise_for_status()
            text = await resp.text()
            if not text.strip():
                return {}
            return json.loads(text)

    async def _gateway_get_json(self, path: str, params: dict | None = None):
        if self.gateway_base_url is None:
            raise RuntimeError("Rust gateway base url is not initialized")
        url = f"{self.gateway_base_url}{path}"
        psrl_logger.debug(f"Making GET request to {url} with params {params}")
        async with self.gateway_client.get(url, params=params) as resp:
            resp.raise_for_status()
            text = await resp.text()
            if not text.strip():
                return {}
            return json.loads(text)

    async def _set_routing_loop_running(self, running: bool):
        path = "/routing_loop/resume" if running else "/routing_loop/pause"
        # When pausing, pass wait=true so the call only returns after the routing loop
        # drains in-flight worker selection (selecting becomes false). This prevents
        # a race where update_currently_syncing_instances and the SYNC command are issued
        # while the loop is still assigning requests using the pre-sync version.
        params = {"wait": "true"} if not running else {}
        psrl_logger.info(f"Setting routing loop running state to {running} via {path}")
        data = await self._gateway_post_json(path, payload={}, params=params)
        expected_paused = not running
        actual_paused = data.get("paused")
        if actual_paused is not None and bool(actual_paused) != expected_paused:
            raise RuntimeError(
                f"Unexpected routing loop pause state after {path}: expected={expected_paused}, got={actual_paused}"
            )
        if not running and bool(data.get("selecting", False)):
            raise RuntimeError("Routing loop is still selecting after pause wait")
        if not running and int(data.get("active_dispatch_handoffs", 0)) != 0:
            raise RuntimeError("Routing loop still has active dispatch handoffs after pause wait")

    async def _is_selecting(self) -> bool:
        """Check if the router is currently selecting a worker for queued requests.

        Returns:
            bool: True if any worker-selection stage is active, False otherwise.
        """
        if self.use_rust_gateway:
            data = await self._gateway_get_json("/routing_loop/status")
            is_selecting = bool(data.get("selecting", False))
        else:
            is_selecting = await self.rollout_router.is_routing.remote()
        return is_selecting
