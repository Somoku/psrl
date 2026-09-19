import asyncio
import logging
import time
from collections import deque

import aiohttp

from psrl.utils.kv_cache.config import LMCacheConfig
from psrl.utils.kv_cache.runtime import LMCacheMPRuntime

psrl_logger = logging.getLogger(__file__)

# Per-request timeout for management calls to the MP server.
_MANAGEMENT_TIMEOUT_S = 30.0

# How long a peer-registry snapshot is reused before re-querying the coordinator.
_PEER_REGISTRY_TTL_S = 5.0

# Destination-side warm-prefetch polling.
_PREFETCH_POLL_INTERVAL_S = 0.2
_PREFETCH_TIMEOUT_S = 60.0

# Legacy in-process backend label SMG sends. MP transfer is tier-agnostic.
_LEGACY_BACKEND = "LocalCPUBackend"


class KVCacheManager:
    """
    Manage PSRL KV cache operations against the LMCache MP server.

    The MP server owns all cache state. This manager owns the node-local
    server process, the management HTTP client, and the GPU pin budget.
    """

    def __init__(self, config: LMCacheConfig) -> None:
        """
        Initialize the KV cache manager.

        Args:
            config (LMCacheConfig): The resolved LMCache configuration.
        """
        self.config = config
        self._gpu_pin_budget: int = config.gpu_pin_block_budget
        self._pinned_gpu_blocks: int = 0
        self._gpu_pinned_order: deque[list[int]] = deque()

        self.current_version: int = 0

        # Parallel geometry used when the server must resolve token sequences
        # to the same per-rank keys the connector stored them under.
        self.model_name: str = ""
        self.kv_world_size: int = 1

        # Token sequences pinned on the MP server, keyed by token tuple. The
        # stored body lets unpin reuse the same identity tags the pin used.
        self._pinned_backend: dict[tuple[int, ...], dict] = {}

        self._inference_engine = None
        self._runtime: LMCacheMPRuntime | None = None
        self._http: aiohttp.ClientSession | None = None

        # Maps a peer instance_id to its MP HTTP base URL. The list shape
        # matches the peer-registry contract SMG already expects.
        self.peer_registry: dict[str, list[str]] = {}
        self._peer_registry_refreshed_at: float = 0.0

        # Last transfer failure reason, surfaced by the SMG servicer.
        self._last_transfer_error: str = ""

        self._log_init_status()

    # --- Initialization helpers ---

    def _log_init_status(self) -> None:
        """
        Log LMCache initialization status and parameters at INFO level.
        """
        if not self.config.enable:
            psrl_logger.info("[LMCache] KV cache offloading is DISABLED.")
            return

        psrl_logger.info("[LMCache] KV cache offloading is ENABLED with the following parameters:")
        psrl_logger.info(f"offload_size_gb = {self.config.offload_size_gb}")
        psrl_logger.info(f"chunk_size = {self.config.chunk_size}")
        psrl_logger.info(f"hash_algorithm = {self.config.hash_algorithm!r}")
        psrl_logger.info(f"eviction_policy = {self.config.eviction_policy!r}")
        psrl_logger.info(f"clear_on_weight_update = {self.config.clear_on_weight_update}")
        psrl_logger.info(f"multi_version_kv = {self.config.multi_version_kv}")
        psrl_logger.info(f"gpu_pin_block_budget = {self.config.gpu_pin_block_budget}")
        if self.config.enable_p2p:
            psrl_logger.info(f"enable_p2p = {self.config.enable_p2p}")
            psrl_logger.info(f"p2p_transfer_engine = {self.config.p2p_transfer_engine!r}")
        if self.l2_adapters_enabled:
            psrl_logger.info(f"l2_adapters = {len(self.config.l2_adapters)} configured")
        self._verify_lmcache_importable()

    def _verify_lmcache_importable(self) -> None:
        """
        Verify that the lmcache package is importable and log its version.
        """
        try:
            import lmcache  # type: ignore[import-untyped]

            version = getattr(lmcache, "__version__", "unknown")
            psrl_logger.info(f"[LMCache] lmcache package is importable, version={version!r}.")
        except ImportError:
            psrl_logger.error(
                "[LMCache] lmcache package is NOT importable! "
                "KV cache offloading will NOT work. "
                "Run `bash scripts/install_lmcache.sh` to install it."
            )

    @property
    def l2_adapters_enabled(self) -> bool:
        """Whether any L2 adapter is configured on the MP server."""
        return bool(self.config.l2_adapters)

    # --- Runtime attachment ---

    def attach_runtime(self, runtime: LMCacheMPRuntime) -> None:
        """
        Attach the node-local MP server runtime.

        Args:
            runtime (LMCacheMPRuntime): The started server runtime whose
                HTTP endpoint serves management calls.
        """
        self._runtime = runtime
        psrl_logger.info(f"[LMCache] KVCacheManager: MP runtime attached at {runtime.base_url}.")

    def attach_engine(self, inference_engine) -> None:
        """
        Attach the vLLM AsyncLLM engine after it has been initialised.

        The engine is only needed for GPU block pinning, which runs in the
        EngineCore process rather than on the MP server.

        Args:
            inference_engine: The `AsyncLLM` (or compatible) engine object whose
                `engine_core.call_utility_async` dispatches to the scheduler.
        """
        self._inference_engine = inference_engine
        psrl_logger.info("[LMCache] KVCacheManager: Engine attached.")

    def set_current_version(self, version: int) -> None:
        """
        Set the current model version used to tag new KV cache entries.

        Called locally by the gen server after this replica completes a weight
        pull and reads back its actual model version. The version is stamped
        onto each request's `lmcache.tag.model_version`, so it must be
        committed only after this replica's weights are current.

        Args:
            version (int): The new model version number (monotonically increasing).
        """
        self.current_version = version
        psrl_logger.info(f"[LMCache] current_version set to {version}.")

    def set_parallel_geometry(self, model_name: str, kv_world_size: int) -> None:
        """
        Record the KV parallelism needed to address this replica's cached keys.

        The MP server resolves token sequences to per-rank object keys, so
        callers must supply the same rank fan-out the connector used. A wrong
        value degrades to "no chunks found" (the pin or transfer is a no-op)
        rather than corrupting cache state.

        Args:
            model_name (str): Model identifier used in cache keys.
            kv_world_size (int): Number of KV pieces one token chunk's cache is
                split into across this replica's MP servers.
        """
        assert kv_world_size >= 1, f"kv_world_size must be >= 1 (got {kv_world_size})."
        self.model_name = model_name
        self.kv_world_size = kv_world_size
        psrl_logger.info(f"[LMCache] Parallel geometry set: model_name={model_name!r}, kv_world_size={kv_world_size}.")

    @property
    def is_attached(self) -> bool:
        """Whether the inference engine has been attached via `attach_engine`."""
        return self._inference_engine is not None

    @property
    def enabled(self) -> bool:
        """Whether LMCache offloading is enabled."""
        return self.config.enable

    @property
    def should_clear_on_weight_update(self) -> bool:
        """Whether to clear the LMCache backend on model weight updates from PS."""
        return self.config.enable and self.config.clear_on_weight_update

    def get_engine_kwargs(self) -> dict:
        """
        Get vLLM engine kwargs for LMCache integration.

        Returns:
            dict: Key-value pairs to merge into vLLM engine arguments.
        """
        kwargs = self.config.to_engine_kwargs()
        if kwargs:
            psrl_logger.info("[LMCache] Injecting engine kwargs into vLLM.")
        else:
            psrl_logger.info("[LMCache] No engine kwargs to inject (LMCache disabled).")
        return kwargs

    # --- HTTP plumbing ---

    def _assert_runtime(self) -> LMCacheMPRuntime:
        """
        Return the attached runtime or fail loudly.

        Returns:
            LMCacheMPRuntime: The attached runtime.
        """
        assert self._runtime is not None, (
            "LMCache MP runtime is not attached. Call attach_runtime() before cache operations."
        )
        return self._runtime

    @property
    def coordinator_url(self) -> str:
        """Base URL of the shared MP coordinator."""
        return f"http://{self.config.coordinator_host}:{self.config.coordinator_port}"

    async def _session(self) -> aiohttp.ClientSession:
        """
        Return the lazily created HTTP client for management calls.

        Returns:
            aiohttp.ClientSession: Shared client session.
        """
        if self._http is None or self._http.closed:
            timeout = aiohttp.ClientTimeout(total=_MANAGEMENT_TIMEOUT_S)
            self._http = aiohttp.ClientSession(timeout=timeout)
        return self._http

    async def _request_url(self, method: str, url: str, payload: dict | None = None) -> dict:
        """
        Issue one management request to an absolute URL.

        Args:
            method (str): HTTP method.
            url (str): Absolute URL including path.
            payload (dict | None): JSON body, when the method accepts one.

        Returns:
            dict: Decoded JSON response, or an empty dict for empty bodies.

        Raises:
            aiohttp.ClientResponseError: If the server returned a non-2xx status.
        """
        session = await self._session()
        async with session.request(method, url, json=payload) as resp:
            resp.raise_for_status()
            if resp.content_length == 0:
                return {}
            return await resp.json(content_type=None)

    async def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        """
        Issue one management request to this node's MP server.

        Args:
            method (str): HTTP method.
            path (str): Path on the MP server, beginning with a slash.
            payload (dict | None): JSON body, when the method accepts one.

        Returns:
            dict: Decoded JSON response, or an empty dict for empty bodies.
        """
        runtime = self._assert_runtime()
        return await self._request_url(method, f"{runtime.base_url}{path}", payload)

    async def close(self) -> None:
        """Release the HTTP client and stop the node-local MP server."""
        if self._http is not None and not self._http.closed:
            await self._http.close()
        self._http = None
        if self._runtime is not None:
            self._runtime.stop()
            self._runtime = None

    # --- Peer discovery ---

    def set_peer_registry(self, registry: dict[str, list[str]]) -> None:
        """
        Merge peer entries into the local registry.

        Kept for SMG's existing contract: it seeds a destination entry when the
        destination is unknown, and the authoritative entries are refreshed
        from the coordinator.

        Args:
            registry (dict[str, list[str]]): Maps instance_id to a list whose
                first element is that instance's MP HTTP base URL.
        """
        self.peer_registry.update(registry)
        psrl_logger.info(
            f"[LMCache] Peer registry updated ({len(registry)} entries this call, {len(self.peer_registry)} total)."
        )

    async def refresh_peer_registry(self, max_age_s: float = _PEER_REGISTRY_TTL_S) -> None:
        """
        Refresh instance endpoints from the coordinator's instance registry.

        Args:
            max_age_s (float): Reuse the cached registry when it was refreshed
                more recently than this many seconds.
        """
        if not self.config.coordinator_host:
            return
        now = time.monotonic()
        if self.peer_registry and now - self._peer_registry_refreshed_at < max_age_s:
            return
        try:
            resp = await self._request_url("GET", f"{self.coordinator_url}/instances")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            psrl_logger.warning(f"[LMCache] Failed to refresh peer registry: {e}")
            return

        instances = resp.get("instances", []) if isinstance(resp, dict) else resp
        refreshed: dict[str, list[str]] = {}
        for instance in instances or []:
            instance_id = instance.get("instance_id")
            ip = instance.get("ip")
            http_port = instance.get("http_port")
            if instance_id and ip and http_port:
                refreshed[instance_id] = [f"http://{ip}:{http_port}"]
        if refreshed:
            self.peer_registry.update(refreshed)
        self._peer_registry_refreshed_at = now
        psrl_logger.debug(f"[LMCache] Peer registry refreshed: {len(refreshed)} instances registered.")

    # --- Cache operations ---

    async def clear(self) -> None:
        """
        Clear all cached KV on this node's MP server.

        Called after a model weight update to drop stale entries when
        `clear_on_weight_update` is enabled.
        """
        if not self.config.enable:
            return
        await self._request("POST", "/cache/clear")
        psrl_logger.info("[LMCache] Cleared MP server cache for this node.")

    async def pin(self, tokens: list[int], targets: list[str]) -> bool:
        """
        Pin the cached prefix of a trajectory to prevent LRU eviction.

        Supported targets: `"gpu"` (vLLM block pool) and `"backend"` (LMCache).
        GPU pinning is subject to `gpu_pin_block_budget`. If the budget is
        exceeded, the oldest-pinned entry is unpinned first (PSRL-side LRU).

        Args:
            tokens (list[int]): Full token sequence for the trajectory.
            targets (list[str]): Subset of `["gpu", "backend"]`.

        Returns:
            bool: True if all requested pin operations succeeded.
        """
        self._assert_engine()
        assert tokens, "tokens must be a non-empty list."
        assert targets, "targets must be a non-empty list."
        assert all(t in ("gpu", "backend") for t in targets), (
            f"Invalid pin targets: {targets!r}. Must be a subset of ['gpu', 'backend']."
        )
        ok = True
        if "gpu" in targets:
            ok = ok and await self._pin_gpu(tokens)
        if "backend" in targets:
            ok = ok and await self._pin_backend(tokens)
        return ok

    async def unpin(self, tokens: list[int], targets: list[str]) -> bool:
        """
        Unpin the cached prefix of a trajectory, allowing LRU eviction.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.
            targets (list[str]): Subset of `["gpu", "backend"]`.

        Returns:
            bool: True if all unpin operations completed without error.
        """
        self._assert_engine()
        assert tokens, "tokens must be a non-empty list."
        assert targets, "targets must be a non-empty list."
        assert all(t in ("gpu", "backend") for t in targets), (
            f"Invalid unpin targets: {targets!r}. Must be a subset of ['gpu', 'backend']."
        )
        ok = True
        if "gpu" in targets:
            ok = ok and await self._unpin_gpu(tokens)
        if "backend" in targets:
            ok = ok and await self._unpin_backend(tokens)
        return ok

    async def transfer_direct(
        self,
        tokens: list[int],
        src: tuple[str, str],
        dst: tuple[str, str],
        copy: bool = False,
        dst_model_version: int = -1,
    ) -> bool:
        """
        Transfer a cached prefix to another rollout instance.

        MP transfer is a pull: this method asks the destination's MP server to
        warm its own L1 from whichever peer holds the prefix. The source is
        therefore never targeted, and a partial result leaves the destination to
        re-prefill the remainder.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.
            src (tuple[str, str]): Source `(lmcache_instance_id, backend)`.
            dst (tuple[str, str]): Destination `(lmcache_instance_id, backend)`.
            copy (bool): If False, best-effort delete the prefix from the source
                once the destination confirms it. The delete is refused while
                the source prefix is pinned.
            dst_model_version (int): Model version to resolve the prefix under at
                the destination. -1 means version-agnostic (no tag).

        Returns:
            bool: True if the destination acquired the whole prefix.
        """
        self._last_transfer_error = ""
        if not self.config.enable_p2p:
            self._last_transfer_error = "enable_p2p is False"
            psrl_logger.warning("[LMCache] transfer_direct() called but enable_p2p is False.")
            return False
        assert tokens, "tokens must be a non-empty list."
        assert self.model_name, "LMCache model_name is unset. Call set_parallel_geometry() before transfers."

        dst_instance_id = dst[0]
        await self.refresh_peer_registry()
        dst_urls = self.peer_registry.get(dst_instance_id)
        if not dst_urls or not dst_urls[0]:
            self._last_transfer_error = f"no MP endpoint known for destination {dst_instance_id!r}"
            psrl_logger.warning(f"[LMCache] {self._last_transfer_error}. The destination will re-prefill.")
            return False

        for label, backend in (("src", src[1]), ("dst", dst[1])):
            if backend and backend != _LEGACY_BACKEND:
                psrl_logger.warning(f"[LMCache] Ignoring {label}_backend={backend!r}: MP transfer is tier-agnostic.")

        body = {
            "model_name": self.model_name,
            "world_size": self.kv_world_size,
            "token_ids": tokens,
            "cache_salt": "",
            "source_tier": "l2",
            "target_tier": "l1",
            "request_configs": (
                {"lmcache.tag.model_version": str(dst_model_version)} if dst_model_version >= 0 else None
            ),
        }
        dst_base = dst_urls[0]

        try:
            submitted = await self._request_url("POST", f"{dst_base}/cache/prefetches", body)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            self._last_transfer_error = f"destination prefetch submit failed: {e}"
            psrl_logger.warning(f"[LMCache] {self._last_transfer_error}. The destination will re-prefill.")
            return False

        chunks = int(submitted.get("chunks", 0))
        request_id = submitted.get("request_id")
        if not request_id or chunks == 0:
            self._last_transfer_error = "token sequence is shorter than one chunk"
            return False

        found = await self._await_prefetch(str(dst_base), str(request_id))
        if found < chunks:
            self._last_transfer_error = (
                f"destination acquired {found}/{chunks} chunks. It will re-prefill the remainder"
            )
            psrl_logger.info(f"[LMCache] {self._last_transfer_error}")
            return False

        if not copy:
            await self._delete_at_source(src[0], body)
        psrl_logger.debug(f"[LMCache] Transferred {chunks} chunks to {dst_instance_id!r} (copy={copy}).")
        return True

    async def _await_prefetch(self, dst_base: str, request_id: str) -> int:
        """
        Poll a destination warm prefetch until it completes or times out.

        Args:
            dst_base (str): Destination MP server base URL.
            request_id (str): Prefetch request id returned by the destination.

        Returns:
            int: Number of chunks loaded into the destination's L1, or 0 on
            timeout or error.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _PREFETCH_TIMEOUT_S
        while loop.time() < deadline:
            try:
                status = await self._request_url("GET", f"{dst_base}/cache/prefetches/{request_id}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                psrl_logger.warning(f"[LMCache] Destination prefetch poll failed: {e}")
                return 0
            state = status.get("status")
            if state == "completed":
                return int(status.get("found_keys", 0))
            if state not in ("pending", "submitted"):
                psrl_logger.warning(f"[LMCache] Unexpected prefetch status {state!r}.")
                return 0
            await asyncio.sleep(_PREFETCH_POLL_INTERVAL_S)
        psrl_logger.warning(
            f"[LMCache] Destination prefetch {request_id!r} timed out after {_PREFETCH_TIMEOUT_S:.0f}s."
        )
        return 0

    async def _delete_at_source(self, src_instance_id: str, body: dict) -> None:
        """
        Best-effort delete of a transferred prefix from the source instance.

        The coordinator resolves the token sequence and forwards the delete to
        the named instance. Pinned or locked keys are refused by the node, so a
        source that is still pinned keeps its copy.

        Args:
            src_instance_id (str): Instance to delete from.
            body (dict): Prefetch body whose identity fields are reused.
        """
        if not src_instance_id or not self.config.coordinator_host:
            return
        request = {
            "instance_id": src_instance_id,
            "model_name": body["model_name"],
            "world_size": body["world_size"],
            "token_ids": body["token_ids"],
            "cache_salt": body["cache_salt"],
            "request_configs": body["request_configs"],
            "tier": "l1",
            # Pins are an explicit client contract, so do not bypass them.
            "force": False,
        }
        try:
            result = await self._request_url("POST", f"{self.coordinator_url}/cache/delete", request)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            psrl_logger.warning(f"[LMCache] Source delete after transfer failed (best effort, non-atomic): {e}")
            return
        skipped = int(result.get("skipped", 0))
        if skipped:
            psrl_logger.info(
                f"[LMCache] Source delete skipped {skipped} keys (locked or pinned). The source keeps its copy."
            )

    # --- Private helpers ---

    def _assert_engine(self) -> None:
        assert self._inference_engine is not None, (
            "KVCacheManager inference engine is not attached. Call attach_engine() after the rollout is initialised."
        )

    async def _utility(self, method: str, *args) -> object:
        """
        Call a `psrl_*` method on the vLLM `EngineCore` via `call_utility_async`.

        GPU block-pool operations must run in the EngineCore process, next to
        `block_pool`, so they are dispatched there rather than to a worker.
        This is safe for TP>1 because the state is never copied across
        process boundaries.

        Args:
            method (str): The `RolloutScheduler` method name to call on EngineCore.
            *args: Positional arguments forwarded to the method.

        Returns:
            object: The return value of the called method.
        """
        return await self._inference_engine.engine_core.call_utility_async(method, *args)

    # --- Backend pin internals ---

    def _l1_pin_body(self, tokens: list[int]) -> dict:
        """
        Build the L1 pin/unpin body for one token sequence.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.

        Returns:
            dict: Request body for `POST`/`DELETE /cache/l1/pins`.
        """
        assert self.model_name, (
            "LMCache model_name is unset. Call set_parallel_geometry() before backend pin operations."
        )
        request_configs = (
            {"lmcache.tag.model_version": str(self.current_version)} if self.config.multi_version_kv else None
        )
        return {
            "model_name": self.model_name,
            "world_size": self.kv_world_size,
            "token_ids": tokens,
            "cache_salt": "",
            "request_configs": request_configs,
        }

    async def _pin_backend(self, tokens: list[int]) -> bool:
        """
        Pin the trajectory's cached chunks in the MP server's L1.

        Chunks that are not resident are reported by the server as missing and
        are not treated as an error, so a pinned prefix shorter than `tokens`
        behaves the same as before. Only a pin that retained at least one chunk
        is remembered, so a later pin can retry chunks that were not stored yet.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.

        Returns:
            bool: True if the request succeeded.
        """
        key = tuple(tokens)
        if key in self._pinned_backend:
            # Repeated pins must not increment the server's pin count.
            return True

        if not self.config.enable:
            return True

        body = self._l1_pin_body(tokens)
        result = await self._request("POST", "/cache/l1/pins", body)
        pinned = int(result.get("pinned", 0))
        if pinned > 0:
            self._pinned_backend[key] = body
        elif int(result.get("missing", 0)) > 0:
            psrl_logger.debug(
                f"[LMCache] Backend pin found no resident chunks for {len(tokens)} tokens. "
                "the prefix may not have been stored yet."
            )
        return True

    async def _unpin_backend(self, tokens: list[int]) -> bool:
        """
        Release the trajectory's L1 pins on the MP server.

        Unpinning reuses the body recorded at pin time so the release happens
        under the same identity tags even if the model version advanced.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.

        Returns:
            bool: True if the request succeeded.
        """
        body = self._pinned_backend.pop(tuple(tokens), None)
        if body is None:
            psrl_logger.debug("[LMCache] Backend unpin skipped: sequence was not pinned.")
            return True
        await self._request("DELETE", "/cache/l1/pins", body)
        return True

    # --- GPU pin budget internals ---

    async def _pin_gpu(self, tokens: list[int]) -> bool:
        """
        Pin GPU prefix-cache blocks for `tokens`, enforcing the budget.

        `psrl_pin_gpu` returns the number of blocks actually pinned by PSRL.
        Budget enforcement uses that authoritative count, avoiding any separate
        cache-info query path.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.

        Returns:
            bool: True if the pin succeeded, False if the budget cannot accommodate it.
        """
        pinned: int = await self._utility("psrl_pin_gpu", tokens)
        if pinned <= 0:
            psrl_logger.debug("[LMCache] GPU pin: no matching prefix-cache blocks pinned.")
            return True

        self._pinned_gpu_blocks += pinned
        self._gpu_pinned_order.append(tokens)

        if self._gpu_pin_budget > 0:
            # Evict oldest-pinned trajectories until the budget is satisfied.
            newest_evicted = False
            while self._pinned_gpu_blocks > self._gpu_pin_budget and self._gpu_pinned_order:
                oldest_tokens = self._gpu_pinned_order.popleft()
                if oldest_tokens is tokens:
                    newest_evicted = True
                freed: int = await self._utility("psrl_unpin_gpu", oldest_tokens)
                self._pinned_gpu_blocks = max(0, self._pinned_gpu_blocks - freed)
                psrl_logger.debug(
                    f"[LMCache] GPU pin budget: evicted oldest trajectory "
                    f"({freed} blocks freed, budget={self._gpu_pin_budget})."
                )

            # If the newest trajectory alone exceeds the budget, the eviction
            # loop eventually unpins it. Report the budget miss to the caller.
            if newest_evicted:
                psrl_logger.warning(
                    f"[LMCache] GPU pin budget exceeded after pinning {pinned} blocks (budget={self._gpu_pin_budget})."
                )
                return False

        psrl_logger.debug(
            f"[LMCache] GPU pin: {pinned} blocks pinned, "
            f"total={self._pinned_gpu_blocks}, budget={self._gpu_pin_budget}."
        )
        return True

    async def _unpin_gpu(self, tokens: list[int]) -> bool:
        """
        Unpin GPU prefix-cache blocks for `tokens`.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.

        Returns:
            bool: True if the unpin succeeded.
        """
        freed: int = await self._utility("psrl_unpin_gpu", tokens)
        self._pinned_gpu_blocks = max(0, self._pinned_gpu_blocks - freed)
        # Remove from order tracking (`deque` does not support arbitrary removal,
        # so rebuild without the evicted entry).
        self._gpu_pinned_order = deque(t for t in self._gpu_pinned_order if t != tokens)
        psrl_logger.debug(f"[LMCache] GPU unpin: {freed} blocks freed, total={self._pinned_gpu_blocks}.")
        return True
