import json
import logging
from dataclasses import dataclass, field

_logger = logging.getLogger(__file__)

# Use the LMCache package connector rather than the copy bundled with vLLM,
# so both sides agree on the MP wire protocol.
MP_CONNECTOR_NAME = "LMCacheMPConnector"
MP_CONNECTOR_MODULE = "lmcache.integration.vllm.lmcache_mp_connector"

# Transfer engines the MP server's P2P subsystem registers.
SUPPORTED_TRANSFER_ENGINES = ("nixl", "mooncake_te")

# L1 eviction policies the MP server accepts.
SUPPORTED_EVICTION_POLICIES = ("LRU", "IsolatedLRU", "noop")

# Alignment the MP server accepts. RDMA reads want a larger power of two.
MIN_L1_ALIGN_BYTES = 4096
RECOMMENDED_P2P_ALIGN_BYTES = 65536


@dataclass
class LMCacheConfig:
    """
    Configure LMCache KV cache offloading for PSRL.

    LMCache runs as a standalone multiprocess (MP) server per node, and vLLM
    workers are its clients. `offload_size_gb` is that server's L1 capacity,
    so it covers every vLLM rank connected to it and is not divided.
    """

    # --- Core ---

    # Whether to enable LMCache KV cache offloading.
    enable: bool = False

    # --- L1 cache (MP server) ---

    # L1 capacity in GiB for one MP server. Shared by all local KV ranks.
    offload_size_gb: float = 100.0

    # Token chunk size for hash-based KV indexing.
    # Must match the connector's block-derived chunking on every rank.
    chunk_size: int = 256

    # Hash algorithm for chunk keys ("blake3" or "sha256").
    hash_algorithm: str = "blake3"

    # L1 allocation alignment in bytes. Larger alignment is required for
    # efficient RDMA reads when P2P is enabled.
    l1_align_bytes: int = 4096

    # L1 eviction policy: "LRU", "IsolatedLRU", or "noop".
    eviction_policy: str = "LRU"

    # --- L2 (optional local/remote tiers) ---

    # L2 adapters as JSON-serializable dicts, one per adapter.
    l2_adapters: list = field(default_factory=list)

    # L2 store and prefetch policies (LMCache adapter registry names).
    l2_store_policy: str = "default"
    l2_prefetch_policy: str = "default"

    # --- MP server runtime (endpoints allocated at launch) ---

    # Host the MP server binds for the worker message queue.
    server_host: str = "127.0.0.1"

    # Host the management HTTP server binds. Empty reuses `server_host`.
    # Set to a routable address when other instances call this server.
    http_host: str = ""

    # Host other instances use to reach this server's management HTTP API.
    # Empty reuses `http_host` then `server_host`.
    http_advertise_host: str = ""

    # ZMQ message-queue port. Zero means "allocate at launch".
    server_port: int = 0

    # Management HTTP port (clear/pin/prefetch). Zero means "allocate".
    http_port: int = 0

    # Instance identifier registered with the coordinator and advertised to
    # SMG. Set by the server actor before the MP server starts.
    lmcache_instance_id: str = ""

    # --- P2P transfer ---

    # Enable peer-to-peer KV sharing between instances.
    enable_p2p: bool = False

    # Node IP advertised for the RDMA transfer channel.
    p2p_advertise_host: str = ""

    # Transfer-channel server port. Zero means "allocate at launch".
    p2p_transfer_port: int = 0

    # Transfer engine: "nixl" (default) or "mooncake_te".
    p2p_transfer_engine: str = "nixl"

    # P2P lookup and load timeouts in seconds.
    p2p_lookup_timeout_s: float = 30.0
    p2p_load_timeout_s: float = 30.0

    # --- Coordinator ---

    # Host where the MP coordinator runs (defaults to ps_manager_ip).
    coordinator_host: str = ""

    # Coordinator HTTP port.
    coordinator_port: int = 9300

    # Coordinator startup timeout in seconds for contended manager nodes.
    coordinator_health_timeout_s: int = 300

    # Report cache events to the coordinator. Required for fleet-level
    # views and for the LMCache-tier KV event stream.
    coordinator_event_reporting: bool = False

    # --- Connector ---

    # Timeout in seconds for MP message-queue requests.
    mq_timeout_s: int = 300

    # Interval in seconds between connector heartbeats to the MP server.
    heartbeat_interval_s: int = 10

    # Submit the LMCache lookup as soon as a request is scheduled.
    eager_prefetch: bool = False

    # --- Cache behaviour ---

    # Clear the LMCache backend after a model weight update (not with P2P).
    clear_on_weight_update: bool = False

    # Tag KV entries with the model version so weights never collide.
    # Required with P2P, where the tier cannot be cleared on weight sync.
    multi_version_kv: bool = True

    # Publish KV cache events so routing can score the off-GPU cache tier.
    enable_kv_events: bool = False

    # --- GPU pin budget (PSRL-side) ---

    # Maximum simultaneously pinned GPU KV blocks. Zero disables the limit.
    gpu_pin_block_budget: int = 0

    # --- Derived configuration ---

    @property
    def resolved_http_host(self) -> str:
        """Host the management HTTP server binds to."""
        return self.http_host or self.server_host

    @property
    def resolved_http_advertise_host(self) -> str:
        """Host other instances use to reach the management HTTP server."""
        return self.http_advertise_host or self.resolved_http_host

    @property
    def http_base_url(self) -> str:
        """Base URL of this node's management HTTP API."""
        return f"http://{self.resolved_http_advertise_host}:{self.http_port}"

    @property
    def event_stream_url(self) -> str:
        """Endpoint consumers subscribe to for the LMCache-tier event stream."""
        return f"{self.http_base_url}/cache/events/stream"

    def to_connector_extra_config(self) -> dict:
        """
        Build the vLLM connector extra config for the MP client.

        Returns:
            dict: Keys consumed by `LMCacheMPConnector` via
                `kv_transfer_config.kv_connector_extra_config`.
        """
        return {
            "lmcache.mp.host": self.server_host,
            "lmcache.mp.port": self.server_port,
            "lmcache.mp.mq_timeout": self.mq_timeout_s,
            "lmcache.mp.heartbeat_interval": self.heartbeat_interval_s,
            "lmcache.mp.eager_prefetch": self.eager_prefetch,
        }

    def to_engine_kwargs(self) -> dict:
        """
        Translate this config into vLLM engine kwargs.

        Returns:
            dict: Key-value pairs to merge into `llm_kwargs`.
        """
        if not self.enable:
            return {}

        assert self.server_port > 0, (
            "LMCache MP server port is unset. Allocate runtime ports before building engine kwargs."
        )

        return {
            "kv_transfer_config": {
                "kv_connector": MP_CONNECTOR_NAME,
                "kv_connector_module_path": MP_CONNECTOR_MODULE,
                "kv_role": "kv_both",
                "kv_connector_extra_config": self.to_connector_extra_config(),
            },
            # LMCacheMPConnector does not implement HMA in this version.
            "disable_hybrid_kv_cache_manager": True,
        }

    def to_server_argv(self) -> list[str]:
        """
        Build the `lmcache server` argv for one node's MP server.

        Runtime endpoints (`server_port`, `http_port`) must be allocated
        before calling this.

        Returns:
            list[str]: Arguments excluding the `lmcache server` prefix.
        """
        assert self.server_port > 0, "MP server port must be allocated before launch."
        assert self.http_port > 0, "MP server HTTP port must be allocated before launch."
        assert self.eviction_policy in SUPPORTED_EVICTION_POLICIES, (
            f"eviction_policy must be one of {SUPPORTED_EVICTION_POLICIES}, got {self.eviction_policy!r}."
        )
        assert self.l1_align_bytes >= MIN_L1_ALIGN_BYTES, (
            f"l1_align_bytes must be at least {MIN_L1_ALIGN_BYTES}, got {self.l1_align_bytes}."
        )
        if self.enable_p2p:
            assert self.p2p_transfer_engine in SUPPORTED_TRANSFER_ENGINES, (
                f"p2p_transfer_engine must be one of {SUPPORTED_TRANSFER_ENGINES}, got {self.p2p_transfer_engine!r}."
            )
            if self.l1_align_bytes < RECOMMENDED_P2P_ALIGN_BYTES:
                _logger.warning(
                    "[LMCache] l1_align_bytes=%d is below the %d recommended for "
                    "P2P RDMA reads. Transfers still work but are less efficient.",
                    self.l1_align_bytes,
                    RECOMMENDED_P2P_ALIGN_BYTES,
                )

        argv = [
            "--l1-size-gb",
            str(self.offload_size_gb),
            "--host",
            self.server_host,
            "--port",
            str(self.server_port),
            "--http-host",
            self.resolved_http_host,
            "--http-port",
            str(self.http_port),
            "--chunk-size",
            str(self.chunk_size),
            "--hash-algorithm",
            self.hash_algorithm,
            "--l1-align-bytes",
            str(self.l1_align_bytes),
            "--eviction-policy",
            self.eviction_policy,
        ]

        if self.lmcache_instance_id:
            argv += ["--instance-id", self.lmcache_instance_id]

        if self.l2_adapters:
            for adapter in self.l2_adapters:
                argv += ["--l2-adapter", _to_json(adapter)]
            argv += ["--l2-store-policy", self.l2_store_policy]
            argv += ["--l2-prefetch-policy", self.l2_prefetch_policy]

        if self.enable_p2p:
            assert self.p2p_advertise_host, "p2p_advertise_host is required when enable_p2p is set."
            assert self.p2p_transfer_port > 0, "p2p_transfer_port must be allocated when enable_p2p is set."
            assert self.coordinator_host, "coordinator_host is required when enable_p2p is set."
            transfer_url = f"{self.p2p_advertise_host}:{self.p2p_transfer_port}"
            argv += [
                "--p2p-advertise-url",
                transfer_url,
                "--p2p-listen-url",
                transfer_url,
                "--p2p-transfer-engine",
                self.p2p_transfer_engine,
                "--p2p-lookup-timeout",
                str(self.p2p_lookup_timeout_s),
                "--p2p-load-timeout",
                str(self.p2p_load_timeout_s),
            ]

        if self.coordinator_host:
            argv += [
                "--coordinator-url",
                f"http://{self.coordinator_host}:{self.coordinator_port}",
            ]

        if self.coordinator_event_reporting:
            argv += ["--coordinator-event-reporting"]
            if self.enable_kv_events:
                argv += [
                    "--coordinator-event-stream-enable",
                    "--coordinator-metadata",
                    _to_json({"event_stream_url": self.event_stream_url}),
                ]

        return argv

    def to_coordinator_argv(self) -> list[str]:
        """
        Build the `lmcache coordinator` argv for the cluster-wide coordinator.

        Returns:
            list[str]: Arguments excluding the `lmcache coordinator` prefix.
        """
        assert self.coordinator_host, "coordinator_host must be set before launch."
        return [
            "--host",
            "0.0.0.0",
            "--port",
            str(self.coordinator_port),
        ]


def _to_json(value) -> str:
    """
    Serialize an L2 adapter spec to the JSON string the CLI expects.

    Args:
        value: Adapter spec as a mapping or an already-serialized string.

    Returns:
        str: JSON text.
    """
    return value if isinstance(value, str) else json.dumps(value)
