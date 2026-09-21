import json

import pytest
from psrl.utils.kv_cache.config import (
    MP_CONNECTOR_MODULE,
    MP_CONNECTOR_NAME,
    LMCacheConfig,
)
from psrl.utils.kv_cache.manager import KVCacheManager


def _runtime_config(**overrides) -> LMCacheConfig:
    """Build a config whose runtime endpoints are already allocated."""
    base = dict(
        enable=True,
        server_host="127.0.0.1",
        server_port=5555,
        http_port=8080,
        offload_size_gb=20.0,
        chunk_size=128,
    )
    base.update(overrides)
    return LMCacheConfig(**base)


class TestLMCacheConfig:
    def test_disabled_by_default(self):
        config = LMCacheConfig()
        assert config.enable is False
        assert config.to_engine_kwargs() == {}

    # --- to_engine_kwargs ---

    def test_engine_kwargs_requires_allocated_port(self):
        config = LMCacheConfig(enable=True)
        with pytest.raises(AssertionError):
            config.to_engine_kwargs()

    def test_engine_kwargs_use_mp_connector(self):
        config = LMCacheConfig(enable=True, server_port=5555)
        kwargs = config.to_engine_kwargs()
        transfer = kwargs["kv_transfer_config"]
        assert transfer["kv_connector"] == MP_CONNECTOR_NAME
        assert transfer["kv_connector_module_path"] == MP_CONNECTOR_MODULE
        assert transfer["kv_role"] == "kv_both"
        assert transfer["kv_connector_extra_config"]["lmcache.mp.port"] == 5555

    def test_engine_kwargs_leave_hma_to_vllm(self):
        """LMCacheMPConnector declares SupportsHMA, so it must not be forced off."""
        kwargs = LMCacheConfig(enable=True, server_port=5555).to_engine_kwargs()
        assert "disable_hybrid_kv_cache_manager" not in kwargs

    def test_engine_kwargs_single_server_by_default(self):
        config = LMCacheConfig(enable=True, server_port=5555)
        assert config.n_servers == 1
        assert "lmcache.mp.server_urls" not in config.to_connector_extra_config()

    def test_engine_kwargs_advertise_multiple_servers(self):
        urls = ["tcp://10.0.0.1:5555", "tcp://10.0.0.2:5555"]
        config = LMCacheConfig(enable=True, server_port=5555, mp_server_urls=urls)
        assert config.n_servers == 2
        assert config.to_connector_extra_config()["lmcache.mp.server_urls"] == ",".join(urls)

    def test_engine_kwargs_reject_blank_server_url(self):
        with pytest.raises(AssertionError):
            LMCacheConfig(enable=True, server_port=5555, mp_server_urls=[""]).to_engine_kwargs()

    def test_engine_kwargs_omit_offloading_backend(self):
        """Capacity is owned by the MP server, not passed through vLLM."""
        config = LMCacheConfig(enable=True, server_port=5555, offload_size_gb=42.0)
        kwargs = config.to_engine_kwargs()
        assert "kv_offloading_backend" not in kwargs
        assert "kv_offloading_size" not in kwargs

    # --- to_server_argv ---

    def test_server_argv_core_flags(self):
        argv = _runtime_config().to_server_argv()
        assert argv[argv.index("--l1-size-gb") + 1] == "20.0"
        assert argv[argv.index("--port") + 1] == "5555"
        assert argv[argv.index("--http-port") + 1] == "8080"
        assert argv[argv.index("--chunk-size") + 1] == "128"

    def test_server_argv_requires_allocated_ports(self):
        with pytest.raises(AssertionError):
            LMCacheConfig(enable=True, http_port=8080).to_server_argv()

    def test_server_argv_omits_p2p_when_disabled(self):
        argv = _runtime_config().to_server_argv()
        assert "--p2p-advertise-url" not in argv
        assert "--coordinator-url" not in argv

    def test_server_argv_p2p_flags(self):
        argv = _runtime_config(
            enable_p2p=True,
            p2p_advertise_host="10.0.0.1",
            p2p_transfer_port=18200,
            coordinator_host="10.0.0.2",
        ).to_server_argv()
        assert argv[argv.index("--p2p-advertise-url") + 1] == "10.0.0.1:18200"
        assert argv[argv.index("--p2p-listen-url") + 1] == "10.0.0.1:18200"
        assert argv[argv.index("--p2p-transfer-engine") + 1] == "nixl"
        assert argv[argv.index("--coordinator-url") + 1] == "http://10.0.0.2:9300"

    def test_server_argv_event_reporting_flag(self):
        argv = _runtime_config(coordinator_event_reporting=True).to_server_argv()
        assert "--coordinator-event-reporting" in argv

    def test_server_argv_l2_adapters(self):
        argv = _runtime_config(
            l2_adapters=[{"type": "fs", "path": "/mnt/kv"}],
            l2_store_policy="default",
            l2_prefetch_policy="retain",
        ).to_server_argv()
        assert argv[argv.index("--l2-adapter") + 1] == '{"type": "fs", "path": "/mnt/kv"}'
        assert argv[argv.index("--l2-prefetch-policy") + 1] == "retain"

    def test_server_argv_instance_id(self):
        argv = _runtime_config(lmcache_instance_id="psrl_instance_3").to_server_argv()
        assert argv[argv.index("--instance-id") + 1] == "psrl_instance_3"

    def test_server_argv_l1_lazy_defaults(self):
        argv = _runtime_config().to_server_argv()
        assert argv[argv.index("--l1-init-size-gb") + 1] == "20"
        assert "--l1-use-lazy" in argv
        assert "--no-l1-use-lazy" not in argv

    def test_server_argv_can_disable_l1_lazy(self):
        argv = _runtime_config(l1_use_lazy=False).to_server_argv()
        assert "--no-l1-use-lazy" in argv
        assert "--l1-use-lazy" not in argv

    def test_server_argv_rejects_init_size_above_capacity(self):
        with pytest.raises(AssertionError):
            _runtime_config(l1_init_size_gb=64, offload_size_gb=20.0).to_server_argv()

    def test_server_argv_rejects_zero_init_size(self):
        with pytest.raises(AssertionError):
            _runtime_config(l1_init_size_gb=0).to_server_argv()

    def test_shipped_yaml_matches_the_config_dataclass(self):
        """The server actor splats the yaml, so a stale key would raise TypeError."""
        import dataclasses
        from pathlib import Path

        known = {f.name for f in dataclasses.fields(LMCacheConfig)}
        yaml_path = Path(__file__).resolve().parents[2] / "psrl/trainer/config/psrl/lmcache.yaml"
        named = {
            line.split(":", 1)[0].strip()
            for line in yaml_path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#") and ":" in line
        }
        assert named <= known, f"lmcache.yaml names unknown fields: {sorted(named - known)}"

    def test_coordinator_metadata_carries_replica(self):
        config = _runtime_config(
            coordinator_host="10.0.0.2",
            replica_id="psrl_instance_3",
            lmcache_instance_id="psrl_instance_3_n1",
        )
        argv = config.to_server_argv()
        metadata = json.loads(argv[argv.index("--coordinator-metadata") + 1])
        assert metadata["replica_id"] == "psrl_instance_3"
        assert metadata["node_instance_id"] == "psrl_instance_3_n1"

    def test_coordinator_metadata_skipped_without_coordinator(self):
        argv = _runtime_config(replica_id="psrl_instance_3").to_server_argv()
        assert "--coordinator-metadata" not in argv

    # --- management HTTP host resolution ---

    def test_http_host_defaults_to_server_host(self):
        config = _runtime_config()
        assert config.resolved_http_host == "127.0.0.1"
        assert config.resolved_http_advertise_host == "127.0.0.1"
        assert config.http_base_url == "http://127.0.0.1:8080"

    def test_http_host_advertised_for_cross_instance_calls(self):
        config = _runtime_config(http_host="0.0.0.0", http_advertise_host="10.0.0.1")
        argv = config.to_server_argv()
        assert argv[argv.index("--http-host") + 1] == "0.0.0.0"
        assert config.http_base_url == "http://10.0.0.1:8080"

    # --- to_coordinator_argv ---

    def test_coordinator_argv(self):
        argv = LMCacheConfig(coordinator_host="10.0.0.2", coordinator_port=9400).to_coordinator_argv()
        assert argv[argv.index("--port") + 1] == "9400"
        assert argv[argv.index("--host") + 1] == "0.0.0.0"


class TestKVCacheManager:
    def test_disabled_manager(self):
        manager = KVCacheManager(LMCacheConfig(enable=False))
        assert manager.enabled is False
        assert manager.get_engine_kwargs() == {}

    def test_enabled_manager(self):
        manager = KVCacheManager(LMCacheConfig(enable=True, server_port=5555, offload_size_gb=15.0))
        assert manager.enabled is True
        assert manager.get_engine_kwargs()["kv_transfer_config"]["kv_connector"] == MP_CONNECTOR_NAME

    def test_clear_on_weight_update_flag(self):
        enabled = KVCacheManager(LMCacheConfig(enable=True, clear_on_weight_update=True))
        assert enabled.should_clear_on_weight_update is True

        disabled = KVCacheManager(LMCacheConfig(enable=True, clear_on_weight_update=False))
        assert disabled.should_clear_on_weight_update is False

    def test_clear_on_weight_update_requires_enable(self):
        manager = KVCacheManager(LMCacheConfig(enable=False, clear_on_weight_update=True))
        assert manager.should_clear_on_weight_update is False

    def test_set_current_version(self):
        manager = KVCacheManager(LMCacheConfig(enable=True))
        assert manager.current_version == 0
        manager.set_current_version(7)
        assert manager.current_version == 7

    def test_pin_requires_attached_engine(self):
        manager = KVCacheManager(LMCacheConfig(enable=True))
        assert manager.is_attached is False
        with pytest.raises(AssertionError):
            import asyncio

            asyncio.run(manager.pin([1, 2, 3], ["gpu"]))

    def test_backend_target_is_ignored_on_pin(self, monkeypatch):
        """L1 retention moved to pin groups, so the retired target is a no-op."""
        import asyncio

        manager = KVCacheManager(LMCacheConfig(enable=True))
        manager.attach_engine(object())
        calls = []

        async def fake_utility(method, *args):
            calls.append(method)
            return 2

        monkeypatch.setattr(manager, "_utility", fake_utility)
        assert asyncio.run(manager.pin([1, 2, 3], ["gpu", "backend"])) is True
        assert calls == ["psrl_pin_gpu"]
        # A backend-only request does no work rather than failing the caller.
        assert asyncio.run(manager.pin([1, 2, 3], ["backend"])) is True
        assert calls == ["psrl_pin_gpu"]

    def test_backend_target_is_ignored_on_unpin(self, monkeypatch):
        import asyncio

        manager = KVCacheManager(LMCacheConfig(enable=True))
        manager.attach_engine(object())
        calls = []

        async def fake_utility(method, *args):
            calls.append(method)
            return 1

        monkeypatch.setattr(manager, "_utility", fake_utility)
        assert asyncio.run(manager.unpin([1, 2, 3], ["gpu", "backend"])) is True
        assert calls == ["psrl_unpin_gpu"]

    def test_invalid_pin_target_is_rejected(self):
        import asyncio

        manager = KVCacheManager(LMCacheConfig(enable=True))
        manager.attach_engine(object())
        with pytest.raises(AssertionError):
            asyncio.run(manager.pin([1, 2, 3], ["disk"]))

    def test_pin_group_admin_is_disabled_without_lmcache(self):
        import asyncio

        manager = KVCacheManager(LMCacheConfig(enable=False))
        assert asyncio.run(manager.release_pin_groups()) == {}
        assert asyncio.run(manager.pin_group_stats()) == {}

    def test_pin_group_admin_targets_the_management_api(self):
        import asyncio

        manager = KVCacheManager(LMCacheConfig(enable=True))
        seen = []

        async def fake_request(method, path, payload=None):
            seen.append((method, path))
            return {"released": 2} if method == "DELETE" else {"policy": "all"}

        manager._request = fake_request
        assert asyncio.run(manager.release_pin_groups()) == {"released": 2}
        assert asyncio.run(manager.pin_group_stats()) == {"policy": "all"}
        assert seen == [
            ("DELETE", "/cache/l1/pins/groups"),
            ("GET", "/cache/l1/pins/stats"),
        ]

    def test_event_stream_url_is_served_from_the_management_port(self):
        config = _runtime_config(http_host="0.0.0.0", http_advertise_host="10.0.0.1")
        assert config.event_stream_url == "http://10.0.0.1:8080/cache/events/stream"

    def test_event_stream_flags(self):
        argv = _runtime_config(coordinator_event_reporting=True, enable_kv_events=True).to_server_argv()
        assert "--coordinator-event-reporting" in argv
        assert "--coordinator-event-stream-enable" in argv
        metadata = json.loads(argv[argv.index("--coordinator-metadata") + 1])
        assert metadata["event_stream_url"].endswith("/cache/events/stream")

    def test_event_stream_disabled_without_kv_events(self):
        argv = _runtime_config(coordinator_event_reporting=True).to_server_argv()
        assert "--coordinator-event-reporting" in argv
        assert "--coordinator-event-stream-enable" not in argv

    def test_transfer_requires_p2p(self):
        import asyncio

        manager = KVCacheManager(LMCacheConfig(enable=True, enable_p2p=False))
        manager.set_parallel_geometry("org/model", 1)
        assert asyncio.run(manager.transfer_direct([1, 2, 3], ("a", ""), ("b", ""))) is False

    def test_transfer_without_known_destination(self):
        import asyncio

        manager = KVCacheManager(LMCacheConfig(enable=True, enable_p2p=True))
        manager.set_parallel_geometry("org/model", 1)
        # No coordinator configured, so the peer registry cannot be refreshed.
        assert asyncio.run(manager.transfer_direct([1, 2, 3], ("a", ""), ("b", ""))) is False
        assert "no MP endpoint known" in manager._last_transfer_error

    def test_set_peer_registry_merges(self):
        manager = KVCacheManager(LMCacheConfig(enable=True))
        manager.set_peer_registry({"psrl_instance_1": ["http://10.0.0.1:8080"]})
        manager.set_peer_registry({"psrl_instance_2": ["http://10.0.0.2:8080"]})
        assert set(manager.peer_registry) == {"psrl_instance_1", "psrl_instance_2"}

    # --- multi-node replica handling ---

    def _multi_node_manager(self) -> KVCacheManager:
        """A manager whose replica spans two nodes, as the registry would build it."""
        manager = KVCacheManager(LMCacheConfig(enable=True, enable_p2p=True, coordinator_host="10.0.0.254"))
        manager.set_parallel_geometry("org/model", 8)
        manager.peer_registry = {
            "psrl_instance_0_n0": ["http://10.0.0.1:9001"],
            "psrl_instance_0_n1": ["http://10.0.0.2:9001"],
        }
        manager.replica_registry = {"psrl_instance_0": ["http://10.0.0.1:9001", "http://10.0.0.2:9001"]}
        manager.replica_nodes = {"psrl_instance_0": ["psrl_instance_0_n0", "psrl_instance_0_n1"]}
        manager._replica_of_node = {
            "psrl_instance_0_n0": "psrl_instance_0",
            "psrl_instance_0_n1": "psrl_instance_0",
        }
        # Treat the registry as freshly refreshed so no coordinator call is made.
        manager._peer_registry_refreshed_at = float("inf")
        return manager

    def test_node_id_resolves_to_its_whole_replica(self):
        manager = self._multi_node_manager()
        assert manager._node_urls("psrl_instance_0") == manager.replica_registry["psrl_instance_0"]
        # Each node holds part of the KV, so a transfer has to reach all of them.
        assert manager._node_urls("psrl_instance_0_n1") == manager.replica_registry["psrl_instance_0"]
        assert manager._node_ids("psrl_instance_0_n0") == manager.replica_nodes["psrl_instance_0"]
        assert manager._node_ids("unknown") == ["unknown"]

    def test_transfer_reaches_every_node_of_the_replica(self):
        import asyncio

        manager = self._multi_node_manager()
        submitted, deleted = [], []

        async def fake_request(method, url, payload=None):
            if "/cache/prefetches" in url and method == "POST":
                host = url.split("//")[1].split(":")[0]
                submitted.append(host)
                return {"request_id": f"r-{host}", "chunks": 4}
            if "/cache/prefetches/" in url:
                return {"status": "completed", "found_keys": 4}
            if url.endswith("/cache/delete"):
                deleted.append(payload["instance_id"])
                return {"skipped": 0}
            raise AssertionError(url)

        manager._request_url = fake_request
        transferred = asyncio.run(
            manager.transfer_direct(
                [1, 2, 3],
                ("psrl_instance_0", "LocalCPUBackend"),
                ("psrl_instance_0", "LocalCPUBackend"),
                copy=False,
            )
        )
        assert transferred is True
        assert sorted(submitted) == ["10.0.0.1", "10.0.0.2"]
        assert sorted(deleted) == ["psrl_instance_0_n0", "psrl_instance_0_n1"]

    def test_transfer_fails_when_one_node_stays_incomplete(self):
        import asyncio

        manager = self._multi_node_manager()

        async def fake_request(method, url, payload=None):
            if "/cache/prefetches" in url and method == "POST":
                host = url.split("//")[1].split(":")[0]
                return {"request_id": f"r-{host}", "chunks": 4}
            if "/cache/prefetches/" in url:
                host = url.split("//")[1].split(":")[0]
                return {"status": "completed", "found_keys": 4 if host == "10.0.0.1" else 1}
            raise AssertionError(url)

        manager._request_url = fake_request
        assert (
            asyncio.run(manager.transfer_direct([1, 2, 3], ("psrl_instance_0", ""), ("psrl_instance_0", ""))) is False
        )
        assert "incomplete prefix" in manager._last_transfer_error

    # --- store to pin race ---

    # --- pin policy configuration ---

    def test_pin_policy_defaults_to_off(self):
        argv = _runtime_config().to_server_argv()
        assert argv[argv.index("--l1-pin-policy") + 1] == "off"
        assert argv[argv.index("--l1-pin-budget-ratio") + 1] == "0.25"
        assert argv[argv.index("--l1-pin-ttl-seconds") + 1] == "600.0"

    def test_pin_policy_is_forwarded(self):
        argv = _runtime_config(pin_policy="all", pin_budget_ratio=0.5, pin_ttl_seconds=60.0).to_server_argv()
        assert argv[argv.index("--l1-pin-policy") + 1] == "all"
        assert argv[argv.index("--l1-pin-budget-ratio") + 1] == "0.5"
        assert argv[argv.index("--l1-pin-ttl-seconds") + 1] == "60.0"

    def test_pin_policy_rejects_noop_eviction(self):
        """Pinned objects cannot be reclaimed without eviction."""
        with pytest.raises(AssertionError):
            _runtime_config(pin_policy="all", eviction_policy="noop").to_server_argv()

    def test_pin_policy_validates_its_own_values(self):
        with pytest.raises(AssertionError):
            _runtime_config(pin_policy="sometimes").to_server_argv()
        with pytest.raises(AssertionError):
            _runtime_config(pin_budget_ratio=1.5).to_server_argv()
        with pytest.raises(AssertionError):
            _runtime_config(pin_ttl_seconds=0).to_server_argv()
