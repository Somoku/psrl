import pytest
from psrl.utils.kv_cache.config import (
    MP_CONNECTOR_MODULE,
    MP_CONNECTOR_NAME,
    LMCacheConfig,
)
from psrl.utils.kv_cache.manager import KVCacheManager


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
        # HMA is disabled for this connector version.
        assert kwargs["disable_hybrid_kv_cache_manager"] is True

    def test_engine_kwargs_omit_offloading_backend(self):
        """Capacity is owned by the MP server, not passed through vLLM."""
        config = LMCacheConfig(enable=True, server_port=5555, offload_size_gb=42.0)
        kwargs = config.to_engine_kwargs()
        assert "kv_offloading_backend" not in kwargs
        assert "kv_offloading_size" not in kwargs

    # --- to_server_argv ---

    def _runtime_config(self, **overrides) -> LMCacheConfig:
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

    def test_server_argv_core_flags(self):
        argv = self._runtime_config().to_server_argv()
        assert argv[argv.index("--l1-size-gb") + 1] == "20.0"
        assert argv[argv.index("--port") + 1] == "5555"
        assert argv[argv.index("--http-port") + 1] == "8080"
        assert argv[argv.index("--chunk-size") + 1] == "128"

    def test_server_argv_requires_allocated_ports(self):
        with pytest.raises(AssertionError):
            LMCacheConfig(enable=True, http_port=8080).to_server_argv()

    def test_server_argv_omits_p2p_when_disabled(self):
        argv = self._runtime_config().to_server_argv()
        assert "--p2p-advertise-url" not in argv
        assert "--coordinator-url" not in argv

    def test_server_argv_p2p_flags(self):
        argv = self._runtime_config(
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
        argv = self._runtime_config(coordinator_event_reporting=True).to_server_argv()
        assert "--coordinator-event-reporting" in argv

    def test_server_argv_l2_adapters(self):
        argv = self._runtime_config(
            l2_adapters=[{"type": "fs", "path": "/mnt/kv"}],
            l2_store_policy="default",
            l2_prefetch_policy="retain",
        ).to_server_argv()
        assert argv[argv.index("--l2-adapter") + 1] == '{"type": "fs", "path": "/mnt/kv"}'
        assert argv[argv.index("--l2-prefetch-policy") + 1] == "retain"

    def test_server_argv_instance_id(self):
        argv = self._runtime_config(lmcache_instance_id="psrl_instance_3").to_server_argv()
        assert argv[argv.index("--instance-id") + 1] == "psrl_instance_3"

    # --- management HTTP host resolution ---

    def test_http_host_defaults_to_server_host(self):
        config = self._runtime_config()
        assert config.resolved_http_host == "127.0.0.1"
        assert config.resolved_http_advertise_host == "127.0.0.1"
        assert config.http_base_url == "http://127.0.0.1:8080"

    def test_http_host_advertised_for_cross_instance_calls(self):
        config = self._runtime_config(http_host="0.0.0.0", http_advertise_host="10.0.0.1")
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

    def test_backend_pin_requires_geometry(self):
        manager = KVCacheManager(LMCacheConfig(enable=True))
        with pytest.raises(AssertionError):
            manager._l1_pin_body([1, 2, 3])

    def test_backend_pin_body_carries_version_tag(self):
        manager = KVCacheManager(LMCacheConfig(enable=True, multi_version_kv=True))
        manager.set_parallel_geometry("org/model", 4)
        manager.set_current_version(7)
        body = manager._l1_pin_body([1, 2, 3])
        assert body["model_name"] == "org/model"
        assert body["world_size"] == 4
        assert body["request_configs"] == {"lmcache.tag.model_version": "7"}

    def test_backend_pin_body_untagged_when_multi_version_off(self):
        manager = KVCacheManager(LMCacheConfig(enable=True, multi_version_kv=False))
        manager.set_parallel_geometry("org/model", 1)
        assert manager._l1_pin_body([1])["request_configs"] is None

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
