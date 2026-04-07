import os
from unittest.mock import patch

import pytest

from psrl.utils.kv_cache.config import LMCacheConfig
from psrl.utils.kv_cache.manager import KVCacheManager
from psrl.utils.kv_cache.types import KVCacheBackend, KVCacheStatus


class TestKVCacheBackend:
    def test_cpu_backend(self):
        assert KVCacheBackend("cpu") == KVCacheBackend.CPU

    def test_disk_backend(self):
        assert KVCacheBackend("disk") == KVCacheBackend.DISK

    def test_remote_backend(self):
        assert KVCacheBackend("remote") == KVCacheBackend.REMOTE

    def test_invalid_backend(self):
        with pytest.raises(ValueError):
            KVCacheBackend("invalid")


class TestKVCacheStatus:
    def test_disabled_status(self):
        status = KVCacheStatus(enabled=False)
        assert status.enabled is False
        assert status.backend is None
        assert status.offload_size_gb == 0.0

    def test_enabled_status(self):
        status = KVCacheStatus(
            enabled=True,
            backend=KVCacheBackend.CPU,
            offload_size_gb=10.0,
        )
        assert status.enabled is True
        assert status.backend == KVCacheBackend.CPU
        assert status.offload_size_gb == 10.0


class TestLMCacheConfig:
    def test_disabled_by_default(self):
        config = LMCacheConfig()
        assert config.enable is False
        assert config.to_engine_kwargs() == {}
        assert config.to_env_vars() == {}

    def test_cpu_engine_kwargs(self):
        config = LMCacheConfig(enable=True, backend="cpu", offload_size_gb=20.0)
        kwargs = config.to_engine_kwargs()
        assert kwargs["kv_offloading_backend"] == "lmcache"
        assert kwargs["kv_offloading_size"] == 20.0

    def test_cpu_env_vars(self):
        config = LMCacheConfig(enable=True, backend="cpu", chunk_size=128)
        env_vars = config.to_env_vars()
        assert env_vars["LMCACHE_CHUNK_SIZE"] == "128"

    def test_disk_env_vars(self):
        config = LMCacheConfig(
            enable=True,
            backend="disk",
            offload_size_gb=10.0,
            disk_path="/tmp/lmcache_disk",
            max_disk_size_gb=50.0,
        )
        env_vars = config.to_env_vars()
        assert env_vars["LMCACHE_LOCAL_DISK"] == "/tmp/lmcache_disk"
        assert env_vars["LMCACHE_MAX_LOCAL_DISK_SIZE"] == "50.0"

    def test_config_file_override(self):
        config = LMCacheConfig(
            enable=True,
            config_file="/path/to/lmcache.yaml",
        )
        env_vars = config.to_env_vars()
        assert env_vars["LMCACHE_CONFIG_FILE"] == "/path/to/lmcache.yaml"

    def test_get_backend_enum(self):
        config = LMCacheConfig(enable=True, backend="cpu")
        assert config.get_backend_enum() == KVCacheBackend.CPU

    def test_invalid_backend(self):
        config = LMCacheConfig(enable=True, backend="invalid")
        with pytest.raises(ValueError):
            config.get_backend_enum()


class TestKVCacheManager:
    def test_disabled_manager(self):
        config = LMCacheConfig(enable=False)
        manager = KVCacheManager(config)
        assert manager.enabled is False
        assert manager.get_status().enabled is False
        assert manager.get_engine_kwargs() == {}

    def test_enabled_manager(self):
        config = LMCacheConfig(enable=True, backend="cpu", offload_size_gb=15.0)
        manager = KVCacheManager(config)
        assert manager.enabled is True
        status = manager.get_status()
        assert status.enabled is True
        assert status.backend == KVCacheBackend.CPU
        assert status.offload_size_gb == 15.0

    def test_get_engine_kwargs(self):
        config = LMCacheConfig(enable=True, backend="cpu", offload_size_gb=20.0)
        manager = KVCacheManager(config)
        kwargs = manager.get_engine_kwargs()
        assert kwargs["kv_offloading_backend"] == "lmcache"
        assert kwargs["kv_offloading_size"] == 20.0

    def test_apply_env_vars(self):
        config = LMCacheConfig(enable=True, backend="cpu", chunk_size=512)
        manager = KVCacheManager(config)
        with patch.dict(os.environ, {}, clear=False):
            manager.apply_env_vars()
            assert os.environ["LMCACHE_CHUNK_SIZE"] == "512"

    def test_apply_env_vars_disabled(self):
        config = LMCacheConfig(enable=False)
        manager = KVCacheManager(config)
        original_env = dict(os.environ)
        manager.apply_env_vars()
        for key in os.environ:
            if key.startswith("LMCACHE_") and key not in original_env:
                pytest.fail(f"Unexpected env var set: {key}")

    def test_clear_on_weight_update_flag(self):
        config = LMCacheConfig(enable=True, clear_on_weight_update=True)
        manager = KVCacheManager(config)
        assert manager.should_clear_on_weight_update is True

        config2 = LMCacheConfig(enable=True, clear_on_weight_update=False)
        manager2 = KVCacheManager(config2)
        assert manager2.should_clear_on_weight_update is False
