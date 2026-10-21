from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import psrl.workers.gen.vllm_async_server as server_module
import pytest
from psrl.workers.gen.vllm_async_server import PSRL_vLLMHttpServer


class _FakeLock:
    """Stand-in for the shared pull lock, reporting the commit under test."""

    def __init__(self, reader=None, committed_version=None):
        self.reader = reader
        self.committed_version = committed_version

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


def _pull_context(_, reader=None):
    return _FakeLock(reader, committed_version=getattr(_pull_context, "committed_version", None))


def _make_server(current_version: int, actual_version: int):
    server = object.__new__(PSRL_vLLMHttpServer)
    server.curr_rollout_instance_model_version = [current_version, current_version]
    server.get_instance_num = Mock(return_value=2)
    server.base_worker_id = "replica-0"
    server.pull_model = AsyncMock()
    server.kv_cache_manager = Mock()
    server.stat_collector = None
    server.psrl_config = SimpleNamespace(nixl={"preempt_repull_attempts": 2})
    server.gen_interface = SimpleNamespace(
        ps_manager_handle=SimpleNamespace(
            get_rollout_instance_model_version=SimpleNamespace(
                remote=AsyncMock(return_value=actual_version),
            ),
            begin_read=SimpleNamespace(remote=AsyncMock()),
            end_read=SimpleNamespace(remote=AsyncMock()),
        ),
    )
    return server


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pull_model_commits_actual_version_to_local_kv_manager(monkeypatch):
    monkeypatch.setattr(server_module, "shared_pull_model_context_async", _pull_context)
    monkeypatch.setattr(server_module, "log_dual_events", lambda *args, **kwargs: nullcontext())
    _pull_context.committed_version = 3
    server = _make_server(current_version=1, actual_version=3)

    version = await PSRL_vLLMHttpServer.pull_model_for_sync(server, ps_version=2)

    assert version == 3
    assert server.curr_rollout_instance_model_version == [3, 3]
    server.kv_cache_manager.set_current_version.assert_called_once_with(3)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pull_model_recommits_current_version_when_pull_is_not_needed():
    server = _make_server(current_version=4, actual_version=5)

    version = await PSRL_vLLMHttpServer.pull_model_for_sync(server, ps_version=3)

    assert version == 4
    server.pull_model.assert_not_awaited()
    server.kv_cache_manager.set_current_version.assert_called_once_with(4)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pull_model_keeps_serving_when_the_actual_version_is_too_old(monkeypatch):
    monkeypatch.setattr(server_module, "shared_pull_model_context_async", _pull_context)
    monkeypatch.setattr(server_module, "log_dual_events", lambda *args, **kwargs: nullcontext())
    _pull_context.committed_version = 1
    server = _make_server(current_version=1, actual_version=1)

    # Falling short is a degradation: the replica keeps serving the version it
    # actually holds instead of quarantining itself.
    version = await PSRL_vLLMHttpServer.pull_model_for_sync(server, ps_version=2)

    assert version == 1
    server.kv_cache_manager.set_current_version.assert_called_once_with(1)
