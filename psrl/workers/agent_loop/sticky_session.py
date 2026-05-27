"""Sticky session context manager for rollout routing.

Usage example in MultiTurnAgentLoop:

    from psrl.workers.agent_loop import sticky_session

    # In the run method - using convenience function:
    async with sticky_session(self.rollout_router, request):
        # All generate_async calls within this context will use the same rollout instance
        output = await self.rollout_router.generate_async.remote(
            self.agent_data.prepare_generation_request(request)
        )

    # Or using class directly:
    from psrl.workers.agent_loop import StickySession

    request_id = request.non_tensor_batch["uid"][0]
    async with StickySession(self.rollout_router, request_id):
        output = await self.rollout_router.generate_async.remote(...)

    # Config-gated helper (preferred when sticky session is opt-in):

    async with maybe_sticky_session(
        self.rollout_router,
        request.non_tensor_batch["uid"][0],
        self.config.psrl.agentic_rl.sticky_session,
    ):
        output = await self.rollout_router.generate_async.remote(...)
"""

from contextlib import asynccontextmanager


class StickySession:
    """Context manager for sticky session routing.

    Ensures that requests maintain the same rollout instance during the session.
    """

    def __init__(self, rollout_router_handle, request_id: int):
        """Initialize sticky session context.

        Args:
            rollout_router_handle: Ray actor handle to RolloutRouter.
            request_id (int): The request ID for this session.
        """
        self.rollout_router_handle = rollout_router_handle
        self.request_id = request_id

    async def __aenter__(self):
        """Enter sticky session context."""
        await self.rollout_router_handle.enter_sticky_session.remote(self.request_id)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit sticky session context."""
        await self.rollout_router_handle.exit_sticky_session.remote(self.request_id)
        return False


def sticky_session(rollout_router_handle, request):
    """Convenience function to create sticky session from request.

    Args:
        rollout_router_handle: Ray actor handle to RolloutRouter.
        request: DataProto containing request with 'uid' in non_tensor_batch.

    Returns:
        StickySession: Context manager for sticky session.
    """
    request_id = request.non_tensor_batch["uid"][0]
    return StickySession(rollout_router_handle, request_id)


@asynccontextmanager
async def null_async_context():
    """No-op async context manager.

    Used as the ``else`` branch when sticky session routing is disabled, so the
    surrounding ``async with`` site stays uniform.
    """
    yield


def maybe_sticky_session(rollout_router_handle, request_id, enabled: bool):
    """Return a sticky session if ``enabled``, else a no-op async context.

    Lets call sites stay a single ``async with`` line regardless of whether
    sticky-session routing is on or off.

    Args:
        rollout_router_handle: Ray actor handle to ``RolloutRouter``.
        request_id: The request ID for this session (typically
            ``request.non_tensor_batch["uid"][0]``).
        enabled (bool): Whether sticky-session routing is enabled (typically
            ``config.psrl.agentic_rl.sticky_session``).

    Returns:
        An async context manager: either ``StickySession`` or
        ``null_async_context``.
    """
    if enabled:
        return StickySession(rollout_router_handle, request_id)
    return null_async_context()
