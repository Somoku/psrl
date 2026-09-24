"""Cancellation boundaries and lock helpers for resource ownership transitions."""

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


async def acquire_nowait(lock: asyncio.Lock) -> bool:
    """Take a lock without waiting, reporting whether it was free.

    Used where waiting would defeat the point: an idle pass that blocked on a
    sandbox's exec lock would hold the pass open for the whole of a long command,
    when the right answer is to skip that sandbox and decide again next time.
    """
    if lock.locked():
        return False
    await lock.acquire()
    return True


async def complete_cleanup(operation: Awaitable[T]) -> T:
    """
    Finish an ownership transition before propagating any caller cancellation.

    Repeated cancellation must not interrupt the underlying operation. Callers
    retain ownership and retry if the operation itself fails.
    """
    task = asyncio.ensure_future(operation)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result
