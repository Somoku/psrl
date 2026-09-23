"""Cancellation boundaries for resource ownership transitions."""

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


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
