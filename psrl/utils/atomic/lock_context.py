import ray
import asyncio
from collections import deque

def add_lock(cls):
    """
    通用的类装饰器：给任意一个类注入 _locked、_waiters，以及 acquire/release 方法。
    """
    original_init = getattr(cls, "__init__", None)

    def __init__(self, *args, **kwargs):
        if original_init is not None:
            original_init(self, *args, **kwargs)
        else:
            super(cls, self).__init__(*args, **kwargs)

        self._locked = False
        self._waiters = deque()

    async def acquire(self):
        if not self._locked:
            self._locked = True
            return
        fut = asyncio.get_event_loop().create_future()
        self._waiters.append(fut)
        await fut

    async def release(self):
        if self._waiters:
            fut = self._waiters.popleft()
            fut.set_result(None)
        else:
            self._locked = False

    setattr(cls, "__init__", __init__)
    setattr(cls, "acquire", acquire)
    setattr(cls, "release", release)
    return cls

class RayLock:
    """Synchronous context manager around a LockActor."""
    def __init__(self, lock_actor):
        self._lock = lock_actor
        
    def acquire(self):
        """Acquire the lock, blocking until it is available."""
        ray.get(self._lock.acquire.remote())
        
    def release(self):
        """Release the lock."""
        ray.get(self._lock.release.remote())

    def __enter__(self):
        # block until acquired
        self.acquire()

    def __exit__(self, exc_type, exc, tb):
        # release regardless of exception
        self.release()