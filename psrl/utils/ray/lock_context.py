import ray
import asyncio
import time
from collections import deque


def add_lock(cls):
    """
    A general class decorator that injects _locked, _waiters, and acquire/release methods into any class.
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


def add_busy_polling_lock(cls):
    """
    A general class decorator that injects _locked and synchronous acquire/release methods into any class.
    Since Ray actor calls are serialized, we can use a simple attribute without threading primitives.
    """
    original_init = getattr(cls, "__init__", None)

    def __init__(self, *args, **kwargs):
        if original_init is not None:
            original_init(self, *args, **kwargs)
        else:
            super(cls, self).__init__(*args, **kwargs)

        self._locked = False

    def acquire(self):
        """Acquire the lock synchronously. Returns True if acquired, False if already locked."""
        if self._locked:
            return False
        self._locked = True
        return True

    def release(self):
        """Release the lock synchronously."""
        assert self._locked, "Lock is not locked"
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
        

class AsyncRayLock:
    """Asynchronous context manager around a LockActor."""
    def __init__(self, lock_actor):
        self._lock = lock_actor
        
    async def acquire(self):
        """Acquire the lock, blocking until it is available."""
        await self._lock.acquire.remote()
        
    async def release(self):
        """Release the lock."""
        await self._lock.release.remote()

    async def __aenter__(self):
        # block until acquired
        await self.acquire()

    async def __aexit__(self, exc_type, exc, tb):
        # release regardless of exception
        await self.release()


class BusyPollingRayLock:
    """
    Synchronous context manager around a LockActor using busy polling.
    Uses busy polling with a specified interval to check if the lock is available.
    """
    def __init__(self, lock_actor, poll_interval=0.01):
        """
        Initialize the BusyPollingRayLock.
        
        Args:
            lock_actor: The actor handle with lock methods (acquire, release). The actor should have acquire() method to check lock state.
            poll_interval: The interval (in seconds) between polling checks. Default is 0.01 seconds.
        """
        self._lock = lock_actor
        self._poll_interval = poll_interval
        
    def acquire(self):
        """
        Acquire the lock using busy polling, blocking until it is available.
        Continuously polls the acquire() method at the specified interval.
        """
        # Busy polling loop: check if lock is available
        while True:
            # Check if lock is available by checking acquire() method
            got_locked = ray.get(self._lock.acquire.remote())
            if got_locked:
                return
            # Lock is still held, wait for poll_interval before checking again
            time.sleep(self._poll_interval)
        
    def release(self):
        """Release the lock."""
        ray.get(self._lock.release.remote())

    def __enter__(self):
        # block until acquired using busy polling
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        # release regardless of exception
        self.release()


class AsyncBusyPollingRayLock:
    """
    Asynchronous context manager around a LockActor using busy polling.
    Uses busy polling with a specified interval to check if the lock is available.
    """
    def __init__(self, lock_actor, poll_interval=0.01):
        """
        Initialize the AsyncBusyPollingRayLock.
        
        Args:
            lock_actor: The actor handle with lock methods (acquire, release). The actor should have acquire() method to check lock state.
            poll_interval: The interval (in seconds) between polling checks. Default is 0.01 seconds.
        """
        self._lock = lock_actor
        self._poll_interval = poll_interval
        
    async def acquire(self):
        """
        Acquire the lock using busy polling, blocking until it is available.
        Continuously polls the acquire() method at the specified interval.
        """
        while True:
            # Check if lock is available by checking acquire() method
            got_locked = await self._lock.acquire.remote()
            if got_locked:
                return
            # Lock is still held, wait for poll_interval before checking again
            await asyncio.sleep(self._poll_interval)
        
    async def release(self):
        """Release the lock."""
        await self._lock.release.remote()

    async def __aenter__(self):
        # block until acquired using busy polling
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # release regardless of exception
        await self.release()