from __future__ import annotations

import asyncio
import threading
import time


class TokenBucket:
    """A small, thread-safe token bucket rate limiter.

    It's designed for both:
    - sync "try-acquire" style (`acquire`)
    - async "wait until allowed" style (`async_acquire`)

    Think of a bucket that accumulates "tokens" over time.
    Each operation costs some tokens (default: 1). If the bucket has
    enough tokens, the operation is allowed immediately; otherwise it is
    rejected (sync) or waits (async) until enough tokens have accumulated.

    This gives you:
    - a long-term average rate limit controlled by `rate`
    - a short-term burst capacity controlled by `capacity`

    Roughly:
        max instantaneous burst  ~= capacity
        max long-run throughput  ~= rate (tokens/second)

    Args:
        rate: tokens replenished per second.
        capacity: maximum burst size. Defaults to `rate`.
        init_tokens: initial tokens. Defaults to `capacity`.

    Notes:
        - Uses `time.monotonic()` to avoid wall-clock jumps.
        - Internal state is protected by a `threading.Lock`.
    """

    def __init__(self, rate: float, capacity: float | None = None, init_tokens: float | None = None):
        if rate <= 0:
            raise ValueError("rate must be > 0")

        self._rate = rate
        self._capacity = capacity if capacity is not None else rate

        # Current available tokens.
        # We clamp it into [0, capacity] to keep invariants sane.
        self._tokens = init_tokens if init_tokens is not None else self._capacity
        self._tokens = max(0.0, min(self._capacity, self._tokens))

        # Timestamp (monotonic) of the last refill. Using `monotonic()` protects
        # us from wall-clock jumps (NTP adjustments / manual time changes).
        self._updated_at = time.monotonic()

        self._lock = threading.Lock()

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def capacity(self) -> float:
        return self._capacity

    def _refill_locked(self, now: float) -> None:
        """Refill tokens based on elapsed time since last update.

        IMPORTANT: Callers must hold `_lock`.
        """
        elapsed = now - self._updated_at
        if elapsed <= 0:
            return
        self._updated_at = now
        # lazy refilling
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)

    def acquire(self, tokens: float = 1.0) -> bool:
        """Try to acquire tokens immediately.

        Returns True if successful, False otherwise.
        """
        if tokens <= 0:
            return True

        now = time.monotonic()
        with self._lock:
            self._refill_locked(now)
            # If enough tokens are available right now, consume them and allow.
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def time_to_availability(self, tokens: float = 1.0) -> float:
        """Return minimal seconds to wait until `tokens` can be acquired."""
        if tokens <= 0:
            return 0.0

        now = time.monotonic()
        with self._lock:
            self._refill_locked(now)
            missing = tokens - self._tokens
            if missing <= 0:
                return 0.0
            return missing / self._rate

    async def async_acquire(self, tokens: float = 1.0, *, max_sleep: float = 0.1) -> None:
        """Wait until tokens are available and then acquire.

        Args:
            tokens: number of tokens to acquire.
            max_sleep: cap the sleep time to keep latency/jitter under control.
        """
        if tokens <= 0:
            return

        # Fast path
        if self.acquire(tokens):
            return

        # Slow path: wait until enough tokens.
        while True:
            # Compute the minimum wait needed at this moment.
            wait_s = self.time_to_availability(tokens)
            await asyncio.sleep(min(max_sleep, max(0.0, wait_s)))

            # Another attempt after sleeping.
            if self.acquire(tokens):
                return
