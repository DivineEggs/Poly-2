#!/usr/bin/env python3
"""
Rate Limiter — Token bucket rate limiter for API calls.

Prevents hitting API rate limits by throttling outbound requests.
Supports:
- Token bucket algorithm (smooth rate limiting with burst capacity)
- 429 response handling with Retry-After parsing
- Async-compatible waiting
"""
import asyncio
import time
import threading

from src.logger import get_logger

logger = get_logger("rate_limiter")


class RateLimiter:
    """
    Token bucket rate limiter.

    Tokens are added at a fixed rate up to a maximum (burst) capacity.
    Each request consumes one token. If no tokens are available, the
    caller waits until one is replenished.

    Thread-safe for use from both sync and async contexts.

    Args:
        name: Identifier for logging (e.g., "gamma", "clob").
        rate: Maximum sustained requests per second.
        burst: Maximum burst capacity (tokens in bucket).
    """

    def __init__(self, name: str, rate: float, burst: int = None):
        self.name = name
        self.rate = rate
        self.burst = burst or int(rate * 2)
        self._tokens = float(self.burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()
        self._total_waits = 0
        self._total_requests = 0

    def _refill(self):
        """Add tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)

    def acquire(self, timeout: float = 30.0) -> bool:
        """
        Acquire a token (blocking). Returns True if acquired, False on timeout.

        Args:
            timeout: Maximum seconds to wait for a token.

        Returns:
            True if token acquired, False if timed out.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    self._total_requests += 1
                    return True
                # Calculate wait time for next token
                wait_time = (1.0 - self._tokens) / self.rate

            if time.monotonic() + wait_time > deadline:
                logger.warning("%s: rate limit acquire timed out after %.1fs", self.name, timeout)
                return False

            self._total_waits += 1
            if self._total_waits % 10 == 1:
                logger.debug("%s: rate limited, waiting %.2fs (total waits: %d)",
                             self.name, wait_time, self._total_waits)
            time.sleep(min(wait_time, 0.5))

    async def acquire_async(self, timeout: float = 30.0) -> bool:
        """
        Acquire a token (async). Returns True if acquired, False on timeout.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    self._total_requests += 1
                    return True
                wait_time = (1.0 - self._tokens) / self.rate

            if time.monotonic() + wait_time > deadline:
                logger.warning("%s: rate limit acquire timed out after %.1fs", self.name, timeout)
                return False

            self._total_waits += 1
            await asyncio.sleep(min(wait_time, 0.5))

    def handle_429(self, retry_after: str = None) -> float:
        """
        Handle a 429 Too Many Requests response.

        Drains remaining tokens and returns the wait time.

        Args:
            retry_after: Value of the Retry-After header (seconds or date).

        Returns:
            Number of seconds to wait before retrying.
        """
        with self._lock:
            self._tokens = 0.0  # Drain all tokens

        if retry_after:
            try:
                wait = float(retry_after)
                logger.warning("%s: 429 rate limited, Retry-After: %.1fs", self.name, wait)
                return wait
            except ValueError:
                pass

        # Default: wait 2 seconds
        default_wait = 2.0
        logger.warning("%s: 429 rate limited, waiting %.1fs (no Retry-After)", self.name, default_wait)
        return default_wait

    @property
    def stats(self) -> dict:
        """Return rate limiter statistics."""
        return {
            "name": self.name,
            "rate": self.rate,
            "burst": self.burst,
            "total_requests": self._total_requests,
            "total_waits": self._total_waits,
        }
