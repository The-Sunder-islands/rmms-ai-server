from __future__ import annotations

import time
import threading
from typing import Optional


class TokenBucket:
    def __init__(self, max_tokens: int, refill_rate: float):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.tokens = float(max_tokens)
        self.last_refill = time.monotonic()

    def consume(self, tokens: int = 1) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    @property
    def retry_after(self) -> float:
        if self.tokens <= 0:
            needed = 1.0 - self.tokens
            return max(1.0, needed / self.refill_rate)
        return 0.0


class RateLimiter:
    def __init__(self, max_tokens: int = 60, refill_rate: float = 10.0):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.monotonic()

    def is_allowed(self, client_id: str) -> tuple[bool, Optional[float]]:
        with self._lock:
            self._cleanup_if_needed()
            bucket = self._buckets.get(client_id)
            if bucket is None:
                bucket = TokenBucket(self.max_tokens, self.refill_rate)
                self._buckets[client_id] = bucket

            if bucket.consume():
                return True, None
            return False, bucket.retry_after

    def _cleanup_if_needed(self):
        now = time.monotonic()
        if now - self._last_cleanup < 300:
            return
        self._last_cleanup = now
        stale = []
        for cid, bucket in self._buckets.items():
            if bucket.tokens >= self.max_tokens:
                stale.append(cid)
        for cid in stale:
            del self._buckets[cid]
