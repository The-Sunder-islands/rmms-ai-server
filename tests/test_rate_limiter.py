from __future__ import annotations

import pytest
from rmms_ai_server.core.rate_limiter import RateLimiter, TokenBucket


class TestTokenBucket:
    def test_initial_tokens_available(self):
        bucket = TokenBucket(max_tokens=10, refill_rate=1.0)
        assert bucket.consume() is True

    def test_deplete_tokens(self):
        bucket = TokenBucket(max_tokens=3, refill_rate=0.0)
        assert bucket.consume() is True
        assert bucket.consume() is True
        assert bucket.consume() is True
        assert bucket.consume() is False

    def test_retry_after_when_exhausted(self):
        bucket = TokenBucket(max_tokens=1, refill_rate=0.1)
        bucket.consume()
        assert bucket.consume() is False
        assert bucket.retry_after > 0

    def test_refill(self):
        import time
        bucket = TokenBucket(max_tokens=5, refill_rate=100.0)
        bucket.consume()
        bucket.consume()
        time.sleep(0.05)
        assert bucket.consume() is True

    def test_max_tokens_cap(self):
        bucket = TokenBucket(max_tokens=5, refill_rate=1000.0)
        time.sleep(0.05)
        assert bucket.tokens <= 5


class TestRateLimiter:
    def test_allow_first_request(self):
        limiter = RateLimiter(max_tokens=10, refill_rate=1.0)
        allowed, retry = limiter.is_allowed("client-1")
        assert allowed is True
        assert retry is None

    def test_block_after_exhaustion(self):
        limiter = RateLimiter(max_tokens=3, refill_rate=0.0)
        for _ in range(3):
            allowed, _ = limiter.is_allowed("client-2")
            assert allowed is True
        allowed, retry = limiter.is_allowed("client-2")
        assert allowed is False
        assert retry is not None

    def test_different_clients_independent(self):
        limiter = RateLimiter(max_tokens=1, refill_rate=0.0)
        assert limiter.is_allowed("client-a")[0] is True
        assert limiter.is_allowed("client-a")[0] is False
        assert limiter.is_allowed("client-b")[0] is True

    def test_retry_after_positive(self):
        limiter = RateLimiter(max_tokens=1, refill_rate=0.5)
        limiter.is_allowed("client-c")
        allowed, retry = limiter.is_allowed("client-c")
        assert allowed is False
        assert retry > 0
        assert retry >= 1.0

    def test_is_allowed_cleanup(self):
        import time
        limiter = RateLimiter(max_tokens=10, refill_rate=100.0)
        limiter.is_allowed("cleanup-test")
        limiter._last_cleanup = time.monotonic() - 600
        assert limiter.is_allowed("cleanup-test")[0] is True
