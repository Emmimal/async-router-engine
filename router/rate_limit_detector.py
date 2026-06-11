"""
rate_limit_detector.py
----------------------
Detects rate limit events from provider responses and tracks
per-provider throttle windows using pure Python.

No external dependencies. Uses time.monotonic() for decay windows.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ThrottleReason(Enum):
    RATE_LIMIT_429    = "rate_limit_429"
    QUOTA_EXHAUSTED   = "quota_exhausted"
    PROVIDER_TIMEOUT  = "provider_timeout"
    CONTEXT_OVERFLOW  = "context_overflow"
    NONE              = "none"


@dataclass
class ThrottleEvent:
    provider_id: str
    reason: ThrottleReason
    timestamp: float
    retry_after_seconds: float
    request_id: Optional[str] = None


@dataclass
class ProviderWindow:
    provider_id: str
    window_seconds: float = 60.0
    max_requests: int = 100
    _requests: List[float] = field(default_factory=list)
    _throttle_events: List[ThrottleEvent] = field(default_factory=list)
    _backoff_until: float = 0.0

    def record_request(self) -> None:
        now = time.monotonic()
        self._requests.append(now)
        self._requests = [t for t in self._requests
                          if now - t < self.window_seconds]

    def record_throttle(self, event: ThrottleEvent) -> None:
        self._throttle_events.append(event)
        self._backoff_until = time.monotonic() + event.retry_after_seconds

    def is_throttled(self) -> bool:
        return time.monotonic() < self._backoff_until

    def backoff_remaining(self) -> float:
        remaining = self._backoff_until - time.monotonic()
        return max(0.0, remaining)

    def request_rate(self) -> float:
        now = time.monotonic()
        recent = [t for t in self._requests if now - t < self.window_seconds]
        if not recent:
            return 0.0
        return len(recent) / self.window_seconds

    def throttle_count(self) -> int:
        return len(self._throttle_events)


RATE_LIMIT_PATTERNS = [
    "429", "rate limit", "too many requests",
    "quota exceeded", "rate_limit_exceeded",
]
QUOTA_PATTERNS = [
    "quota exhausted", "billing", "insufficient_quota",
    "exceeded your current quota",
]
TIMEOUT_PATTERNS = [
    "timeout", "timed out", "connection error",
    "service unavailable", "503", "502",
]
CONTEXT_PATTERNS = [
    "context length", "maximum context", "token limit",
    "context_length_exceeded", "max_tokens",
]


class RateLimitDetector:
    def __init__(self, window_seconds: float = 60.0, max_requests: int = 100):
        self._windows: Dict[str, ProviderWindow] = {}
        self._window_seconds = window_seconds
        self._max_requests = max_requests

    def _get_window(self, provider_id: str) -> ProviderWindow:
        if provider_id not in self._windows:
            self._windows[provider_id] = ProviderWindow(
                provider_id=provider_id,
                window_seconds=self._window_seconds,
                max_requests=self._max_requests,
            )
        return self._windows[provider_id]

    def inspect(
        self,
        provider_id: str,
        error_message: str,
        retry_after: float = 5.0,
        request_id: Optional[str] = None,
    ) -> Optional[ThrottleEvent]:
        msg = error_message.lower()
        reason = ThrottleReason.NONE

        if any(p in msg for p in RATE_LIMIT_PATTERNS):
            reason = ThrottleReason.RATE_LIMIT_429
        elif any(p in msg for p in QUOTA_PATTERNS):
            reason = ThrottleReason.QUOTA_EXHAUSTED
            retry_after = 30.0
        elif any(p in msg for p in TIMEOUT_PATTERNS):
            reason = ThrottleReason.PROVIDER_TIMEOUT
        elif any(p in msg for p in CONTEXT_PATTERNS):
            reason = ThrottleReason.CONTEXT_OVERFLOW
            retry_after = 0.0

        if reason == ThrottleReason.NONE:
            return None

        return ThrottleEvent(
            provider_id=provider_id,
            reason=reason,
            timestamp=time.monotonic(),
            retry_after_seconds=retry_after,
            request_id=request_id,
        )

    def record(self, event: ThrottleEvent) -> None:
        self._get_window(event.provider_id).record_throttle(event)

    def record_request(self, provider_id: str) -> None:
        self._get_window(provider_id).record_request()

    def is_throttled(self, provider_id: str) -> bool:
        return self._get_window(provider_id).is_throttled()

    def backoff_remaining(self, provider_id: str) -> float:
        return self._get_window(provider_id).backoff_remaining()

    def request_rate(self, provider_id: str) -> float:
        return self._get_window(provider_id).request_rate()

    def stats(self) -> Dict[str, dict]:
        return {
            pid: {
                "throttled": w.is_throttled(),
                "backoff_remaining": round(w.backoff_remaining(), 2),
                "request_rate": round(w.request_rate(), 4),
                "throttle_count": w.throttle_count(),
            }
            for pid, w in self._windows.items()
        }
