"""Retry and heartbeat helpers for remote STT uploads."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

# HTTP statuses worth another attempt (timeouts / overload / transient).
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 529})
_RETRYABLE_EXC_NAMES = frozenset(
    {
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "ReadTimeout",
        "WriteTimeout",
        "ConnectTimeout",
        "ConnectError",
        "RemoteProtocolError",
        "PoolTimeout",
    }
)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """App-owned retry knobs for one remote STT upload attempt loop."""

    max_retries: int
    base_delay: float
    heartbeat_seconds: float
    delay_cap: float
    sleep: Callable[[float], None] = time.sleep


def is_retryable_stt_error(exc: BaseException) -> bool:
    """Whether a remote STT failure should be retried with backoff."""
    name = type(exc).__name__
    if name in _RETRYABLE_EXC_NAMES:
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    return isinstance(status, int) and status in _RETRYABLE_STATUS


def retry_delay_seconds(attempt: int, base_delay: float, *, delay_cap: float) -> float:
    """Exponential backoff for attempt 0, 1, … capped by delay_cap."""
    return min(base_delay * (2**attempt), delay_cap)


def call_with_retry[T](
    operation: Callable[[], T],
    *,
    policy: RetryPolicy,
    on_status: Callable[[str], None],
    label: str,
) -> T:
    """Run operation with app-owned retries and an in-flight heartbeat.

    ``policy.max_retries`` is the number of *retries* after the first failure
    (total attempts = max_retries + 1), matching the summarizer convention.
    """
    last_exc: BaseException | None = None
    for attempt in range(policy.max_retries + 1):
        try:
            with _Heartbeat(on_status, label, policy.heartbeat_seconds):
                return operation()
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            last_exc = exc
            if not is_retryable_stt_error(exc) or attempt == policy.max_retries:
                raise
            delay = retry_delay_seconds(attempt, policy.base_delay, delay_cap=policy.delay_cap)
            on_status(
                f"OpenAI STT: {label} failed ({type(exc).__name__}); "
                f"retrying in {delay:.0f}s "
                f"(attempt {attempt + 1}/{policy.max_retries})..."
            )
            policy.sleep(delay)
    if last_exc is None:  # pragma: no cover
        raise RuntimeError("remote STT retry loop exited without a result")
    raise last_exc


class _Heartbeat:
    """Emit a status line every ``interval`` seconds until the context exits."""

    def __init__(
        self,
        on_status: Callable[[str], None],
        label: str,
        interval: float,
    ) -> None:
        self._on_status = on_status
        self._label = label
        self._interval = interval
        self._stop = threading.Event()
        self._started = 0.0
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _Heartbeat:
        if self._interval <= 0:
            return self
        self._started = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="stt-heartbeat", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            elapsed = time.monotonic() - self._started
            minutes = elapsed / 60.0
            self._on_status(
                f"OpenAI STT: still waiting on {self._label} ({minutes:.1f}m elapsed)..."
            )
