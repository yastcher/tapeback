"""Remote STT retry / heartbeat helpers."""

from tapeback._stt_retry import (
    RetryPolicy,
    call_with_retry,
    is_retryable_stt_error,
    retry_delay_seconds,
)


class _Timeout(Exception):
    """Stand-in for openai.APITimeoutError without importing the SDK."""


_Timeout.__name__ = "APITimeoutError"


class _BadRequest(Exception):
    status_code = 400


_BadRequest.__name__ = "BadRequestError"


class _ServerError(Exception):
    status_code = 503


def test_is_retryable_timeout_by_name():
    assert is_retryable_stt_error(_Timeout()) is True


def test_is_retryable_http_status():
    assert is_retryable_stt_error(_ServerError()) is True
    assert is_retryable_stt_error(_BadRequest()) is False


def test_retry_delay_caps_at_60():
    assert retry_delay_seconds(0, 5.0, delay_cap=60.0) == 5.0
    assert retry_delay_seconds(1, 5.0, delay_cap=60.0) == 10.0
    assert retry_delay_seconds(10, 5.0, delay_cap=60.0) == 60.0


def test_call_with_retry_succeeds_after_timeouts():
    attempts = {"n": 0}
    status: list[str] = []

    def _op():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _Timeout("slow")
        return "ok"

    slept: list[float] = []
    result = call_with_retry(
        _op,
        policy=RetryPolicy(
            max_retries=5,
            base_delay=5.0,
            heartbeat_seconds=0,
            delay_cap=60.0,
            sleep=slept.append,
        ),
        on_status=status.append,
        label="transcribe chunk 1/2",
    )
    assert result == "ok"
    assert attempts["n"] == 3
    assert slept == [5.0, 10.0]
    assert any("retrying" in line for line in status)


def test_call_with_retry_does_not_retry_bad_request():
    attempts = {"n": 0}

    def _op():
        attempts["n"] += 1
        raise _BadRequest("invalid")

    try:
        call_with_retry(
            _op,
            policy=RetryPolicy(
                max_retries=5,
                base_delay=5.0,
                heartbeat_seconds=0,
                delay_cap=60.0,
                sleep=lambda _d: None,
            ),
            on_status=lambda _m: None,
            label="upload",
        )
    except _BadRequest:
        pass
    else:
        raise AssertionError("expected BadRequest")
    assert attempts["n"] == 1
