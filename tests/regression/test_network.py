"""Tests for the network retry helper in kpfpipe.utils.network.

The network is never touched: every test drives ``retry_request`` with a mock
callable and patches ``time.sleep``, so the backoff is inspected, not waited out.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
from pyvo.dal import DALQueryError, DALServiceError

from kpfpipe.utils.network import _RETRY_WAITS, retry_request


def _patch_sleep():
    """Patch the helper's time.sleep; its call args are the backoff waits."""
    return patch("kpfpipe.utils.network.time.sleep")


def _slept(sleep_mock):
    """The delays passed to the patched sleep, in order."""
    return [call.args[0] for call in sleep_mock.call_args_list]


class TestRetryRequest:
    def test_success_returns_immediately(self):
        func = MagicMock(return_value="ok")
        with _patch_sleep() as sleep:
            assert retry_request(func, "Service") == "ok"
        assert func.call_count == 1
        assert sleep.call_count == 0

    def test_retries_then_succeeds(self):
        func = MagicMock(
            side_effect=[ConnectionError("down"), ConnectionError("down"), "ok"]
        )
        with _patch_sleep() as sleep:
            assert retry_request(func, "Service") == "ok"
        assert func.call_count == 3
        assert sleep.call_count == 2

    def test_exhausted_retries_raises_last_exception(self):
        func = MagicMock(side_effect=ConnectionError("still down"))
        with (
            _patch_sleep() as sleep,
            pytest.raises(ConnectionError, match="still down"),
        ):
            retry_request(func, "Service")
        # One attempt per wait, plus the final attempt outside the loop.
        assert func.call_count == len(_RETRY_WAITS) + 1
        assert sleep.call_count == len(_RETRY_WAITS)

    @pytest.mark.parametrize(
        "exc", [ValueError("bad input"), DALQueryError("malformed ADQL")]
    )
    def test_non_transient_exception_is_not_retried(self, exc):
        # A bad query or bad input never succeeds on a retry, so it must fail at
        # once rather than burn the full backoff.
        func = MagicMock(side_effect=exc)
        with _patch_sleep() as sleep, pytest.raises(type(exc)):
            retry_request(func, "Service")
        assert func.call_count == 1
        assert sleep.call_count == 0

    @pytest.mark.parametrize(
        "exc", [TimeoutError("timed out"), OSError("reset"), DALServiceError("503")]
    )
    def test_transient_exception_types_are_retried(self, exc):
        func = MagicMock(side_effect=[exc, "ok"])
        with _patch_sleep():
            assert retry_request(func, "Service") == "ok"
        assert func.call_count == 2

    def test_backoff_schedule_with_equal_jitter(self):
        func = MagicMock(side_effect=ConnectionError("down"))
        with _patch_sleep() as sleep, pytest.raises(ConnectionError):
            retry_request(func, "Service")
        delays = _slept(sleep)
        assert len(delays) == len(_RETRY_WAITS)
        # Equal jitter: half the nominal wait is fixed, the other half random.
        for delay, wait in zip(delays, _RETRY_WAITS, strict=True):
            assert wait / 2 <= delay <= wait
        # Only the nominal schedule is monotonic; realized delays can cross where
        # adjacent waits' jitter ranges overlap.
        assert list(_RETRY_WAITS) == sorted(_RETRY_WAITS)

    def test_retry_logs_warning(self, caplog):
        func = MagicMock(side_effect=[ConnectionError("down"), "ok"])
        with _patch_sleep(), caplog.at_level(logging.WARNING):
            retry_request(func, "Gaia DR3")
        assert "Gaia DR3 request failed" in caplog.text
        assert "ConnectionError" in caplog.text
        assert "retrying in" in caplog.text
