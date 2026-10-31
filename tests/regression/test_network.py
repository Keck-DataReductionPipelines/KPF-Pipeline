"""Tests for the network helpers in kpfpipe.utils.network.

No external host is contacted. The retry and failover tests drive ``retry_request``
and ``query_mirrors`` with mock callables and patch ``time.sleep``, so the waits are
inspected, not waited out. The
timeout tests need a real socket that stalls, so they use a loopback listener --
which the conftest guard allows, and which no amount of mocking could stand in for.
"""

import http.client
import itertools
import logging
import socket
import threading
import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import requests
from pyvo.dal import DALFormatError, DALQueryError, DALServiceError

from kpfpipe.utils.network import (
    _RETRY_WAITS,
    _RETRYABLE,
    cfht_archive,
    gaia_client,
    query_mirrors,
    retry_request,
    simbad_client,
)


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
        "exc",
        [
            TimeoutError("timed out"),
            OSError("reset"),
            DALServiceError("503"),
            DALFormatError("read timed out mid-response"),
        ],
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


class TestQueryMirrors:
    """Ordered failover: the first mirror to answer wins; two passes, capped."""

    def test_first_mirror_answers_and_the_rest_are_untouched(self):
        func = MagicMock(return_value="ok")
        assert query_mirrors(func, ("a", "b"), "Service") == "ok"
        assert [c.args[0] for c in func.call_args_list] == ["a"]

    def test_transient_failure_fails_over_to_next_mirror(self, caplog):
        func = MagicMock(side_effect=[ConnectionError("a down"), "ok"])
        with caplog.at_level(logging.INFO):
            assert query_mirrors(func, ("a", "b"), "Service") == "ok"
        assert [c.args[0] for c in func.call_args_list] == ["a", "b"]
        assert "Service mirror a failed (ConnectionError: a down)" in caplog.text
        assert "Service answered by b" in caplog.text

    def test_non_transient_error_does_not_fail_over(self):
        # A malformed query is wrong on every mirror; trying the next would hide it.
        func = MagicMock(side_effect=DALQueryError("bad ADQL"))
        with pytest.raises(DALQueryError):
            query_mirrors(func, ("a", "b"), "Service")
        assert func.call_count == 1

    def test_all_mirrors_down_tries_two_passes_then_raises_last(self):
        errors = [ConnectionError(f"down {i}") for i in range(4)]
        func = MagicMock(side_effect=errors)
        with _patch_sleep() as sleep, pytest.raises(ConnectionError, match="down 3"):
            query_mirrors(func, ("a", "b"), "Service")
        assert [c.args[0] for c in func.call_args_list] == ["a", "b", "a", "b"]
        assert _slept(sleep) == [0.0, 2.0]

    def test_no_attempt_starts_past_the_budget(self):
        # The clock reads 0 for the deadline and the first check, then 61 -- past 60 s.
        clock = itertools.chain([0.0, 0.0], itertools.repeat(61.0))
        func = MagicMock(side_effect=ConnectionError("down"))
        with (
            _patch_sleep(),
            patch("kpfpipe.utils.network.time.monotonic", lambda: next(clock)),
            pytest.raises(ConnectionError),
        ):
            query_mirrors(func, ("a", "b", "c"), "Service", budget=60.0)
        assert func.call_count == 1


# ------------------------------------------------------------------
# Timeouts -- two transports, two seams
# ------------------------------------------------------------------


@contextmanager
def _stalled_listener():
    """A loopback address that accepts a connection and then never answers.

    The listen backlog completes the TCP handshake without anyone calling
    ``accept``, so the client blocks on the read -- the shape of the hang these
    timeouts exist to bound.
    """
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    try:
        yield server.getsockname()
    finally:
        server.close()


def _raises_within(seconds, func):
    """Run ``func`` in a daemon thread; return the exception it raised.

    Both seams fail by *not returning*, so calling ``func`` inline would hang the
    suite instead of failing it. The thread is abandoned on timeout -- it is stuck
    on a loopback read and dies with the interpreter.
    """
    outcome = []
    thread = threading.Thread(
        target=lambda: outcome.append(_capture(func)), daemon=True
    )
    started = time.monotonic()
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), (
        f"still blocked after {seconds:.0f}s: nothing bounds it"
    )
    assert time.monotonic() - started < seconds
    return outcome[0]


def _capture(func):
    try:
        func()
    except BaseException as exc:  # noqa: BLE001 -- the exception is the result
        return exc
    return None


class _FakeSimbadClass:
    """SimbadClass stand-in: the real one validates fields over the network.

    Carries the one attribute ``simbad_client`` touches, so the adapter it mounts
    is exercised on a genuine requests.Session.
    """

    def __init__(self):
        self._session = requests.Session()
        self.server = None

    def add_votable_fields(self, *fields):
        pass


class TestTimeoutBounds:
    """The two transports are bounded by two different seams -- assert both.

    ``retry_request``'s ``socket.setdefaulttimeout`` bounds an ``http.client``
    transport but not the catalog clients, which go through pyvo over ``requests`` and
    pass an explicit ``timeout=None`` down to ``sock.settimeout``. Those are bounded by
    the session adapter their factories mount, so each factory is asserted directly.
    """

    def test_timeout_bounds_the_http_client_transport(self):
        with _stalled_listener() as (host, port):

            def request():
                conn = http.client.HTTPConnection(host, port)
                try:
                    conn.request("GET", "/")
                    conn.getresponse()
                finally:
                    conn.close()

            with _patch_sleep():
                raised = _raises_within(
                    0.5 * (len(_RETRY_WAITS) + 1) + 10,
                    lambda: retry_request(request, "stall", timeout=0.5),
                )

        # Classification matters as much as the bound: a timeout the retry loop
        # did not recognise would escape on the first attempt instead of retrying.
        assert isinstance(raised, _RETRYABLE)

    def test_simbad_timeout_bounds_a_requests_call(self):
        with _stalled_listener() as (host, port):
            with patch("astroquery.simbad.SimbadClass", _FakeSimbadClass):
                client = simbad_client("simbad.harvard.edu", ("B",), timeout=0.5)
            assert client.server == "simbad.harvard.edu"

            # The point of the separate seam: nothing here is holding a socket
            # default, so only the mounted adapter can end this call.
            assert socket.getdefaulttimeout() is None
            raised = _raises_within(
                10, lambda: client._session.get(f"http://{host}:{port}/")
            )

        assert isinstance(raised, requests.exceptions.Timeout)

    def test_gaia_timeout_bounds_a_tap_query(self):
        with _stalled_listener() as (host, port):
            client = gaia_client(f"http://{host}:{port}/tap", timeout=0.5)
            assert socket.getdefaulttimeout() is None
            raised = _raises_within(10, lambda: client.run_sync("SELECT 1"))

        # pyvo wraps the timeout; it must still be one query_mirrors fails over on.
        assert isinstance(raised, _RETRYABLE)

    def test_socket_default_restored_after_success(self):
        socket.setdefaulttimeout(7.0)
        try:
            retry_request(lambda: "ok", "Service", timeout=1.0)
            assert socket.getdefaulttimeout() == 7.0
        finally:
            socket.setdefaulttimeout(None)

    def test_socket_default_restored_after_failure(self):
        # The non-transient path: one attempt, then the raise escapes the loop.
        socket.setdefaulttimeout(7.0)
        try:
            with pytest.raises(ValueError):
                retry_request(_raise(ValueError("bad")), "Service", timeout=1.0)
            assert socket.getdefaulttimeout() == 7.0
        finally:
            socket.setdefaulttimeout(None)

    def test_none_timeout_leaves_the_default_alone(self):
        assert socket.getdefaulttimeout() is None
        retry_request(lambda: "ok", "Service")
        assert socket.getdefaulttimeout() is None


def _raise(exc):
    def func():
        raise exc

    return func


class TestCfhtArchive:
    """Byte-range reads of the CFHT tower's one-file-per-year weather archive."""

    @staticmethod
    def _response(text="rows\n", total=4096):
        response = MagicMock()
        response.text = text
        response.headers = {"Content-Range": f"bytes 0-{len(text) - 1}/{total}"}
        return response

    def test_range_counts_from_the_start(self):
        with patch("kpfpipe.utils.network.requests.get") as get:
            get.return_value = self._response()
            cfht_archive(2024, 1000, 256)
        assert get.call_args.kwargs["headers"] == {"Range": "bytes=1000-1255"}
        assert "cfht-wx.2024.dat" in get.call_args.args[0]

    def test_a_negative_start_reads_the_tail(self):
        with patch("kpfpipe.utils.network.requests.get") as get:
            get.return_value = self._response()
            cfht_archive(2024, -256, 256)
        assert get.call_args.kwargs["headers"] == {"Range": "bytes=-256"}

    def test_the_files_total_length_is_returned(self):
        with patch("kpfpipe.utils.network.requests.get") as get:
            get.return_value = self._response(total=21886)
            assert cfht_archive(2024, 0, 256) == ("rows\n", 21886)

    def test_http_error_raises(self):
        response = MagicMock()
        response.raise_for_status.side_effect = requests.HTTPError("416")
        with patch("kpfpipe.utils.network.requests.get", return_value=response):
            with pytest.raises(requests.HTTPError):
                cfht_archive(2024, 0, 256)
