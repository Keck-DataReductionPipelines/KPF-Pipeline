"""Tests for the network helpers in kpfpipe.utils.network.

No external host is contacted. The retry tests drive ``retry_request`` with a mock
callable and patch ``time.sleep``, so the backoff is inspected, not waited out. The
timeout tests need a real socket that stalls, so they use a loopback listener --
which the conftest guard allows, and which no amount of mocking could stand in for.
"""

import http.client
import json
import logging
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import requests
from pyvo.dal import DALQueryError, DALServiceError

from kpfpipe.utils.network import _RETRY_WAITS, _RETRYABLE, retry_request, simbad_client

from ._scripts import REPO_ROOT


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


# ---------------------------------------------------------------------------
# Import purity -- witnessed in a child, because this suite's conftest hides it
# ---------------------------------------------------------------------------

# Records every outbound connect, imports the module, and reports what it saw.
# Run through `python -c` so no conftest loads: the suite's own network guard
# would mask exactly the connection this is here to detect.
_IMPORT_PROBE = """
import json, socket, sys
hits = []
socket.socket.connect = lambda self, addr, *a, **k: hits.append(addr)
socket.create_connection = lambda addr, *a, **k: hits.append(addr)
import kpfpipe.modules.astro_query
print(json.dumps({
    "hits": [str(h) for h in hits],
    "astroquery": sorted(m for m in sys.modules if m.startswith("astroquery")),
}))
"""


@pytest.mark.cli
def test_importing_astro_query_touches_no_network():
    """Importing the module must not open a connection, nor pull in a client.

    ``astroquery.gaia`` builds a module-level ``GaiaClass()`` whose constructor
    fetches ESA status messages, so importing it *is* a connection. Both client
    imports are deferred into the factories to keep that off every code path that
    never queries a catalog -- this is the only thing that witnesses it.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": REPO_ROOT, "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    seen = json.loads(proc.stdout.strip().splitlines()[-1])

    assert seen["hits"] == []
    # astroquery.exceptions is imported at module scope for the retryable-error
    # tuple; it is cheap and reaches nothing. The two client modules are what must
    # stay out, so name them rather than the package.
    assert "astroquery.gaia" not in seen["astroquery"]
    assert "astroquery.simbad" not in seen["astroquery"]


# ---------------------------------------------------------------------------
# Timeouts -- two transports, two seams
# ---------------------------------------------------------------------------


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

    def add_votable_fields(self, *fields):
        pass


class TestTimeoutBounds:
    """The two transports are bounded by two different seams -- assert both.

    ``socket.setdefaulttimeout`` bounds Gaia's ``http.client`` transport but not
    SIMBAD's, which goes through pyvo over ``requests`` and passes an explicit
    ``timeout=None`` down to ``sock.settimeout``. Covering only the Gaia path would
    let the SIMBAD half regress in silence.
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
                client = simbad_client(("B",), timeout=0.5)

            # The point of the separate seam: nothing here is holding a socket
            # default, so only the mounted adapter can end this call.
            assert socket.getdefaulttimeout() is None
            raised = _raises_within(
                10, lambda: client._session.get(f"http://{host}:{port}/")
            )

        assert isinstance(raised, requests.exceptions.Timeout)

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
