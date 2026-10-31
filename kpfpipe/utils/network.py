"""Network helpers for the pipeline's calls out to external services."""

import logging
import random
import socket
import time

import requests
from astroquery.exceptions import RemoteServiceError
from astroquery.exceptions import TimeoutError as AstroqueryTimeoutError
from pyvo.dal import DALProtocolError
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

# Nominal seconds to wait before each retry; len() sets the retry count, so a call
# is attempted len(_RETRY_WAITS) + 1 = 5 times (~18 s of waiting worst case).
_RETRY_WAITS = (1.0, 2.0, 5.0, 10.0)

# Transient failures worth another attempt. OSError covers the builtin
# ConnectionError/TimeoutError and every requests exception (RequestException
# subclasses OSError); DALProtocolError covers pyvo's service/transport failures
# (DALServiceError) and unreadable responses (DALFormatError, which is also how pyvo
# wraps a timeout mid-read). DALQueryError -- a malformed query -- is outside it and
# deliberately absent, so a bad ADQL query fails at once instead of failing over.
_RETRYABLE = (OSError, AstroqueryTimeoutError, RemoteServiceError, DALProtocolError)


def retry_request(func, description, timeout=None):
    """Call ``func``, retrying transient network failures with backoff.

    Waits follow ``_RETRY_WAITS`` with equal jitter. A non-transient exception, or the
    last attempt's, propagates to the caller unchanged -- the AstroQuery callers turn
    that into their fail-soft warning + None.

    ``timeout`` bounds each HTTP request, not the call as a whole: one attempt makes
    several round trips, so the ceiling is ``timeout`` per round trip over 5 attempts
    plus the backoff -- minutes at a 30 s timeout, but finite.

    The catalog lookups do not use this: they go through ``query_mirrors``, bounded by
    the session adapter their clients install (pyvo passes an explicit
    ``timeout=None`` to ``requests``, which would override the socket default set here).

    ``socket.setdefaulttimeout`` is process-global and not thread-safe; the
    ``try/finally`` and the tight window around ``func()`` are what make it acceptable.
    Queries run single-threaded inside each child process today (``_dispatch.py``
    threads supervise child *processes*, not queries); revisit this if that changes.

    Parameters
    ----------
    func : callable
        Zero-argument callable performing the request. Build the service client inside
        it: both clients reach the network while being constructed, and only what runs
        inside ``func`` is bounded.
    description : str
        Service name used in the retry log messages.
    timeout : float, optional
        Seconds to bound each request. None leaves the process default untouched.

    Returns
    -------
    object
        Whatever ``func`` returns.

    Raises
    ------
    Exception
        Whatever ``func`` raises, once retries are exhausted or immediately if the
        failure is not transient. A tripped socket timeout is a ``TimeoutError``, which
        ``_RETRYABLE`` covers via OSError, so it retries like any other.
    """

    def attempt():
        previous = socket.getdefaulttimeout()
        socket.setdefaulttimeout(timeout)
        try:
            return func()
        finally:
            socket.setdefaulttimeout(previous)

    for wait in _RETRY_WAITS:
        try:
            return attempt()
        except _RETRYABLE as e:
            # Equal jitter -- half the nominal wait plus a random half -- so parallel
            # workers that failed together do not retry in lockstep.
            delay = wait / 2 + random.uniform(0, wait / 2)
            logger.warning(
                "%s request failed (%s: %s); retrying in %.1f s",
                description,
                type(e).__name__,
                e,
                delay,
            )
            time.sleep(delay)
    return attempt()


class _TimeoutAdapter(HTTPAdapter):
    """requests adapter supplying a default timeout when the caller passes none."""

    def __init__(self, timeout, **kwargs):
        self._timeout = timeout
        super().__init__(**kwargs)

    def send(self, request, stream=False, timeout=None, **kwargs):
        if timeout is None:
            timeout = self._timeout
        return super().send(request, stream=stream, timeout=timeout, **kwargs)


def _bound(session, timeout):
    """Mount a ``_TimeoutAdapter`` on ``session``: 5 s to connect, ``timeout`` to read.

    A short connect bound lets a refused or unrouted mirror fail over quickly; the read
    bound covers a mirror that accepts the connection and then hangs.
    """
    adapter = _TimeoutAdapter((5.0, timeout))
    session.mount("https://", adapter)
    session.mount("http://", adapter)


def query_mirrors(func, mirrors, description, budget=60.0):
    """Return ``func(mirror)`` from the first of ``mirrors`` that answers.

    Mirrors are tried in order, in two passes 2 s apart, so a blip that hits every
    mirror at once still recovers. Only transient failures (``_RETRYABLE``) fail over;
    anything else -- a malformed query, an unknown table -- propagates at once. No new
    attempt starts once ``budget`` seconds have passed, so the stall is bounded by
    ``budget`` plus one in-flight attempt however many mirrors are listed.

    Parameters
    ----------
    func : callable
        One-argument callable performing the request against the mirror it is given.
        Build the service client inside it, so its construction is bounded too.
    mirrors : sequence
        Mirrors in priority order; each is passed to ``func`` as is.
    description : str
        Service name used in the log messages.
    budget : float
        Seconds after which no new attempt starts.

    Returns
    -------
    object
        Whatever ``func`` returns.

    Raises
    ------
    Exception
        The last transient failure once mirrors or budget are exhausted, or a
        non-transient one immediately.
    """
    deadline = time.monotonic() + budget
    error = None
    for wait in (0.0, 2.0):
        time.sleep(wait)
        for mirror in mirrors:
            if time.monotonic() >= deadline:
                break
            try:
                result = func(mirror)
            except _RETRYABLE as e:
                logger.warning(
                    "%s mirror %s failed (%s: %s)",
                    description,
                    mirror,
                    type(e).__name__,
                    e,
                )
                error = e
                continue
            logger.info("%s answered by %s", description, mirror)
            return result
    raise error


def gaia_client(url, timeout):
    """Return a TAP client for the Gaia mirror at ``url``, bounded by ``timeout``.

    pyvo's generic client rather than ``astroquery.gaia``, whose ``GaiaClass`` is tied
    to ESA and whose import alone contacts ESA.
    """
    # Deferred: import cost, and to keep pyvo off the paths that never query a catalog.
    import pyvo

    session = requests.Session()
    _bound(session, timeout)
    return pyvo.dal.TAPService(url, session=session)


def simbad_client(server, fields, timeout):
    """Return a client for the SIMBAD mirror ``server`` requesting ``fields``.

    The timeout goes on before ``add_votable_fields``, which validates the names
    server-side and is itself a network call.

    ``SimbadClass()`` is constructed explicitly because astroquery's ``Simbad`` is an
    instance, not a class; calling it works only via ``BaseQuery.__call__``.
    """
    # Deferred: import cost, and to keep astroquery off the paths that never query a
    # catalog.
    from astroquery.simbad import SimbadClass

    client = SimbadClass()
    client.server = server
    _bound(client._session, timeout)
    client.add_votable_fields(*fields)
    return client


def cfht_archive(year, start, length, timeout=30):
    """Return ``length`` bytes of the CFHT tower's weather archive for ``year``.

    ``start`` counts back from the end of the file when negative. Returned with the
    file's total length, which the range response reports, so a caller seeking
    through the file need not ask for it separately.
    """
    span = f"-{length}" if start < 0 else f"{start}-{start + length - 1}"
    response = requests.get(
        f"http://mkwc.ifa.hawaii.edu/archive/wx/cfht/cfht-wx.{year}.dat",
        headers={"Range": f"bytes={span}"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.text, int(response.headers["Content-Range"].rpartition("/")[2])
