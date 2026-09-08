"""Network helpers for the pipeline's calls out to external services."""

import logging
import random
import socket
import time

import requests
from astroquery.exceptions import RemoteServiceError
from astroquery.exceptions import TimeoutError as AstroqueryTimeoutError
from pyvo.dal import DALServiceError
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

# Nominal seconds to wait before each retry; len() sets the retry count, so a call
# is attempted len(_RETRY_WAITS) + 1 = 5 times (~18 s of waiting worst case).
_RETRY_WAITS = (1.0, 2.0, 5.0, 10.0)

# Transient failures worth another attempt. OSError covers the builtin
# ConnectionError/TimeoutError and every requests exception (RequestException
# subclasses OSError); DALServiceError is a pyvo service/transport failure. Its
# sibling DALQueryError -- a malformed query -- is deliberately absent, so a bad ADQL
# query or an unknown object fails at once instead of burning the full backoff.
_RETRYABLE = (OSError, AstroqueryTimeoutError, RemoteServiceError, DALServiceError)


def retry_request(func, description, timeout=None):
    """Call ``func``, retrying transient network failures with backoff.

    Waits follow ``_RETRY_WAITS`` with equal jitter. A non-transient exception, or the
    last attempt's, propagates to the caller unchanged -- the AstroQuery callers turn
    that into their fail-soft warning + None.

    ``timeout`` bounds each HTTP request, not the call as a whole: one attempt makes
    several round trips, so the ceiling is ``timeout`` per round trip over 5 attempts
    plus the backoff -- minutes at the 30 s default, but finite.

    It covers only Gaia. Gaia's TAP transport is ``http.client``, which honours the
    process-wide socket default set here; SIMBAD's is ``requests`` via pyvo, which
    passes an explicit ``timeout=None`` down to ``sock.settimeout`` and so *overrides*
    that default. SIMBAD is bounded instead by the adapter ``simbad_client`` installs.
    Do not collapse the two -- dropping that adapter leaves SIMBAD unbounded.

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


def gaia_client():
    """Return a Gaia TAP client that does not fetch server status messages.

    ``astroquery.gaia`` reaches ESA twice: once building its module-level
    ``Gaia = GaiaClass()``, whose constructor fetches status messages, and again per
    query. ``show_server_messages=False`` kills the second. The first is not gone, only
    moved out of the pipeline's import graph -- importing the module still triggers
    it -- so call this inside the callable handed to ``retry_request``, where it is
    bounded.
    """
    # Deferred: this import *is* the connection to ESA, so it must not run until a
    # caller has decided to query Gaia.
    from astroquery.gaia import GaiaClass

    return GaiaClass(show_server_messages=False)


def simbad_client(fields, timeout=None):
    """Return a SIMBAD client requesting ``fields``, bounded by ``timeout``.

    A session-level default rather than a socket one, because SIMBAD is reached through
    pyvo over ``requests`` -- see ``retry_request``. It goes on before
    ``add_votable_fields``, which validates the names server-side and is itself a
    network call.

    ``SimbadClass()`` is constructed explicitly because astroquery's ``Simbad`` is an
    instance, not a class; calling it works only via ``BaseQuery.__call__``.
    """
    # Deferred: import cost, and to keep astroquery off the paths that never query a
    # catalog.
    from astroquery.simbad import SimbadClass

    client = SimbadClass()
    if timeout is not None:
        adapter = _TimeoutAdapter(timeout)
        client._session.mount("https://", adapter)
        client._session.mount("http://", adapter)
    client.add_votable_fields(*fields)
    return client


def cfht_weather(date, timeout=30):
    """The Mauna Kea Weather Center's CFHT tower readings for one HST ``date``.

    Returned as the archive's own text: one row per minute, fields documented at
    http://mkwc.ifa.hawaii.edu/archive/wx/cfht/format.txt. The archive stores a
    whole year per file, so the day is cut out with a Range request rather than
    downloading 21 MB -- rows average 42.3 bytes and the running offset drifts up
    to 131 kB from that over a year, which the 256 kB of padding covers. The rows
    bounding the range are cut in half and so match no timestamp.
    """
    day = date.timetuple().tm_yday - 1
    start = max(int(day * 1440 * 42.3) - 262144, 0)
    end = int((day + 1) * 1440 * 42.3) + 262144
    url = f"http://mkwc.ifa.hawaii.edu/archive/wx/cfht/cfht-wx.{date.year}.dat"

    def fetch():
        response = requests.get(
            url, headers={"Range": f"bytes={start}-{end}"}, timeout=timeout
        )
        response.raise_for_status()
        return response.text

    return retry_request(fetch, "CFHT weather")
