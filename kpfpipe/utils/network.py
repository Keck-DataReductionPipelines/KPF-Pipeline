"""Network helpers for the pipeline's calls out to external services."""

import logging
import random
import time

from astroquery.exceptions import RemoteServiceError
from astroquery.exceptions import TimeoutError as AstroqueryTimeoutError
from pyvo.dal import DALServiceError

logger = logging.getLogger(__name__)

# Nominal seconds to wait before each retry; len() sets the retry count, so a call
# is attempted len(_RETRY_WAITS) + 1 times (~44 s of waiting worst case).
_RETRY_WAITS = (1.0, 3.0, 10.0, 30.0)

# Transient failures worth another attempt. OSError covers the builtin
# ConnectionError/TimeoutError and every requests exception (RequestException
# subclasses OSError). pyvo's DALServiceError is a service/transport failure; its
# sibling DALQueryError -- a malformed query -- is deliberately absent, as is every
# other exception, so a bad ADQL query or an unknown object fails at once instead of
# burning the full backoff.
_RETRYABLE = (OSError, AstroqueryTimeoutError, RemoteServiceError, DALServiceError)


def retry_request(func, description):
    """Call ``func``, retrying transient network failures with backoff.

    ``description`` names the service in the retry warnings (e.g. "Gaia DR3").
    Waits follow ``_RETRY_WAITS`` with equal jitter. A non-transient exception, or
    the last attempt's exception, propagates to the caller unchanged -- the
    AstroQuery callers turn that into their existing fail-soft warning + None.

    Note there is no per-attempt timeout: astroquery 0.4.11 exposes no socket
    timeout for either client (``Gaia.launch_job`` takes none and its TAP transport
    builds an ``http.client`` connection without one; SIMBAD's ``timeout`` is a
    server-side execution duration), so a hung connection is not bounded here.

    Parameters
    ----------
    func : callable
        Zero-argument callable performing the request.
    description : str
        Service name used in the retry log messages.

    Returns
    -------
    object
        Whatever ``func`` returns.

    Raises
    ------
    Exception
        Whatever ``func`` raises, once retries are exhausted or immediately if the
        failure is not transient.
    """
    for wait in _RETRY_WAITS:
        try:
            return func()
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
    return func()
