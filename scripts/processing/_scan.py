"""Up-front L0 mini-database cache warming for the batch orchestrators.

The masters and science orchestrators warm the on-disk mini-database cache
(``{KPF_DATA_INPUT}/vNext/mini_db/{datecode}_L0.csv``) once, up front, in parallel
by datecode (``cache="rw"``) before fanning reductions out; every downstream
``reduce`` then reads it read-only (``cache="r"``). Writing thus happens in one
place, one thread per distinct cache file, so a future analysis pulling frames from
multiple nights can never race two writers onto the same file.

``FileHandler`` is not thread-safe (it carries the scanned night on
``self._mini_db``), so each night gets its own instance; header scans are I/O-bound
(``fits.getheader`` releases the GIL), so nights fan out across a thread pool. This
lives apart from ``_dispatch.py`` (the subprocess engine) to keep that module free
of a ``kpfpipe.utils.io`` import.
"""

import concurrent.futures
import logging

from kpfpipe.utils.io import FileHandler

logger = logging.getLogger(__name__)


def scan_night_to_cache(data_input, datecode, cache="rw"):
    """Scan one night's L0 headers under the given mini-database `cache` mode.

    Builds a fresh ``FileHandler`` for `datecode` and calls
    ``build_mini_database(datecode, cache=cache)`` -- with the default ``"rw"``,
    reuse a current cache else rescan and write. Returns the scanned ``DataFrame``,
    or ``None`` for an empty/absent night (which ``build_mini_database`` signals
    with ``ValueError`` -- warned and skipped, not fatal); any other error
    propagates.
    """
    file_handler = FileHandler({"KPF_DATA_INPUT": data_input})
    try:
        return file_handler.build_mini_database(datecode, cache=cache)
    except ValueError as e:
        logger.warning("  skipping night %s: %s", datecode, e)
        return None


def scan_datecodes(datecodes, jobs, worker, *, label="scanning"):
    """Fan `worker(dc)` out over `datecodes` in a thread pool; return the results.

    A generic parallel dispatcher for the per-night header scans (I/O bound, so a
    thread pool overlaps the NFS latency). `worker(dc)` returns ``(result, note)``:
    `result` is appended to the returned list, `note` is an optional string
    appended to that night's heartbeat, logged in completion order.
    """
    n = len(datecodes)
    logger.info("%s %d night(s) (%d workers)...", label, n, min(jobs, n) if n else 0)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {pool.submit(worker, dc): dc for dc in datecodes}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            dc = futures[future]
            result, note = future.result()
            results.append(result)
            logger.info("  [%d/%d] %s%s", i, n, dc, note)
    return results


def warm_mini_db_caches(data_input, datecodes, jobs, cache="rw"):
    """Warm the mini-database cache for every datecode, up front and in parallel.

    The masters/science entry point: writes each night's cache so the fan-out's
    ``reduce`` subprocesses read it read-only. A read-only `cache` mode (no ``"w"``)
    warms nothing, so the pre-scan is skipped entirely and ``(0, len(datecodes))``
    is returned. Otherwise, side-effect only (DataFrames discarded); returns
    ``(n_written, n_skipped)``. Fail-soft -- an empty night is skipped and any
    pool-level failure is swallowed (returns ``(0, len(datecodes))``) so warming
    never aborts the batch; each reduction falls back to an in-process scan on a
    miss.
    """
    if "w" not in cache:
        logger.info("cache=%s: read-only, skipping mini-db pre-scan", cache)
        return 0, len(datecodes)

    def _worker(dc):
        df = scan_night_to_cache(data_input, dc, cache=cache)
        return (df is not None, "" if df is not None else " (empty; skipped)")

    logger.info("pre-scanning L0 mini-db caches for %d night(s)...", len(datecodes))
    try:
        flags = scan_datecodes(datecodes, jobs, _worker, label="pre-scanning")
    except Exception as e:
        logger.warning("mini-db pre-scan failed (%s); reduces will scan in-process", e)
        return 0, len(datecodes)

    n_written = sum(1 for ok in flags if ok)
    n_skipped = len(flags) - n_written
    logger.info("pre-scan done: %d night(s) cached, %d skipped", n_written, n_skipped)
    return n_written, n_skipped
