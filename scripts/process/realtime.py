#!/usr/bin/env python3
"""Continuous-mode L0 watcher (``kpfpipe realtime``): reduce frames as they land.

The WMKO DRP-RUN-14 "real-time script": started once, it runs unattended, polling
the current and previous UT night directories under ``{KPF_DATA_INPUT}/L0`` for
newly landed raw frames and reducing each science frame end-to-end (L0 -> L4) via
the ``kpfpipe run`` leaf, exactly once, in a bounded process pool:

    kpfpipe realtime                     # watch tonight + last night, forever
    kpfpipe realtime --once              # one scan pass, reduce what is new, exit
    kpfpipe realtime --poll_interval 30 --settle 120 --jobs 4

It reimplements no pipeline logic and builds no masters; it only decides *which
frames to hand to the leaf and when*. Three mechanisms make that safe on a live
NFS mount:

* **Polling, not inotify.** The L0 tree is an NFS mount, where kernel file events
  do not fire, so the watcher rescans every ``--poll_interval`` seconds with plain
  ``os.scandir``. No third-party dependency.
* **Settle time.** A frame is dispatched only once its mtime is at least
  ``--settle`` seconds old, so a file still being written is never reduced.
* **A persisted ledger.** Every frame seen is recorded in a JSON ledger keyed on
  ``(path, mtime, size)``; restarts and rescans are harmless, and a re-delivered
  frame (new mtime) is treated as a new frame.

Only science frames (``OBSTYPE == 'Target'``, the same test the timeseries wrapper
uses) are reduced. Other frames are recorded as ``calibration``, ``junk``
(observer junk list), or ``rejected`` (unreadable, invalid or unclassifiable
header, or a single-CCD exposure the DRP does not yet support). A
heartbeat status file (``--status_file``, default ``{log_dir}/realtime_status.json``)
is rewritten every pass so operations tooling can see the watcher is alive and what
it has done.
"""

import argparse
import concurrent.futures
import json
import logging
import os
import signal
import socket
import sys
import time

import kpfpipe
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.header import HeaderError, HeaderParser
from kpfpipe.utils.io import load_junk_obs_ids
from kpfpipe.utils.kpf import get_obs_id
from kpfpipe.utils.logger import setup_batch_logging
from scripts._argparse import (
    config_parser,
    data_dirs_parser,
    logging_parser,
    pool_parser,
    resolve_dir_shortcuts,
    resolve_log_settings,
    resolve_pool,
)
from scripts._dispatch import (
    _default_single_frame_jobs,
    _interrupted,
    _run_one,
    _terminate_all_children,
    configure_runtime,
)
from scripts.process import DEFAULT_SCIENCE_CONFIG
from scripts.process.science import _cli_task

logger = logging.getLogger(__name__)


def _write_json_atomic(path, data):
    """Write ``data`` as JSON to ``path`` atomically (temp file + ``os.replace``),
    so a reader of the ledger or status file never sees a partial write."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


STATUS_SCHEMA = 1
LEDGER_SCHEMA = 1

# Launch spacing for the pool (same reason as science.py: the per-frame L0 pointing
# QC queries SIMBAD/Gaia at startup). Default for --launch_interval.
_LAUNCH_INTERVAL = 1.0

# Seconds to wait for in-flight reductions after a stop request before killing them.
_SHUTDOWN_GRACE = 30.0

_JOBS_HELP = (
    "max concurrent reductions; left unset, defaults to a cores-based value "
    "(~25%% of CPUs, but up to 16)"
)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="kpfpipe realtime",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[
            config_parser(),
            data_dirs_parser(science_output=True),
            logging_parser(),
            pool_parser(jobs_help=_JOBS_HELP, launch_interval=_LAUNCH_INTERVAL),
        ],
    )
    ap.add_argument(
        "--poll_interval",
        type=float,
        default=15.0,
        help="seconds between L0 directory scans (default: %(default)s)",
    )
    ap.add_argument(
        "--settle",
        type=float,
        default=60.0,
        help="a frame is dispatched only once its mtime is at least this many "
        "seconds old, so a file still being written is never reduced "
        "(default: %(default)s)",
    )
    ap.add_argument(
        "--nights",
        type=int,
        default=2,
        help="how many UT night directories to watch, counting back from today "
        "(default: %(default)s, i.e. tonight and last night)",
    )
    ap.add_argument(
        "--once",
        action="store_true",
        help="scan once, reduce whatever is new, wait for it, and exit (1 if any "
        "frame failed, else 130 if stopped first); for cron-style use and tests",
    )
    ap.add_argument(
        "--status_file",
        default=None,
        help="heartbeat JSON path (default: {log_dir}/realtime_status.json)",
    )
    ap.add_argument(
        "--ledger",
        default=None,
        help="exactly-once ledger JSON path (default: {log_dir}/realtime_ledger.json)",
    )
    args = ap.parse_args(argv)
    if args.poll_interval <= 0:
        ap.error("--poll_interval must be > 0")
    if args.settle < 0:
        ap.error("--settle must be >= 0")
    if args.nights < 1:
        ap.error("--nights must be >= 1")
    args = resolve_pool(ap, args, _default_single_frame_jobs)
    return resolve_dir_shortcuts(args)


def watch_datecodes(now=None, nights=2):
    """UT datecodes to watch: today first, then ``nights - 1`` earlier nights.

    Parameters
    ----------
    now : float or None
        Epoch seconds; ``None`` means ``time.time()``.
    nights : int
        Number of consecutive UT dates, newest first.
    """
    now = time.time() if now is None else now
    return [
        time.strftime("%Y%m%d", time.gmtime(now - 86400 * i)) for i in range(nights)
    ]


def scan_settled_frames(l0_dirs, settle, now=None):
    """Regular ``KP.*.fits`` files under ``l0_dirs`` whose mtime is >= ``settle`` s old.

    Missing directories are skipped silently (a night dir appears with its first
    frame). Returns ``(path, mtime, size)`` tuples sorted by path.
    """
    now = time.time() if now is None else now
    found = []
    for d in l0_dirs:
        try:
            entries = list(os.scandir(d))
        except FileNotFoundError:
            continue
        for e in entries:
            if not (e.name.startswith("KP.") and e.name.endswith(".fits")):
                continue
            try:
                st = e.stat()
            except FileNotFoundError:
                continue
            if not e.is_file():
                continue
            if now - st.st_mtime < settle:
                continue
            found.append((e.path, st.st_mtime, st.st_size))
    found.sort()
    return found


def classify_frame(path, junk=frozenset()):
    """Return ``(state, reason)``: ``('science', None)`` for a frame to reduce
    (``OBSTYPE == 'Target'``), ``('calibration', reason)`` for a good calibration
    frame, ``('junk', reason)`` for an obs_id in `junk`, or ``('rejected', reason)``.

    A rejected frame cannot be read (``OSError``; frames are settled, so not a
    partial write), has a header ``HeaderParser`` finds nonsensical
    (``HeaderError``) or cannot classify (OBSMODE ``Unknown``; the reason is its
    warnings), or did not read out both GREEN and RED (unsupported for now). The
    ledger keeps the reason so the frame can be repaired at KOA or junked. Each
    junk/rejected verdict is logged here, at INFO/ERROR (single-CCD at WARNING).
    Any other exception is a pipeline bug and raises.
    """
    obs_id = get_obs_id(path)
    if obs_id in junk:
        logger.info("junk %s (junk frame)", obs_id)
        return "junk", "junk frame"
    try:
        parser = HeaderParser(path)
    except OSError as e:
        reason = f"unreadable L0: {e}"
    except HeaderError as e:
        reason = f"invalid header: {e}"
    else:
        keywords = parser.DRP_KEYWORDS
        if keywords["OBSMODE"] == "Unknown":
            reason = "; ".join(parser.warnings)
        elif not (keywords.get("GREEN") and keywords.get("RED")):
            reason = "DRP does not yet support single-CCD exposures"
            logger.warning("reject %s (%s)", obs_id, reason)
            return "rejected", reason
        elif keywords["OBSTYPE"] == "Target":
            return "science", None
        else:
            return "calibration", f"OBJECT={keywords['OBJECT']}"
    logger.error("reject %s (%s)", obs_id, reason)
    return "rejected", reason


def _utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Ledger:
    """The exactly-once ledger: one entry per ``(path, mtime, size)`` ever seen.

    States: ``calibration``, ``junk``, ``rejected`` (see ``classify_frame``),
    ``running`` (dispatched, not yet reaped), ``succeeded``, ``failed``,
    ``interrupted`` (dispatched but cut short by a stop, or left ``running`` by a
    watcher that died; not seen, so retried). Persisted atomically after every
    change so a restart resumes where it left off.
    """

    def __init__(self, path):
        self.path = path
        self.entries = {}

    def load(self):
        if not os.path.isfile(self.path):
            return self
        with open(self.path, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("schema") != LEDGER_SCHEMA:
            raise ValueError(
                f"{self.path}: ledger schema {data.get('schema')!r}, "
                f"expected {LEDGER_SCHEMA}"
            )
        self.entries = data.get("entries", {})
        # No reduction outlives the watcher that dispatched it.
        for e in self.entries.values():
            if e.get("state") == "running":
                e["state"] = "interrupted"
        return self

    def save(self):
        _write_json_atomic(
            self.path, {"schema": LEDGER_SCHEMA, "entries": self.entries}
        )

    @staticmethod
    def key(path, mtime, size):
        return f"{path}|{mtime:.6f}|{size}"

    def seen(self, key):
        return self.entries.get(key, {}).get("state", "interrupted") != "interrupted"

    def mark(self, key, **fields):
        entry = self.entries.setdefault(key, {})
        entry.update(fields)
        self.save()
        return entry

    def counts(self):
        out = {
            "calibration": 0,
            "junk": 0,
            "rejected": 0,
            "running": 0,
            "succeeded": 0,
            "failed": 0,
            "interrupted": 0,
        }
        for e in self.entries.values():
            state = e.get("state")
            if state in out:
                out[state] += 1
        out["dispatched"] = (
            out["running"] + out["succeeded"] + out["failed"] + out["interrupted"]
        )
        return out


class Realtime:
    """The watcher: scan -> classify -> dispatch -> reap -> heartbeat, on a loop."""

    def __init__(self, args, config, *, log_dir, log_path, run_id=None):
        self.args = args
        self.log_dir = log_dir
        self.data_input = (
            args.kpf_data_input or config.get_params(["DATA_DIRS"])["KPF_DATA_INPUT"]
        )
        self.l0_root = os.path.join(self.data_input, "L0")
        self.status_file = args.status_file or os.path.join(
            log_dir, "realtime_status.json"
        )
        self.ledger = Ledger(
            args.ledger or os.path.join(log_dir, "realtime_ledger.json")
        ).load()
        self.forward = []
        for value, flag in (
            (args.kpf_data_input, "--kpf_data_input"),
            (args.kpf_masters_output, "--kpf_masters_output"),
            (args.kpf_science_output, "--kpf_science_output"),
            # Both halves of the run directory, which each child rejoins.
            (log_dir, "--log_dir"),
            (run_id, "--run_id"),
            (args.log_level, "--log_level"),
        ):
            if value:
                self.forward += [flag, value]
        self.log_path = log_path
        self.host = socket.gethostname()
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs)
        self.futures = {}  # future -> ledger key
        self.reaped = {"succeeded": 0, "failed": 0, "interrupted": 0}  # this run
        self.stop = False
        self.started_utc = _utc_now()
        self.files_seen = 0
        self.last_scan_utc = None
        self.last_dispatch_utc = None

    # -- one pass -----------------------------------------------------------

    def watched_dirs(self, now=None):
        return [
            os.path.join(self.l0_root, dc)
            for dc in watch_datecodes(now, self.args.nights)
        ]

    def tick(self, now=None):
        """One pass: scan, dispatch new frames, reap finished ones, heartbeat."""
        dirs = self.watched_dirs(now)
        frames = scan_settled_frames(dirs, self.args.settle, now)
        self.files_seen = len(frames)
        self.last_scan_utc = _utc_now()
        for path, mtime, size in frames:
            if self.stop:
                break
            key = Ledger.key(path, mtime, size)
            if self.ledger.seen(key):
                continue
            self._admit(key, path, mtime, size)
        self._reap()
        self.write_status(dirs)

    def _admit(self, key, path, mtime, size):
        try:
            obs_id = get_obs_id(path)
        except ValueError as exc:
            logger.error("rejecting %s: %s", path, exc)
            self.ledger.mark(
                key,
                path=path,
                mtime=mtime,
                size=size,
                state="rejected",
                reason=str(exc),
            )
            return
        state, reason = classify_frame(path, load_junk_obs_ids(self.data_input))
        if state != "science":
            if state == "calibration":
                logger.info("calibration %s (%s)", obs_id, reason)
            self.ledger.mark(
                key,
                path=path,
                obs_id=obs_id,
                mtime=mtime,
                size=size,
                state=state,
                reason=reason,
            )
            return
        _tag, argv = _cli_task(obs_id, self.forward, config=self.args.config)
        logger.info("dispatch %s", obs_id)
        self.ledger.mark(
            key,
            path=path,
            obs_id=obs_id,
            mtime=mtime,
            size=size,
            state="running",
            dispatched_utc=_utc_now(),
        )
        self.last_dispatch_utc = _utc_now()
        future = self.pool.submit(
            _run_one, argv, self.args.job_timeout, self.args.launch_interval
        )
        self.futures[future] = key

    def _reap(self):
        for future in [f for f in self.futures if f.done()]:
            key = self.futures.pop(future)
            rc = None if future.cancelled() else future.result()[0]
            if rc is None or (self.stop and (rc == 130 or rc < 0)):
                state = "interrupted"  # never launched, or killed by the stop
            else:
                state = "succeeded" if rc == 0 else "failed"
            self.reaped[state] += 1
            entry = self.ledger.mark(
                key, state=state, exit_status=rc, ended_utc=_utc_now()
            )
            log = logger.info if rc == 0 else logger.warning
            log("%s %s (exit %s)", state, entry.get("obs_id"), rc)

    def write_status(self, dirs):
        c = self.ledger.counts()
        _write_json_atomic(
            self.status_file,
            {
                "schema": STATUS_SCHEMA,
                "pid": os.getpid(),
                "host": self.host,
                "started_utc": self.started_utc,
                "updated_utc": _utc_now(),
                "last_scan_utc": self.last_scan_utc,
                "last_dispatch_utc": self.last_dispatch_utc,
                "watched_dirs": dirs,
                "poll_interval": self.args.poll_interval,
                "settle": self.args.settle,
                "jobs": self.args.jobs,
                "files_seen": self.files_seen,
                "dispatched": c["dispatched"],
                # Schema 1 (KPF-Ops): skipped = deliberately not reduced.
                "skipped": c["calibration"] + c["junk"],
                "rejected": c["rejected"],
                "running": sum(1 for f in self.futures if f.running()),
                "queued": sum(1 for f in self.futures if not f.running()),
                "succeeded": c["succeeded"],
                "failed": c["failed"],
                "interrupted": c["interrupted"],
                "ledger": self.ledger.path,
                "log_path": self.log_path,
            },
        )

    # -- lifecycle ----------------------------------------------------------

    def wait_for_inflight(self, timeout=None):
        """Block until every dispatched reduction has been reaped (or ``timeout``)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.futures:
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            done, _ = concurrent.futures.wait(
                list(self.futures),
                timeout=remaining,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            self._reap()
            if not done and deadline is not None and time.monotonic() >= deadline:
                return False
        return True

    def finish(self):
        """Reap, write the final status; return the exit status: 1 if a frame
        failed in this run (earlier runs' failures stay in the ledger), else 130
        if one was interrupted or a ``--once`` pass was stopped short."""
        self._reap()
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.write_status(self.watched_dirs())
        r = self.reaped
        logger.info(
            "done: %d succeeded, %d failed, %d interrupted",
            r["succeeded"],
            r["failed"],
            r["interrupted"],
        )
        if r["failed"]:
            return 1
        if r["interrupted"] or (self.args.once and self.stop):
            return 130
        return 0

    def shutdown(self):
        """Launch nothing more, give in-flight reductions a grace period, then kill
        them; the unfinished are left ``interrupted`` for the next start."""
        _interrupted.set()  # throttled pool workers return 130 instead of launching
        for future in self.futures:
            future.cancel()  # queued, never started
        logger.warning(
            "stop requested; waiting up to %.0fs for in-flight reduction(s)",
            _SHUTDOWN_GRACE,
        )
        if not self.wait_for_inflight(timeout=_SHUTDOWN_GRACE):
            logger.warning("grace period over; terminating remaining children")
            _terminate_all_children()
            self.wait_for_inflight(timeout=5.0)

    def run(self):
        """The main loop; returns the process exit status."""
        _interrupted.clear()  # module state; a prior run in-process may have set it
        if self.args.once:
            self.tick()
            # In slices, so a stop request is honoured as in the loop below.
            while self.futures and not self.stop:
                self.wait_for_inflight(timeout=0.5)
        else:
            logger.info(
                "watching %s every %.0fs (settle %.0fs, jobs %d); "
                "SIGINT/SIGTERM to stop",
                self.l0_root,
                self.args.poll_interval,
                self.args.settle,
                self.args.jobs,
            )
            while not self.stop:
                self.tick()
                # Sleep in short slices so a stop request is honoured promptly.
                deadline = time.monotonic() + self.args.poll_interval
                while not self.stop and time.monotonic() < deadline:
                    time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
        if self.stop:
            self.shutdown()
        return self.finish()


def main(argv=None):
    configure_runtime()
    args = parse_args(argv)

    config = ConfigHandler(args.config or DEFAULT_SCIENCE_CONFIG)
    logger_params = config.get_params(["LOGGER"])
    # One run, one log directory: the watcher's own log and every frame's
    # per-reduction log land together in {log_dir}/{run_id}.
    log_dir, level = resolve_log_settings(args, logger_params)
    run_id, log_path = setup_batch_logging(
        log_dir, "realtime", args.run_id, level=level
    )

    logger.info("kpfpipe %s realtime starting", kpfpipe.__version__)
    logger.info("argv: %s", " ".join(sys.argv))
    logger.info("config: %s", args.config or DEFAULT_SCIENCE_CONFIG)
    logger.info("log: %s", log_path)

    rt = Realtime(args, config, log_dir=log_dir, log_path=log_path, run_id=run_id)
    logger.info("status file: %s", rt.status_file)
    logger.info("ledger: %s", rt.ledger.path)

    def _request_stop(signum, frame):
        rt.stop = True

    # configure_runtime routed SIGTERM/SIGHUP to KeyboardInterrupt (the batch
    # drivers' teardown path); the watcher instead stops gracefully on all three.
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGHUP, _request_stop)

    exit_status = rt.run()
    if exit_status:
        sys.exit(exit_status)


if __name__ == "__main__":
    main()
