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

Only science frames (PRIMARY ``IMTYPE == 'Object'``, the same test the timeseries
wrapper uses) are reduced; calibration frames are recorded as ``skipped``. A
heartbeat status file (``--status_file``, default ``{log_dir}/realtime_status.json``)
is rewritten every pass so operations tooling can see the watcher is alive and what
it has done. The daemon's own run.json (kind ``realtime``) is the parent of every
child reduction's record.
"""

import argparse
import concurrent.futures
import json
import logging
import os
import signal
import sys
import time

from astropy.io import fits

import kpfpipe
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.kpf import get_obs_id
from kpfpipe.utils.logger import setup_batch_logging
from kpfpipe.utils.run_record import (
    PARENT_ENV,
    RunRecord,
    collect_child_records,
    write_json_atomic,
)
from scripts._argparse import (
    data_dirs_parser,
    logging_parser,
    pool_parser,
    recipe_and_config_parser,
    resolve_dir_shortcuts,
    resolve_log_settings,
)
from scripts._dispatch import (
    _default_science_jobs,
    _run_one,
    _terminate_all_children,
    configure_runtime,
)
from scripts.processing import DEFAULT_SCIENCE_CONFIG
from scripts.processing.science import _cli_task

logger = logging.getLogger(__name__)

STATUS_SCHEMA = 1
LEDGER_SCHEMA = 1

# Launch spacing for the pool (same reason as science.py: the per-frame L0 pointing
# QC queries SIMBAD/Gaia at startup).
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
            recipe_and_config_parser(),
            data_dirs_parser(science_output=True),
            logging_parser(),
            pool_parser(jobs_help=_JOBS_HELP),
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
        help="scan once, reduce whatever is new, wait for it, and exit (nonzero if "
        "any frame failed); for cron-style use and tests",
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
    if args.job_timeout < 1:
        ap.error("--job_timeout must be >= 1")
    if args.jobs is None:
        args.jobs = _default_science_jobs()
    elif args.jobs < 1:
        ap.error("--jobs must be >= 1")
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


def classify_frame(path):
    """Return ``('science', None)`` for an ``IMTYPE == 'Object'`` frame, else
    ``('cal', reason)``.

    A header that cannot be read is classified ``cal`` with the error as the
    reason, so a corrupt or partial file is recorded and skipped rather than
    dispatched to fail loudly later -- the ledger keeps the reason.
    """
    try:
        imtype = str(fits.getheader(path, 0).get("IMTYPE", "")).strip()
    except Exception as exc:  # astropy raises a zoo of types for a bad file
        return "cal", f"unreadable header: {exc}"
    if imtype == "Object":
        return "science", None
    return "cal", f"IMTYPE={imtype or '<missing>'}"


def _utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Ledger:
    """The exactly-once ledger: one entry per ``(path, mtime, size)`` ever seen.

    States: ``skipped`` (calibration frame or unreadable), ``running``
    (dispatched, not yet reaped), ``succeeded``, ``failed``. Persisted atomically
    after every change so a restart resumes where it left off.
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
        return self

    def save(self):
        write_json_atomic(self.path, {"schema": LEDGER_SCHEMA, "entries": self.entries})

    @staticmethod
    def key(path, mtime, size):
        return f"{path}|{mtime:.6f}|{size}"

    def seen(self, key):
        return key in self.entries

    def mark(self, key, **fields):
        entry = self.entries.setdefault(key, {})
        entry.update(fields)
        self.save()
        return entry

    def counts(self):
        out = {"skipped": 0, "running": 0, "succeeded": 0, "failed": 0}
        for e in self.entries.values():
            state = e.get("state")
            if state in out:
                out[state] += 1
        out["dispatched"] = out["running"] + out["succeeded"] + out["failed"]
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
        self.record = RunRecord.start(
            log_path,
            kind="realtime",
            recipe="science",
            target="realtime",
            config=args.config or DEFAULT_SCIENCE_CONFIG,
        )
        self._prior_parent = os.environ.get(PARENT_ENV)
        os.environ[PARENT_ENV] = self.record.path
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs)
        self.futures = {}  # future -> ledger key
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
            logger.warning("skipping %s: %s", path, exc)
            self.ledger.mark(
                key, path=path, mtime=mtime, size=size, state="skipped", reason=str(exc)
            )
            return
        kind, reason = classify_frame(path)
        if kind != "science":
            logger.info("skip %s (%s)", obs_id, reason)
            self.ledger.mark(
                key,
                path=path,
                obs_id=obs_id,
                mtime=mtime,
                size=size,
                state="skipped",
                reason=reason,
            )
            return
        _tag, argv = _cli_task(
            obs_id, self.forward, config=self.args.config, recipe=self.args.recipe
        )
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
            _run_one, argv, self.args.job_timeout, _LAUNCH_INTERVAL
        )
        self.futures[future] = key

    def _reap(self):
        for future in [f for f in self.futures if f.done()]:
            key = self.futures.pop(future)
            rc, _stderr = future.result()
            state = "succeeded" if rc == 0 else "failed"
            entry = self.ledger.mark(
                key, state=state, exit_status=rc, ended_utc=_utc_now()
            )
            log = logger.info if rc == 0 else logger.warning
            log("%s %s (exit %d)", state, entry.get("obs_id"), rc)

    def write_status(self, dirs):
        c = self.ledger.counts()
        write_json_atomic(
            self.status_file,
            {
                "schema": STATUS_SCHEMA,
                "pid": os.getpid(),
                "host": self.record.record["host"],
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
                "skipped": c["skipped"],
                "running": sum(1 for f in self.futures if f.running()),
                "queued": sum(1 for f in self.futures if not f.running()),
                "succeeded": c["succeeded"],
                "failed": c["failed"],
                "ledger": self.ledger.path,
                "run_json": self.record.path,
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
        """Reap, restore the environment, finalize the record; return exit status."""
        self._reap()
        if self._prior_parent is None:
            os.environ.pop(PARENT_ENV, None)
        else:
            os.environ[PARENT_ENV] = self._prior_parent
        self.pool.shutdown(wait=False, cancel_futures=True)
        c = self.ledger.counts()
        self.write_status(self.watched_dirs())
        exit_status = 0 if c["failed"] == 0 else 1
        self.record.finish(
            exit_status,
            done=c["succeeded"],
            failed=c["failed"],
            skipped=c["skipped"],
            children=collect_child_records(self.log_dir, self.record.path),
        )
        logger.info(
            "done: %d succeeded, %d failed, %d skipped",
            c["succeeded"],
            c["failed"],
            c["skipped"],
        )
        return exit_status

    def run(self):
        """The main loop; returns the process exit status."""
        if self.args.once:
            self.tick()
            self.wait_for_inflight()
            return self.finish()
        logger.info(
            "watching %s every %.0fs (settle %.0fs, jobs %d); SIGINT/SIGTERM to stop",
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
        logger.warning(
            "stop requested; waiting up to %.0fs for %d in-flight reduction(s)",
            _SHUTDOWN_GRACE,
            len(self.futures),
        )
        if not self.wait_for_inflight(timeout=_SHUTDOWN_GRACE):
            logger.warning("grace period over; terminating remaining children")
            _terminate_all_children()
            self.wait_for_inflight(timeout=5.0)
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
    logger.info("run record: %s", rt.record.path)
    logger.info("status file: %s", rt.status_file)
    logger.info("ledger: %s", rt.ledger.path)

    def _request_stop(signum, frame):
        rt.stop = True

    # configure_runtime routed SIGTERM to KeyboardInterrupt (the batch drivers'
    # teardown path); the watcher instead stops gracefully on both signals.
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    exit_status = rt.run()
    if exit_status:
        sys.exit(exit_status)


if __name__ == "__main__":
    main()
