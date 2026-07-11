#!/usr/bin/env python3
"""Build nightly master calibrations for a set of datecodes (``kpfpipe masters``).

A lightweight, fail-loud orchestrator: it dispatches each night as a separate
``python -m scripts.processing.reduce --masters -d <datecode>`` subprocess (the
``kpfpipe run`` leaf), so every night gets its own log file, clean process state,
and independent exit code. It reimplements no pipeline logic; the caller supplies
which nights to build, via two mutually exclusive input forms:

    kpfpipe masters --dates 20240405 20240712       # explicit datecode(s)
    kpfpipe masters --dates nights.txt              # a file of datecodes
    kpfpipe masters --date_range 20240101 20240131  # every L0 night in range

The range form enumerates the datecode dirs present under ``{KPF_DATA_INPUT}/L0``
within [START, END].

Builds run in a bounded process pool. The L0 mini-database caches are warmed up
front by a parallel per-datecode pre-scan; the first night then runs alone as a
canary to warm the *other* shared caches (barycorrpy leap-seconds, astropy IERS,
matplotlib fonts, bytecode) before the rest fan out. The run is fail-soft (a night
that fails to build is reported and the others continue) but exits nonzero if any
night failed.
"""

import argparse
import logging
import os
import sys

import kpfpipe
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.io import datecode_dirs_in_range, read_token_file
from kpfpipe.utils.kpf_utils import is_datecode
from kpfpipe.utils.logger import setup_batch_logging
from scripts.processing import DEFAULT_MASTERS_CONFIG, DEFAULT_MASTERS_RECIPE
from scripts.processing._argparse import (
    data_dirs_parser,
    logging_parser,
    pool_parser,
    recipe_and_config_parser,
    resolve_dir_shortcuts,
)
from scripts.processing._dispatch import (
    _MASTERS_JOBS,
    _default_masters_jobs,
    configure_runtime,
    run_stage,
)
from scripts.processing._scan import warm_mini_db_caches

logger = logging.getLogger(__name__)

# Minimum spacing (seconds) between fanned-out subprocess launches. A masters build
# opens with an I/O-heavy stack phase -- reading and assembling the night's bias,
# dark and ThAr L0 frames (~100 s solo) before any WLS fitting. Launched in lockstep,
# a full pool marches through that read phase together, so the disk sees `jobs`-deep
# read bursts and the shared phase balloons ~4-5x (measured ~465 s at 16-wide on
# shrek), pushing the slowest nights past --job_timeout. Spacing the launches apart
# desynchronizes the read bursts -- staggered 16-wide jobs held that phase near the
# ~100 s solo cost -- without lowering steady-state concurrency: at 5 s the 16-job
# first wave spreads over ~75 s (most of the read window) while staying well under
# the throughput ceiling (~job_duration/jobs), so the pool still fills. (Tuned from
# the 2026-07-09 shrek logs; confirm against a real run before trusting the exact
# value.) Passed to run_stage as its launch_interval.
_LAUNCH_INTERVAL = 5.0

# --jobs help is command-specific (the fixed masters cap, not the cores default),
# so it is passed into the shared pool_parser rather than baked into it.
_JOBS_HELP = (
    "max concurrent masters builds; left unset, defaults to the "
    f"{_MASTERS_JOBS}-job cap (stacking degrades from OS memory contention when too "
    "many run at once), floored by cores and RAM on small machines"
)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="kpfpipe masters",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[
            recipe_and_config_parser(),
            data_dirs_parser(science_output=False),
            logging_parser(),
            pool_parser(jobs_help=_JOBS_HELP),
        ],
    )
    ap.add_argument(
        "--dates",
        nargs="*",
        default=None,
        metavar="DATECODE_OR_FILE",
        help="one or more datecodes to build, or a text file listing one datecode "
        "per line, e.g. --dates 20240405 20240712 or --dates nights.txt (mutually "
        "exclusive with --date_range)",
    )
    ap.add_argument(
        "--date_range",
        nargs=2,
        metavar=("START", "END"),
        help="inclusive datecode range; builds every L0 night in it, e.g. "
        "--date_range 20240101 20240131 (mutually exclusive with --dates)",
    )
    args = ap.parse_args(argv)

    # Exactly one input form: an explicit datecode list, or a range.
    if bool(args.dates) == bool(args.date_range):
        ap.error("give either --dates or --date_range, not both or neither")

    if args.date_range:
        start, end = args.date_range
        for dc in (start, end):
            if not is_datecode(dc):
                ap.error(f"--date_range value is not a valid datecode: {dc!r}")
        if start > end:
            ap.error(f"--date_range START must be <= END (got {start} > {end})")
    else:
        # Each --dates value is a datecode (built as-is) or a text file of
        # datecodes; expand file entries in place. A valid datecode is always read
        # as such, even if a like-named file exists.
        datecodes = []
        for entry in args.dates:
            if is_datecode(entry):
                datecodes.append(entry)
            elif os.path.isfile(entry):
                for dc in read_token_file(entry):
                    if not is_datecode(dc):
                        ap.error(f"not a valid datecode in {entry}: {dc!r}")
                    datecodes.append(dc)
            else:
                ap.error(
                    f"--dates entry is neither a datecode nor a readable file: "
                    f"{entry!r}"
                )
        if not datecodes:
            ap.error(f"--dates produced no datecodes (empty file?): {args.dates}")
        args.dates = sorted(set(datecodes))

    if args.job_timeout < 1:
        ap.error("--job_timeout must be >= 1")
    if args.jobs is None:
        args.jobs = _default_masters_jobs()
    elif args.jobs < 1:
        ap.error("--jobs must be >= 1")
    return resolve_dir_shortcuts(args)


def resolve_datecodes(args, data_input):
    """The datecodes to build, from either input form.

    The explicit list is already validated and sorted in parse_args. A range is
    expanded here, since it needs the resolved L0 input root: the datecode dirs
    under {data_input}/L0 within the range. Either way an empty result is fatal.
    """
    if args.dates:
        return args.dates
    l0_root = os.path.join(data_input, "L0")
    if not os.path.isdir(l0_root):
        sys.exit(f"error: L0 input directory not found: {l0_root}")
    start, end = args.date_range
    nights = datecode_dirs_in_range(l0_root, start, end)
    if not nights:
        sys.exit(f"error: no datecode dirs under {l0_root} in range {start}..{end}")
    return nights


def _cli_task(datecode, forward, config=None, recipe=None):
    """Build a (tag, argv) task building one night via the ``kpfpipe run`` leaf.

    Passes recipe/config to the leaf explicitly (the masters defaults unless
    `recipe`/`config` override) rather than leaning on its ``--masters`` shortcut,
    so the orchestrator is the single owner of what it runs. `forward` is the
    resolved dir/log overrides appended to every invocation; the tag (datecode)
    keys the sentinel/log paths.
    """
    argv = [
        sys.executable,
        "-m",
        "scripts.processing.reduce",
        "-r",
        recipe or DEFAULT_MASTERS_RECIPE,
        "-c",
        config or DEFAULT_MASTERS_CONFIG,
        "-d",
        datecode,
        *forward,
    ]
    return datecode, argv


def main(argv=None):
    configure_runtime()
    args = parse_args(argv)

    # Read the effective config (the masters default unless -c overrides) to resolve
    # the L0 input root and log dir; subprocesses get recipe/config via -r/-c.
    config = ConfigHandler(args.config or DEFAULT_MASTERS_CONFIG)
    logger_params = config.get_params(["LOGGER"])
    data_input = (
        args.kpf_data_input or config.get_params(["DATA_DIRS"])["KPF_DATA_INPUT"]
    )
    log_dir = args.log_dir or logger_params.get("log_dir")
    if not log_dir:
        sys.exit(
            "error: no log directory configured; set [LOGGER] log_dir in the "
            "config file or pass --log_dir"
        )

    # The batch-summary log: this orchestrator's own DRP-RUN-08 decision trail,
    # echoed live to stdout, alongside each night's per-reduction log.
    level = args.log_level or logger_params.get("log_level", "INFO")
    log_path = setup_batch_logging(log_dir, "masters", level=level)

    forward = []
    for value, flag in (
        (args.kpf_data_input, "--kpf_data_input"),
        (args.kpf_masters_output, "--kpf_masters_output"),
        (args.log_dir, "--log_dir"),
        (args.log_level, "--log_level"),
    ):
        if value:
            forward += [flag, value]

    # Batch banner: the start of this invocation's decision trail.
    logger.info("kpfpipe %s masters batch starting", kpfpipe.__version__)
    logger.info("argv: %s", " ".join(sys.argv))
    logger.info("config: %s", args.config or DEFAULT_MASTERS_CONFIG)
    logger.info("data input: %s", data_input)
    logger.info("jobs: %s", args.jobs)
    logger.info("batch log: %s", log_path)

    datecodes = resolve_datecodes(args, data_input)
    logger.info(
        "building masters for %d night(s): %s", len(datecodes), ", ".join(datecodes)
    )

    # Warm the L0 mini-db caches up front, one thread per night
    warm_mini_db_caches(data_input, datecodes, args.jobs)

    tasks = [
        _cli_task(dc, forward, config=args.config, recipe=args.recipe)
        for dc in datecodes
    ]
    failed = run_stage(
        "masters",
        tasks,
        args.jobs,
        log_dir,
        job_timeout=args.job_timeout,
        abort_on_failure=False,
        launch_interval=_LAUNCH_INTERVAL,
    )

    built = len(datecodes) - len(failed)
    logger.info("done: built masters for %d/%d night(s)", built, len(datecodes))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
