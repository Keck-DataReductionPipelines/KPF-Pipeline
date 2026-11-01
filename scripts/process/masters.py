#!/usr/bin/env python3
"""Build nightly master calibrations for a set of datecodes (``kpfpipe masters``).

A lightweight orchestrator: it dispatches each night as a separate
``python -m scripts.process.reduce -r <masters recipe> -d <datecode>`` subprocess (the
``kpfpipe run`` leaf), so every night gets its own log file, clean process state,
and independent exit code. It reimplements no pipeline logic; the caller supplies
which nights to build, via two mutually exclusive input forms:

    kpfpipe masters --dates 20240405 20240712       # explicit datecode(s)
    kpfpipe masters --dates nights.txt              # a file of datecodes
    kpfpipe masters --date_range 20240101 20240131  # every L0 night in range

The range form enumerates the datecode dirs present under ``{KPF_DATA_INPUT}/L0``
within [START, END].

Builds run stage by stage: every night's bias, then every night's dark, and so on
through the stages enabled in config [STAGES] (``--stages`` overrides), each stage a
bounded process pool.
A night's masters are associated from nearby nights' (``*_search_window_days``),
so each stage starts only after every night has finished the previous one -- the
masters chosen are then those a chronological one-night-at-a-time run would choose,
however wide the window. All of a night's stages append to its one log in the run
directory. The L0 mini-database caches are warmed up front by a parallel
per-datecode pre-scan (``--cache``, ``rw`` by default; ``r`` skips the pre-scan
and leaves each reduce to read-only); within each stage the first night runs alone
as a canary before the rest fan out. The masters recipe imports neither
barycorrpy, astropy IERS nor matplotlib, so that canary warms only bytecode; the
reference-data caches it warms for the science stage have no counterpart here. The
run is fail-soft (a night that fails a stage is reported, and it and the others
continue to the next stage) but exits nonzero if any night failed any stage.
"""

import argparse
import glob
import logging
import os
import sys

import kpfpipe
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.logger import setup_batch_logging
from recipes.kpf_drp_masters import RECIPE_STAGES
from scripts._argparse import (
    cache_parser,
    config_parser,
    data_dirs_parser,
    dates_parser,
    logging_parser,
    pool_parser,
    resolve_dates,
    resolve_dir_shortcuts,
    resolve_log_settings,
    stages_override,
    stages_parser,
)
from scripts._dispatch import (
    _MASTERS_JOBS,
    _default_masters_jobs,
    configure_runtime,
    run_stage,
)
from scripts._scan import datecodes_in_range, warm_mini_db_caches
from scripts.process import DEFAULT_MASTERS_CONFIG, DEFAULT_MASTERS_RECIPE

logger = logging.getLogger(__name__)

# Minimum spacing (seconds) between fanned-out subprocess launches. A masters build
# opens with a stack phase -- reading and assembling the night's bias, dark and
# ThAr L0 frames (~100 s solo) before any WLS fitting. A light stagger keeps a full
# pool from starting that phase in lockstep; at 1 s a 64-job first wave spreads over
# ~64 s. The earlier ~4-5x slowdown of this phase at 16-wide was numpy hugepage
# compaction, not disk (see configure_runtime), so a long stagger is no longer
# needed. Passed to run_stage as its launch_interval.
_LAUNCH_INTERVAL = 1.0

# --jobs help is command-specific (the fixed masters cap, not the cores default),
# so it is passed into the shared pool_parser rather than baked into it.
_JOBS_HELP = (
    "max concurrent masters builds; left unset, defaults to the "
    f"{_MASTERS_JOBS}-job cap (too many concurrent builds contend for memory), "
    "floored by cores and RAM on small machines"
)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="kpfpipe masters",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[
            config_parser(),
            data_dirs_parser(science_output=False),
            logging_parser(),
            pool_parser(jobs_help=_JOBS_HELP),
            cache_parser(default="rw"),
            dates_parser("builds every L0 night in it"),
            stages_parser(RECIPE_STAGES),
        ],
    )
    args = resolve_dates(ap, ap.parse_args(argv))

    if args.job_timeout < 1:
        ap.error("--job_timeout must be >= 1")
    if args.jobs is None:
        args.jobs = _default_masters_jobs()
    elif args.jobs < 1:
        ap.error("--jobs must be >= 1")
    return resolve_dir_shortcuts(args)


def _cli_task(datecode, forward, *, stage, config=None):
    """Build a (tag, argv) task building one night via the ``kpfpipe run`` leaf.

    Passes the masters recipe and config (the default unless `config` overrides) to
    the leaf explicitly rather than leaning on its ``--masters`` shortcut,
    so the orchestrator is the single owner of what it runs. `forward` is the
    resolved dir/log overrides appended to every invocation; the tag (datecode)
    keys the sentinel/log paths. `stage` becomes ``--stages``, one stage per task.
    """
    argv = [
        sys.executable,
        "-m",
        "scripts.process.reduce",
        "-r",
        DEFAULT_MASTERS_RECIPE,
        "-c",
        config or DEFAULT_MASTERS_CONFIG,
        "-d",
        datecode,
        *forward,
        "--stages",
        stage,
    ]
    return datecode, argv


def main(argv=None):
    configure_runtime()
    args = parse_args(argv)

    # Read the effective config (the masters default unless -c overrides) to resolve
    # the L0 input root and log dir; subprocesses get recipe/config via -r/-c.
    config = ConfigHandler(
        args.config or DEFAULT_MASTERS_CONFIG,
        overrides=stages_override(args, RECIPE_STAGES) or None,
    )
    logger_params = config.get_params(["LOGGER"])
    data_input = (
        args.kpf_data_input or config.get_params(["DATA_DIRS"])["KPF_DATA_INPUT"]
    )
    # One run, one log directory: this batch's own DRP-RUN-08 decision trail and
    # every night's per-reduction log land together in {log_dir}/{run_id}.
    log_dir, level = resolve_log_settings(args, logger_params)
    run_id, log_path = setup_batch_logging(log_dir, "masters", args.run_id, level=level)
    run_dir = os.path.join(log_dir, run_id)

    forward = []
    for value, flag in (
        (args.kpf_data_input, "--kpf_data_input"),
        (args.kpf_masters_output, "--kpf_masters_output"),
        # Both halves of the run directory, which the child rejoins.
        (log_dir, "--log_dir"),
        (run_id, "--run_id"),
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

    datecodes = args.dates or datecodes_in_range(data_input, *args.date_range)
    logger.info(
        "building masters for %d night(s): %s", len(datecodes), ", ".join(datecodes)
    )
    # A night with no L0 FITS has nothing to build; skip it rather than fail it.
    empty = [
        dc
        for dc in datecodes
        if not glob.glob(os.path.join(data_input, "L0", dc, "*.fits"))
    ]
    if empty:
        logger.warning(
            "skipping %d empty night(s) (no L0 FITS): %s", len(empty), ", ".join(empty)
        )
        datecodes = [dc for dc in datecodes if dc not in empty]

    # Warm the L0 mini-db caches up front, one thread per night (--cache, rw
    # default; --cache r skips the pre-scan and leaves reduces to read-only).
    warm_mini_db_caches(data_input, datecodes, args.jobs, cache=args.cache)

    failed = set()
    stages = [s for s in RECIPE_STAGES if config.get_params(["STAGES"])[s]]
    for stage in stages:
        logger.info("stage %s: all %d night(s)", stage, len(datecodes))
        tasks = [
            _cli_task(
                dc,
                forward,
                config=args.config,
                stage=stage,
            )
            for dc in datecodes
        ]
        failed |= run_stage(
            "masters",
            tasks,
            args.jobs,
            run_dir,
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
