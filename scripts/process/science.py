#!/usr/bin/env python3
"""Reduce a set of science frames end-to-end, L0 -> L4 (``kpfpipe science``).

A lightweight orchestrator: it dispatches each frame as a separate
``python -m scripts.process.reduce -r <science recipe> -o <obs_id>`` subprocess (the
``kpfpipe run`` leaf), so every frame gets its own log file, clean process state,
and independent exit code. It reimplements no pipeline logic; the caller supplies
which frames to reduce (discovering a target's frames from the L0 tree lives in
the timeseries wrapper, not here), via one of the input forms:

    kpfpipe science --obs_ids KP.20240405.40113.57 KP.20240405.40237.36
    kpfpipe science --obs_ids frames.txt        # a file of obs_ids
    kpfpipe science --dates 20240405 20240712   # every Target frame of these nights
    kpfpipe science --date_range 20240401 20240430
    kpfpipe science --dates 20240405 --no-socal    # skip the night's SoCal frames

Reductions run in a bounded process pool. The L0 mini-database caches for the
nights the frames span are warmed up front by a parallel per-datecode pre-scan
(``--cache``, ``rw`` by default; ``r`` skips the pre-scan and leaves each reduce to
read-only); the first frame then runs alone as a canary to warm the *other* shared
caches, then the rest fan out (paced apart, since the per-frame L0 pointing QC
queries SIMBAD/Gaia). ``--stages`` is forwarded to every frame, overriding the
config's [STAGES]. The run is fail-soft (a frame that fails to reduce is reported
and the others continue) but exits nonzero if any frame failed.
"""

import argparse
import logging
import os
import sys

import kpfpipe
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.io import FileHandler
from kpfpipe.utils.kpf import get_datecode, get_obs_id
from kpfpipe.utils.logger import setup_batch_logging
from recipes.kpf_drp_science import RECIPE_STAGES
from scripts._argparse import (
    cache_parser,
    config_parser,
    data_dirs_parser,
    dates_parser,
    logging_parser,
    obs_ids_parser,
    pool_parser,
    resolve_dir_shortcuts,
    resolve_frame_selection,
    resolve_log_settings,
    resolve_pool,
    stages_parser,
)
from scripts._dispatch import (
    _default_single_frame_jobs,
    configure_runtime,
    run_stage,
)
from scripts._scan import datecodes_in_range, obs_ids_in_nights, warm_mini_db_caches
from scripts.process import DEFAULT_SCIENCE_CONFIG, DEFAULT_SCIENCE_RECIPE
from scripts.process.reduce import remove_outputs, science_output_paths

logger = logging.getLogger(__name__)

# Minimum spacing (seconds) between fanned-out subprocess launches. The per-frame
# L0 pointing QC fires rapid SIMBAD/Gaia catalog queries at startup, so a burst of
# pool workers launching at once would hammer those services; pacing the launches
# rate-limits them. Default for --launch_interval.
_LAUNCH_INTERVAL = 1.0

# --jobs help is command-specific (the cores-based default, not the masters cap),
# so it is passed into the shared pool_parser rather than baked into it.
_JOBS_HELP = (
    "max concurrent science reductions; left unset, defaults to a cores-based "
    "value (~25%% of CPUs, but up to 16)"
)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="kpfpipe science",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[
            config_parser(),
            data_dirs_parser(science_output=True),
            logging_parser(),
            pool_parser(jobs_help=_JOBS_HELP, launch_interval=_LAUNCH_INTERVAL),
            cache_parser(default="rw"),
            dates_parser("reduces the Target frames of every L0 night in it"),
            obs_ids_parser(),
            stages_parser(RECIPE_STAGES),
        ],
    )
    ap.add_argument(
        "--socal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="reduce the SoCal frames of the requested night(s) (default: on; pass "
        "--no-socal to skip); --dates/--date_range only",
    )
    args = resolve_frame_selection(ap, ap.parse_args(argv))

    # Default None so an explicit flag alongside --obs_ids can be rejected.
    if args.socal is not None and args.obs_ids is not None:
        ap.error("--socal/--no-socal applies only to --dates/--date_range")
    if args.socal is None:
        args.socal = True

    args = resolve_pool(ap, args, _default_single_frame_jobs)
    return resolve_dir_shortcuts(args)


def _cli_task(obs_id, forward, config=None):
    """Build a (tag, argv) task reducing one frame via the ``kpfpipe run`` leaf.

    Passes the science recipe and config (the default unless `config` overrides) to
    the leaf explicitly rather than leaning on its ``--science`` shortcut, so the
    orchestrator is the single owner of what it runs. `forward` is the
    resolved dir/log overrides appended to every invocation; the tag (obs_id)
    keys the sentinel/log paths.
    """
    argv = [
        sys.executable,
        "-m",
        "scripts.process.reduce",
        "-r",
        DEFAULT_SCIENCE_RECIPE,
        "-c",
        config or DEFAULT_SCIENCE_CONFIG,
        "-o",
        obs_id,
        *forward,
    ]
    return obs_id, argv


def main(argv=None):
    configure_runtime()
    args = parse_args(argv)

    # Read the effective config (the science default unless -c overrides) only for
    # the log dir; subprocesses get recipe/config via -r/-c.
    config = ConfigHandler(args.config or DEFAULT_SCIENCE_CONFIG)
    logger_params = config.get_params(["LOGGER"])
    data_input = (
        args.kpf_data_input or config.get_params(["DATA_DIRS"])["KPF_DATA_INPUT"]
    )
    # One run, one log directory: this batch's own DRP-RUN-08 decision trail and
    # every frame's per-reduction log land together in {log_dir}/{run_id}.
    log_dir, level = resolve_log_settings(args, logger_params)
    run_id, log_path = setup_batch_logging(log_dir, "science", args.run_id, level=level)
    run_dir = os.path.join(log_dir, run_id)

    forward = []
    for value, flag in (
        (args.kpf_data_input, "--kpf_data_input"),
        (args.kpf_masters_output, "--kpf_masters_output"),
        (args.kpf_science_output, "--kpf_science_output"),
        # Both halves of the run directory, which the child rejoins.
        (log_dir, "--log_dir"),
        (run_id, "--run_id"),
        (args.log_level, "--log_level"),
    ):
        if value:
            forward += [flag, value]
    if args.stages:
        forward += ["--stages", *args.stages]

    # Nights expand to every frame in them; the gate below keeps only Targets.
    by_night = args.obs_ids is None
    if by_night:
        nights = args.dates or datecodes_in_range(data_input, *args.date_range)
        obs_ids = obs_ids_in_nights(data_input, nights)
    else:
        obs_ids = args.obs_ids

    # Batch banner: the start of this invocation's decision trail.
    logger.info("kpfpipe %s science batch starting", kpfpipe.__version__)
    logger.info("argv: %s", " ".join(sys.argv))
    logger.info("config: %s", args.config or DEFAULT_SCIENCE_CONFIG)
    logger.info("jobs: %s", args.jobs)
    logger.info("batch log: %s", log_path)
    if by_night:
        logger.info("reducing the Target frames of night(s): %s", ", ".join(nights))
    else:
        logger.info(
            "reducing %d science frame(s): %s", len(obs_ids), ", ".join(obs_ids)
        )

    # Warm the L0 mini-db caches up front, one thread per night (--cache, rw
    # default; --cache r skips the pre-scan and leaves reduces to read-only).
    datecodes = sorted({get_datecode(o) for o in obs_ids})
    warm_mini_db_caches(data_input, datecodes, args.jobs, cache=args.cache)

    # Gate junk and single-CCD frames out of the science recipe (deleting a junk
    # frame's earlier products), and, for a night, every frame that is not a Target.
    # A frame absent from the mini-db (rejected by the scan) is in none of the sets:
    # not a known Target, but not gated either.
    science_output = (
        args.kpf_science_output
        or config.get_params(["DATA_DIRS"])["KPF_SCIENCE_OUTPUT"]
    )
    junk, single_ccd, targets = set(), set(), set()
    for dc in datecodes:
        mini_db = FileHandler({"KPF_DATA_INPUT": data_input}).build_mini_database(
            dc, cache="r"
        )
        ids = mini_db["FILENAME"].map(get_obs_id)
        junk |= set(ids[mini_db["ISJUNK"].astype(bool)])
        single_ccd |= set(ids[~mini_db[["GREEN", "RED"]].eq(True).all(axis=1)])
        is_target = mini_db["DRP_OBSTYPE"] == "Target"
        if not args.socal:
            is_target &= mini_db["DRP_OBJECT"] != "SoCal"
        night_targets = set(ids[is_target])
        targets |= night_targets
        if by_night:
            logger.info("%s: %d Target frame(s)", dc, len(night_targets))
            if not night_targets:
                logger.warning("skipping %s: no Target frames", dc)
    if by_night:
        obs_ids = [o for o in obs_ids if o in targets]
    requested = len(obs_ids)
    kept = []
    for o in obs_ids:
        if o in junk:
            logger.info("skipping %s: junk frame; clearing its products", o)
            remove_outputs(science_output_paths(o, science_output))
        elif o in single_ccd:
            logger.warning(
                "skipping %s: DRP does not yet support single-CCD exposures", o
            )
        else:
            kept.append(o)
    obs_ids = kept
    if by_night and not obs_ids:
        sys.exit("error: no Target frames to reduce in the requested night(s)")

    tasks = [_cli_task(o, forward, config=args.config) for o in obs_ids]
    failed = run_stage(
        "science",
        tasks,
        args.jobs,
        run_dir,
        job_timeout=args.job_timeout,
        abort_on_failure=False,
        launch_interval=args.launch_interval,
    )

    reduced = len(obs_ids) - len(failed)
    logger.info(
        "done: reduced %d/%d frame(s), %d skipped",
        reduced,
        len(obs_ids),
        requested - len(obs_ids),
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
