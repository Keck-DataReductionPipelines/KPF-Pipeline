#!/usr/bin/env python3
"""Reduce a set of frames for engineering, L0 -> L2 (``kpfpipe engineering``).

A lightweight orchestrator: it dispatches each frame as a separate
``python -m scripts.process.reduce -r <engineering recipe> -o <obs_id>`` subprocess
(the ``kpfpipe run`` leaf), so every frame gets its own log file, clean process
state, and independent exit code. It reimplements no pipeline logic; the caller
supplies which frames to reduce, via one of the input forms:

    kpfpipe engineering --obs_ids KP.20240405.40113.57 KP.20240405.40237.36
    kpfpipe engineering --obs_ids frames.txt        # a file of obs_ids
    kpfpipe engineering --dates 20240405 20240712   # every frame of these nights
    kpfpipe engineering --date_range 20240401 20240430

Reductions run in a bounded process pool; the first frame runs alone as a canary
to warm the shared caches, then the rest fan out, paced apart. Unlike ``kpfpipe
science``, no frame is gated out (junk and single-CCD frames are reduced too).
``--stages`` is forwarded to every frame, overriding the config's [STAGES]. The run
is fail-soft (a frame that fails to reduce is reported and the others continue) but
exits nonzero if any frame failed.
"""

import argparse
import logging
import os
import sys

import kpfpipe
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.logger import setup_batch_logging
from recipes.kpf_drp_engineering import RECIPE_STAGES
from scripts._argparse import (
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
from scripts._dispatch import _default_single_frame_jobs, configure_runtime, run_stage
from scripts._scan import datecodes_in_range, obs_ids_in_nights
from scripts.process import DEFAULT_ENGINEERING_CONFIG, DEFAULT_ENGINEERING_RECIPE

logger = logging.getLogger(__name__)

# Minimum spacing (seconds) between fanned-out subprocess launches. The engineering
# recipe makes no catalog queries today, so this is precautionary: it keeps the
# fan-out paced like science's should a future per-frame step hit a shared service.
# Default for --launch_interval.
_LAUNCH_INTERVAL = 1.0

_JOBS_HELP = (
    "max concurrent engineering reductions; left unset, defaults to a cores-based "
    "value (~25%% of CPUs, but up to 16)"
)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="kpfpipe engineering",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[
            config_parser(),
            data_dirs_parser(science_output=False, engineering_output=True),
            logging_parser(),
            pool_parser(jobs_help=_JOBS_HELP, launch_interval=_LAUNCH_INTERVAL),
            dates_parser("reduces every L0 frame of every night in it"),
            obs_ids_parser(),
            stages_parser(RECIPE_STAGES),
        ],
    )
    args = resolve_frame_selection(ap, ap.parse_args(argv))
    args = resolve_pool(ap, args, _default_single_frame_jobs)
    return resolve_dir_shortcuts(args)


def _cli_task(obs_id, forward, config=None):
    """Build a (tag, argv) task reducing one frame via the ``kpfpipe run`` leaf.

    Passes the engineering recipe and config (the default unless `config`
    overrides) to the leaf explicitly, so the orchestrator is the single owner of
    what it runs. `forward` is the resolved dir/log overrides appended to every
    invocation; the tag (obs_id) keys the sentinel/log paths.
    """
    argv = [
        sys.executable,
        "-m",
        "scripts.process.reduce",
        "-r",
        DEFAULT_ENGINEERING_RECIPE,
        "-c",
        config or DEFAULT_ENGINEERING_CONFIG,
        "-o",
        obs_id,
        *forward,
    ]
    return obs_id, argv


def main(argv=None):
    configure_runtime()
    args = parse_args(argv)

    # Read the effective config (the engineering default unless -c overrides) for the
    # L0 input root and log dir; subprocesses get recipe/config via -r/-c.
    config = ConfigHandler(args.config or DEFAULT_ENGINEERING_CONFIG)
    data_input = (
        args.kpf_data_input or config.get_params(["DATA_DIRS"])["KPF_DATA_INPUT"]
    )
    # One run, one log directory: this batch's own DRP-RUN-08 decision trail and
    # every frame's per-reduction log land together in {log_dir}/{run_id}.
    log_dir, level = resolve_log_settings(args, config.get_params(["LOGGER"]))
    run_id, log_path = setup_batch_logging(
        log_dir, "engineering", args.run_id, level=level
    )
    run_dir = os.path.join(log_dir, run_id)

    forward = []
    for value, flag in (
        (args.kpf_data_input, "--kpf_data_input"),
        # The ThAr master lookup reads the masters root.
        (args.kpf_masters_output, "--kpf_masters_output"),
        (args.kpf_engineering_output, "--kpf_engineering_output"),
        # Both halves of the run directory, which the child rejoins.
        (log_dir, "--log_dir"),
        (run_id, "--run_id"),
        (args.log_level, "--log_level"),
    ):
        if value:
            forward += [flag, value]
    if args.stages:
        forward += ["--stages", *args.stages]

    # Nights expand to every frame in them.
    if args.obs_ids is None:
        nights = args.dates or datecodes_in_range(data_input, *args.date_range)
        obs_ids = obs_ids_in_nights(data_input, nights)
    else:
        obs_ids = args.obs_ids

    # Batch banner: the start of this invocation's decision trail.
    logger.info("kpfpipe %s engineering batch starting", kpfpipe.__version__)
    logger.info("argv: %s", " ".join(sys.argv))
    logger.info("config: %s", args.config or DEFAULT_ENGINEERING_CONFIG)
    logger.info("jobs: %s", args.jobs)
    logger.info("batch log: %s", log_path)
    logger.info("reducing %d frame(s): %s", len(obs_ids), ", ".join(obs_ids))

    tasks = [_cli_task(o, forward, config=args.config) for o in obs_ids]
    failed = run_stage(
        "engineering",
        tasks,
        args.jobs,
        run_dir,
        job_timeout=args.job_timeout,
        abort_on_failure=False,
        launch_interval=args.launch_interval,
    )

    logger.info(
        "done: reduced %d/%d frame(s)", len(obs_ids) - len(failed), len(obs_ids)
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
