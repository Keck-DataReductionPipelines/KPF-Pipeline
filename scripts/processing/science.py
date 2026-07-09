#!/usr/bin/env python3
"""Reduce a set of science frames end-to-end, L0 -> L4 (``kpfpipe science``).

A lightweight, fail-loud orchestrator. Given an explicit list of obs_ids, it
dispatches each frame as a separate
``python -m scripts.processing.reduce --science -o <obs_id>`` subprocess (the
``kpfpipe run`` leaf), so every frame gets its own log file, clean process state,
and an independent exit code.

It reimplements no pipeline logic and does not decide *which* frames to reduce:
the caller supplies the obs_ids (discovering a target's frames from the L0 tree
lives in the orchestrator, not here). The single input form is:

    kpfpipe science --obs_id_list KP.20240405.40113.57 KP.20240405.40237.36

Reductions run in a bounded process pool: the first frame runs alone as a canary
to warm the shared on-disk caches, then the rest fan out (paced apart, since the
per-frame L0 pointing QC queries SIMBAD/Gaia). The run is fail-soft -- a frame
that fails to reduce is reported and the run continues with the others -- but the
script exits nonzero if any frame failed, so a caller gets a meaningful exit code.
"""

import argparse
import logging
import sys

import kpfpipe
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.kpf_utils import is_obs_id
from kpfpipe.utils.logger import setup_batch_logging
from scripts.processing import DEFAULT_SCIENCE_CONFIG, DEFAULT_SCIENCE_RECIPE
from scripts.processing._argparse import (
    data_dirs_parser,
    logging_parser,
    pool_parser,
    recipe_parser,
)
from scripts.processing._dispatch import _default_jobs, configure_runtime, run_stage

logger = logging.getLogger(__name__)

# Minimum spacing (seconds) between fanned-out subprocess launches. The per-frame
# L0 pointing QC fires rapid SIMBAD/Gaia catalog queries at startup, so a burst of
# pool workers launching at once would hammer those services; pacing the launches
# rate-limits them. Passed to run_stage as its launch_interval.
_LAUNCH_INTERVAL = 1.0

# --jobs help is command-specific (the cores-based default, not the masters cap),
# so it is passed into the shared pool_parser rather than living in it.
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
            recipe_parser(),
            data_dirs_parser(science_output=True),
            logging_parser(),
            pool_parser(jobs_help=_JOBS_HELP),
        ],
    )
    ap.add_argument(
        "--obs_id_list",
        nargs="+",
        required=True,
        metavar="OBS_ID",
        help="one or more obs_ids to reduce, e.g. --obs_id_list "
        "KP.20240405.40113.57 KP.20240405.40237.36",
    )
    args = ap.parse_args(argv)

    for obs_id in args.obs_id_list:
        if not is_obs_id(obs_id):
            ap.error(f"not a valid obs_id: {obs_id!r}")
    args.obs_id_list = sorted(set(args.obs_id_list))

    if args.job_timeout < 1:
        ap.error("--job_timeout must be >= 1")
    if args.jobs is None:
        args.jobs = _default_jobs()
    elif args.jobs < 1:
        ap.error("--jobs must be >= 1")
    return args


def _cli_task(obs_id, forward, config=None, recipe=None):
    """Build a (tag, argv) task reducing one frame via the ``kpfpipe run`` leaf.

    Runs ``python -m scripts.processing.reduce -r <recipe> -c <config> -o <obs_id>``
    with the science defaults (``DEFAULT_SCIENCE_RECIPE``/``DEFAULT_SCIENCE_CONFIG``)
    unless `recipe`/`config` override them. The orchestrator passes the recipe/config
    explicitly -- rather than leaning on the leaf's ``--science`` shortcut -- so it is
    the single owner of what it runs. `forward` is the resolved dir/log overrides
    appended to every invocation. The tag is the obs_id, which the sentinel/log paths
    key on.
    """
    argv = [
        sys.executable,
        "-m",
        "scripts.processing.reduce",
        "-r",
        recipe or DEFAULT_SCIENCE_RECIPE,
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
    # the log dir used in the failure sentinels; each subprocess gets the resolved
    # recipe/config explicitly via -r/-c (see _cli_task).
    config = ConfigHandler(args.config or DEFAULT_SCIENCE_CONFIG)
    logger_params = config.get_params(["LOGGER"])
    log_dir = args.log_dir or logger_params.get("log_dir")
    if not log_dir:
        sys.exit(
            "error: no log directory configured; set [LOGGER] log_dir in the "
            "config file or pass --log_dir"
        )

    # The batch-summary log: the orchestrator's own DRP-RUN-08 decision trail
    # (dispatch banner, per-frame ok/FAILED, summary), echoed live to stdout and
    # persisted alongside each frame's own per-reduction log.
    level = args.log_level or logger_params.get("log_level", "INFO")
    log_path = setup_batch_logging(log_dir, "science", level=level)

    forward = []
    for value, flag in (
        (args.kpf_data_input, "--kpf_data_input"),
        (args.kpf_masters_output, "--kpf_masters_output"),
        (args.kpf_science_output, "--kpf_science_output"),
        (args.log_dir, "--log_dir"),
        (args.log_level, "--log_level"),
    ):
        if value:
            forward += [flag, value]

    obs_ids = args.obs_id_list

    # Batch banner: the start of this invocation's decision trail.
    logger.info("kpfpipe %s science batch starting", kpfpipe.__version__)
    logger.info("argv: %s", " ".join(sys.argv))
    logger.info("config: %s", args.config or DEFAULT_SCIENCE_CONFIG)
    logger.info("jobs: %s", args.jobs)
    logger.info("batch log: %s", log_path)
    logger.info("reducing %d science frame(s): %s", len(obs_ids), ", ".join(obs_ids))

    tasks = [
        _cli_task(o, forward, config=args.config, recipe=args.recipe) for o in obs_ids
    ]
    failed = run_stage(
        "science",
        tasks,
        args.jobs,
        log_dir,
        job_timeout=args.job_timeout,
        abort_on_failure=False,
        launch_interval=_LAUNCH_INTERVAL,
    )

    reduced = len(obs_ids) - len(failed)
    logger.info("done: reduced %d/%d frame(s)", reduced, len(obs_ids))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
