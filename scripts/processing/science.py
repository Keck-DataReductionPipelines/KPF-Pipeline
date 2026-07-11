#!/usr/bin/env python3
"""Reduce a set of science frames end-to-end, L0 -> L4 (``kpfpipe science``).

A lightweight, fail-loud orchestrator: it dispatches each frame as a separate
``python -m scripts.processing.reduce --science -o <obs_id>`` subprocess (the
``kpfpipe run`` leaf), so every frame gets its own log file, clean process state,
and independent exit code. It reimplements no pipeline logic; the caller supplies
which frames to reduce (discovering a target's frames from the L0 tree lives in
the orchestrator, not here), via the input form:

    kpfpipe science --obs_ids KP.20240405.40113.57 KP.20240405.40237.36
    kpfpipe science --obs_ids frames.txt        # a file of obs_ids

Reductions run in a bounded process pool. The L0 mini-database caches for the
nights the frames span are warmed up front by a parallel per-datecode pre-scan;
the first frame then runs alone as a canary to warm the *other* shared caches,
then the rest fan out (paced apart, since the per-frame L0 pointing QC queries
SIMBAD/Gaia). The run is fail-soft (a frame that fails to reduce is reported and
the others continue) but exits nonzero if any frame failed.
"""

import argparse
import logging
import os
import sys

import kpfpipe
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.io import read_token_file
from kpfpipe.utils.kpf_utils import get_datecode, is_obs_id
from kpfpipe.utils.logger import setup_batch_logging
from scripts.processing import DEFAULT_SCIENCE_CONFIG, DEFAULT_SCIENCE_RECIPE
from scripts.processing._argparse import (
    data_dirs_parser,
    logging_parser,
    pool_parser,
    recipe_and_config_parser,
    resolve_dir_shortcuts,
)
from scripts.processing._dispatch import (
    _default_science_jobs,
    configure_runtime,
    run_stage,
)
from scripts.processing._scan import warm_mini_db_caches

logger = logging.getLogger(__name__)

# Minimum spacing (seconds) between fanned-out subprocess launches. The per-frame
# L0 pointing QC fires rapid SIMBAD/Gaia catalog queries at startup, so a burst of
# pool workers launching at once would hammer those services; pacing the launches
# rate-limits them. Passed to run_stage as its launch_interval.
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
            recipe_and_config_parser(),
            data_dirs_parser(science_output=True),
            logging_parser(),
            pool_parser(jobs_help=_JOBS_HELP),
        ],
    )
    ap.add_argument(
        "--obs_ids",
        nargs="+",
        required=True,
        metavar="OBS_ID_OR_FILE",
        help="one or more obs_ids to reduce, or a text file listing one obs_id per "
        "line, e.g. --obs_ids KP.20240405.40113.57 KP.20240405.40237.36 or "
        "--obs_ids frames.txt",
    )
    args = ap.parse_args(argv)

    # Each --obs_ids value is an obs_id (reduced as-is) or a text file of obs_ids;
    # expand file entries in place. A valid obs_id is always read as such, even if
    # a like-named file exists.
    obs_ids = []
    for entry in args.obs_ids:
        if is_obs_id(entry):
            obs_ids.append(entry)
        elif os.path.isfile(entry):
            for oid in read_token_file(entry):
                if not is_obs_id(oid):
                    ap.error(f"not a valid obs_id in {entry}: {oid!r}")
                obs_ids.append(oid)
        else:
            ap.error(
                f"--obs_ids entry is neither an obs_id nor a readable file: {entry!r}"
            )
    if not obs_ids:
        ap.error(f"--obs_ids produced no obs_ids (empty file?): {args.obs_ids}")
    args.obs_ids = sorted(set(obs_ids))

    if args.job_timeout < 1:
        ap.error("--job_timeout must be >= 1")
    if args.jobs is None:
        args.jobs = _default_science_jobs()
    elif args.jobs < 1:
        ap.error("--jobs must be >= 1")
    return resolve_dir_shortcuts(args)


def _cli_task(obs_id, forward, config=None, recipe=None):
    """Build a (tag, argv) task reducing one frame via the ``kpfpipe run`` leaf.

    Passes recipe/config to the leaf explicitly (the science defaults unless
    `recipe`/`config` override) rather than leaning on its ``--science`` shortcut,
    so the orchestrator is the single owner of what it runs. `forward` is the
    resolved dir/log overrides appended to every invocation; the tag (obs_id)
    keys the sentinel/log paths.
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
    # the log dir; subprocesses get recipe/config via -r/-c.
    config = ConfigHandler(args.config or DEFAULT_SCIENCE_CONFIG)
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
    # echoed live to stdout, alongside each frame's per-reduction log.
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

    obs_ids = args.obs_ids

    # Batch banner: the start of this invocation's decision trail.
    logger.info("kpfpipe %s science batch starting", kpfpipe.__version__)
    logger.info("argv: %s", " ".join(sys.argv))
    logger.info("config: %s", args.config or DEFAULT_SCIENCE_CONFIG)
    logger.info("jobs: %s", args.jobs)
    logger.info("batch log: %s", log_path)
    logger.info("reducing %d science frame(s): %s", len(obs_ids), ", ".join(obs_ids))

    # Warm the L0 mini-db caches up front, one thread per night
    datecodes = sorted({get_datecode(o) for o in obs_ids})
    warm_mini_db_caches(data_input, datecodes, args.jobs)

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
