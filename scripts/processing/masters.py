#!/usr/bin/env python3
"""Build nightly master calibrations for a set of datecodes (``kpfpipe masters``).

A lightweight, fail-loud orchestrator. Given either an explicit list of datecodes
or an inclusive datecode range, it dispatches each night as a separate
``python -m scripts.processing.reduce --masters -d <datecode>`` subprocess (the
``kpfpipe run`` leaf), so every night gets its own log file, clean process state,
and an independent exit code.

It reimplements no pipeline logic and does not decide *which* nights to build:
the caller supplies the datecodes. The two input forms are mutually exclusive:

    kpfpipe masters --dates 20240405 20240712       # explicit datecode(s)
    kpfpipe masters --dates nights.txt              # a file of datecodes
    kpfpipe masters --date_range 20240101 20240131  # every L0 night in range

Each ``--dates`` value is either a datecode built as-is or a path to a text file
listing one datecode per line (blank lines ignored); the two may be mixed, and a
datecode is always read as a datecode even if a like-named file exists.

The range form enumerates the datecode directories present under
``{KPF_DATA_INPUT}/L0`` within [START, END]; the raw L0 location itself is
resolved by the recipe (from its config, or a forwarded --kpf_data_input
override) for every form, single datecode included.

Builds run in a bounded process pool: the first night runs alone as a canary to
warm the shared on-disk caches, then the rest fan out. The build is fail-soft --
a night that fails to build (e.g. the known WLS failure mode) is reported and
the run continues with the other nights -- but the script exits nonzero if any
night failed, so a caller gets a meaningful exit code.
"""

import argparse
import logging
import os
import sys

import kpfpipe
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.io import datecode_dirs_in_range, read_datecodes
from kpfpipe.utils.kpf_utils import is_datecode
from kpfpipe.utils.logger import setup_batch_logging
from scripts.processing import DEFAULT_MASTERS_CONFIG, DEFAULT_MASTERS_RECIPE
from scripts.processing._argparse import (
    data_dirs_parser,
    logging_parser,
    pool_parser,
    recipe_parser,
)
from scripts.processing._dispatch import (
    _MASTERS_JOBS,
    _default_masters_jobs,
    configure_runtime,
    run_stage,
)

logger = logging.getLogger(__name__)

# --jobs help is command-specific (the fixed masters cap, not the cores default),
# so it is passed into the shared pool_parser rather than living in it.
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
            recipe_parser(),
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

    # Exactly one input form: an explicit datecode list, or a range -- not both,
    # not neither.
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
        # Each --dates value is either a datecode (built as-is) or a path to a
        # text file listing one datecode per line; expand file entries in place.
        # A valid datecode is always read as such, even if a like-named file exists.
        datecodes = []
        for entry in args.dates:
            if is_datecode(entry):
                datecodes.append(entry)
            elif os.path.isfile(entry):
                for dc in read_datecodes(entry):
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
    return args


def resolve_datecodes(args, data_input):
    """The datecodes to build, from either input form.

    The explicit list is already validated and sorted in parse_args. A range is
    expanded here (it needs the resolved L0 input root): the datecode dirs present
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

    Runs ``python -m scripts.processing.reduce -r <recipe> -c <config> -d <datecode>``
    with the masters defaults (``DEFAULT_MASTERS_RECIPE``/``DEFAULT_MASTERS_CONFIG``)
    unless `recipe`/`config` override them. The orchestrator passes the recipe/config
    explicitly -- rather than leaning on the leaf's ``--masters`` shortcut -- so it is
    the single owner of what it runs. `forward` is the resolved dir/log overrides
    appended to every invocation. The tag is the datecode, which the sentinel/log
    paths key on.
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
    # the L0 input root + log dir; each subprocess gets the resolved recipe/config
    # explicitly via -r/-c (see _cli_task).
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

    # The batch-summary log: the orchestrator's own DRP-RUN-08 decision trail
    # (dispatch banner, per-night ok/FAILED, summary), echoed live to stdout and
    # persisted alongside each night's own per-reduction log.
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
    )

    built = len(datecodes) - len(failed)
    logger.info("done: built masters for %d/%d night(s)", built, len(datecodes))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
