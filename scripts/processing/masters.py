#!/usr/bin/env python3
"""Build nightly master calibrations for a set of datecodes (``kpfpipe masters``).

A lightweight, fail-loud orchestrator. Given either an explicit list of datecodes
or an inclusive datecode range, it dispatches each night as a separate
``python -m scripts.processing.reduce --masters -d <datecode>`` subprocess (the
``kpfpipe run`` leaf), so every night gets its own log file, clean process state,
and an independent exit code.

It reimplements no pipeline logic and does not decide *which* nights to build:
the caller supplies the datecodes. The two input forms are mutually exclusive:

    kpfpipe masters --datecode_list 20240405 20240712   # explicit datecode(s)
    kpfpipe masters --date_range 20240101 20240131      # every L0 night in range

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
import os
import sys

from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.kpf_utils import is_datecode
from scripts.processing import DEFAULT_MASTERS_CONFIG, DEFAULT_MASTERS_RECIPE
from scripts.processing._argparse import (
    data_dirs_parser,
    logging_parser,
    pool_parser,
    recipe_parser,
)
from scripts.processing._common import _datecode_dirs
from scripts.processing._dispatch import (
    _MASTERS_JOBS,
    _default_masters_jobs,
    configure_runtime,
    run_stage,
)

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
        "--datecode_list",
        nargs="*",
        default=None,
        metavar="DATECODE",
        help="one or more datecodes to build, e.g. --datecode_list 20240405 "
        "20240712 (mutually exclusive with --date_range)",
    )
    ap.add_argument(
        "--date_range",
        nargs=2,
        metavar=("START", "END"),
        help="inclusive datecode range; builds every L0 night in it, e.g. "
        "--date_range 20240101 20240131 (mutually exclusive with --datecode_list)",
    )
    args = ap.parse_args(argv)

    # Exactly one input form: an explicit datecode list, or a range -- not both,
    # not neither.
    if bool(args.datecode_list) == bool(args.date_range):
        ap.error("give either --datecode_list or --date_range, not both or neither")

    if args.date_range:
        start, end = args.date_range
        for dc in (start, end):
            if not is_datecode(dc):
                ap.error(f"--date_range value is not a valid datecode: {dc!r}")
        if start > end:
            ap.error(f"--date_range START must be <= END (got {start} > {end})")
    else:
        for dc in args.datecode_list:
            if not is_datecode(dc):
                ap.error(f"not a valid datecode: {dc!r}")
        args.datecode_list = sorted(set(args.datecode_list))

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
    if args.datecode_list:
        return args.datecode_list
    l0_root = os.path.join(data_input, "L0")
    if not os.path.isdir(l0_root):
        sys.exit(f"error: L0 input directory not found: {l0_root}")
    start, end = args.date_range
    nights = _datecode_dirs(l0_root, start, end)
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
    data_input = (
        args.kpf_data_input or config.get_params(["DATA_DIRS"])["KPF_DATA_INPUT"]
    )
    log_dir = args.log_dir or config.get_params(["LOGGER"]).get("log_dir")

    forward = []
    for value, flag in (
        (args.kpf_data_input, "--kpf_data_input"),
        (args.kpf_masters_output, "--kpf_masters_output"),
        (args.log_dir, "--log_dir"),
        (args.log_level, "--log_level"),
    ):
        if value:
            forward += [flag, value]

    datecodes = resolve_datecodes(args, data_input)
    print(f"building masters for {len(datecodes)} night(s): {', '.join(datecodes)}")

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
    print(f"\ndone: built masters for {built}/{len(datecodes)} night(s)")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
