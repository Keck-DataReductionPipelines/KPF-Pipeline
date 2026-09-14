"""Shared argparse parent parsers for the ``kpfpipe`` CLI commands.

Common flag groups -- recipe/config selection, data-dir overrides, logging
overrides, the fan-out pool controls, and the mini-db ``--cache`` mode -- factored
out so the subcommand parsers (``reduce``/``masters``/``science``) compose them via
``parents=[...]`` instead of each re-declaring the same flags. Every factory
returns a fresh ``add_help=False`` parser to slot in as a parent. The
``kpfpipe analyze`` scripts compose one parent, `analysis_parser`, which is itself
built from these same groups.

This layer parses arguments and hands the values on; it never asks the filesystem
what data exists. Resolving a datecode range against the data tree is ``_scan.py``'s
job. Depends only on stdlib + ``kpfpipe`` -- never on ``tools`` -- so, like
``_dispatch.py``, the scripts layer stays ignorant of the CLI dispatcher above it.
"""

import argparse
import os
import sys

from kpfpipe.utils.io import read_token_file
from kpfpipe.utils.kpf import is_datecode


def recipe_and_config_parser():
    """Recipe + config selection (``-r``/``-c``), shared by all three commands.

    They override the command's default recipe/config pair: for the orchestrators
    that default is their fixed kind's pair; for ``run`` it comes from a
    ``--masters``/``--science`` shortcut (or, given neither, ``-r``/``-c`` are the
    base specification).
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "-r",
        "--recipe",
        default=None,
        help="recipe .py to run; overrides the default recipe",
    )
    p.add_argument(
        "-c",
        "--config",
        default=None,
        help="TOML config to use; overrides the default config",
    )
    return p


def dates_parser(range_action, *, dates=True, default_date_range=None):
    """The night-selection flags, declared and documented once for every command.

    `range_action` is the only command-specific wording: the predicate describing
    what a range covers, e.g. "builds every L0 night in it". The surrounding help --
    the example, the mutual-exclusion note, the stated default -- is assembled here,
    so the same flag reads the same way everywhere. Validated by `resolve_dates`.

    Commands offering both input forms take the default `dates=True` and must give
    exactly one. A range-only command passes ``dates=False``, which makes
    ``--date_range`` required unless `default_date_range` supplies a ``(START, END)``
    fallback.
    """
    p = argparse.ArgumentParser(add_help=False)
    if dates:
        p.add_argument(
            "--dates",
            nargs="*",
            default=None,
            metavar="DATECODE_OR_FILE",
            help="one or more datecodes, or a text file listing one datecode per "
            "line, e.g. --dates 20240405 20240712 or --dates nights.txt (mutually "
            "exclusive with --date_range)",
        )
    range_help = (
        f"inclusive datecode range; {range_action}, e.g. --date_range 20240101 20240131"
    )
    if dates:
        range_help += " (mutually exclusive with --dates)"
    if default_date_range:
        range_help += f" (default: {' '.join(default_date_range)})"
    p.add_argument(
        "--date_range",
        nargs=2,
        metavar=("START", "END"),
        required=not dates and default_date_range is None,
        default=list(default_date_range) if default_date_range else None,
        help=range_help,
    )
    return p


def resolve_dates(ap, args):
    """Validate a command's night selection, expanding ``--dates`` in place.

    A command offering both forms must give exactly one; a range-only command has no
    ``--dates`` attribute, and argparse has already required its range. A range is
    checked but not expanded -- that needs a data root, so `_scan.datecodes_in_range`
    does it later. Each ``--dates`` value is either a datecode, used as-is, or a text
    file of datecodes expanded in place; a valid datecode is always read as such, even
    if a like-named file exists. The result is sorted and deduplicated.

    Takes the command's own `ap` so a bad value reports as its usage error. Returns
    `args`.
    """
    if hasattr(args, "dates") and bool(args.dates) == bool(args.date_range):
        ap.error("give either --dates or --date_range, not both or neither")

    if args.date_range:
        start, end = args.date_range
        for dc in (start, end):
            if not is_datecode(dc):
                ap.error(f"--date_range value is not a valid datecode: {dc!r}")
        if start > end:
            ap.error(f"--date_range START must be <= END (got {start} > {end})")
        return args

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
                f"--dates entry is neither a datecode nor a readable file: {entry!r}"
            )
    if not datecodes:
        ap.error(f"--dates produced no datecodes (empty file?): {args.dates}")
    args.dates = sorted(set(datecodes))
    return args


def data_dirs_parser(science_output=True):
    """[DATA_DIRS] override flags, plus the ``--input_dir``/``--output_dir``
    convenience shortcuts. ``--kpf_science_output`` is included only when
    `science_output` is true (masters produces no science output). ``--input_dir``
    aliases ``--kpf_data_input``; ``--output_dir`` is fanned out post-parse by
    `resolve_dir_shortcuts`."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--kpf_data_input",
        "--input_dir",
        dest="kpf_data_input",
        help="override [DATA_DIRS] KPF_DATA_INPUT (--input_dir is an alias)",
    )
    p.add_argument(
        "--kpf_masters_output", help="override [DATA_DIRS] KPF_MASTERS_OUTPUT"
    )
    if science_output:
        p.add_argument(
            "--kpf_science_output", help="override [DATA_DIRS] KPF_SCIENCE_OUTPUT"
        )
    p.add_argument(
        "--output_dir",
        default=None,
        help="shortcut: root for every output directory not given its own explicit "
        "flag -- the masters output, the science output (where applicable), the log "
        "dir ({output_dir}/logs), and (timeseries) the plot dir "
        "({output_dir}/QLP/timeseries)",
    )
    return p


# Where each --output_dir slot lands beneath the given root. The masters/science
# outputs take the root verbatim (their path builders add the substructure); every
# directory a flag names outright -- log, plot, analysis -- gets its conventional
# subdirectory here, so --output_dir yields the same layout an explicit
# --log_dir/--plot_dir/--analysis_dir would (the plot subdir matches the timeseries
# default of {science_output}/QLP/timeseries).
_OUTPUT_DIR_SLOTS = {
    "kpf_masters_output": (),
    "kpf_science_output": (),
    "log_dir": ("logs",),
    "plot_dir": ("QLP", "timeseries"),
    "analysis_dir": ("analysis",),
}


def resolve_dir_shortcuts(args):
    """Fan ``--output_dir`` out to each output directory the command left unset.

    Slots and their subdirectories are defined by ``_OUTPUT_DIR_SLOTS``. It is a
    fallback, never an override -- any slot an explicit flag already set keeps its
    value; only unset slots, and only those the command actually has (masters has
    no science output or plot dir), inherit ``--output_dir``. The input dir is
    untouched (use ``--input_dir``/``--kpf_data_input``). Returns `args`; each
    command's ``parse_args`` calls it post-parse.
    """
    out = getattr(args, "output_dir", None)
    if not out:
        return args
    for name, subdir in _OUTPUT_DIR_SLOTS.items():
        if hasattr(args, name) and getattr(args, name) is None:
            setattr(args, name, os.path.join(out, *subdir))
    return args


def logging_parser():
    """[LOGGER] log_dir / log_level overrides, plus the ``--run_id`` a launching
    script forwards to bind its children to its own run directory."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--log_dir", help="override [LOGGER] log_dir")
    p.add_argument("--log_level", help="override [LOGGER] log_level (e.g. DEBUG)")
    p.add_argument(
        "--run_id",
        help="name of the run directory under --log_dir, instead of minting a "
        "fresh one; set automatically when one script launches another",
    )
    return p


def resolve_log_settings(args, params):
    """This run's ``(log_dir, level)``: the CLI flags over the ``[LOGGER]`` config.

    Exits when neither supplies a log directory (DRP-RUN-07). Absolute, because the
    orchestrators forward this to children that run from ``kpfpipe.REPO_ROOT``, not
    the operator's cwd -- relative, it would name two different places.
    """
    log_dir = args.log_dir or params.get("log_dir")
    if not log_dir:
        sys.exit(
            "error: no log directory configured; set [LOGGER] log_dir in the "
            "config file or pass --log_dir"
        )
    return os.path.abspath(log_dir), args.log_level or params.get("log_level", "INFO")


def cache_parser(default="r"):
    """L0 mini-db ``--cache`` mode flag, shared by the batch orchestrators.

    Controls how a command's up-front pre-scan warms the on-disk mini-database
    cache. The factory default is ``"r"`` (read-only, no warm); the orchestrators
    that own cache writing pass ``default="rw"``. The leaf ``reduce`` deliberately
    has no such flag -- recipes read the cache but never write it.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--cache",
        choices=["r", "w", "rw", "wr"],
        default=default,
        help="L0 mini-db cache mode: 'r' read-only (skip the pre-scan), 'w' rescan "
        "and write, 'rw' reuse a current cache else write (default: %(default)s)",
    )
    return p


def pool_parser(jobs_help):
    """Fan-out pool controls (``--jobs``/``--job_timeout``) for the orchestrators.

    ``--jobs`` help differs per command (masters cites its fixed cap, science its
    cores-based default), so the caller passes it in; ``--job_timeout`` is
    identical across masters and science and lives here so it is written once. Both
    default to ``None``/``1200`` and are validated + resolved post-parse by each
    command's ``parse_args``.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--jobs", type=int, default=None, help=jobs_help)
    p.add_argument(
        "--job_timeout",
        type=int,
        default=1200,
        help="per-job wall-clock limit (seconds) for each fanned-out recipe "
        "subprocess (default: %(default)s). A job exceeding this is treated as "
        "wedged: its process group is killed and the job counts as a failure "
        "rather than hanging the whole batch",
    )
    return p


def analysis_parser(default_date_range=None):
    """The single parent parser the ``kpfpipe analyze`` scripts compose: their own
    date range and directories, over the logging/pool/cache groups above.

    A script whose natural scope is the whole archive passes `default_date_range`
    as a ``(START, END)`` pair, which makes ``--date_range`` optional.

    The analysis scripts read no recipe TOML (those configs belong to the recipes),
    so their directories come from the CLI alone: ``--input_dir`` is required, and
    ``--analysis_dir``/``--log_dir`` -- or the ``--output_dir`` they fall back to --
    say where the reports and logs land. Validated post-parse by each script's
    ``parse_args``, as with the processing commands.
    """
    p = argparse.ArgumentParser(
        add_help=False,
        parents=[
            logging_parser(),
            pool_parser(jobs_help="max concurrent per-night header scans"),
            cache_parser(default="rw"),
            dates_parser(
                "analyzes every L0 night in it",
                dates=False,
                default_date_range=default_date_range,
            ),
        ],
    )
    p.add_argument(
        "--kpf_data_input",
        "--input_dir",
        dest="kpf_data_input",
        required=True,
        help="root of the raw L0 input tree to read (--input_dir is an alias)",
    )
    p.add_argument(
        "--analysis_dir",
        default=None,
        help="directory for the analysis reports (default: {output_dir}/analysis)",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="shortcut: root for every output directory not given its own explicit "
        "flag -- the analysis dir ({output_dir}/analysis) and the log dir "
        "({output_dir}/logs)",
    )
    return p
