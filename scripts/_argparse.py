"""Shared argparse parent parsers for the ``kpfpipe`` CLI commands.

Common flag groups -- config selection, data-dir overrides, logging
overrides, stage selection, the fan-out pool controls, and the mini-db ``--cache``
mode -- factored out so the subcommand parsers
(``reduce``/``masters``/``science``/``engineering``) compose them via
``parents=[...]`` instead of each re-declaring the same flags. Every factory
returns a fresh ``add_help=False`` parser to slot in as a parent. The
``kpfpipe analyze`` scripts compose `analysis_parser`, itself built from these same
groups, adding the input and pool/cache flags each one needs.

This layer parses arguments and hands the values on; it never asks the filesystem
what data exists. Resolving a datecode range against the data tree is ``_scan.py``'s
job. Depends only on stdlib + ``kpfpipe`` -- never on ``tools`` -- so, like
``_dispatch.py``, the scripts layer stays ignorant of the CLI dispatcher above it.
"""

import argparse
import os
import sys
import time

from kpfpipe.utils.io import read_token_file
from kpfpipe.utils.kpf import is_datecode, is_obs_id


def config_parser():
    """Config selection (``-c``), shared by every recipe-running command.

    Overrides the command's default config. The recipe is never overridden here:
    each orchestrator runs its fixed recipe, and ``run`` selects its own.
    """
    p = argparse.ArgumentParser(add_help=False)
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


def obs_ids_parser():
    """The ``--obs_ids`` flag of the frame commands; validated by `resolve_obs_ids`
    (via `resolve_frame_selection`)."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--obs_ids",
        nargs="+",
        metavar="OBS_ID_OR_FILE",
        help="one or more obs_ids to reduce, or a text file listing one obs_id per "
        "line, e.g. --obs_ids KP.20240405.40113.57 KP.20240405.40237.36 or "
        "--obs_ids frames.txt (mutually exclusive with --dates/--date_range)",
    )
    return p


def resolve_obs_ids(ap, args):
    """Expand ``--obs_ids`` in place.

    Each value is either an obs_id, used as-is, or a text file of obs_ids expanded in
    place; a valid obs_id is always read as such, even if a like-named file exists.
    The result is sorted and deduplicated. Takes the command's own `ap` so a bad
    value reports as its usage error. Returns `args`.
    """
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
    return args


def resolve_frame_selection(ap, args):
    """Validate a frame command's selection: exactly one of ``--obs_ids``,
    ``--dates``, or ``--date_range``, resolved by `resolve_obs_ids` or
    `resolve_dates`. Nights expand to obs_ids later, once the data root is known
    (`_scan.obs_ids_in_nights`). Returns `args`.
    """
    if sum(v is not None for v in (args.obs_ids, args.dates, args.date_range)) != 1:
        ap.error("give exactly one of --obs_ids, --dates, or --date_range")
    if args.obs_ids is not None:
        return resolve_obs_ids(ap, args)
    return resolve_dates(ap, args)


def data_dirs_parser(science_output=True, engineering_output=False):
    """[DATA_DIRS] override flags, plus the ``--input_dir``/``--output_dir``
    convenience shortcuts. ``--kpf_science_output`` is included only when
    `science_output` is true (masters produces no science output), and
    ``--kpf_engineering_output`` only when `engineering_output` is true.
    ``--input_dir`` aliases ``--kpf_data_input``; ``--output_dir`` is fanned out
    post-parse by `resolve_dir_shortcuts`."""
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
    if engineering_output:
        p.add_argument(
            "--kpf_engineering_output",
            help="override [DATA_DIRS] KPF_ENGINEERING_OUTPUT",
        )
    p.add_argument(
        "--output_dir",
        default=None,
        help="shortcut: root for every output directory not given its own explicit "
        "flag -- the masters output, the science and engineering outputs (where "
        "applicable), the log dir ({output_dir}/logs), and (timeseries) the plot dir "
        "({output_dir}/QLP/timeseries)",
    )
    return p


# Where each --output_dir slot lands beneath the given root. The
# masters/science/engineering outputs take the root verbatim (their path builders
# add the substructure); every directory a flag names outright -- log, plot,
# analysis -- gets its conventional subdirectory here, so --output_dir yields the
# same layout an explicit --log_dir/--plot_dir/--analysis_dir would (the plot
# subdir matches the timeseries default of {science_output}/QLP/timeseries).
_OUTPUT_DIR_SLOTS = {
    "kpf_masters_output": (),
    "kpf_science_output": (),
    "kpf_engineering_output": (),
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


def stages_parser(stages=None):
    """``--stages``, which overrides the config's [STAGES] switches.

    The caller passes the recipe's stage names as choices (this layer never imports
    recipes); ``None`` when the recipe is only known after parsing.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--stages",
        nargs="+",
        choices=stages,
        default=None,
        help="run only these stages, overriding config [STAGES] (default: as "
        "configured)",
    )
    return p


def stages_override(args, stages):
    """The config override for ``--stages``: only the named stages switched on, or
    ``{}`` when the flag is unset."""
    if not args.stages:
        return {}
    return {"STAGES": {stage: stage in args.stages for stage in stages}}


def pool_parser(jobs_help, timeout=True, launch_interval=None):
    """Fan-out pool controls (``--jobs``/``--job_timeout``/``--launch_interval``).

    ``--jobs`` help differs per command (masters cites its fixed cap, science its
    cores-based default), so the caller passes it in; ``--job_timeout`` is
    identical across the orchestrators and lives here so it is written once.
    ``timeout=False`` omits ``--job_timeout`` for a command whose pool runs
    in-process threads, with no subprocess to time out. ``--launch_interval`` is
    added only when the caller passes its default (its ``_LAUNCH_INTERVAL``).
    Validated + resolved post-parse by `resolve_pool`.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--jobs", type=int, default=None, help=jobs_help)
    if timeout:
        p.add_argument(
            "--job_timeout",
            type=int,
            default=1200,
            help="per-job wall-clock limit (seconds) for each fanned-out recipe "
            "subprocess (default: %(default)s). A job exceeding this is treated as "
            "wedged: its process group is killed and the job counts as a failure "
            "rather than hanging the whole batch",
        )
    if launch_interval is not None:
        p.add_argument(
            "--launch_interval",
            type=float,
            default=launch_interval,
            help="minimum seconds between fanned-out job launches "
            "(default: %(default)s)",
        )
    return p


def resolve_pool(ap, args, default_jobs=None):
    """Validate the `pool_parser` flags present on `args`; fill an unset ``--jobs``
    from ``default_jobs()``, or leave it ``None`` when no callable is given (the
    caller resolves it later). Returns `args`."""
    if getattr(args, "job_timeout", 1) < 1:
        ap.error("--job_timeout must be >= 1")
    if (getattr(args, "launch_interval", None) or 0) < 0:
        ap.error("--launch_interval must be >= 0")
    if args.jobs is None:
        if default_jobs is not None:
            args.jobs = default_jobs()
    elif args.jobs < 1:
        ap.error("--jobs must be >= 1")
    return args


def default_date_range():
    """First light through tomorrow -- the whole archive.

    The end is UTC-dated a day ahead so it never falls behind the newest HST night
    on disk; a datecode dir that does not exist yet simply contributes nothing.
    """
    return "20221109", time.strftime("%Y%m%d", time.gmtime(time.time() + 86400))


def analysis_parser(range_action):
    """The parent parser every ``kpfpipe analyze`` script composes: a date range
    defaulting to the whole archive, the logging group above, and where the reports
    land. `range_action` words the ``--date_range`` help, as in `dates_parser`.

    The analysis scripts read no recipe TOML (those configs belong to the recipes),
    so their directories come from the CLI alone: ``--analysis_dir``/``--log_dir``
    -- or the ``--output_dir`` they fall back to -- say where the reports and logs
    land. Each script declares its own required input flag (they read different
    trees) and any pool/cache groups it needs. Validated post-parse by each
    script's ``parse_args``, as with the processing commands.
    """
    p = argparse.ArgumentParser(
        add_help=False,
        parents=[
            logging_parser(),
            dates_parser(
                range_action, dates=False, default_date_range=default_date_range()
            ),
        ],
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
