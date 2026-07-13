"""Shared argparse parent parsers for the batch processing CLI commands.

Common flag groups -- recipe/config selection, data-dir overrides, logging
overrides, the fan-out pool controls, and the mini-db ``--cache`` mode -- factored
out so the subcommand parsers (``reduce``/``masters``/``science``) compose them via
``parents=[...]`` instead of each re-declaring the same flags. Every factory
returns a fresh ``add_help=False`` parser to slot in as a parent.

Depends only on ``argparse`` -- never on ``tools`` -- so, like ``_dispatch.py``,
the scripts layer stays ignorant of the CLI dispatcher above it.
"""

import argparse
import os


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
# outputs take the root verbatim (their path builders add the substructure); the
# log dir and plot dir each get a conventional subdirectory so --output_dir yields
# the same layout an explicit --log_dir/--plot_dir would (the plot subdir matches
# the timeseries default of {science_output}/QLP/timeseries).
_OUTPUT_DIR_SLOTS = {
    "kpf_masters_output": (),
    "kpf_science_output": (),
    "log_dir": ("logs",),
    "plot_dir": ("QLP", "timeseries"),
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
    """[LOGGER] log_dir / log_level override flags, shared by all three commands."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--log_dir", help="override [LOGGER] log_dir")
    p.add_argument("--log_level", help="override [LOGGER] log_level (e.g. DEBUG)")
    return p


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
    default to ``None``/``600`` and are validated + resolved post-parse by each
    command's ``parse_args``.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--jobs", type=int, default=None, help=jobs_help)
    p.add_argument(
        "--job_timeout",
        type=int,
        default=600,
        help="per-job wall-clock limit (seconds) for each fanned-out recipe "
        "subprocess (default: %(default)s). A recipe normally runs in ~2 min, so a "
        "job exceeding this is treated as wedged: its process group is killed and "
        "the job counts as a failure rather than hanging the whole batch",
    )
    return p
