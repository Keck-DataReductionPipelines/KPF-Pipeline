"""Shared argparse parent parsers for the batch processing CLI commands.

Common flag groups -- recipe/config selection, data-dir overrides, logging
overrides, and the fan-out pool controls -- factored out so the three subcommand
parsers (``reduce``/``masters``/``science``) compose them via argparse's
``parents=[...]`` instead of each re-declaring the same flags. Every factory
returns a fresh ``add_help=False`` parser to slot in as a parent. (The default
``masters``/``science`` recipe/config path constants live in the package
``__init__``, not here.)

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
    base specification). "Overrides the default" reads correctly in every case.
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
    `science_output` is true -- masters produces no science output, so its parser
    omits it. ``--input_dir`` is a plain alias of ``--kpf_data_input``;
    ``--output_dir`` is fanned out post-parse by `resolve_dir_shortcuts`."""
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
# outputs take the root verbatim (their path builders add masters/{datecode} and
# L{N}/{datecode} substructure), while the log dir and plot dir each get a
# conventional subdirectory so --output_dir yields the same layout an explicit
# --log_dir/--plot_dir would -- in particular the plot subdir matches the
# timeseries default of {science_output}/QLP/timeseries.
_OUTPUT_DIR_SLOTS = {
    "kpf_masters_output": (),
    "kpf_science_output": (),
    "log_dir": ("logs",),
    "plot_dir": ("QLP", "timeseries"),
}


def resolve_dir_shortcuts(args):
    """Fan ``--output_dir`` out to each output directory the command left unset.

    ``--output_dir`` is a convenience: one root standing in for the masters output,
    the science output, the log dir, and (timeseries) the plot dir. The masters and
    science outputs take the root directly; the log dir lands at
    ``{output_dir}/logs`` and the plot dir at ``{output_dir}/QLP/timeseries`` (see
    ``_OUTPUT_DIR_SLOTS``), so the shortcut reproduces the layout an explicit
    ``--log_dir``/``--plot_dir`` would give rather than dumping logs and plots in the
    root. It is a fallback, never an override -- any slot an explicit flag already
    set keeps its value; only the unset ones, and only those the command actually
    has (masters has no science output or plot dir), inherit ``--output_dir``. The
    input dir is untouched (use ``--input_dir``/``--kpf_data_input``). Returns
    `args`; each command's ``parse_args`` calls it post-parse.
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
        "the job counts as a failure, rather than hanging the whole batch. The "
        "serial canary uses a larger, separate limit since it warms cold caches on "
        "the first run",
    )
    return p
