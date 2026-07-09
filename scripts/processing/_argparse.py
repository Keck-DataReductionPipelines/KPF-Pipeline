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


def recipe_parser():
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
    """[DATA_DIRS] override flags. ``--kpf_science_output`` is included only when
    `science_output` is true -- masters produces no science output, so its parser
    omits it."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--kpf_data_input", help="override [DATA_DIRS] KPF_DATA_INPUT")
    p.add_argument(
        "--kpf_masters_output", help="override [DATA_DIRS] KPF_MASTERS_OUTPUT"
    )
    if science_output:
        p.add_argument(
            "--kpf_science_output", help="override [DATA_DIRS] KPF_SCIENCE_OUTPUT"
        )
    return p


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
