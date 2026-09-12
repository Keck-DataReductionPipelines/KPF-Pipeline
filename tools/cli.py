"""KPF Pipeline CLI entry point: the ``kpfpipe`` command dispatcher.

``kpfpipe`` is the single front door to the pipeline. It is a thin, git-style
dispatcher that routes a subcommand to its ``processing/`` orchestrator under
``scripts/`` and forwards the remaining arguments verbatim -- the subcommand owns
its own argument parsing:

    kpfpipe run         -- reduce one recipe on one unit, in-process (the leaf)
    kpfpipe masters     -- build nightly master calibrations for a set of datecodes
    kpfpipe science     -- reduce a set of science frames end-to-end (L0 -> L4)
    kpfpipe timeseries  -- reduce a star's RV timeseries over a datecode range
    kpfpipe analyze     -- track instrument/calibrator behavior over a date range

``analyze`` is the one command with a second level: it reduces nothing, and routes
to an ``scripts/analysis`` script named by its subject (``analyze thar``,
``analyze flat``, ...), each of which reports on how that subject changes over time.
    kpfpipe realtime    -- watch the L0 tree and reduce new science frames as they land

Examples:

    kpfpipe run --masters -d 20240405                 # one night, in-process
    kpfpipe masters --dates 20240405 20240712         # batch (fans out `run`)
    kpfpipe science --obs_ids KP.20240405.40113.57
    kpfpipe timeseries --target 10700 --date_range 20240101 20240131
    kpfpipe analyze thar --date_range 20240727 20241022

Run ``kpfpipe <command> -h`` for a command's own options.

Dependencies flow downward only: this interface layer imports the ``scripts``
layer; the scripts never import ``tools`` (see CLAUDE.md, "CLI architecture").
"""

import sys

from scripts.analysis import thar
from scripts.processing import masters, realtime, reduce, science, timeseries

# The `analyze` subcommands, keyed by subject. Kept here rather than in
# scripts/analysis so the scripts stay ignorant of the dispatcher above them, as
# the processing drivers are.
_ANALYSES = {
    "thar": thar.main,
}


def _analyze(argv):
    """Route ``kpfpipe analyze <subject>`` to its analysis script.

    A second dispatcher of the same shape as `main`: it owns only the subject
    lookup and forwards the rest verbatim, so each analysis script parses its own
    options and stays runnable as ``python -m scripts.analysis.<subject>``.
    """
    if not argv or argv[0] in ("-h", "--help"):
        print(_analyze_usage())
        return 0

    subject, rest = argv[0], argv[1:]
    if subject not in _ANALYSES:
        print(f"kpfpipe analyze: unknown subject {subject!r}\n", file=sys.stderr)
        print(_analyze_usage(), file=sys.stderr)
        raise SystemExit(2)

    return _ANALYSES[subject](rest)


_COMMANDS = {
    "run": reduce.main,
    "masters": masters.main,
    "science": science.main,
    "timeseries": timeseries.main,
    "analyze": _analyze,
    "realtime": realtime.main,
}


def _usage():
    """The top-level usage banner listing the available subcommands."""
    return (
        "usage: kpfpipe <command> [options]\n\n"
        "commands:\n"
        "  run         reduce one recipe on one unit, in-process (the leaf)\n"
        "  masters     build nightly master calibrations for a set of datecodes\n"
        "  science     reduce a set of science frames end-to-end (L0 -> L4)\n"
        "  timeseries  reduce a star's RV timeseries over a datecode range\n"
        "  analyze     track instrument/calibrator behavior over a date range\n\n"
        "  realtime    watch the L0 tree and reduce new science frames as they land\n\n"
        "Run `kpfpipe <command> -h` for a command's own options."
    )


def _analyze_usage():
    """The ``analyze`` usage banner listing the available subjects."""
    return (
        "usage: kpfpipe analyze <subject> [options]\n\n"
        "subjects:\n"
        "  thar        which ThAr lamp (HCLSN) was in use, night by night\n\n"
        "Run `kpfpipe analyze <subject> -h` for a subject's own options."
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help"):
        print(_usage())
        return 0

    command, rest = argv[0], argv[1:]
    if command not in _COMMANDS:
        print(f"kpfpipe: unknown command {command!r}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        raise SystemExit(2)

    # Forward the remaining args verbatim; the subcommand parses its own options.
    return _COMMANDS[command](rest)


if __name__ == "__main__":
    main()
