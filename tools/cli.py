"""KPF Pipeline CLI entry point: the ``kpfpipe`` command dispatcher.

``kpfpipe`` is the single front door to the pipeline. It is a thin, git-style
dispatcher that routes a subcommand to its implementation under
``scripts/processing/`` and forwards the remaining arguments verbatim -- the
subcommand owns its own argument parsing:

    kpfpipe run         -- reduce one recipe on one unit, in-process (the leaf)
    kpfpipe masters     -- build nightly master calibrations for a set of datecodes
    kpfpipe science     -- reduce a set of science frames end-to-end (L0 -> L4)
    kpfpipe timeseries  -- reduce a star's RV timeseries over a datecode range

Examples:

    kpfpipe run --masters -d 20240405                 # one night, in-process
    kpfpipe masters --dates 20240405 20240712         # batch (fans out `run`)
    kpfpipe science --obs_ids KP.20240405.40113.57
    kpfpipe timeseries --target 10700 --date_range 20240101 20240131

Run ``kpfpipe <command> -h`` for a command's own options.

Dependencies flow downward only: this interface layer imports the ``scripts``
layer; the scripts never import ``tools`` (see CLAUDE.md, "CLI architecture").
"""

import sys

# tools (interface) -> scripts (orchestration): a downward dependency. The
# scripts never import back up into tools.
from scripts.processing import masters, reduce, science, timeseries

_COMMANDS = {
    "run": reduce.main,
    "masters": masters.main,
    "science": science.main,
    "timeseries": timeseries.main,
}


def _usage():
    """The top-level usage banner listing the available subcommands."""
    return (
        "usage: kpfpipe <command> [options]\n\n"
        "commands:\n"
        "  run         reduce one recipe on one unit, in-process (the leaf)\n"
        "  masters     build nightly master calibrations for a set of datecodes\n"
        "  science     reduce a set of science frames end-to-end (L0 -> L4)\n"
        "  timeseries  reduce a star's RV timeseries over a datecode range\n\n"
        "Run `kpfpipe <command> -h` for a command's own options."
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
