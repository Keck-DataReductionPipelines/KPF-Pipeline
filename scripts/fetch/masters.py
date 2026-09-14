#!/usr/bin/env python3
"""Copy nightly master calibrations off the remote host (``kpfpipe fetch masters``).

A developer convenience, not a pipeline stage: it reduces nothing and writes no
log: it shells out to ``rsync``, one invocation per night, and lets rsync's own
progress output go straight to the terminal. Nights are selected exactly as
``kpfpipe masters`` selects them:

    kpfpipe fetch masters -u <user> --flat --dates 20240405 --local_dir ~/data
    kpfpipe fetch masters -u <user> --all --dates nights.txt --local_dir ~/data
    kpfpipe fetch masters -u <user> --bias --dark \\
        --date_range 20240101 20240131 --local_dir ~/data

A night holds every calibration kind together, and they differ in size by orders
of magnitude, so at least one kind must be named: ``--all``, or any combination of
``--bias``, ``--dark``, ``--flat``, ``--order_trace``, ``--thar``, ``--lfc`` and
``--etalon``. ``--order_trace`` is the CSV the flat stage writes alongside the
flat, selectable on its own because it is a few tens of KB against the flat's
~300 MB. ``--thar`` brings the per-exposure ``thar_L2/`` sidecar directory and the
diagnostics alongside it.

``--lfc`` and ``--etalon`` are accepted but match nothing today: those wavelength
solutions are not yet produced. They are wired up on the ``--thar`` pattern so
they start working as soon as the masters pipeline writes them.

The range form enumerates the datecode dirs present under the *remote* masters
root within [START, END], mirroring how ``kpfpipe masters`` enumerates local L0.

Each night lands in ``{local_dir}/masters/{datecode}/``, the same layout it has on the
remote. The whole run rides on one multiplexed SSH connection, so a batch
authenticates once. rsync skips files already present at the right size and time,
so re-running is cheap and an interrupted run resumes where it stopped. The run is
fail-soft (a night that fails to transfer is reported and the others continue) but
exits nonzero if any night failed.
"""

import sys

from scripts._argparse import resolve_dates
from scripts.fetch import _fetch

SUBJECT = "masters"
DEFAULT_REMOTE_DIR = "/data/kpf/vNext"

# What each --<kind> flag pulls out of a night, as rsync include patterns. Masters
# are named KP.<datecode>.<seconds>_master_<kind>_<level>.<ext>, so one glob per
# kind suffices. A wavelength solution also writes a per-exposure sidecar dir and a
# diagnostics file, so those kinds name three patterns; the trailing `/**` is what
# lets rsync descend into the sidecar once the exclude-everything rule is in force.
_SELECTIONS = {
    "bias": ["*_master_bias_L1.fits"],
    "dark": ["*_master_dark_L1.fits"],
    "flat": ["*_master_flat_L1.fits"],
    "order_trace": ["*_master_order_trace.csv"],
    "thar": ["*_master_thar_*", "thar_L2/", "thar_L2/**"],
    "lfc": ["*_master_lfc_*", "lfc_L2/", "lfc_L2/**"],
    "etalon": ["*_master_etalon_*", "etalon_L2/", "etalon_L2/**"],
}


def parse_args(argv=None):
    ap = _fetch.subject_parser(SUBJECT, DEFAULT_REMOTE_DIR, __doc__)
    ap.add_argument(
        "--all",
        action="store_true",
        help="fetch the whole night directory, every calibration kind",
    )
    for kind in _SELECTIONS:
        ap.add_argument(
            f"--{kind}",
            action="store_true",
            help=f"fetch the {kind.replace('_', ' ')} masters",
        )
    args = resolve_dates(ap, ap.parse_args(argv))

    if not args.all and not any(getattr(args, kind) for kind in _SELECTIONS):
        ap.error(
            "name at least one calibration kind: --all, or any of "
            + ", ".join(f"--{kind}" for kind in _SELECTIONS)
        )
    return args


def rsync_filters(args):
    """The rsync rules selecting the named kinds; empty means the whole night.

    ``--all`` takes the directory verbatim, so it needs no rules at all. Otherwise
    every selected kind's patterns are included and everything else excluded --
    the trailing ``--exclude=*`` is what makes the include list exhaustive.
    """
    if args.all:
        return []
    filters = []
    for kind, patterns in _SELECTIONS.items():
        if getattr(args, kind):
            filters += [f"--include={pattern}" for pattern in patterns]
    return filters + ["--exclude=*"]


def main(argv=None):
    args = parse_args(argv)
    return _fetch.run(args, filters=rsync_filters(args))


if __name__ == "__main__":
    sys.exit(main())
