#!/usr/bin/env python3
"""Copy reduced L2 data off the remote host (``kpfpipe fetch L2``).

A developer convenience, not a pipeline stage: it reduces nothing and writes no
log: it shells out to ``rsync``, one invocation per night, and lets rsync's own
progress output go straight to the terminal. Nights are selected exactly as
``kpfpipe fetch masters`` selects them:

    kpfpipe fetch L2 -u <user> --dates 20240405 20240712 --local_dir ~/data
    kpfpipe fetch L2 -u <user> --dates nights.txt --local_dir ~/data
    kpfpipe fetch L2 -u <user> --date_range 20240101 20240131 --local_dir ~/data

The range form enumerates the datecode dirs present under the *remote* L2 root
within [START, END].

Each night lands in ``{local_dir}/L2/{datecode}/``, the same layout it has on the
remote. The whole run rides on one multiplexed SSH connection, so a batch
authenticates once. rsync skips files already present at the right size and time,
so re-running is cheap and an interrupted run resumes where it stopped. The run is
fail-soft (a night that fails to transfer is reported and the others continue) but
exits nonzero if any night failed.
"""

import sys

from scripts._argparse import resolve_dates
from scripts.fetch import _fetch

SUBJECT = "L2"
DEFAULT_REMOTE_DIR = "/data/kpf/vNext"


def main(argv=None):
    ap = _fetch.subject_parser(SUBJECT, DEFAULT_REMOTE_DIR, __doc__)
    return _fetch.run(resolve_dates(ap, ap.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
