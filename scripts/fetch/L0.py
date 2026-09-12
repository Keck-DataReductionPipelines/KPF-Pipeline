#!/usr/bin/env python3
"""Copy raw L0 data off the remote host (``kpfpipe fetch L0``).

A developer convenience, not a pipeline stage: it reduces nothing and writes no
log: it shells out to ``rsync``, one invocation per night, and lets rsync's own
progress output go straight to the terminal. Nights are selected exactly as
``kpfpipe fetch masters`` selects them:

    kpfpipe fetch L0 -u <user> --dates 20240405 20240712 --local_dir ~/L0
    kpfpipe fetch L0 -u <user> --dates nights.txt --local_dir ~/L0
    kpfpipe fetch L0 -u <user> --date_range 20240101 20240131 --local_dir ~/L0

The range form enumerates the datecode dirs present under the *remote* L0 root
within [START, END]. Unlike the reduced products, L0 is the shared raw archive
rather than a vNext output tree, so a night here is whole-night raw data and the
transfer is correspondingly large.

Each night lands in ``{local_dir}/{datecode}/``, the same layout it has on the
remote. The whole run rides on one multiplexed SSH connection, so a batch
authenticates once. rsync skips files already present at the right size and time,
so re-running is cheap and an interrupted run resumes where it stopped. The run is
fail-soft (a night that fails to transfer is reported and the others continue) but
exits nonzero if any night failed.
"""

import sys

from scripts.fetch import _fetch

SUBJECT = "L0"
DEFAULT_REMOTE_DIR = "/data/kpf/L0"


def main(argv=None):
    return _fetch.main(argv, SUBJECT, DEFAULT_REMOTE_DIR, __doc__)


if __name__ == "__main__":
    sys.exit(main())
