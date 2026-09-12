#!/usr/bin/env python3
"""Copy nightly master calibrations off the remote host (``kpfpipe fetch masters``).

A developer convenience, not a pipeline stage: it reduces nothing and writes no
log: it shells out to ``rsync``, one invocation per night, and lets rsync's own
progress output go straight to the terminal. Nights are selected exactly as
``kpfpipe masters`` selects them:

    kpfpipe fetch masters -u <user> --dates 20240405 20240712 --local_dir ~/masters
    kpfpipe fetch masters -u <user> --dates nights.txt --local_dir ~/masters
    kpfpipe fetch masters -u <user> --date_range 20240101 20240131 --local_dir ~/masters

The range form enumerates the datecode dirs present under the *remote* masters
root within [START, END], mirroring how ``kpfpipe masters`` enumerates local L0.

Each night lands in ``{local_dir}/{datecode}/``, the same layout it has on the
remote. The whole run rides on one multiplexed SSH connection, so a batch
authenticates once. rsync skips files already present at the right size and time,
so re-running is cheap and an interrupted run resumes where it stopped. The run is
fail-soft (a night that fails to transfer is reported and the others continue) but
exits nonzero if any night failed.
"""

import argparse
import os
import shlex
import subprocess
import sys

from kpfpipe.utils.kpf import is_datecode
from scripts._argparse import dates_parser, resolve_dates

# ``{user}`` is filled in from --user.
REMOTE_HOST = "{user}@shrek.caltech.edu"
DEFAULT_REMOTE_DIR = "/data/kpf/vNext/masters"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="kpfpipe fetch masters",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[
            dates_parser("fetches every night present on the remote in it"),
        ],
    )
    ap.add_argument(
        "-u",
        "--user",
        required=True,
        help="your username on the remote host",
    )
    ap.add_argument(
        "--local_dir",
        required=True,
        help="local root to fetch into; each night lands in {local_dir}/{datecode}/",
    )
    ap.add_argument(
        "--remote_dir",
        default=DEFAULT_REMOTE_DIR,
        help="masters root on the remote host (default: %(default)s)",
    )
    return resolve_dates(ap, ap.parse_args(argv))


def ssh_command():
    """The ``ssh`` argv prefix that multiplexes this run over one connection.

    Every rsync and the range listing share a single authenticated connection, so a
    batch prompts at most once instead of once per night. The control socket is
    keyed to this process; `close_ssh_connection` tears it down on the way out.
    Kept short and in /tmp because a control path is a Unix socket, which is capped
    near 104 bytes on macOS.
    """
    socket = f"/tmp/kpf_fetch.{os.getpid()}.sock"
    return [
        "ssh",
        "-o",
        "ControlMaster=auto",
        "-o",
        f"ControlPath={socket}",
        "-o",
        "ControlPersist=60",
    ]


def close_ssh_connection(ssh, remote):
    """Drop the multiplexed master connection; a closed one is not an error."""
    subprocess.run(
        ssh + ["-O", "exit", remote],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def remote_datecodes(ssh, remote, remote_dir, start, end):
    """The datecode dirs present under `remote_dir` within [start, end].

    The remote counterpart of ``_scan.datecodes_in_range``: a range names the nights
    that actually exist, so a gap in the archive is skipped rather than becoming a
    failed transfer. An empty result is fatal.
    """
    listing = subprocess.run(
        ssh + [remote, f"ls -1 {shlex.quote(remote_dir)}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if listing.returncode != 0:
        sys.exit(
            f"error: cannot list {remote}:{remote_dir} "
            f"(exit {listing.returncode}): {listing.stderr.strip()}"
        )
    nights = sorted(
        name
        for name in listing.stdout.split()
        if is_datecode(name) and start <= name <= end
    )
    if not nights:
        sys.exit(
            f"error: no datecode dirs under {remote}:{remote_dir} "
            f"in range {start}..{end}"
        )
    return nights


def fetch_night(ssh, remote, remote_dir, datecode, local_dir):
    """rsync one night's masters dir into ``{local_dir}/{datecode}/``; True on success.

    Inherits stdout/stderr so rsync's ``--progress`` reports live. ``--partial``
    keeps a part-transferred file so an interrupted run resumes into it.
    """
    source = f"{remote}:{shlex.quote(f'{remote_dir}/{datecode}')}/"
    destination = os.path.join(local_dir, datecode)
    result = subprocess.run(
        [
            "rsync",
            "-avh",
            "--partial",
            "--progress",
            "-e",
            shlex.join(ssh),
            source,
            destination,
        ],
        check=False,
    )
    return result.returncode == 0


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.local_dir, exist_ok=True)

    remote = REMOTE_HOST.format(user=args.user)
    ssh = ssh_command()
    try:
        datecodes = args.dates or remote_datecodes(
            ssh, remote, args.remote_dir, *args.date_range
        )
        failed = []
        for datecode in datecodes:
            print(f"=== {datecode}")
            if not fetch_night(ssh, remote, args.remote_dir, datecode, args.local_dir):
                print(f"  !! {datecode}: transfer failed", file=sys.stderr)
                failed.append(datecode)
    except KeyboardInterrupt:
        return 130
    finally:
        close_ssh_connection(ssh, remote)

    print(f"\n=== {len(datecodes)} night(s) -> {args.local_dir}")
    if failed:
        print(f"failed: {' '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
