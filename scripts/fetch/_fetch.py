"""The shared machinery behind every ``kpfpipe fetch <subject>`` script.

The fetch scripts differ only in which remote tree they pull from, so everything
else -- the flags, the multiplexed SSH connection, the remote listing, the rsync
per night and the fail-soft driver loop -- lives here. A subject script supplies
its name, its default remote directory and its own module docstring, and is
otherwise a few lines long.

Depends only on stdlib + ``kpfpipe``/``scripts`` -- never on ``tools``, like the
rest of the scripts layer.
"""

import argparse
import os
import shlex
import subprocess
import sys

from kpfpipe.utils.kpf import is_datecode
from scripts._argparse import dates_parser

# ``{user}`` is filled in from --user.
REMOTE_HOST = "{user}@shrek.caltech.edu"


def subject_parser(subject, default_remote_dir, description, gb_per_night=None):
    """The argument parser for a fetch subject, complete but not yet parsed.

    A subject with a `gb_per_night` estimate also gets ``--yes``, which skips the
    size confirmation `confirm_volume` would otherwise ask for. A subject needing
    flags of its own adds them to this and parses for itself (see ``masters.py``).

    The subject's name and estimate ride along on the parsed namespace, so `run`
    takes nothing the parser did not already know.
    """
    p = argparse.ArgumentParser(
        prog=f"kpfpipe fetch {subject}",
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[
            dates_parser("fetches every night present on the remote in it"),
        ],
    )
    p.add_argument(
        "-u",
        "--user",
        required=True,
        help="your username on the remote host",
    )
    p.add_argument(
        "--local_dir",
        required=True,
        help="local root to fetch into; each night lands in "
        f"{{local_dir}}/{subject}/{{datecode}}/",
    )
    p.add_argument(
        "--remote_dir",
        default=default_remote_dir,
        help="data root on the remote host (default: %(default)s)",
    )
    if gb_per_night:
        p.add_argument(
            "-y",
            "--yes",
            action="store_true",
            help=f"skip the size confirmation (~{gb_per_night} GB per night)",
        )
    p.set_defaults(subject=subject, gb_per_night=gb_per_night)
    return p


def confirm_volume(subject, datecodes, gb_per_night, local_dir):
    """Report the estimated transfer size and require a yes before continuing.

    Exits rather than prompting when stdin is not a terminal: a fetch this large
    should never start unattended by accident, and a piped run would otherwise read
    EOF and look like a refusal.
    """
    total = len(datecodes) * gb_per_night
    print(
        f"fetch {subject}: {len(datecodes)} night(s) x ~{gb_per_night} GB "
        f"= ~{total} GB into {local_dir}"
    )
    if not sys.stdin.isatty():
        sys.exit("error: refusing to start unattended; re-run with --yes")
    if input("continue? [y/N] ").strip().lower() not in ("y", "yes"):
        sys.exit("aborted")


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


def fetch_night(ssh, remote, remote_dir, datecode, local_dir, filters=()):
    """rsync one night's dir into ``{local_dir}/{datecode}/``; True on success.

    Inherits stdout/stderr so rsync's ``--progress`` reports live. ``--partial``
    keeps a part-transferred file so an interrupted run resumes into it. `filters`
    are extra rsync arguments -- the ``--include``/``--exclude`` rules a subject
    uses to pull part of a night; empty means the whole directory.
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
            *filters,
            source,
            destination,
        ],
        check=False,
    )
    return result.returncode == 0


def run(args, filters=()):
    """Fetch every selected night from a parsed `subject_parser` namespace.

    A subject carrying a ``gb_per_night`` estimate gates the transfer behind
    `confirm_volume` once the night count is known. `filters` are rsync rules
    narrowing what each night yields (see `fetch_night`).
    """
    remote = REMOTE_HOST.format(user=args.user)
    remote_dir = f"{args.remote_dir}/{args.subject}"
    local_dir = os.path.join(args.local_dir, args.subject)
    ssh = ssh_command()
    try:
        datecodes = args.dates or remote_datecodes(
            ssh, remote, remote_dir, *args.date_range
        )
        if args.gb_per_night and not args.yes:
            confirm_volume(args.subject, datecodes, args.gb_per_night, local_dir)
        os.makedirs(local_dir, exist_ok=True)
        failed = []
        for datecode in datecodes:
            print(f"=== {datecode}")
            if not fetch_night(ssh, remote, remote_dir, datecode, local_dir, filters):
                print(f"  !! {datecode}: transfer failed", file=sys.stderr)
                failed.append(datecode)
    except KeyboardInterrupt:
        return 130
    finally:
        close_ssh_connection(ssh, remote)

    print(f"\n=== {len(datecodes)} night(s) -> {local_dir}")
    if failed:
        print(f"failed: {' '.join(failed)}", file=sys.stderr)
        return 1
    return 0
