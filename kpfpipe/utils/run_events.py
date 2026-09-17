"""Machine-readable run provenance inside the log: ``RUN_START`` and ``RUN_END``.

Every ``kpfpipe`` invocation emits exactly two INFO records through the normal
logging stack: a ``RUN_START {json}`` line as soon as its log is open and a
``RUN_END {json}`` line when it finishes. Together they say what ran, on what
code, where, when, and how it ended, without a second file on disk -- the log
stays the single artifact per invocation (DRP-RUN-08/09).

Linkage is by command line, never by environment: an orchestrator passes
``--parent_run <its own log path>`` to each child it launches and forwards any
``--flow_run_id`` it was itself given (an opaque id from an external orchestrator
such as KPF-Ops). The child records both in its ``RUN_START``.

``RUN_END`` is written by ``finish``; if the process exits without it (Ctrl-C,
an unhandled SystemExit) an ``atexit`` hook emits ``RUN_END`` with status
``interrupted``. A SIGKILL leaves no ``RUN_END`` at all -- an honest partial log.
"""

import atexit
import json
import logging
import os
import socket
import subprocess
import sys
import time

import kpfpipe

logger = logging.getLogger(__name__)

SCHEMA = 1
START_TAG = "RUN_START "
END_TAG = "RUN_END "

# Record kinds: the single-unit leaf, a batch orchestrator, the realtime watcher.
KINDS = ("run", "batch", "realtime")
STATUSES = ("succeeded", "failed", "interrupted")

_GIT_TIMEOUT_S = 5


def git_sha(repo_root=kpfpipe.REPO_ROOT):
    """The checked-out commit of `repo_root`, or None if git cannot say.

    Never raises: a source tarball, a missing ``git`` binary, or a slow
    filesystem all yield None rather than failing the run.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


def _utc_now():
    """ISO-8601 UT timestamp with a trailing Z, second resolution."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class RunEvents:
    """Emit this invocation's ``RUN_START`` now and its ``RUN_END`` at the end."""

    def __init__(self, fields):
        self.fields = fields
        self._finished = False

    @classmethod
    def start(
        cls,
        log_path,
        *,
        kind,
        recipe,
        target,
        config,
        parent=None,
        flow_run_id=None,
        argv=None,
    ):
        """Log ``RUN_START`` for this invocation and arm the ``atexit`` hook.

        Parameters
        ----------
        log_path : str
            This invocation's log file (from ``setup_logging``); children name it
            as their ``parent``.
        kind : {'run', 'batch', 'realtime'}
        recipe : str
            Short recipe/orchestrator name, e.g. ``science``/``masters``.
        target : str
            obs_id, datecode, ``batch`` or ``realtime``.
        config : str
            Path of the TOML config in force.
        parent : str or None
            The launching orchestrator's log path (``--parent_run``).
        flow_run_id : str or None
            Opaque external orchestrator id (``--flow_run_id``), forwarded verbatim.
        argv : list of str or None
            The command line; ``None`` means ``sys.argv``.
        """
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        argv = list(sys.argv if argv is None else argv)
        fields = {
            "schema": SCHEMA,
            "kind": kind,
            "recipe": recipe,
            "target": target,
            "command": " ".join(argv),
            "config": config,
            "git_sha": git_sha(),
            "drptag": kpfpipe.__version__,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "cwd": os.getcwd(),
            "started_utc": _utc_now(),
            "log_path": log_path,
            "parent": parent or None,
            "flow_run_id": flow_run_id or None,
        }
        self = cls(fields)
        logger.info("%s%s", START_TAG, json.dumps(fields, sort_keys=True))
        atexit.register(self._on_exit)
        return self

    def finish(self, exit_status, *, done=0, failed=0, skipped=0):
        """Log ``RUN_END``: ``succeeded`` for exit 0, else ``failed``.

        Raises
        ------
        RuntimeError
            If called twice.
        """
        if self._finished:
            raise RuntimeError("run events already finished")
        self._end(
            "succeeded" if exit_status == 0 else "failed",
            int(exit_status),
            {"done": done, "failed": failed, "skipped": skipped},
        )

    def is_finished(self):
        return self._finished

    def _on_exit(self):
        """atexit hook: a run that never called ``finish`` ended ``interrupted``."""
        if not self._finished:
            self._end("interrupted", None, None)

    def _end(self, status, exit_status, counts):
        self._finished = True
        fields = {
            "schema": SCHEMA,
            "status": status,
            "exit_status": exit_status,
            "ended_utc": _utc_now(),
            "counts": counts,
        }
        logger.info("%s%s", END_TAG, json.dumps(fields, sort_keys=True))


def parse_run_events(lines):
    """``(start, end)`` dicts from an iterable of log lines; either may be None.

    The reader side of the contract, for tests and for tooling that would rather
    not re-derive the tags. Takes the first ``RUN_START`` and the last ``RUN_END``.
    """
    start = end = None
    for line in lines:
        if start is None and START_TAG in line:
            start = _payload(line, START_TAG)
        elif END_TAG in line:
            end = _payload(line, END_TAG)
    return start, end


def _payload(line, tag):
    try:
        return json.loads(line[line.index(tag) + len(tag) :])
    except ValueError:
        return None
