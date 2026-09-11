"""Per-invocation ``run.json`` run records.

Every ``kpfpipe`` invocation -- the ``run`` leaf and the ``masters``/``science``
batch orchestrators -- writes one machine-readable record beside its log file
(same directory, same stem, ``.run.json`` instead of ``.log``). The record is
written at start with ``status: running`` and rewritten at exit, so a crash
leaves an honest partial record rather than nothing. It exists so operations
tooling (KPF-Ops) and shell users can answer "what ran, from which pipeline
commit, and how did it end" without parsing log text.

Stdlib only, and independent of the logging stack: the record must still be
written when logging setup itself has failed.
"""

import atexit
import glob
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import time

import kpfpipe

logger = logging.getLogger(__name__)

SCHEMA = 1

# Record kinds: the single-unit leaf, a batch orchestrator, the realtime watcher.
KINDS = ("run", "batch", "realtime")

# Environment variables that link records together. An orchestrator exports
# PARENT_ENV (its own run.json path) before fanning out; each child records it as
# ``parent``. FLOW_RUN_ENV is set by KPF-Ops flows so a record can be joined to the
# Prefect flow run that launched it; absent outside KPF-Ops.
PARENT_ENV = "KPFPIPE_PARENT_RUN"
FLOW_RUN_ENV = "KPFOPS_FLOW_RUN_ID"

# ``x.log`` -> ``x.run.json``; a collision-suffixed ``x.log.3`` -> ``x.3.run.json``.
_LOG_SUFFIX = re.compile(r"\.log(?:\.(\d+))?$")

_GIT_TIMEOUT_S = 5


def run_json_path(log_path):
    """Return the sidecar path for a log file path.

    Parameters
    ----------
    log_path : str
        A log path produced by ``kpfpipe.utils.logger.setup_logging`` (ending in
        ``.log`` or a collision-suffixed ``.log.N``).

    Returns
    -------
    str

    Raises
    ------
    ValueError
        If ``log_path`` does not end in ``.log`` / ``.log.N``.
    """
    m = _LOG_SUFFIX.search(log_path)
    if m is None:
        raise ValueError(f"not a kpfpipe log path (no .log suffix): {log_path!r}")
    stem = log_path[: m.start()]
    suffix = f".{m.group(1)}" if m.group(1) else ""
    return f"{stem}{suffix}.run.json"


def git_sha(repo_root=kpfpipe.REPO_ROOT):
    """Return the HEAD commit sha of ``repo_root``, or None if unavailable.

    Never raises: a missing ``git`` binary, a damaged or absent ``.git``, or a
    timeout all yield None, which the record stores as ``null``. A null sha is an
    honest "unknown"; it is never replaced with a guess.
    """
    if shutil.which("git") is None:
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    sha = out.stdout.strip()
    return sha or None


def write_json_atomic(path, data):
    """Write ``data`` as JSON to ``path`` atomically (temp file + ``os.replace``).

    Creates the parent directory as needed. A reader never observes a partial
    file: it sees either the previous complete record or the new one.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def read_run_record(path):
    """Load a run record, validating its ``schema`` field.

    Raises
    ------
    ValueError
        If the file's ``schema`` is not the one this module writes.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("schema") != SCHEMA:
        raise ValueError(
            f"{path}: run record schema {data.get('schema')!r}, expected {SCHEMA}"
        )
    return data


def _utc_now():
    """ISO-8601 UT timestamp with a trailing Z, second resolution."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class RunRecord:
    """The run.json record for one invocation: start now, finish at exit.

    ``start`` writes the record with ``status: running`` and registers an
    ``atexit`` hook; if the process exits without ``finish`` (``sys.exit(130)``
    on Ctrl-C, an unhandled SystemExit) the hook rewrites it as ``interrupted``.
    A SIGKILL leaves the ``running`` record in place -- an honest partial record.
    """

    def __init__(self, path, record):
        self.path = path
        self.record = record
        self._finished = False

    @classmethod
    def start(cls, log_path, *, kind, recipe, target, config, argv=None):
        """Create and write the ``running`` record for this invocation.

        Parameters
        ----------
        log_path : str
            The invocation's log file (from ``setup_logging``); the record is
            written beside it (``run_json_path``).
        kind : {'run', 'batch', 'realtime'}
            ``run`` for the single-unit leaf, ``batch`` for an orchestrator,
            ``realtime`` for the continuous-mode watcher.
        recipe : str
            Short recipe/orchestrator name, e.g. ``science``/``masters``.
        target : str
            obs_id, datecode, or ``batch``.
        config : str
            Path of the TOML config in force.
        argv : list of str or None
            The command line; ``None`` means ``sys.argv``.
        """
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        argv = list(sys.argv if argv is None else argv)
        record = {
            "schema": SCHEMA,
            "kind": kind,
            "status": "running",
            "recipe": recipe,
            "target": target,
            "command": " ".join(argv),
            "argv": argv,
            "config": config,
            "git_sha": git_sha(),
            "drptag": kpfpipe.__version__,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "cwd": os.getcwd(),
            "started_utc": _utc_now(),
            "ended_utc": None,
            "exit_status": None,
            "counts": {"done": 0, "failed": 0, "skipped": 0},
            "log_path": log_path,
            "parent": os.environ.get(PARENT_ENV),
            "flow_run_id": os.environ.get(FLOW_RUN_ENV),
            "children": [],
        }
        self = cls(run_json_path(log_path), record)
        write_json_atomic(self.path, self.record)
        atexit.register(self._on_exit)
        return self

    def finish(self, exit_status, *, done=0, failed=0, skipped=0, children=None):
        """Finalize the record: ``succeeded`` for exit 0, else ``failed``.

        Raises
        ------
        RuntimeError
            If called twice.
        """
        if self._finished:
            raise RuntimeError(f"run record already finished: {self.path}")
        self.record["status"] = "succeeded" if exit_status == 0 else "failed"
        self.record["exit_status"] = int(exit_status)
        self.record["ended_utc"] = _utc_now()
        self.record["counts"] = {"done": done, "failed": failed, "skipped": skipped}
        if children is not None:
            self.record["children"] = list(children)
        self._finished = True
        write_json_atomic(self.path, self.record)

    def is_finished(self):
        return self._finished

    def _on_exit(self):
        """atexit hook: mark a still-running record ``interrupted``."""
        if self._finished:
            return
        self.record["status"] = "interrupted"
        self.record["ended_utc"] = _utc_now()
        self._finished = True
        write_json_atomic(self.path, self.record)


def collect_child_records(log_dir, parent_path):
    """List the child run records under ``log_dir`` that name ``parent_path``.

    Scans ``{log_dir}/*/*.run.json`` (the one-level date layout ``setup_logging``
    writes). Files that are not valid run records are skipped with a WARNING
    rather than aborting the batch's own bookkeeping. Returned entries carry the
    child's ``target`` as ``tag`` and are sorted by it.
    """
    children = []
    for path in glob.glob(os.path.join(log_dir, "*", "*.run.json")):
        try:
            data = read_run_record(path)
        except (OSError, ValueError) as exc:
            # json.JSONDecodeError is a ValueError subclass.
            logger.warning("skipping unreadable run record %s: %s", path, exc)
            continue
        if data.get("parent") != parent_path:
            continue
        children.append(
            {
                "tag": data.get("target"),
                "exit_status": data.get("exit_status"),
                "status": data.get("status"),
                "run_json": path,
            }
        )
    children.sort(key=lambda c: str(c["tag"]))
    return children
