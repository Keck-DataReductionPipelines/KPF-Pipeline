#!/usr/bin/env python3
"""Build nightly master calibrations for a set of datecodes.

A lightweight, fail-loud wrapper around the masters recipe. Given either an
explicit list of datecodes or an inclusive datecode range, it dispatches each
night as a separate ``python -m tools.cli --masters -d <datecode>`` subprocess
(the pipeline's own command-line entry point), so every night gets its own log
file, clean process state, and an independent exit code.

It reimplements no pipeline logic and does not decide *which* nights to build:
the caller supplies the datecodes. The two input forms are mutually exclusive:

    masters.py --datecode_list 20240405 20240712   # explicit datecode(s)
    masters.py --date_range 20240101 20240131      # every L0 night in the range

The range form enumerates the datecode directories present under
``{KPF_DATA_INPUT}/L0`` within [START, END]; the raw L0 location itself is
resolved by the recipe (from its config, or a forwarded --kpf_data_input
override) for every form, single datecode included.

Builds run in a bounded process pool: the first night runs alone as a canary to
warm the shared on-disk caches, then the rest fan out. The build is fail-soft --
a night that fails to build (e.g. the known WLS failure mode) is reported and
the run continues with the other nights -- but the script exits nonzero if any
night failed, so a caller gets a meaningful exit code.
"""

import argparse
import concurrent.futures
import os
import signal
import subprocess
import sys
import threading
import time

import kpfpipe
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.kpf_utils import is_datecode
from tools.cli import shortcut_paths

_REPO = kpfpipe.REPO_ROOT

# Fixed cap on concurrent masters (stacking) jobs. The stacking stage does NOT
# bottleneck on cores or RAM -- measured on shrek (256 cores, 2 TiB), a masters
# fan-out left the CPUs ~75% idle with 1.4 TiB free and never swapped, yet every
# job crawled to the --job_timeout: 1 job alone is fast, ~56 at once wedge. The
# limit is the operating system's own memory bookkeeping (page faults / mapping
# churn) coordinated across all cores, a cost that grows with the number of
# concurrent jobs, not with the work each does. So cap masters concurrency at a
# fixed, empirically tuned value rather than a cores- or RAM-derived one. 16 is a
# ~2x margin below the ~32 where such degradation has been observed on similar
# pipelines. Override per machine with --jobs.
_MASTERS_JOBS = 16

# Approximate resident RAM (GiB) of one masters job at its peak with the L1 cache
# enabled (~4.8 GiB measured cache-off + ~1.3 GiB for the retained assembled
# frames). Only used as a small-machine *floor* on the fixed cap above (so a
# high-concurrency, cache-on run can't overcommit a modest box's RAM); on a big
# host it never binds.
_MASTERS_JOB_GIB = 6


def _default_jobs():
    """Cores-based job count: at most 25% of the CPUs, but always allow up to 16.

    The base for the masters default. Keeps a many-core shared machine from being
    monopolised (the 25% cap), while still letting a laptop use up to 16 cores
    even when that is a large fraction of them. Never exceeds the actual CPU
    count. os.cpu_count() returns None when it cannot be determined, so fall back
    to 1 (a valid positive default).
    """
    n = os.cpu_count() or 1
    return min(n, max(16, n // 4))


def _default_masters_jobs():
    """Masters default job count: the fixed _MASTERS_JOBS cap, floored for small
    machines.

    _MASTERS_JOBS (16) is the real limit -- see its definition for why masters
    concurrency is a tuned constant rather than a cores/RAM formula. Two floors
    keep it sane on a small machine: never exceed the cores-based default, and
    never run so many cache-enabled jobs that their combined footprint
    (~_MASTERS_JOB_GIB each) overcommits physical RAM. On a big host neither floor
    binds and the result is simply _MASTERS_JOBS. Falls back to just the cores
    floor when physical RAM can't be determined (e.g. os.sysconf unavailable).
    """
    cap = min(_MASTERS_JOBS, _default_jobs())
    try:
        ram_gib = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
    except (ValueError, OSError, AttributeError):
        return cap
    return max(1, min(cap, int(ram_gib // _MASTERS_JOB_GIB)))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--datecode_list",
        nargs="*",
        default=None,
        metavar="DATECODE",
        help="one or more datecodes to build, e.g. --datecode_list 20240405 "
        "20240712 (mutually exclusive with --date_range)",
    )
    ap.add_argument(
        "--date_range",
        nargs=2,
        metavar=("START", "END"),
        help="inclusive datecode range; builds every L0 night in it, e.g. "
        "--date_range 20240101 20240131 (mutually exclusive with --datecode_list)",
    )
    ap.add_argument(
        "-c",
        "--config",
        default=None,
        help="masters recipe TOML override (default: the --masters shortcut's "
        "configs/kpf_drp_masters.toml)",
    )
    ap.add_argument("--kpf_data_input", help="override [DATA_DIRS] KPF_DATA_INPUT")
    ap.add_argument(
        "--kpf_masters_output", help="override [DATA_DIRS] KPF_MASTERS_OUTPUT"
    )
    ap.add_argument("--log_dir", help="override [LOGGER] log_dir")
    ap.add_argument(
        "--log_level", help="override [LOGGER] log_level for each recipe (e.g. DEBUG)"
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=None,
        help=(
            "max concurrent masters builds; left unset, defaults to the "
            f"{_MASTERS_JOBS}-job cap (stacking degrades from OS memory contention "
            "when too many run at once), floored by cores and RAM on small machines"
        ),
    )
    ap.add_argument(
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
    args = ap.parse_args(argv)

    # Exactly one input form: an explicit datecode list, or a range -- not both,
    # not neither.
    if bool(args.datecode_list) == bool(args.date_range):
        ap.error("give either --datecode_list or --date_range, not both or neither")

    if args.date_range:
        start, end = args.date_range
        for dc in (start, end):
            if not is_datecode(dc):
                ap.error(f"--date_range value is not a valid datecode: {dc!r}")
        if start > end:
            ap.error(f"--date_range START must be <= END (got {start} > {end})")
    else:
        for dc in args.datecode_list:
            if not is_datecode(dc):
                ap.error(f"not a valid datecode: {dc!r}")
        args.datecode_list = sorted(set(args.datecode_list))

    if args.job_timeout < 1:
        ap.error("--job_timeout must be >= 1")
    if args.jobs is None:
        args.jobs = _default_masters_jobs()
    elif args.jobs < 1:
        ap.error("--jobs must be >= 1")
    return args


def _dir_params(config_path, section):
    """Flattened dict for one config section (DATA_DIRS or LOGGER)."""
    return ConfigHandler(config_path).get_params([section])


def _datecode_dirs(root, start, end):
    """Sorted datecode subdirs of `root` within the inclusive [start, end] range."""
    return [
        d
        for d in sorted(os.listdir(root))
        if is_datecode(d) and start <= d <= end and os.path.isdir(os.path.join(root, d))
    ]


def resolve_datecodes(args, data_input):
    """The datecodes to build, from either input form.

    The explicit list is already validated and sorted in parse_args. A range is
    expanded here (it needs the resolved L0 input root): the datecode dirs present
    under {data_input}/L0 within the range. Either way an empty result is fatal.
    """
    if args.datecode_list:
        return args.datecode_list
    l0_root = os.path.join(data_input, "L0")
    if not os.path.isdir(l0_root):
        sys.exit(f"error: L0 input directory not found: {l0_root}")
    start, end = args.date_range
    nights = _datecode_dirs(l0_root, start, end)
    if not nights:
        sys.exit(f"error: no datecode dirs under {l0_root} in range {start}..{end}")
    return nights


# Live child recipe processes. Each is its own session leader
# (start_new_session=True); on interrupt the orchestrator kills every in-flight
# child's group via os.killpg, and `_interrupted` stops the pool launching more.
_live_procs = set()
_live_lock = threading.Lock()
_interrupted = threading.Event()


def _handle_termination_signal(signum, frame):
    """Turn SIGTERM into a KeyboardInterrupt so `kill` cleans up like Ctrl+C.

    SIGINT already raises KeyboardInterrupt in the main thread; routing SIGTERM
    through the same path lets `run_stage` catch both and terminate children.
    """
    raise KeyboardInterrupt


def _kill_group(proc, sig=signal.SIGKILL):
    """Signal a child's whole process group, ignoring an already-dead child."""
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _terminate_all_children(grace=5.0):
    """SIGTERM every live child's process group, then SIGKILL any survivor."""
    with _live_lock:
        procs = list(_live_procs)
    for p in procs:
        _kill_group(p, signal.SIGTERM)
    deadline = time.monotonic() + grace
    for p in procs:
        try:
            p.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for p in procs:
        if p.poll() is None:
            _kill_group(p, signal.SIGKILL)


def _run_one(argv, timeout=None):
    """Run one recipe subprocess; return (returncode, captured stderr).

    The child starts in its own session (process group) and is tracked in
    `_live_procs` so `run_stage` can kill the whole tree on interrupt. `timeout`
    (seconds), used for the serial canary and each fanned-out job, bounds the run:
    on expiry the child's process group is killed and (124, note) returned -- 124
    being the conventional timeout exit status. Returns (130, "") without
    launching once teardown has begun (`_interrupted`), and also kills-then-returns
    130 if teardown begins in the window between launching and tracking the child.
    """
    if _interrupted.is_set():
        return 130, ""
    proc = subprocess.Popen(
        argv,
        cwd=_REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    with _live_lock:
        _live_procs.add(proc)
    try:
        # Close the launch-vs-track race: _terminate_all_children snapshots
        # _live_procs under _live_lock, and _interrupted is set before it. If that
        # snapshot ran between the Popen above and the add, it missed this child --
        # but then _interrupted is already set (it precedes the snapshot, which
        # the lock orders before our add), so re-checking here catches it and we
        # kill our own child. From here on the child is tracked, so any later
        # teardown sees it directly.
        if _interrupted.is_set():
            _kill_group(proc)
            proc.communicate()  # best-effort reap
            return 130, ""
        try:
            _out, stderr = proc.communicate(timeout=timeout)
            return proc.returncode, stderr
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            _out, stderr = proc.communicate()  # reap the killed group
            return 124, f"timed out after {timeout}s; child process group killed"
        except BaseException:
            # KeyboardInterrupt during the serial canary unwinds here before the
            # finally untracks proc -- kill the child now so it can't outlive us.
            _kill_group(proc)
            proc.communicate()  # best-effort reap
            raise
    finally:
        with _live_lock:
            _live_procs.discard(proc)


def run_stage(label, tasks, jobs, log_dir, job_timeout=600, canary_timeout=1800):
    """Run `tasks`: the first serially as a canary, then the rest in a pool.

    tasks: list of (tag, argv). The first unit runs alone (bounded by
    `canary_timeout` seconds) before any fan-out. This warms the shared on-disk
    caches every job lazily builds -- astropy IERS tables, compiled bytecode -- so
    the parallel jobs hit a warm cache instead of stampeding a cold one (the
    thundering-herd failure mode).

    Each fanned-out job is bounded by `job_timeout` seconds (the canary gets the
    larger `canary_timeout`, since it alone pays the cold-cache cost): a job that
    overruns is a wedged subprocess (a stalled NFS read, a runaway recipe), so its
    process group is killed and it returns 124 -- counted as a failure -- so one
    stuck unit can't hang the whole batch.

    The build is fail-soft: every night is attempted regardless (a failed canary
    still fans out, since one bad night must not stop the others), and failures
    are reported without raising. Returns the set of failed tags -- the caller
    decides the exit code. On Ctrl+C (SIGINT) or SIGTERM, cancel the queue and
    terminate every still-running child before exiting (130).
    """
    if not tasks:
        return set()
    print(
        f"[{label}] dispatching {len(tasks)} job(s): 1 serial canary, "
        f"then up to {jobs} at once"
    )
    failures = []
    pool = None
    try:
        canary_tag, canary_argv = tasks[0]
        print(f"  [{label}] canary {canary_tag} (serial; warms shared caches)...")
        rc, stderr = _run_one(canary_argv, timeout=canary_timeout)
        if rc != 0:
            failures.append((label, canary_tag, rc, stderr))
            print(f"  [{label}] FAILED: canary {canary_tag} (exit {rc}); continuing")
        else:
            print(f"  [{label}] ok: {canary_tag}")

        # Always fan out the rest -- a bad canary night must not block the others.
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=jobs)
        futures = {
            pool.submit(_run_one, argv, job_timeout): tag for tag, argv in tasks[1:]
        }
        for future in concurrent.futures.as_completed(futures):
            tag = futures[future]
            try:
                rc, stderr = future.result()
            except concurrent.futures.CancelledError:
                continue
            if rc == 0:
                print(f"  [{label}] ok: {tag}")
                continue
            failures.append((label, tag, rc, stderr))
            print(f"  [{label}] FAILED: {tag} (exit {rc}); continuing")
    except KeyboardInterrupt:
        _interrupted.set()
        print(
            f"\n[{label}] interrupted -- cancelling queued jobs and terminating "
            f"running ones...",
            file=sys.stderr,
        )
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        _terminate_all_children()
        print(f"[{label}] all child processes stopped; exiting", file=sys.stderr)
        sys.exit(130)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    if failures:
        _report_failures(
            failures,
            log_dir,
            header=f"WARNING: {len(failures)} {label} night(s) failed",
        )
    return {tag for _, tag, _, _ in failures}


def _cli_task(datecode, forward, config=None):
    """Build a (tag, argv) task building one night via `python -m tools.cli`.

    Uses the CLI's --masters shortcut (which resolves the masters recipe + default
    config); `config` overrides the shortcut's default config only when given.
    `forward` is the resolved dir/log overrides appended to every invocation. The
    tag is the datecode itself, which the sentinel/log paths key on.
    """
    argv = [sys.executable, "-m", "tools.cli", "--masters"]
    if config:
        argv += ["-c", config]
    argv += ["-d", datecode, *forward]
    return datecode, argv


def _report_failures(failures, log_dir, *, header):
    """Print an actionable sentinel for each failed subprocess, to stderr."""
    print(f"\n{'=' * 72}", file=sys.stderr)
    print(header, file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    for label, tag, rc, stderr in failures:
        hint = os.path.join(log_dir, "*", f"kpf_{label}_{tag}_*.log")
        print(f"\nFAILED [{label}] {tag} (exit {rc})", file=sys.stderr)
        print(f"  inspect log: {hint}", file=sys.stderr)
        tail = (stderr or "").strip().splitlines()[-20:]
        if tail:
            print("  --- last stderr lines ---", file=sys.stderr)
            for line in tail:
                print(f"  {line}", file=sys.stderr)


def main(argv=None):
    # Route SIGTERM through the KeyboardInterrupt path so `kill` tears down
    # children like Ctrl+C (see run_stage). Here, not at import, so importing
    # this module doesn't mutate signal handlers.
    signal.signal(signal.SIGTERM, _handle_termination_signal)

    # Pin each recipe subprocess to single-threaded BLAS/OpenMP: we already run
    # one process per job (`--jobs`), so an OpenBLAS pool inside each would be
    # jobs x cores threads and thrash. setdefault lets an explicit caller win.
    for _var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(_var, "1")

    args = parse_args(argv)

    # Resolve the config the subprocesses will use: the --masters shortcut default
    # (tools.cli owns that path) unless overridden. We read it to resolve effective
    # dirs; the subprocess gets the config via the shortcut, plus -c only when
    # overridden (see _cli_task).
    masters_config = args.config or shortcut_paths("masters")[1]
    data_input = (
        args.kpf_data_input
        or _dir_params(masters_config, "DATA_DIRS")["KPF_DATA_INPUT"]
    )
    log_dir = args.log_dir or _dir_params(masters_config, "LOGGER").get("log_dir")

    forward = []
    for value, flag in (
        (args.kpf_data_input, "--kpf_data_input"),
        (args.kpf_masters_output, "--kpf_masters_output"),
        (args.log_dir, "--log_dir"),
        (args.log_level, "--log_level"),
    ):
        if value:
            forward += [flag, value]

    datecodes = resolve_datecodes(args, data_input)
    print(f"building masters for {len(datecodes)} night(s): {', '.join(datecodes)}")

    tasks = [_cli_task(dc, forward, config=args.config) for dc in datecodes]
    failed = run_stage(
        "masters", tasks, args.jobs, log_dir, job_timeout=args.job_timeout
    )

    built = len(datecodes) - len(failed)
    print(f"\ndone: built masters for {built}/{len(datecodes)} night(s)")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
