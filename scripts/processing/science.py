#!/usr/bin/env python3
"""Reduce a set of science frames end-to-end (L0 -> L4).

A lightweight, fail-loud wrapper around the science recipe. Given an explicit
list of obs_ids, it dispatches each frame as a separate
``python -m tools.cli --science -o <obs_id>`` subprocess (the pipeline's own
command-line entry point), so every frame gets its own log file, clean process
state, and an independent exit code.

It reimplements no pipeline logic and does not decide *which* frames to reduce:
the caller supplies the obs_ids (discovering a target's frames from the L0 tree
lives in the orchestrator, not here). The single input form is:

    science.py --obs_id_list KP.20240405.40113.57 KP.20240405.40237.36

Reductions run in a bounded process pool: the first frame runs alone as a canary
to warm the shared on-disk caches, then the rest fan out (paced apart, since the
per-frame L0 pointing QC queries SIMBAD/Gaia). The run is fail-soft -- a frame
that fails to reduce is reported and the run continues with the others -- but the
script exits nonzero if any frame failed, so a caller gets a meaningful exit code.
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
from kpfpipe.utils.kpf_utils import is_obs_id
from tools.cli import shortcut_paths

_REPO = kpfpipe.REPO_ROOT

# Minimum spacing (seconds) between fanned-out subprocess launches. The per-frame
# L0 pointing QC fires rapid SIMBAD/Gaia catalog queries at startup, so a burst of
# pool workers launching at once would hammer those services; pacing the launches
# rate-limits them.
_LAUNCH_INTERVAL = 1.0


def _default_jobs():
    """Cores-based default job count: at most 25% of the CPUs, but always allow
    up to 16.

    Keeps a many-core shared machine from being monopolised (the 25% cap), while
    still letting a laptop use up to 16 cores even when that is a large fraction of
    them. Never exceeds the actual CPU count. os.cpu_count() returns None when it
    cannot be determined, so fall back to 1 (a valid positive default).
    """
    n = os.cpu_count() or 1
    return min(n, max(16, n // 4))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--obs_id_list",
        nargs="+",
        required=True,
        metavar="OBS_ID",
        help="one or more obs_ids to reduce, e.g. --obs_id_list "
        "KP.20240405.40113.57 KP.20240405.40237.36",
    )
    ap.add_argument(
        "-c",
        "--config",
        default=None,
        help="science recipe TOML override (default: the --science shortcut's "
        "configs/kpf_drp_science.toml)",
    )
    ap.add_argument("--kpf_data_input", help="override [DATA_DIRS] KPF_DATA_INPUT")
    ap.add_argument(
        "--kpf_masters_output", help="override [DATA_DIRS] KPF_MASTERS_OUTPUT"
    )
    ap.add_argument(
        "--kpf_science_output", help="override [DATA_DIRS] KPF_SCIENCE_OUTPUT"
    )
    ap.add_argument("--log_dir", help="override [LOGGER] log_dir")
    ap.add_argument(
        "--log_level", help="override [LOGGER] log_level for each recipe (e.g. DEBUG)"
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="max concurrent science reductions; left unset, defaults to a "
        "cores-based value (~25%% of CPUs, but up to 16)",
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

    for obs_id in args.obs_id_list:
        if not is_obs_id(obs_id):
            ap.error(f"not a valid obs_id: {obs_id!r}")
    args.obs_id_list = sorted(set(args.obs_id_list))

    if args.job_timeout < 1:
        ap.error("--job_timeout must be >= 1")
    if args.jobs is None:
        args.jobs = _default_jobs()
    elif args.jobs < 1:
        ap.error("--jobs must be >= 1")
    return args


def _dir_params(config_path, section):
    """Flattened dict for one config section (DATA_DIRS or LOGGER)."""
    return ConfigHandler(config_path).get_params([section])


# Live child recipe processes. Each is its own session leader
# (start_new_session=True); on interrupt the orchestrator kills every in-flight
# child's group via os.killpg, and `_interrupted` stops the pool launching more.
_live_procs = set()
_live_lock = threading.Lock()
_interrupted = threading.Event()

# Launch throttle: serializes subprocess launches to >= _LAUNCH_INTERVAL apart, so
# a burst of pool workers doesn't hammer SIMBAD/Gaia with the rapid catalog queries
# the per-frame L0 pointing QC fires. `_last_launch` holds the monotonic time of
# the most recent launch.
_launch_lock = threading.Lock()
_last_launch = [0.0]


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

    Every launch is spaced >= _LAUNCH_INTERVAL apart: the lock is held across the
    sleep so concurrent pool workers launch one interval apart, pacing the
    SIMBAD/Gaia queries the L0 pointing QC makes. The canary is the first launch,
    so it never waits (_last_launch starts at 0).
    """
    if _interrupted.is_set():
        return 130, ""
    with _launch_lock:
        wait = _LAUNCH_INTERVAL - (time.monotonic() - _last_launch[0])
        if wait > 0:
            time.sleep(wait)
        _last_launch[0] = time.monotonic()
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
    caches every job lazily builds -- barycorrpy leap-second / astropy IERS tables,
    the matplotlib font cache, compiled bytecode -- so the parallel jobs hit a warm
    cache instead of stampeding a cold one (the thundering-herd failure mode).

    Each fanned-out job is bounded by `job_timeout` seconds (the canary gets the
    larger `canary_timeout`, since it alone pays the cold-cache cost): a job that
    overruns is a wedged subprocess (a stalled NFS read, a runaway recipe), so its
    process group is killed and it returns 124 -- counted as a failure -- so one
    stuck unit can't hang the whole batch.

    The run is fail-soft: every frame is attempted regardless (a failed canary
    still fans out, since one bad frame must not stop the others), and failures are
    reported without raising. Returns the set of failed tags -- the caller decides
    the exit code. On Ctrl+C (SIGINT) or SIGTERM, cancel the queue and terminate
    every still-running child before exiting (130).
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

        # Always fan out the rest -- a bad canary frame must not block the others.
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
            header=f"WARNING: {len(failures)} {label} frame(s) failed",
        )
    return {tag for _, tag, _, _ in failures}


def _cli_task(obs_id, forward, config=None):
    """Build a (tag, argv) task reducing one frame via `python -m tools.cli`.

    Uses the CLI's --science shortcut (which resolves the science recipe + default
    config); `config` overrides the shortcut's default config only when given.
    `forward` is the resolved dir/log overrides appended to every invocation. The
    tag is the obs_id itself, which the sentinel/log paths key on.
    """
    argv = [sys.executable, "-m", "tools.cli", "--science"]
    if config:
        argv += ["-c", config]
    argv += ["-o", obs_id, *forward]
    return obs_id, argv


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

    # Resolve the config the subprocesses will use: the --science shortcut default
    # (tools.cli owns that path) unless overridden. We read it only for the log dir
    # (used in the failure sentinels); the subprocess gets the config via the
    # shortcut, plus -c only when overridden (see _cli_task).
    science_config = args.config or shortcut_paths("science")[1]
    log_dir = args.log_dir or _dir_params(science_config, "LOGGER").get("log_dir")

    forward = []
    for value, flag in (
        (args.kpf_data_input, "--kpf_data_input"),
        (args.kpf_masters_output, "--kpf_masters_output"),
        (args.kpf_science_output, "--kpf_science_output"),
        (args.log_dir, "--log_dir"),
        (args.log_level, "--log_level"),
    ):
        if value:
            forward += [flag, value]

    obs_ids = args.obs_id_list
    print(f"reducing {len(obs_ids)} science frame(s): {', '.join(obs_ids)}")

    tasks = [_cli_task(o, forward, config=args.config) for o in obs_ids]
    failed = run_stage(
        "science", tasks, args.jobs, log_dir, job_timeout=args.job_timeout
    )

    reduced = len(obs_ids) - len(failed)
    print(f"\ndone: reduced {reduced}/{len(obs_ids)} frame(s)")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
