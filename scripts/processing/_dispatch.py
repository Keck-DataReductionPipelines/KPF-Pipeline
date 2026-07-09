"""Shared subprocess fan-out engine for the batch processing orchestrators.

Runs a set of recipe subprocesses -- one serial canary to warm the shared
on-disk caches, then a bounded pool -- with per-job timeout kills, clean
interrupt teardown, and fail-fast/fail-soft failure handling. Hoisted out of the
per-domain drivers (``masters.py``/``science.py``) so they share one engine
rather than each carrying a copy; ``rv_timeseries.py`` will adopt it in a
follow-up. Depends only on stdlib + ``kpfpipe`` -- never on ``tools`` -- so the
scripts stay ignorant of the CLI dispatcher above them.
"""

import concurrent.futures
import logging
import os
import signal
import subprocess
import sys
import threading
import time

import kpfpipe

# Batch narration + failure sentinels flow through this logger, so the
# orchestrator's setup_batch_logging handlers persist them to the batch log and
# echo them live to stdout. With no handlers installed (e.g. direct-call tests)
# the records are simply dropped.
logger = logging.getLogger(__name__)

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

    The base for the science default and the masters cap's cores floor. Keeps a
    many-core shared machine from being monopolised (the 25% cap), while still
    letting a laptop use up to 16 cores even when that is a large fraction of
    them. Never exceeds the actual CPU count. os.cpu_count() returns None when it
    cannot be determined, so fall back to 1 (a valid positive default).
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


# Live child recipe processes. Each is its own session leader
# (start_new_session=True); on interrupt the orchestrator kills every in-flight
# child's group via os.killpg, and `_interrupted` stops the pool launching more.
_live_procs = set()
_live_lock = threading.Lock()
_interrupted = threading.Event()

# Launch throttle: serializes subprocess launches to >= _launch_interval apart
# (science only), so a burst of pool workers doesn't hammer SIMBAD/Gaia with the
# rapid catalog queries the per-frame L0 pointing QC fires. `_last_launch` holds
# the monotonic time of the most recent launch.
_launch_lock = threading.Lock()
_last_launch = [0.0]


def _handle_termination_signal(signum, frame):
    """Turn SIGTERM into a KeyboardInterrupt so `kill` cleans up like Ctrl+C.

    SIGINT already raises KeyboardInterrupt in the main thread; routing SIGTERM
    through the same path lets `run_stage` catch both and terminate children.
    """
    raise KeyboardInterrupt


def configure_runtime():
    """Install the SIGTERM handler and pin subprocess BLAS/OpenMP to one thread.

    Call once at the start of an orchestrator's main() -- not at import, so
    importing this module never mutates signal handlers or the environment.
    Routes SIGTERM through the KeyboardInterrupt path so `kill` tears down
    children like Ctrl+C (see run_stage), and caps the BLAS/OpenMP thread pools at
    1 via setdefault: we already run one process per job (`--jobs`), so an
    OpenBLAS pool inside each would be jobs x cores threads and thrash. setdefault
    lets an explicit caller win.
    """
    signal.signal(signal.SIGTERM, _handle_termination_signal)
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(var, "1")


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


def _run_one(argv, timeout=None, launch_interval=0.0):
    """Run one recipe subprocess; return (returncode, captured stderr).

    The child starts in its own session (process group) and is tracked in
    `_live_procs` so `run_stage` can kill the whole tree on interrupt. `timeout`
    (seconds), used for the serial canary and each fanned-out job, bounds the run:
    on expiry the child's process group is killed and (124, note) returned -- 124
    being the conventional timeout exit status. Returns (130, "") without
    launching once teardown has begun (`_interrupted`), and also kills-then-returns
    130 if teardown begins in the window between launching and tracking the child.

    `launch_interval` (seconds, >0 for science) throttles the actual launch: the
    lock is held across the sleep so concurrent pool workers launch one interval
    apart, pacing the SIMBAD/Gaia queries the L0 pointing QC makes. The canary is
    the first launch, so it never waits (_last_launch starts at 0).
    """
    if _interrupted.is_set():
        return 130, ""
    if launch_interval > 0:
        with _launch_lock:
            wait = launch_interval - (time.monotonic() - _last_launch[0])
            if wait > 0:
                time.sleep(wait)
            _last_launch[0] = time.monotonic()
        if _interrupted.is_set():
            return 130, ""
    proc = subprocess.Popen(
        argv,
        cwd=kpfpipe.REPO_ROOT,
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


def run_stage(
    label,
    tasks,
    jobs,
    log_dir,
    job_timeout=600,
    canary_timeout=1800,
    abort_on_failure=True,
    launch_interval=0.0,
):
    """Run `tasks`: the first serially as a canary, then the rest in a pool.

    tasks: list of (tag, argv). The first unit runs alone (bounded by
    `canary_timeout` seconds) before any fan-out. This warms the shared on-disk
    caches every job lazily builds -- barycorrpy leap-second / astropy IERS
    tables, the matplotlib font cache, compiled bytecode -- so the parallel jobs
    hit a warm cache instead of stampeding a cold one (the thundering-herd
    failure mode).

    Each fanned-out job is bounded by `job_timeout` seconds (the canary gets the
    larger `canary_timeout`, since it alone pays the cold-cache cost): a job that
    overruns is a wedged subprocess (a stalled NFS read, a runaway recipe), so
    its process group is killed and it returns 124 -- counted as a failure -- so
    one stuck unit can't hang the whole batch instead of ever completing.

    `abort_on_failure` controls what a nonzero exit means:

    * True (default): fail-fast. A failed canary skips the fan-out; a failed pool
      job stops new launches (in-flight jobs finish). Any failure prints a
      per-unit sentinel and exits(1).
    * False (the standalone masters/science drivers): fail-soft. Every unit is
      attempted regardless (a failed canary still fans out, since one bad unit
      must not stop the others), and failures are reported as a warning without
      exiting -- the caller inspects the returned failure set to decide the exit
      code.

    `launch_interval` (seconds) spaces the fan-out launches apart (science passes
    1.0 to rate-limit the SIMBAD/Gaia queries the L0 pointing QC makes); the
    canary is unaffected (it runs alone, first).

    Returns the set of failed tags. On Ctrl+C (SIGINT) or SIGTERM, cancel the
    queue and terminate every still-running child before exiting (130), so an
    interrupt stops the whole run rather than leaving the pool draining orphaned
    subprocesses.
    """
    if not tasks:
        return set()
    logger.info(
        "[%s] dispatching %d job(s): 1 serial canary, then up to %d at once",
        label,
        len(tasks),
        jobs,
    )
    failures = []
    pool = None
    try:
        canary_tag, canary_argv = tasks[0]
        logger.info(
            "  [%s] canary %s (serial; warms shared caches)...", label, canary_tag
        )
        rc, stderr = _run_one(canary_argv, timeout=canary_timeout)
        if rc != 0:
            failures.append((label, canary_tag, rc, stderr))
            if abort_on_failure:
                logger.error(
                    "  [%s] FAILED: canary %s (exit %d); not fanning out",
                    label,
                    canary_tag,
                    rc,
                )
            else:
                logger.warning(
                    "  [%s] FAILED: canary %s (exit %d); continuing anyway",
                    label,
                    canary_tag,
                    rc,
                )
        else:
            logger.info("  [%s] ok: %s", label, canary_tag)

        # Fan out when the canary passed, or always in fail-soft mode (a bad
        # unit must not block the rest). Skip only when a fail-fast canary died.
        if rc == 0 or not abort_on_failure:
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=jobs)
            futures = {
                pool.submit(_run_one, argv, job_timeout, launch_interval): tag
                for tag, argv in tasks[1:]
            }
            for future in concurrent.futures.as_completed(futures):
                tag = futures[future]
                try:
                    rc, stderr = future.result()
                except concurrent.futures.CancelledError:
                    continue
                if rc == 0:
                    logger.info("  [%s] ok: %s", label, tag)
                    continue
                failures.append((label, tag, rc, stderr))
                if abort_on_failure:
                    logger.error(
                        "  [%s] FAILED: %s (exit %d); halting new %s jobs",
                        label,
                        tag,
                        rc,
                        label,
                    )
                    for pending in futures:
                        pending.cancel()
                else:
                    logger.warning(
                        "  [%s] FAILED: %s (exit %d); continuing", label, tag, rc
                    )
    except KeyboardInterrupt:
        _interrupted.set()
        logger.warning(
            "[%s] interrupted -- cancelling queued jobs and terminating running "
            "ones...",
            label,
        )
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        _terminate_all_children()
        logger.warning("[%s] all child processes stopped; exiting", label)
        sys.exit(130)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    if failures:
        if abort_on_failure:
            _report_failures(
                failures, log_dir, header=f"ABORTED: {len(failures)} job(s) failed"
            )
            sys.exit(1)
        _report_failures(
            failures,
            log_dir,
            header=(
                f"WARNING: {len(failures)} {label} job(s) failed; continuing "
                f"without them"
            ),
        )
    return {tag for _, tag, _, _ in failures}


def _report_failures(failures, log_dir, *, header):
    """Log an actionable sentinel for each failed subprocess, via the batch logger.

    `header` is the banner line -- an ABORTED line when the stage is fatal, or a
    WARNING line when a fail-soft stage is continuing past the failures. Records
    go through the module logger at ERROR, so the orchestrator's batch log (and its
    stdout echo) capture them; with no handlers installed they are simply dropped.
    """
    rule = "=" * 72
    logger.error("%s\n%s\n%s", rule, header, rule)
    for label, tag, rc, stderr in failures:
        hint = os.path.join(log_dir, "*", f"kpf_{label}_{tag}_*.log")
        lines = [f"FAILED [{label}] {tag} (exit {rc})", f"  inspect log: {hint}"]
        tail = (stderr or "").strip().splitlines()[-20:]
        if tail:
            lines.append("  --- last stderr lines ---")
            lines.extend(f"  {line}" for line in tail)
        logger.error("\n".join(lines))
