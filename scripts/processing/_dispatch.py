"""Shared subprocess fan-out engine for the batch processing orchestrators.

Runs a set of recipe subprocesses -- one serial canary to warm the shared
on-disk caches, then a bounded pool -- with per-job timeout kills, clean
interrupt teardown, and fail-fast/fail-soft failure handling. Shared by
``masters.py``/``science.py``; depends only on stdlib + ``kpfpipe``, never on
``tools``.
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

# Batch narration + failure sentinels flow through this named logger so the
# orchestrator's setup_batch_logging handlers persist and echo them; with no
# handlers (e.g. direct-call tests) the records are dropped.
logger = logging.getLogger(__name__)

# Fixed cap on concurrent masters (stacking) jobs -- deliberately NOT cores/RAM
# derived. The stacking stage bottlenecks on the OS's own memory bookkeeping
# (page-fault / mapping churn coordinated across all cores), a cost that grows
# with the number of concurrent jobs, not the work each does: measured on shrek
# (256 cores, 2 TiB), a wide fan-out left CPUs ~75% idle with 1.4 TiB free and
# never swapped, yet every job crawled to --job_timeout. 16 is a ~2x margin below
# the ~32 where degradation appears. Override per machine with --jobs.
_MASTERS_JOBS = 16

# Approximate peak resident RAM (GiB) of one masters job with the L1 cache on
# (~4.8 GiB cache-off + ~1.3 GiB retained frames). Only a small-machine *floor* on
# the cap above so a cache-on run can't overcommit a modest box; never binds on a
# big host.
_MASTERS_JOB_GIB = 6


def _default_science_jobs():
    """Cores-based science job count: at most 25% of the CPUs, but always allow up
    to 16 (and never more than the CPU count).

    Keeps a many-core shared machine from being monopolised while still letting a
    laptop use up to 16 cores. os.cpu_count() may be None, so fall back to 1.
    """
    n = os.cpu_count() or 1
    return min(n, max(16, n // 4))


def _default_masters_jobs():
    """Masters default job count: the fixed _MASTERS_JOBS cap, floored for small
    machines.

    _MASTERS_JOBS (16) is the real limit (see its definition). Two floors keep it
    sane on a small machine: never exceed the cores-based count (the same formula
    _default_science_jobs uses, inlined so the two defaults stay independent), and
    never let the cache-on footprint (~_MASTERS_JOB_GIB each) overcommit RAM. On a
    big host neither binds. Falls back to the cores floor when physical RAM can't
    be determined.
    """
    n = os.cpu_count() or 1
    cap = min(_MASTERS_JOBS, min(n, max(16, n // 4)))
    try:
        ram_gib = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
    except (ValueError, OSError, AttributeError):
        return cap
    return max(1, min(cap, int(ram_gib // _MASTERS_JOB_GIB)))


# Live child recipe processes, each its own session leader
# (start_new_session=True). On interrupt the orchestrator kills every in-flight
# child's group via os.killpg; `_interrupted` stops the pool launching more.
_live_procs = set()
_live_lock = threading.Lock()
_interrupted = threading.Event()

# Launch throttle: serializes fan-out launches >= _launch_interval apart. Two
# callers, two reasons -- science paces the SIMBAD/Gaia catalog queries the per-frame
# L0 pointing QC fires; masters desynchronizes the I/O-heavy stack read/assemble
# phase so a lockstep wave doesn't saturate the disk. `_last_launch` is the monotonic
# time of the last launch.
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

    Call once at the start of an orchestrator's main(), not at import, so importing
    never mutates signal handlers or the environment. Routes SIGTERM through the
    KeyboardInterrupt path so `kill` tears down children like Ctrl+C. Caps the
    BLAS/OpenMP pools at 1 (via setdefault, so an explicit caller wins): we already
    run one process per job, so an inner OpenBLAS pool would be jobs x cores threads
    and thrash.
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
    (seconds) bounds the run: on expiry the process group is killed and (124, note)
    returned -- 124 being the conventional timeout exit status. Returns (130, "")
    without launching once teardown has begun (`_interrupted`), and also
    kills-then-returns 130 if teardown begins between launching and tracking.

    `launch_interval` (seconds, >0 for science and masters) throttles the launch,
    holding the lock across the sleep so pool workers launch one interval apart
    (science paces the SIMBAD/Gaia queries; masters desynchronizes the disk-read
    phase). The canary launches first, so it never waits.
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
        # Close the launch-vs-track race: teardown snapshots _live_procs under the
        # lock with _interrupted already set, so if it ran between Popen and add it
        # missed this child -- but _interrupted is then set, so re-checking here
        # kills our own child. Past this point the child is tracked.
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
            # KeyboardInterrupt during the canary unwinds here before `finally`
            # untracks proc -- kill the child now so it can't outlive us.
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
    """Run `tasks` (a list of (tag, argv)): the first serially as a canary, then
    the rest in a pool of `jobs` workers. Returns the set of failed tags.

    The canary runs alone (bounded by `canary_timeout`) before any fan-out, warming
    the shared on-disk caches every job lazily builds (barycorrpy leap-seconds,
    astropy IERS, matplotlib fonts, bytecode) so the parallel jobs don't stampede a
    cold cache. Each fanned-out job is bounded by the shorter `job_timeout`; on
    overrun (a stalled NFS read, a runaway recipe) its process group is killed and
    it returns 124, counted as a failure, so one stuck unit can't hang the batch.

    `abort_on_failure` controls what a nonzero exit means:

    * True (default): fail-fast. A failed canary skips the fan-out; a failed pool
      job stops new launches (in-flight jobs finish). Any failure prints a per-unit
      sentinel and exits(1).
    * False (the standalone masters/science drivers): fail-soft. Every unit is
      attempted (a failed canary still fans out), failures are warned but do not
      exit, and the caller inspects the returned failure set for its exit code.

    `launch_interval` (seconds) spaces the fan-out launches apart (science passes
    1.0 to rate-limit the SIMBAD/Gaia queries; masters passes a larger value to
    desynchronize the lockstep disk-read phase); the canary, running first, is
    unaffected. On SIGINT/SIGTERM, cancel the queue and terminate every running
    child before exiting 130, so an interrupt leaves no orphaned subprocesses.
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

        # Fan out when the canary passed, or always in fail-soft mode (one bad
        # unit must not block the rest); skip only when a fail-fast canary died.
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

    `header` is the banner line -- ABORTED when the stage is fatal, WARNING when a
    fail-soft stage continues past the failures. Records go through the module
    logger at ERROR.
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
