#!/usr/bin/env python3
"""Produce an RV timeseries for a single star over a datecode range.

A lightweight, fail-loud wrapper around the existing masters and science
recipes. Given a target star and an inclusive datecode range, it

  1. combs the L0 input tree for that target's raw science frames (obs_ids),
  2. infers the unique nights (datecodes) those frames span,
  3. (optionally) rebuilds the nightly calibration masters for each night,
  4. reduces every science frame end-to-end (L0 -> L4), then
  5. plots the RV timeseries (RV vs BJD_TDB, with RVERR error bars) when a
     --plot_directory is given.

It reimplements no pipeline logic: each night and each frame is dispatched as a
separate ``python -m tools.cli`` subprocess (the pipeline's own command-line
entry point), so every invocation gets its own log file, clean process state,
and an independent exit code. Steps 3 and 4 run in a bounded process pool; the
whole run aborts loudly if any masters night or any science frame fails, naming
the offending obs_id/datecode and its log so it can be investigated. The reduced
L1/L2/L4 products land in the usual output dirs; when --plot_directory is given,
the RV timeseries plot is written there once every frame has reduced.
"""

import argparse
import concurrent.futures
import glob
import os
import signal
import subprocess
import sys
import threading
import time

import numpy as np
from astropy.io import fits

import kpfpipe
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.io import build_filepath, build_mini_database, glob_masters
from kpfpipe.utils.kpf import get_datecode, get_obs_id, is_datecode

# Masters every science frame depends on, as (cal_type, level). Flat masters are
# scaffolded but not implemented, so they are not (yet) required -- this matches
# CalibrationAssociation.perform(["bias", "dark", "thar"]) in the science recipe.
_REQUIRED_MASTERS = [("bias", "L1"), ("dark", "L1"), ("thar", "L2")]

_REPO = kpfpipe.REPO_ROOT
_MASTERS_RECIPE = os.path.join(_REPO, "recipes", "kpf_drp_masters.py")
_SCIENCE_RECIPE = os.path.join(_REPO, "recipes", "kpf_drp_science.py")
_DEFAULT_MASTERS_CONFIG = os.path.join(_REPO, "configs", "kpf_drp_masters.toml")
_DEFAULT_SCIENCE_CONFIG = os.path.join(_REPO, "configs", "kpf_drp_science.toml")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "target", help="star id as it appears in the L0 OBJECT header, e.g. 10700"
    )
    ap.add_argument(
        "--date_range",
        nargs=2,
        metavar=("START", "END"),
        required=True,
        help="inclusive datecode range, e.g. --date_range 20240101 20240131",
    )
    ap.add_argument(
        "--masters_config",
        default=_DEFAULT_MASTERS_CONFIG,
        help="masters recipe TOML (default: configs/kpf_drp_masters.toml)",
    )
    ap.add_argument(
        "--science_config",
        default=_DEFAULT_SCIENCE_CONFIG,
        help="science recipe TOML (default: configs/kpf_drp_science.toml)",
    )
    ap.add_argument(
        "--reprocess_masters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="rebuild masters for every night in range (default: on; pass "
        "--no-reprocess_masters to reuse masters already on disk)",
    )
    ap.add_argument("--kpf_data_input", help="override [DATA_DIRS] KPF_DATA_INPUT")
    ap.add_argument(
        "--kpf_masters_output", help="override [DATA_DIRS] KPF_MASTERS_OUTPUT"
    )
    ap.add_argument(
        "--kpf_science_output", help="override [DATA_DIRS] KPF_SCIENCE_OUTPUT"
    )
    ap.add_argument("--log_directory", help="override [LOGGER] log_directory")
    ap.add_argument(
        "--plot_directory",
        help="directory to write the RV timeseries plot into; if omitted, the "
        "reduction still runs but no plot is produced",
    )
    ap.add_argument(
        "--group_intranight_obs",
        action="store_true",
        help="combine all of a night's RVs into one point before plotting, via "
        "an RVERR-weighted average",
    )
    ap.add_argument(
        "--file_limit",
        type=int,
        default=500,
        help="guardrail: abort if more than this many science frames match "
        "(default: 500)",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="max concurrent recipe subprocesses (default: CPU count)",
    )
    args = ap.parse_args(argv)

    start, end = args.date_range
    for dc in (start, end):
        if not is_datecode(dc):
            ap.error(f"--date_range value is not a valid datecode: {dc!r}")
    if start > end:
        ap.error(f"--date_range START must be <= END (got {start} > {end})")
    if args.file_limit < 1:
        ap.error("--file_limit must be >= 1")
    if args.jobs < 1:
        ap.error("--jobs must be >= 1")
    return args


def _dir_params(config_path, section):
    """Flattened dict for one config section (DATA_DIRS or LOGGER)."""
    return ConfigHandler(config_path).get_params([section])


def discover_science_obs_ids(data_input, target, start, end, file_limit):
    """Raw science obs_ids for `target` over [start, end], from the L0 tree.

    Enumerates the nights (datecode dirs) under {data_input}/L0, scans each one's
    PRIMARY headers via build_mini_database, and keeps the frames whose IMTYPE is
    'Object' and whose OBJECT matches `target`. Every directory under L0 must be a
    valid datecode -- an unexpected entry aborts the run (fail loudly, don't skip).
    """
    l0_root = os.path.join(data_input, "L0")
    if not os.path.isdir(l0_root):
        sys.exit(f"error: L0 input directory not found: {l0_root}")

    subdirs = [
        e
        for e in sorted(os.listdir(l0_root))
        if os.path.isdir(os.path.join(l0_root, e))
    ]
    invalid = [d for d in subdirs if not is_datecode(d)]
    if invalid:
        sys.exit(
            f"error: non-datecode director(y/ies) under {l0_root}: "
            f"{', '.join(invalid)} -- expected one datecode dir per night"
        )

    nights = [d for d in subdirs if start <= d <= end]
    if not nights:
        sys.exit(f"error: no datecode dirs under {l0_root} in range {start}..{end}")

    obs_ids = []
    for dc in nights:
        df = build_mini_database(os.path.join(l0_root, dc))
        is_object = df["IMTYPE"].astype(str).str.strip() == "Object"
        is_target = df["OBJECT"].astype(str).str.strip() == str(target)
        for fn in df.loc[is_object & is_target, "FILENAME"]:
            obs_ids.append(get_obs_id(fn))

    obs_ids = sorted(set(obs_ids))
    if not obs_ids:
        sys.exit(
            f"error: no science frames for target {target!r} in range "
            f"{start}..{end} under {l0_root}"
        )
    if len(obs_ids) > file_limit:
        sys.exit(
            f"error: {len(obs_ids)} science frames exceed --file_limit "
            f"{file_limit}; narrow --date_range or raise the limit"
        )
    return obs_ids


def check_masters_exist(masters_output, datecodes):
    """Abort unless every required master exists for every night in `datecodes`."""
    missing = []
    for dc in datecodes:
        for cal_type, level in _REQUIRED_MASTERS:
            if not glob.glob(glob_masters(masters_output, cal_type, level, dc)):
                missing.append(f"  {dc}: missing {cal_type} {level} master")
    if missing:
        sys.exit(
            "error: required masters missing (aborting before science)\n"
            + "\n".join(missing)
        )


# Live child recipe processes, so an interrupt can tear down the whole tree and
# not just the foreground shell. Each child is its own session/process-group
# leader (start_new_session=True), and this orchestrator is its sole owner: a
# Ctrl+C (SIGINT) or `kill` (SIGTERM handled below) of the orchestrator kills
# every in-flight child via os.killpg. `_interrupted` also stops the pool from
# launching any not-yet-started job once teardown has begun.
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
    (seconds), used for the serial canary, bounds the run: on expiry the child's
    process group is killed and (124, note) returned -- 124 being the
    conventional timeout exit status. Returns (130, "") without launching once
    teardown has begun (`_interrupted`).
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
        try:
            _out, stderr = proc.communicate(timeout=timeout)
            return proc.returncode, stderr
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            _out, stderr = proc.communicate()  # reap the killed group
            return 124, f"timed out after {timeout}s; child process group killed"
        except BaseException:
            # A KeyboardInterrupt (SIGINT/SIGTERM in the main thread, i.e. the
            # serial canary) unwinds here before the finally removes proc from
            # _live_procs -- kill the child now so it can't outlive us, since it
            # is in its own session and would otherwise be untracked.
            _kill_group(proc)
            proc.communicate()  # best-effort reap
            raise
    finally:
        with _live_lock:
            _live_procs.discard(proc)


def run_stage(label, tasks, jobs, log_dir, canary_timeout=1800):
    """Run `tasks`: the first serially as a canary, then the rest in a pool.

    tasks: list of (tag, argv). The first unit runs alone (bounded by
    `canary_timeout` seconds) before any fan-out. This warms the shared on-disk
    caches every job lazily builds -- barycorrpy leap-second / astropy IERS
    tables, the matplotlib font cache, compiled bytecode -- so the parallel jobs
    hit a warm cache instead of stampeding a cold one (the thundering-herd
    failure mode). It doubles as a fail-fast check: if the canary fails, abort
    before committing to the fan-out.

    In the parallel phase, on the first nonzero exit stop launching new jobs
    (cancel not-yet-started ones) but let in-flight jobs finish. Any failure
    prints a sentinel per failing unit -- naming its tag and log file -- and
    exits(1).

    On Ctrl+C (SIGINT) or SIGTERM, cancel the queue and terminate every
    still-running child before exiting (130), so an interrupt stops the whole
    run rather than leaving the pool draining orphaned subprocesses.
    """
    if not tasks:
        return
    print(
        f"[{label}] dispatching {len(tasks)} job(s): 1 serial canary, "
        f"then up to {jobs} at once"
    )
    failures = []
    pool = None
    try:
        # Canary: run the first unit alone to warm shared caches and surface
        # config/env failures before fanning out. Fan out only if it passes.
        canary_tag, canary_argv = tasks[0]
        print(f"  [{label}] canary {canary_tag} (serial; warms shared caches)...")
        rc, stderr = _run_one(canary_argv, timeout=canary_timeout)
        if rc != 0:
            failures.append((label, canary_tag, rc, stderr))
            print(
                f"  [{label}] FAILED: canary {canary_tag} (exit {rc}); not fanning out"
            )
        else:
            print(f"  [{label}] ok: {canary_tag}")
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=jobs)
            futures = {pool.submit(_run_one, argv): tag for tag, argv in tasks[1:]}
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
                print(
                    f"  [{label}] FAILED: {tag} (exit {rc}); halting new {label} jobs"
                )
                for pending in futures:
                    pending.cancel()
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
        _report_failures(failures, log_dir)
        sys.exit(1)


def _report_failures(failures, log_dir):
    """Print an actionable sentinel for each failed subprocess, to stderr."""
    print(f"\n{'=' * 72}", file=sys.stderr)
    print(f"ABORTED: {len(failures)} job(s) failed", file=sys.stderr)
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


def _read_l4_rv(obs_ids, science_output):
    """Read (BJD_TDB, RV, RVERR, datecode) from each frame's L4 product.

    RV and RVERR are km/s, BJDTDB is a Julian day -- all on the L4 PRIMARY header
    (per the EPRV standard). Frames with a non-finite RV/RVERR are skipped with a
    warning (a real reduction should not produce them, but we do not want one bad
    point to blow up the plot).
    """
    times, rvs, errs, nights = [], [], [], []
    for oid in obs_ids:
        hdr = fits.getheader(build_filepath(oid, "L4", data_root=science_output), 0)
        vals = (hdr.get("BJDTDB"), hdr.get("RV"), hdr.get("RVERR"))
        if any(v is None for v in vals) or not np.all(np.isfinite(vals)):
            print(f"  warning: {oid} has no finite RV/RVERR/BJDTDB; skipping")
            continue
        bjd, rv, err = vals
        times.append(bjd)
        rvs.append(rv)
        errs.append(err)
        nights.append(get_datecode(oid))
    return np.array(times), np.array(rvs), np.array(errs), np.array(nights)


def _group_by_night(times, rvs, errs, nights):
    """Collapse each night's frames to one RVERR-weighted mean point.

    Weights are 1/RVERR**2: the combined RV is the weighted mean, its error is
    1/sqrt(sum w), and the epoch is the same weighted mean of BJD_TDB.
    """
    g_times, g_rvs, g_errs = [], [], []
    for night in sorted(set(nights)):
        sel = nights == night
        w = 1.0 / errs[sel] ** 2
        g_times.append(np.sum(w * times[sel]) / np.sum(w))
        g_rvs.append(np.sum(w * rvs[sel]) / np.sum(w))
        g_errs.append(1.0 / np.sqrt(np.sum(w)))
    return np.array(g_times), np.array(g_rvs), np.array(g_errs)


def plot_rv_timeseries(target, obs_ids, science_output, plot_directory, group):
    """Write the RV-vs-BJD_TDB timeseries plot (with RVERR error bars)."""
    import matplotlib

    matplotlib.use("Agg")  # headless: never needs a display (e.g. on the server)
    import matplotlib.pyplot as plt

    times, rvs, errs, nights = _read_l4_rv(obs_ids, science_output)
    if times.size == 0:
        sys.exit("error: no finite RV points to plot")

    title = f"{target} RV timeseries"
    suffix = ""
    if group:
        times, rvs, errs = _group_by_night(times, rvs, errs, nights)
        title += " (nightly weighted mean)"
        suffix = "_nightly"

    order = np.argsort(times)
    os.makedirs(plot_directory, exist_ok=True)
    out_path = os.path.join(plot_directory, f"{target}_rv_timeseries{suffix}.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(times[order], rvs[order], yerr=errs[order], fmt="o", capsize=3)
    ax.set_xlabel("BJD_TDB [day]")
    ax.set_ylabel("RV [km/s]")
    ax.set_title(f"{title} -- {times.size} point(s)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"RV timeseries plot -> {out_path}")


def main(argv=None):
    # Route SIGTERM through the same KeyboardInterrupt path as Ctrl+C so both a
    # terminal interrupt and an explicit `kill` tear down the child processes
    # (see run_stage). Installed here, not at import, so importing this module
    # (e.g. from tests) does not mutate signal handlers.
    signal.signal(signal.SIGTERM, _handle_termination_signal)

    # Pin each recipe subprocess to single-threaded BLAS/OpenMP. We already run
    # one process per CPU (`--jobs`), and numpy/scipy's OpenBLAS backend would
    # otherwise spawn a full thread pool *inside each* -- jobs x cores threads,
    # which thrashes and presents as a hang. setdefault so an explicit caller
    # value wins; set here (after this module's numpy import) so it changes only
    # the inherited env of the children, not our already-initialised numpy.
    for _var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(_var, "1")

    args = parse_args(argv)
    start, end = args.date_range

    # Resolve effective dirs: input/logs from the science config, masters-output
    # from the masters config (where masters are written), each with any CLI
    # override applied. Overrides are also forwarded verbatim to every subprocess.
    data_input = (
        args.kpf_data_input
        or _dir_params(args.science_config, "DATA_DIRS")["KPF_DATA_INPUT"]
    )
    masters_output = (
        args.kpf_masters_output
        or _dir_params(args.masters_config, "DATA_DIRS")["KPF_MASTERS_OUTPUT"]
    )
    log_dir = args.log_directory or _dir_params(args.science_config, "LOGGER").get(
        "log_directory"
    )

    forward = []
    if args.kpf_data_input:
        forward += ["--data_input", args.kpf_data_input]
    if args.kpf_masters_output:
        forward += ["--data_masters", args.kpf_masters_output]
    if args.kpf_science_output:
        forward += ["--data_science", args.kpf_science_output]
    if args.log_directory:
        forward += ["--log_dir", args.log_directory]

    # Steps 1-2: science frames -> unique nights.
    obs_ids = discover_science_obs_ids(
        data_input, args.target, start, end, args.file_limit
    )
    datecodes = sorted({get_datecode(o) for o in obs_ids})
    print(
        f"target {args.target}: {len(obs_ids)} science frame(s) across "
        f"{len(datecodes)} night(s) [{start}..{end}]"
    )

    # Step 3: nightly masters (parallel across nights), or reuse existing.
    if args.reprocess_masters:
        tasks = [
            (
                dc,
                [
                    sys.executable,
                    "-m",
                    "tools.cli",
                    "-r",
                    _MASTERS_RECIPE,
                    "-c",
                    args.masters_config,
                    "-d",
                    dc,
                    *forward,
                ],
            )
            for dc in datecodes
        ]
        run_stage("masters", tasks, args.jobs, log_dir)
    else:
        print("reusing existing masters on disk (--reprocess_masters not set)")

    # Gate: never start science if any required master is missing.
    check_masters_exist(masters_output, datecodes)
    print("all required masters present")

    # Step 4: science reduction (parallel across frames).
    tasks = [
        (
            oid,
            [
                sys.executable,
                "-m",
                "tools.cli",
                "-r",
                _SCIENCE_RECIPE,
                "-c",
                args.science_config,
                "-o",
                oid,
                *forward,
            ],
        )
        for oid in obs_ids
    ]
    # run_stage runs the first frame serially as a canary before fanning out --
    # warming barycorrpy/IERS/matplotlib caches so the parallel jobs don't
    # stampede a cold cache (and failing fast if the first frame is broken).
    run_stage("science", tasks, args.jobs, log_dir)

    science_output = (
        args.kpf_science_output
        or _dir_params(args.science_config, "DATA_DIRS")["KPF_SCIENCE_OUTPUT"]
    )
    print(
        f"\ndone: reduced {len(obs_ids)} frame(s); "
        f"L2/L4 products under {science_output}"
    )

    # Step 5: RV timeseries plot from the freshly written L4 products -- only
    # when a plot directory was given (no --plot_directory => reduction only).
    if args.plot_directory:
        plot_rv_timeseries(
            args.target,
            obs_ids,
            science_output,
            args.plot_directory,
            args.group_intranight_obs,
        )
    else:
        print("no --plot_directory given; skipping RV timeseries plot")


if __name__ == "__main__":
    main()
