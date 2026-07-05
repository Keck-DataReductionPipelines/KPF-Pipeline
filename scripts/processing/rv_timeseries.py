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
        "--skip_existing_masters",
        action="store_true",
        help="skip building a night's masters when all of its required masters "
        "already exist on disk (default: rebuild every night)",
    )
    ap.add_argument(
        "--skip_existing_science",
        action="store_true",
        help="skip reducing a frame that already has an L4 product on disk "
        "(default: reduce every frame)",
    )
    ap.add_argument(
        "--input_dir",
        help="shorthand for --kpf_data_input (the L0 input root)",
    )
    ap.add_argument(
        "--output_dir",
        help="shorthand routing all outputs under one root: sets "
        "--kpf_masters_output and --kpf_science_output to it, "
        "--log_directory to {output_dir}/logs, and --plot_directory to "
        "{output_dir}/QLP/timeseries",
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
        "--group_bursts",
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

    # --input_dir / --output_dir are shorthands that populate the explicit
    # dir overrides below; reject giving both a shorthand and its long form.
    if args.input_dir:
        if args.kpf_data_input:
            ap.error("give either --input_dir or --kpf_data_input, not both")
        args.kpf_data_input = args.input_dir
    if args.output_dir:
        routed = {
            "kpf_masters_output": args.output_dir,
            "kpf_science_output": args.output_dir,
            "log_directory": os.path.join(args.output_dir, "logs"),
            "plot_directory": os.path.join(args.output_dir, "QLP", "timeseries"),
        }
        clashes = [f"--{k}" for k in routed if getattr(args, k)]
        if clashes:
            ap.error(f"--output_dir conflicts with {', '.join(clashes)}")
        for k, v in routed.items():
            setattr(args, k, v)
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


def _missing_masters(masters_output, datecode):
    """Required (cal_type, level) masters absent for `datecode` (empty == complete)."""
    return [
        (cal_type, level)
        for cal_type, level in _REQUIRED_MASTERS
        if not glob.glob(glob_masters(masters_output, cal_type, level, datecode))
    ]


def check_masters_exist(masters_output, datecodes):
    """Abort unless every required master exists for every night in `datecodes`."""
    missing = [
        f"  {dc}: missing {cal_type} {level} master"
        for dc in datecodes
        for cal_type, level in _missing_masters(masters_output, dc)
    ]
    if missing:
        sys.exit(
            "error: required masters missing (aborting before science)\n"
            + "\n".join(missing)
        )


def _science_complete(obs_id, science_output):
    """True if the frame's L4 product already exists (reduced to completion)."""
    return os.path.exists(build_filepath(obs_id, "L4", data_root=science_output))


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
            # KeyboardInterrupt during the serial canary unwinds here before the
            # finally untracks proc -- kill the child now so it can't outlive us.
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


def _cli_task(unit, recipe, config, unit_flag, forward):
    """Build a (tag, argv) task running one recipe `unit` via `python -m tools.cli`.

    `unit` is the datecode (masters) or obs_id (science); `unit_flag` its CLI flag
    (`-d`/`-o`); `forward` the resolved dir/log overrides appended to every
    invocation. The tag is the unit itself, which the sentinel/log paths key on.
    """
    argv = [sys.executable, "-m", "tools.cli", "-r", recipe, "-c", config]
    argv += [unit_flag, unit, *forward]
    return unit, argv


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


def plot_rv_timeseries(target, obs_ids, science_output, plot_directory, group_bursts):
    """Write the RV timeseries plot.

    Plots delta-RV (m/s, relative to the median RV) vs. observation date, with
    RVERR error bars, a zero reference line (the median), an RMS annotation, and
    calendar-date (YYYYMMDD) x tick labels derived from BJD_TDB.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: never needs a display (e.g. on the server)
    import matplotlib.pyplot as plt
    from astropy.time import Time
    from matplotlib.ticker import FuncFormatter

    times, rvs, errs, nights = _read_l4_rv(obs_ids, science_output)
    if times.size == 0:
        sys.exit("error: no finite RV points to plot")

    title = f"{target} RV timeseries"
    suffix = ""
    if group_bursts:
        times, rvs, errs = _group_by_night(times, rvs, errs, nights)
        title += " (nightly weighted mean)"
        suffix = "_nightly"

    # Delta-RV about the median, in m/s (RV/RVERR are stored in km/s per EPRV).
    drv = (rvs - np.median(rvs)) * 1e3
    derr = errs * 1e3
    rms = np.sqrt(np.mean(drv**2))

    order = np.argsort(times)
    os.makedirs(plot_directory, exist_ok=True)
    out_path = os.path.join(plot_directory, f"{target}_rv_timeseries{suffix}.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(0.0, color="0.6", lw=1, zorder=0)  # zero = median RV, guides the eye
    ax.errorbar(times[order], drv[order], yerr=derr[order], fmt="o", capsize=3)

    # Relabel the BJD_TDB axis with human-readable calendar dates (YYYYMMDD); the
    # TDB-vs-UTC offset (~seconds) is irrelevant at day granularity.
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda jd, _p: Time(jd, format="jd", scale="tdb").strftime("%Y%m%d")
        )
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_xlabel("Date [UT]")
    ax.set_ylabel(r"$\Delta$RV [m/s]")
    ax.set_title(f"{title} -- {times.size} point(s)")
    ax.annotate(
        f"RMS = {rms:.2f} m/s",
        xy=(0.02, 0.96),
        xycoords="axes fraction",
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8),
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"RV timeseries plot -> {out_path}  (RMS {rms:.2f} m/s)")


def main(argv=None):
    # Route SIGTERM through the KeyboardInterrupt path so `kill` tears down
    # children like Ctrl+C (see run_stage). Here, not at import, so importing
    # this module doesn't mutate signal handlers.
    signal.signal(signal.SIGTERM, _handle_termination_signal)

    # Pin each recipe subprocess to single-threaded BLAS/OpenMP: we already run
    # one process per CPU (`--jobs`), so an OpenBLAS pool inside each would be
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
    start, end = args.date_range

    # Resolve effective dirs: input/science-output/logs from the science config,
    # masters-output from the masters config (where masters are written), each
    # with any CLI override applied. Overrides are also forwarded verbatim to
    # every subprocess.
    science_dirs = _dir_params(args.science_config, "DATA_DIRS")
    data_input = args.kpf_data_input or science_dirs["KPF_DATA_INPUT"]
    science_output = args.kpf_science_output or science_dirs["KPF_SCIENCE_OUTPUT"]
    masters_output = (
        args.kpf_masters_output
        or _dir_params(args.masters_config, "DATA_DIRS")["KPF_MASTERS_OUTPUT"]
    )
    log_dir = args.log_directory or _dir_params(args.science_config, "LOGGER").get(
        "log_directory"
    )

    forward = []
    for value, flag in (
        (args.kpf_data_input, "--data_input"),
        (args.kpf_masters_output, "--data_masters"),
        (args.kpf_science_output, "--data_science"),
        (args.log_directory, "--log_dir"),
    ):
        if value:
            forward += [flag, value]

    # Steps 1-2: science frames -> unique nights.
    obs_ids = discover_science_obs_ids(
        data_input, args.target, start, end, args.file_limit
    )
    datecodes = sorted({get_datecode(o) for o in obs_ids})
    print(
        f"target {args.target}: {len(obs_ids)} science frame(s) across "
        f"{len(datecodes)} night(s) [{start}..{end}]"
    )

    # Step 3: nightly masters (parallel across nights). Build every night by
    # default; with --skip_existing_masters, skip a night only when all of its
    # required masters are already on disk.
    masters_todo = datecodes
    if args.skip_existing_masters:
        masters_todo = [dc for dc in datecodes if _missing_masters(masters_output, dc)]
        skipped = len(datecodes) - len(masters_todo)
        if skipped:
            print(f"skipping masters for {skipped} night(s) already complete on disk")
    tasks = [
        _cli_task(dc, _MASTERS_RECIPE, args.masters_config, "-d", forward)
        for dc in masters_todo
    ]
    run_stage("masters", tasks, args.jobs, log_dir)

    # Gate: never start science if any required master is missing.
    check_masters_exist(masters_output, datecodes)
    print("all required masters present")

    # Step 4: science reduction (parallel across frames; canary-then-fan-out).
    # Reduce every frame by default; with --skip_existing_science, skip frames
    # that already have an L4 product (reduced to completion).
    science_todo = obs_ids
    if args.skip_existing_science:
        science_todo = [o for o in obs_ids if not _science_complete(o, science_output)]
        skipped = len(obs_ids) - len(science_todo)
        if skipped:
            print(f"skipping {skipped} science frame(s) already reduced to L4")
    tasks = [
        _cli_task(oid, _SCIENCE_RECIPE, args.science_config, "-o", forward)
        for oid in science_todo
    ]
    run_stage("science", tasks, args.jobs, log_dir)

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
            args.group_bursts,
        )
    else:
        print("no --plot_directory given; skipping RV timeseries plot")


if __name__ == "__main__":
    main()
