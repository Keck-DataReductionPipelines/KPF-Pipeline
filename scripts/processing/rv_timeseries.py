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
import subprocess
import sys

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


def _run_one(argv):
    """Run one recipe subprocess; return (returncode, captured stderr)."""
    proc = subprocess.run(argv, cwd=_REPO, capture_output=True, text=True)
    return proc.returncode, proc.stderr


def run_stage(label, tasks, jobs, log_dir):
    """Run `tasks` in a bounded pool; abort loudly if any fails.

    tasks: list of (tag, argv). On the first nonzero exit, stop launching new
    jobs (cancel not-yet-started ones) but let in-flight jobs finish, then print
    a sentinel per failure -- naming the tag and its log file -- and exit(1).
    """
    if not tasks:
        return
    print(f"[{label}] dispatching {len(tasks)} job(s), up to {jobs} at once")
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_run_one, argv): tag for tag, argv in tasks}
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
            print(f"  [{label}] FAILED: {tag} (exit {rc}); halting new {label} jobs")
            for pending in futures:
                pending.cancel()

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
