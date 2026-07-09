#!/usr/bin/env python3
"""Plot a single star's RV timeseries from its L4 products on disk.

A lean, standalone post-reduction plotter. The frames come from one of two
sources: ``--date_range`` combs the L4 output tree for a target's products over an
inclusive datecode range, or ``--obs_ids`` names them explicitly (their L4 paths
are built directly from ``--data_dir``, skipping the scan -- how ``kpfpipe
timeseries`` hands off the set it already discovered). Either way it reads the RV
summary keywords from each L4 PRIMARY header and renders the RV-vs-date plot. It
runs *after* the reduction has written the L4 files; it reads headers only and
never touches the pipeline.

Bursts of rapid-succession frames are always collapsed to one RVERR-weighted
point (revisits stay distinct): the individual frames are drawn as a faint grey
underlay and the burst means overplotted on top, and the RV_RMS/RV_ERR annotation
and outlier flagging reflect the burst means. A per-night panel plot is also
written, but only for nights with more than one observation (a single-frame night
carries no within-night trend, so paneling it is meaningless).

Two PNGs land in ``--plot_dir``: ``{target}_rv_timeseries.png`` always, and
``{target}_rv_nightly.png`` when at least one night has multiple observations.

    python -m scripts.plots.plot_timeseries --target 10700 \\
        --date_range 20240101 20240131 \\
        --data_dir /data/kpf/science --plot_dir ./plots

    python -m scripts.plots.plot_timeseries --target 10700 \\
        --obs_ids KP.20240101.03600.00 KP.20240101.03660.00 \\
        --data_dir /data/kpf/science --plot_dir ./plots
"""

import argparse
import concurrent.futures
import glob
import os
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime

import numpy as np
from astropy.io import fits

import kpfpipe
from kpfpipe.utils.io import datecode_dirs_in_range, kpf_filepath
from kpfpipe.utils.kpf_utils import is_datecode, is_obs_id
from kpfpipe.utils.stats import flag_outliers

_REPO = kpfpipe.REPO_ROOT


def _scan_nights(nights, target, worker, jobs):
    """Fan `worker(dc)` out over `nights` in a thread pool; return the pooled hits.

    Scanning a night reads every L4 PRIMARY header -- slow over NFS but I/O bound,
    so a thread pool overlaps the latency (``getheader`` releases the GIL during
    I/O). `worker(dc)` returns ``(hits, note)``: that night's L4 paths and an
    optional trailing note. A per-night heartbeat is printed in completion order
    (the scan is otherwise silent).
    """
    print(
        f"scanning {len(nights)} night(s) for target {target} "
        f"({min(jobs, len(nights))} workers)...",
        flush=True,
    )
    hits_all = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(worker, dc): dc for dc in nights}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            dc = futures[future]
            hits, note = future.result()
            hits_all.extend(hits)
            print(
                f"  [{i}/{len(nights)}] {dc}: {len(hits)} L4 frame(s){note} "
                f"(total {len(hits_all)})",
                flush=True,
            )
    return hits_all


def discover_l4_files(data_dir, target, start, end, jobs):
    """L4 product paths for `target` over [start, end], scanned from the L4 tree.

    Enumerates the nights (datecode dirs) under {data_dir}/L4, reads each L4
    PRIMARY OBJECT, and keeps this target's frames. Returns a sorted list of L4
    file paths. A frame whose L4 was deleted simply doesn't appear -- the plotter
    reads what is on disk and never depends on the L0 input tree. Exits loudly
    when the L4 tree is missing or nothing matches.
    """
    l4_root = os.path.join(data_dir, "L4")
    if not os.path.isdir(l4_root):
        sys.exit(f"error: L4 output directory not found: {l4_root}")

    nights = datecode_dirs_in_range(l4_root, start, end)
    if not nights:
        sys.exit(f"error: no L4 datecode dirs under {l4_root} in range {start}..{end}")

    def _scan_night(dc):
        hits = []
        for path in sorted(glob.glob(os.path.join(l4_root, dc, "kpf_SL4_*.fits"))):
            try:
                obj = fits.getheader(path, 0).get("OBJECT")
            except OSError as e:
                print(f"  warning: skipping unreadable L4 {path}: {e}", flush=True)
                continue
            if str(obj).strip() == str(target):
                hits.append(path)
        return hits, ""

    l4_paths = _scan_nights(nights, target, _scan_night, jobs)

    l4_paths = sorted(set(l4_paths))
    if not l4_paths:
        sys.exit(
            f"error: no L4 products for target {target!r} in range "
            f"{start}..{end} under {l4_root}"
        )
    return l4_paths


def l4_paths_for_obs_ids(data_dir, obs_ids, target):
    """L4 product paths for an explicit obs_id list, built without a directory scan.

    The counterpart to discover_l4_files for when the caller already knows the
    frames (e.g. timeseries, which discovered and reduced them): each obs_id's L4
    path is built directly with kpf_filepath(obs_id, 'L4', data_root=data_dir), so
    no L4 tree is walked. Every supplied obs_id whose L4 is present is plotted --
    including one whose L4 OBJECT does not match `target` (warned, not dropped, so
    a header/target mismatch is surfaced without silently discarding data). An
    obs_id whose L4 is absent or unreadable (e.g. its reduction failed) is warned
    and skipped. Exits loudly only when none of the supplied obs_ids have an L4.
    """
    paths = []
    for oid in obs_ids:
        path = kpf_filepath(oid, "L4", data_root=data_dir)
        if not os.path.isfile(path):
            print(f"  warning: no L4 product for {oid} at {path}; skipping", flush=True)
            continue
        try:
            obj = fits.getheader(path, 0).get("OBJECT")
        except OSError as e:
            print(f"  warning: skipping unreadable L4 {path}: {e}", flush=True)
            continue
        if str(obj).strip() != str(target):
            print(
                f"  warning: L4 for {oid} has OBJECT {obj!r}, not target "
                f"{target!r}; plotting anyway",
                flush=True,
            )
        paths.append(path)

    paths = sorted(set(paths))
    if not paths:
        sys.exit(
            f"error: none of the {len(obs_ids)} supplied obs_id(s) have an L4 "
            f"product under {data_dir}"
        )
    return paths


def _read_l4_rv(l4_paths):
    """Read (BJD_TDB, RV, RVERR, datecode) from each L4 product.

    RV and RVERR are km/s, BJDTDB is a Julian day -- all on the L4 PRIMARY header
    (per the EPRV standard). The datecode labels each frame's observing night for
    the per-night panels; it is the L4 file's parent directory
    ({data_dir}/L4/{datecode}). Frames with a non-finite RV/RVERR are skipped with
    a warning (a real reduction should not produce them, but we do not want one
    bad point to blow up the plot).
    """
    times, rvs, errs, nights = [], [], [], []
    for path in l4_paths:
        hdr = fits.getheader(path, 0)
        vals = (hdr.get("BJDTDB"), hdr.get("RV"), hdr.get("RVERR"))
        if any(v is None for v in vals) or not np.all(np.isfinite(vals)):
            name = os.path.basename(path)
            print(f"  warning: {name} has no finite RV/RVERR/BJDTDB; skipping")
            continue
        bjd, rv, err = vals
        times.append(bjd)
        rvs.append(rv)
        errs.append(err)
        nights.append(os.path.basename(os.path.dirname(path)))
    return np.array(times), np.array(rvs), np.array(errs), np.array(nights)


# Minimum gap separating one burst from the next. A bright-star burst is ~3-5
# exposures at ~1-min readout cadence; revisits are tens of minutes apart, so
# 15 min cleanly sits between the two (well above intra-burst, below inter-burst).
_BURST_GAP_MINUTES = 15.0


def _group_bursts(times, rvs, errs, gap_minutes=_BURST_GAP_MINUTES):
    """Collapse each burst of rapid-succession frames to one RVERR-weighted point.

    Splits the time-ordered frames wherever consecutive BJD_TDB values differ by
    more than `gap_minutes` (as `build_calibration_stacks` clusters calibrations), then
    combines each burst with 1/RVERR**2 weights: weighted-mean RV, error
    1/sqrt(sum w), epoch the weighted-mean BJD_TDB.
    """
    order = np.argsort(times)
    times, rvs, errs = times[order], rvs[order], errs[order]
    gaps_minutes = np.diff(times) * 1440.0  # BJD_TDB days -> minutes
    breaks = np.nonzero(gaps_minutes > gap_minutes)[0] + 1
    g_times, g_rvs, g_errs = [], [], []
    for sel in np.split(np.arange(times.size), breaks):
        w = 1.0 / errs[sel] ** 2
        g_times.append(np.sum(w * times[sel]) / np.sum(w))
        g_rvs.append(np.sum(w * rvs[sel]) / np.sum(w))
        g_errs.append(1.0 / np.sqrt(np.sum(w)))
    return np.array(g_times), np.array(g_rvs), np.array(g_errs)


def _stamp_provenance(fig):
    """Footer with UT generation time + short git commit (quicklook-style)."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    try:
        commit = (
            subprocess.run(
                ["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            or "unknown"
        )
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    fig.text(
        0.99,
        0.005,
        f"generated {now} UT · {commit}",
        fontsize=8,
        color="darkgray",
        ha="right",
        va="bottom",
    )


def _symmetric_ylim(ax):
    """Set the y-limits symmetric about 0 (equal span above and below)."""
    ymax = max(abs(v) for v in ax.get_ylim())
    ax.set_ylim(-ymax, ymax)


def plot_nightly_panels(target, times, rvs, errs, nights, plot_directory):
    """Write a per-night multi-panel plot of the individual (ungrouped) frames.

    One panel per observing night that has more than one observation -- a
    single-frame night carries no within-night trend, so paneling it is
    meaningless and it is skipped. Each panel shows delta-RV (m/s, about the
    overall median) vs. minutes from that night's first frame, so within-night
    trends are visible; panels share a y-axis to keep nights comparable. When no
    night has multiple observations, nothing is written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Only nights with more than one observation are worth a panel.
    counts = Counter(nights)
    unique = sorted(night for night, n in counts.items() if n > 1)
    if not unique:
        print("no multi-observation nights; skipping nightly panels")
        return

    drv = (rvs - np.median(rvs)) * 1e3
    derr = errs * 1e3

    ncols = min(4, len(unique))
    nrows = int(np.ceil(len(unique) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.3 * ncols, 2.7 * nrows), sharey=True, squeeze=False
    )
    for ax, night in zip(axes.flat, unique, strict=False):
        sel = nights == night
        minutes = (times[sel] - times[sel].min()) * 1440.0
        o = np.argsort(minutes)
        ax.axhline(0.0, color="0.6", lw=1, zorder=0)
        ax.errorbar(
            minutes[o],
            drv[sel][o],
            yerr=derr[sel][o],
            fmt="o",
            ms=4,
            capsize=2,
            color="C3",
        )
        ax.set_title(night, fontsize=9)
        ax.grid(True, alpha=0.3)
    for ax in axes.flat[len(unique) :]:
        ax.set_visible(False)
    _symmetric_ylim(axes.flat[0])  # sharey: one range applies to every panel

    fig.supxlabel("Minutes from first frame of night")
    fig.supylabel(r"$\Delta$RV [m/s]")
    fig.suptitle(target)
    fig.tight_layout()
    _stamp_provenance(fig)
    os.makedirs(plot_directory, exist_ok=True)
    out_path = os.path.join(plot_directory, f"{target}_rv_nightly.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"nightly panels plot -> {out_path}  ({len(unique)} night(s))")


def plot_rv_timeseries(target, l4_paths, plot_directory):
    """Write the RV timeseries plot, with bursts always grouped.

    Plots delta-RV (m/s, relative to the median RV) vs. observation date, with
    RVERR error bars, a zero reference line (the median), an RMS annotation, and
    calendar-date (YYYYMMDD) x tick labels derived from BJD_TDB.

    The individual frames are drawn as a faint grey underlay and the burst-grouped
    means overplotted in colour on top; the RV_RMS/RV_ERR annotation and the
    outlier flagging reflect the burst means. It also writes the per-night panel
    plot from the ungrouped frames, for nights with multiple observations only
    (see plot_nightly_panels).

    Outliers are flagged robustly (only for series of >=10 burst means): each
    point is compared to the local trend of the time-ordered series and >5-sigma
    residuals are dropped (flag_outliers, trend method). RV_RMS is then the std of
    the retained points (an outlier can't inflate it), the y-range is clipped to
    the retained points, and each outlier is drawn as a black triangle at the
    clipped edge (up for positive, down for negative) annotated with its delta-RV,
    so a large outlier is flagged without rescaling the plot.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: never needs a display (e.g. on the server)
    import matplotlib.pyplot as plt
    from astropy.time import Time
    from matplotlib.ticker import FuncFormatter

    times, rvs, errs, nights = _read_l4_rv(l4_paths)
    if times.size == 0:
        sys.exit("error: no finite RV points to plot")

    # Delta-RV about the median of the individual frames, in m/s (RV/RVERR are
    # stored in km/s per EPRV). One reference for both layers keeps them aligned.
    ref = np.median(rvs)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(0.0, color="0.6", lw=1, zorder=0)  # zero = median RV, guides the eye

    # Per-night panels from the individual (ungrouped) frames; self-skips when no
    # night has multiple observations.
    plot_nightly_panels(target, times, rvs, errs, nights, plot_directory)

    # Faint grey underlay: the individual (ungrouped) frames for context.
    g_order = np.argsort(times)
    ax.errorbar(
        times[g_order],
        ((rvs - ref) * 1e3)[g_order],
        yerr=(errs * 1e3)[g_order],
        fmt="o",
        ms=4,
        color="0.6",
        alpha=0.5,
        zorder=1,
        label="individual frames",
    )
    times, rvs, errs = _group_bursts(times, rvs, errs)

    drv = (rvs - ref) * 1e3
    derr = errs * 1e3
    med_err = np.median(derr)  # median per-point photon uncertainty

    order = np.argsort(times)
    os.makedirs(plot_directory, exist_ok=True)
    out_path = os.path.join(plot_directory, f"{target}_rv_timeseries.png")

    # Robust outlier flagging: compare each burst-grouped point to the local trend
    # of the time-ordered series and flag >5-sigma residuals. Gated to >=10 points
    # -- below that the trend is meaningless, so nothing is flagged. RV_RMS is the
    # std of the retained points, so an outlier can't inflate it.
    outlier = np.zeros(drv.shape, dtype=bool)
    if drv.size >= 10:
        outlier[order] = flag_outliers(drv[order], 5.0, kernel_size=5, method="trend")
    rms = float(np.std(drv[~outlier])) if np.any(~outlier) else float(np.std(drv))

    # Clip the y-range to the retained points so a flagged outlier doesn't
    # compress the plot; the outliers are drawn as triangles at the edge instead.
    ylim = None
    if outlier.any():
        span = np.abs(drv[~outlier])
        ref_max = float(span.max()) if span.size else float(np.abs(drv).max())
        if ref_max > 0:
            ylim = 1.1 * ref_max

    # Foreground series (the burst means): larger, black-outlined markers so they
    # stand out over the grey underlay. Draw only the in-range points; flagged
    # outliers are marked separately at the edge.
    fg_kw = dict(
        fmt="o",
        capsize=3,
        zorder=2,
        color="C3",
        ms=8,
        mec="black",
        mew=0.8,
        label="burst mean",
    )
    keep = order[~outlier[order]] if ylim is not None else order
    ax.errorbar(times[keep], drv[keep], yerr=derr[keep], **fg_kw)

    # Off-plot outliers: a black up/down triangle at the clipped edge (clip_on
    # False so the whole marker shows), annotated with the delta-RV.
    if ylim is not None:
        up = outlier & (drv > 0)
        dn = outlier & (drv < 0)
        tri_kw = dict(ls="none", color="black", ms=9, clip_on=False, zorder=4)
        label = r"5$\sigma$ outlier"
        if up.any():
            ax.plot(
                times[up],
                np.clip(drv[up], -ylim, ylim),
                marker="^",
                label=label,
                **tri_kw,
            )
            label = None
        if dn.any():
            ax.plot(
                times[dn],
                np.clip(drv[dn], -ylim, ylim),
                marker="v",
                label=label,
                **tri_kw,
            )
        for i in np.flatnonzero(outlier):
            y = float(np.clip(drv[i], -ylim, ylim))
            off = -12 if drv[i] > 0 else 12
            ax.annotate(
                f"{drv[i]:+.0f} m/s",
                (times[i], y),
                textcoords="offset points",
                xytext=(0, off),
                ha="center",
                va="top" if drv[i] > 0 else "bottom",
                fontsize=7,
            )

    ax.legend(loc="upper right", fontsize=8)

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
    ax.set_title(target)
    ax.annotate(
        f"RV_RMS = {rms:.2f} m/s\nRV_ERR = {med_err:.2f} m/s",
        xy=(0.02, 0.96),
        xycoords="axes fraction",
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8),
    )
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(-ylim, ylim)
    else:
        _symmetric_ylim(ax)
    fig.tight_layout()
    _stamp_provenance(fig)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(
        f"RV timeseries plot -> {out_path}  "
        f"(RV_RMS {rms:.2f} m/s, RV_ERR {med_err:.2f} m/s)"
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="plot_timeseries",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target",
        required=True,
        help="star id as it appears in the L4 OBJECT header, e.g. 10700",
    )
    # Exactly one frame source: scan a datecode range, or an explicit obs_id list
    # (the latter lets a caller that already knows the frames skip the L4 scan).
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--date_range",
        nargs=2,
        metavar=("START", "END"),
        help="inclusive datecode range to scan the L4 tree over, e.g. "
        "--date_range 20240101 20240131 (mutually exclusive with --obs_ids)",
    )
    source.add_argument(
        "--obs_ids",
        nargs="+",
        metavar="OBS_ID",
        help="explicit obs_ids to plot; their L4 paths are built from --data_dir "
        "(no scan), e.g. --obs_ids KP.20240405.40113.57 (exclusive with --date_range)",
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="science output root; L4 products are read from {data_dir}/L4/{datecode}",
    )
    parser.add_argument(
        "--plot_dir",
        required=True,
        help="directory the PNG plots are written to (created if absent)",
    )
    args = parser.parse_args(argv)

    if args.date_range:
        start, end = args.date_range
        for dc in (start, end):
            if not is_datecode(dc):
                parser.error(f"--date_range value is not a valid datecode: {dc!r}")
        if start > end:
            parser.error(f"--date_range START must be <= END (got {start} > {end})")
    else:
        for oid in args.obs_ids:
            if not is_obs_id(oid):
                parser.error(f"--obs_ids value is not a valid obs_id: {oid!r}")
    return args


def main(argv=None):
    args = parse_args(argv)

    if args.obs_ids:
        # Frames known ahead of time: build L4 paths directly, no directory scan.
        l4_paths = l4_paths_for_obs_ids(args.data_dir, args.obs_ids, args.target)
    else:
        start, end = args.date_range
        # The discovery scan reads one PRIMARY header per L4 file; a small thread
        # pool overlaps the (NFS) latency without needing a user-facing --jobs knob.
        jobs = min(8, os.cpu_count() or 1)
        l4_paths = discover_l4_files(args.data_dir, args.target, start, end, jobs)

    plot_rv_timeseries(args.target, l4_paths, args.plot_dir)


if __name__ == "__main__":
    main()
