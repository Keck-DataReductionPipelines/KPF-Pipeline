#!/usr/bin/env python3
"""Measure order-trace stability within each instrument era (``kpfpipe analyze
order_trace``).

KPF is a stabilized spectrograph, so within a science era every order trace is
expected to hold to well under one pixel; offsets between eras are not measured
here. Each ``KP.{datecode}.{seconds}_master_order_trace.csv`` under
``{input_dir}/masters/{datecode}/`` is one epoch, named for the master flat it was
traced from. Epochs are grouped by science era (engineering eras are skipped), and
each trace's movement between the epochs of its era is measured in detector rows
and in fit coefficients. Steps where a chip's traces moved to a new position and
stayed there are detected as jumps -- the dates that split an era into the
``reference/order_traces/`` epochs.

Over an inclusive datecode range this writes, under ``{analysis_dir}/order_trace/``:

- ``order_trace_pixel_{START}_{END}.csv`` -- row scatter per trace;
- ``order_trace_epochs_{START}_{END}.csv`` -- offset per epoch and chip;
- ``order_trace_coefficients_{START}_{END}.csv`` -- coefficient scatter per trace;
- ``order_trace_jumps_{START}_{END}.csv`` -- the detected jumps;
- six PNG figures of the same, and a per-era summary in the log.

``--date_range`` defaults to the whole archive: first light through tomorrow.

    kpfpipe analyze order_trace --date_range 20221109 20250829 \
        --input_dir /data/kpf --output_dir /data/kpf-next
"""

import argparse
import glob
import logging
import os
import re
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

import kpfpipe
from kpfpipe import CHIPS, FIBERS
from kpfpipe.utils.io import datecode_dirs_in_range
from kpfpipe.utils.logger import setup_batch_logging
from scripts._argparse import (
    dates_parser,
    logging_parser,
    resolve_dates,
    resolve_dir_shortcuts,
)

logger = logging.getLogger(__name__)

OBS_ID_RE = re.compile(r"^KP\.(\d{8})\.")
NCOL = 4080  # detector columns; the fits are defined over 0..NCOL-1
NSAMPLE = 81  # columns sampled when comparing centerlines
MIN_EPOCHS = 3  # an era needs this many trace files to have a scatter

# Jump detection, calibrated against the jumps that are visible by eye in
# stability_over_time.png over 2022-2025: 4 in era 1.0, 1 in 2.0, 1 in 2.6, none
# in 3.0, 2 in 4.0. The counts hold for steps of 0.022-0.027 pixels and
# significances of 12-18, so the thresholds sit in the middle of a plateau, not on
# an edge.
JUMP_WINDOW = 8  # epochs compared either side of a candidate boundary
JUMP_MIN_STEP = 0.025  # pixels; a smaller level change is not marked
JUMP_MIN_SIGNIF = 15.0  # ... and it must be this many times the local scatter
JUMP_MERGE_DAYS = 3  # chips stepping this close together are one jump

_PIXEL_COLUMNS = ["Era", "Chip", "Fiber", "Order", "Epochs", "Typical", "Max"]
_EPOCH_COLUMNS = ["Era", "Epoch", "Chip", "Date", "RMS", "P95", "Max", "Shift"]
_COEFF_COLUMNS = ["Era", "Chip", "Fiber", "Order", "Coeff", "Typical", "Max"]
_JUMP_COLUMNS = ["Era", "Chip", "Date", "Epoch", "Step", "Significance"]


def default_date_range():
    """First light through tomorrow -- the whole archive.

    The end is UTC-dated a day ahead so it never falls behind the newest HST night
    on disk; a datecode dir that does not exist yet simply contributes nothing.
    """
    return "20221109", time.strftime("%Y%m%d", time.gmtime(time.time() + 86400))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="kpfpipe analyze order_trace",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[
            logging_parser(),
            dates_parser(
                "analyzes every masters night in it",
                dates=False,
                default_date_range=default_date_range(),
            ),
        ],
    )
    ap.add_argument(
        "--input_dir",
        required=True,
        help="root of the masters tree to read: {input_dir}/masters/{datecode}/",
    )
    ap.add_argument(
        "--analysis_dir",
        default=None,
        help="directory for the analysis reports (default: {output_dir}/analysis)",
    )
    ap.add_argument(
        "--output_dir",
        default=None,
        help="shortcut: root for every output directory not given its own explicit "
        "flag -- the analysis dir ({output_dir}/analysis) and the log dir "
        "({output_dir}/logs)",
    )
    args = resolve_dates(ap, ap.parse_args(argv))
    args = resolve_dir_shortcuts(args)
    if not args.analysis_dir or not args.log_dir:
        ap.error("pass --output_dir, or both --analysis_dir and --log_dir")
    return args


def _pyplot():
    """``pyplot`` with the Agg backend pinned, imported on first use."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def load_science_eras():
    """``[(era, start, end)]`` for the eras that are not engineering/service time."""
    eras = kpfpipe.INSTRUMENT_ERAS
    engineering = eras["Comments"].fillna("").str.lower().str.contains("engineering")
    return [
        (str(row.INSTERA), row.UT_start_date, row.UT_end_date)
        for row in eras[~engineering].itertuples()
    ]


def era_of(datecode, eras):
    """Eras are resolved at 12:00 UT, which falls inside the Keck night."""
    t = pd.Timestamp(datecode) + pd.Timedelta(hours=12)
    for era, start, end in eras:
        if start <= t <= end:
            return era
    return None


def load_traces(input_dir, start, end, eras):
    """Every trace row under ``{input_dir}/masters`` in [start, end], tagged with
    era and epoch.

    One CSV is one epoch, named for the master flat it was traced from, so the
    datecode comes from the filename and a night with several flats
    contributes several epochs. Exits loudly when the tree is missing or holds no
    science-era traces in range -- an empty report would read as "stable".
    """
    root = os.path.join(input_dir, "masters")
    if not os.path.isdir(root):
        sys.exit(f"error: masters directory not found: {root}")
    tables = []
    for night_dir in datecode_dirs_in_range(root, start, end):
        pattern = os.path.join(root, night_dir, "*_master_order_trace.csv")
        for path in sorted(glob.glob(pattern)):
            epoch = os.path.basename(path).split("_master")[0]
            match = OBS_ID_RE.match(epoch)
            if not match:
                logger.warning("skipping unrecognised filename %s", path)
                continue
            night = match.group(1)
            era = era_of(night, eras)
            if era is None:  # engineering / service-mission night
                continue
            table = pd.read_csv(path)
            table["Era"] = era
            table["Night"] = night
            table["Epoch"] = epoch
            tables.append(table)
    if not tables:
        sys.exit(
            f"error: no science-era order-trace CSVs under {root} in {start}..{end}"
        )
    return pd.concat(tables, ignore_index=True)


def pixel_deviations(traces, coeff_cols):
    """Row scatter about each era's median centerline, per trace and per epoch.

    Each epoch's centerline is evaluated on a common column grid, masked
    outside the span it was fitted over, and differenced against the pointwise
    median of that era's epochs. Per trace, `Typical` is the RMS of those
    differences over all epochs and columns and `Max` the single worst
    excursion; the per-epoch table carries the same two, plus `P95`, over that
    one epoch's traces.
    """
    columns = np.linspace(0.0, NCOL - 1.0, NSAMPLE)
    powers = np.vstack([columns**k for k in range(len(coeff_cols))])

    records = []
    # (era, epoch, chip) -> one value per trace
    samples = {name: defaultdict(list) for name in ("Max", "Square", "Shift")}
    keys = ["Era", "Chip", "Fiber", "Order"]
    for (era, chip, fiber, order), group in traces.groupby(keys, sort=False):
        group = group[group["Status"] != "missing"].sort_values("Epoch")
        if group["Epoch"].nunique() < MIN_EPOCHS:
            continue
        rows = group[coeff_cols].to_numpy(float) @ powers
        off_span = (columns < group["X1"].to_numpy(float)[:, None]) | (
            columns > group["X2"].to_numpy(float)[:, None]
        )
        rows[off_span] = np.nan
        # A column no epoch reached is all-NaN; nanmedian/nanmax say so loudly.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            deviation = rows - np.nanmedian(rows, axis=0)
            if not np.isfinite(deviation).any():
                continue
            per_epoch = {
                "Max": np.nanmax(np.abs(deviation), axis=1),
                "Square": np.nanmean(deviation**2, axis=1),
                "Shift": np.nanmean(deviation, axis=1),
            }
            typical = float(np.sqrt(np.nanmean(deviation**2)))
            largest = float(np.nanmax(np.abs(deviation)))
        records.append(
            {
                "Era": era,
                "Chip": chip,
                "Fiber": fiber,
                "Order": order,
                "Epochs": int(group["Epoch"].nunique()),
                "Typical": typical,
                "Max": largest,
            }
        )
        for position, epoch in enumerate(group["Epoch"]):
            if not np.isfinite(per_epoch["Max"][position]):
                continue
            for name, values in per_epoch.items():
                samples[name][(era, epoch, chip)].append(float(values[position]))

    return pd.DataFrame(records, columns=_PIXEL_COLUMNS), epoch_table(samples)


def epoch_table(samples):
    """One row per epoch and chip, plus a both-chips row, dated by observation.

    `samples` maps a field to {(era, epoch, chip): one value per trace}. `P95`
    is the 95th percentile of that epoch's per-trace worst excursions -- the
    era-median offset of the epoch as a whole, unmoved by the one or two edge
    traces that dominate `Max`. `Shift` keeps its sign, so a bulk translation
    of the traces shows up there rather than in `RMS`.
    """
    both = {name: defaultdict(list) for name in samples}
    for name, source in samples.items():
        for (era, epoch, _), values in source.items():
            both[name][(era, epoch, "BOTH")].extend(values)

    rows = []
    for source in (samples, both):
        for key in source["Max"]:
            era, epoch, chip = key
            worst = np.asarray(source["Max"][key])
            rows.append(
                {
                    "Era": era,
                    "Epoch": epoch,
                    "Chip": chip,
                    "Date": pd.Timestamp(epoch.split(".")[1]),
                    "RMS": float(np.sqrt(np.mean(source["Square"][key]))),
                    "P95": float(np.quantile(worst, 0.95)),
                    "Max": float(worst.max()),
                    "Shift": float(np.median(source["Shift"][key])),
                }
            )
    table = pd.DataFrame(rows, columns=_EPOCH_COLUMNS)
    return table.sort_values(["Era", "Epoch", "Chip"])


def find_jumps(epochs):
    """Dates where a chip's traces stepped to a new position and stayed there.

    Each boundary between consecutive epochs is scored by comparing the median
    level of the `JUMP_WINDOW` epochs either side, on whichever of the offset
    (`RMS`) and the signed shift (`Shift`) separates them most sharply. A jump
    has to clear both an absolute size and the local scatter -- the second test
    is what tells a step apart from a noisy stretch or a single bad flat, which
    moves the traces for one epoch and then moves them back.
    """
    jumps = []
    for (era, chip), group in epochs.groupby(["Era", "Chip"]):
        if chip == "BOTH":
            continue
        group = group.sort_values("Epoch").reset_index(drop=True)
        series = [group["RMS"].to_numpy(float), group["Shift"].to_numpy(float)]
        scored = []
        for i in range(JUMP_WINDOW, len(group) - JUMP_WINDOW + 1):
            step, scatter = max(
                (level_step(values, i) for values in series),
                key=lambda pair: pair[0] / pair[1],
            )
            scored.append((step / scatter, step, i))

        taken = []
        for significance, step, i in sorted(scored, reverse=True):
            if significance < JUMP_MIN_SIGNIF:
                break
            if step < JUMP_MIN_STEP or any(abs(i - j) < JUMP_WINDOW for j in taken):
                continue
            taken.append(i)
            jumps.append(
                {
                    "Era": era,
                    "Chip": chip,
                    "Date": group["Date"].iloc[i],
                    "Epoch": group["Epoch"].iloc[i],
                    "Step": step,
                    "Significance": significance,
                }
            )
    table = pd.DataFrame(jumps, columns=_JUMP_COLUMNS)
    return table.sort_values(["Era", "Date", "Chip"])


def level_step(values, i):
    """Level change at boundary `i`, and the scatter it has to stand out from."""
    before, after = values[i - JUMP_WINDOW : i], values[i : i + JUMP_WINDOW]
    low, high = np.median(before), np.median(after)
    scatter = 0.5 * (np.median(np.abs(before - low)) + np.median(np.abs(after - high)))
    return abs(high - low), max(scatter, 2e-4)


def jump_dates(jumps, era):
    """One date per jump, with the two chips' detections of one jump merged."""
    dates = sorted(jumps[jumps["Era"] == era]["Date"].unique())
    merged = []
    for date in dates:
        if not merged or (date - merged[-1]).days > JUMP_MERGE_DAYS:
            merged.append(date)
    return merged


def coefficient_deviations(traces, coeff_cols):
    """Per-trace, per-coefficient scatter about the era's median coefficient.

    Only `full` traces are used: a partial trace is refit at degree 2 with the
    remaining coefficients zeroed, which is a different fit, not a shift.
    """
    full = traces[traces["Status"] == "full"]
    records = []
    keys = ["Era", "Chip", "Fiber", "Order"]
    for (era, chip, fiber, order), group in full.groupby(keys, sort=False):
        if group["Epoch"].nunique() < MIN_EPOCHS:
            continue
        values = group[coeff_cols].to_numpy(float)
        deviation = values - np.median(values, axis=0)
        for k, name in enumerate(coeff_cols):
            records.append(
                {
                    "Era": era,
                    "Chip": chip,
                    "Fiber": fiber,
                    "Order": order,
                    "Coeff": name,
                    "Typical": float(np.sqrt(np.mean(deviation[:, k] ** 2))),
                    "Max": float(np.max(np.abs(deviation[:, k]))),
                }
            )
    return pd.DataFrame(records, columns=_COEFF_COLUMNS)


def log_bins(values, count=40):
    """Log-spaced bins spanning `values`, floored six decades below the top."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return np.logspace(-6, 0, count)
    top = np.log10(values.max())
    return np.logspace(max(np.log10(values.min()), top - 6), top, count)


def draw_hist(axis, values, bins, label, color):
    values = np.clip(np.asarray(values, float), bins[0], bins[-1])
    axis.hist(values, bins=bins, histtype="step", lw=1.6, label=label, color=color)


def era_colors(eras):
    palette = _pyplot().get_cmap("tab10")
    return {era: palette(i % 10) for i, era in enumerate(eras)}


def plot_pixel_by_era(pixel, eras, colors, out_dir):
    """Row scatter per trace, one histogram per era, split by chip."""
    fig, axes = _pyplot().subplots(2, 2, figsize=(11, 7), sharex=True)
    bins = log_bins(pd.concat([pixel["Typical"], pixel["Max"]]))
    for row, chip in enumerate(CHIPS):
        for col, metric in enumerate(["Typical", "Max"]):
            axis = axes[row, col]
            for era in eras:
                subset = pixel[(pixel["Chip"] == chip) & (pixel["Era"] == era)]
                draw_hist(axis, subset[metric], bins, f"era {era}", colors[era])
            axis.axvline(1.0, color="k", ls="--", lw=1)
            axis.set_xscale("log")
            axis.set_title(f"{chip}: {metric.lower()} deviation per trace")
            axis.set_ylabel("traces")
            if row == 1:
                axis.set_xlabel("deviation from era median centerline [pixels]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Order-trace row stability within an era (dashed line: 1 pixel)")
    save(fig, out_dir, "pixel_deviation_by_era.png")


def plot_pixel_by_fiber(pixel, eras, colors, out_dir):
    """The same row scatter, resolved by fiber, to expose a bad orderlet."""
    fig, axes = _pyplot().subplots(
        2, len(FIBERS), figsize=(4 * len(FIBERS), 7), sharex=True, sharey="row"
    )
    bins = log_bins(pixel["Max"])
    for row, chip in enumerate(CHIPS):
        for col, fiber in enumerate(FIBERS):
            axis = axes[row, col]
            for era in eras:
                subset = pixel[
                    (pixel["Chip"] == chip)
                    & (pixel["Fiber"] == fiber)
                    & (pixel["Era"] == era)
                ]
                draw_hist(axis, subset["Max"], bins, f"era {era}", colors[era])
            axis.axvline(1.0, color="k", ls="--", lw=1)
            axis.set_xscale("log")
            axis.set_title(f"{chip} {fiber}")
            if col == 0:
                axis.set_ylabel("traces")
            if row == 1:
                axis.set_xlabel("max deviation [pixels]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Maximum row deviation per trace, by chip and fiber")
    save(fig, out_dir, "pixel_deviation_by_fiber.png")


def plot_pixel_by_order(pixel, eras, colors, out_dir):
    """Where on the detector the scatter sits: deviation against order index."""
    fig, axes = _pyplot().subplots(2, 2, figsize=(12, 7), sharex="row")
    for row, chip in enumerate(CHIPS):
        for col, metric in enumerate(["Typical", "Max"]):
            axis = axes[row, col]
            for era in eras:
                subset = pixel[(pixel["Chip"] == chip) & (pixel["Era"] == era)]
                per_order = subset.groupby("Order")[metric].max()
                axis.plot(
                    per_order.index,
                    per_order.to_numpy(),
                    marker="o",
                    ms=3,
                    lw=1,
                    label=f"era {era}",
                    color=colors[era],
                )
            axis.axhline(1.0, color="k", ls="--", lw=1)
            axis.set_yscale("log")
            axis.set_title(f"{chip}: worst {metric.lower()} deviation in each order")
            axis.set_xlabel("order index")
            axis.set_ylabel("deviation [pixels]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Row deviation against order index (worst fiber of each order)")
    save(fig, out_dir, "pixel_deviation_by_order.png")


def plot_coefficients(coeffs, coeff_cols, eras, colors, out_dir):
    """Coefficient-space scatter: one panel per coefficient, per chip."""
    fig, axes = _pyplot().subplots(
        2, len(coeff_cols), figsize=(3.6 * len(coeff_cols), 7), sharey="row"
    )
    for row, chip in enumerate(CHIPS):
        for col, name in enumerate(coeff_cols):
            axis = axes[row, col]
            here = coeffs[(coeffs["Chip"] == chip) & (coeffs["Coeff"] == name)]
            bins = log_bins(here["Typical"])
            for era in eras:
                draw_hist(
                    axis,
                    here[here["Era"] == era]["Typical"],
                    bins,
                    f"era {era}",
                    colors[era],
                )
            axis.set_xscale("log")
            axis.set_title(f"{chip} {name}")
            if col == 0:
                axis.set_ylabel("traces")
            axis.set_xlabel("RMS deviation from era median")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Polynomial-coefficient stability within an era (full traces only)")
    save(fig, out_dir, "coefficient_deviation_by_era.png")


def mark_jumps(axis, dates, color="k"):
    """Dashed line and date label at each jump."""
    for date in dates:
        axis.axvline(date, color=color, ls="--", lw=1, alpha=0.7)
        axis.annotate(
            pd.Timestamp(date).strftime("%Y-%m-%d"),
            xy=(date, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(2, -2),
            textcoords="offset points",
            va="top",
            fontsize=7,
            color=color,
        )


def plot_stability_over_time(epochs, jumps, eras, out_dir):
    """Each epoch's offset against date, which is where settling shows up.

    An era that settles decays over its first days; an era that steps holds a
    plateau and then drops between two consecutive flats. Detected jumps are
    marked.
    """
    fig, axes = _pyplot().subplots(len(eras), 1, figsize=(12, 2.4 * len(eras)))
    series = (("Max", "tab:red"), ("P95", "tab:orange"), ("RMS", "tab:blue"))
    for axis, era in zip(np.atleast_1d(axes), eras, strict=True):
        subset = epochs[(epochs["Era"] == era) & (epochs["Chip"] == "BOTH")]
        for metric, color in series:
            axis.plot(
                subset["Date"],
                subset[metric],
                ".",
                ms=4,
                color=color,
                label=metric.lower(),
            )
        dates = jump_dates(jumps, era)
        mark_jumps(axis, dates)
        axis.set_yscale("log")
        axis.set_ylabel("deviation [pixels]")
        axis.set_title(
            f"era {era}: {len(subset)} epochs, {len(dates)} jumps", fontsize=10
        )
        axis.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=8, ncol=3, loc="lower left")
    fig.suptitle("Per-epoch offset from the era median centerline, over time")
    save(fig, out_dir, "stability_over_time.png")


def plot_stability_over_time_by_chip(epochs, jumps, eras, out_dir):
    """The same track split by chip, to show which detector moved."""
    fig, axes = _pyplot().subplots(len(eras), 1, figsize=(12, 2.4 * len(eras)))
    colors = {"GREEN": "tab:green", "RED": "tab:red"}
    for axis, era in zip(np.atleast_1d(axes), eras, strict=True):
        for chip in CHIPS:
            subset = epochs[(epochs["Era"] == era) & (epochs["Chip"] == chip)]
            axis.plot(
                subset["Date"], subset["P95"], ".", ms=4, color=colors[chip], label=chip
            )
            found = jumps[(jumps["Era"] == era) & (jumps["Chip"] == chip)]
            mark_jumps(axis, found["Date"], color=colors[chip])
        axis.set_yscale("log")
        axis.set_ylabel("p95 deviation [pixels]")
        axis.set_title(f"era {era}", fontsize=10)
        axis.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=8, ncol=2, loc="lower left")
    fig.suptitle("Per-epoch offset by chip, over time; jumps marked per chip")
    save(fig, out_dir, "stability_over_time_by_chip.png")


def save(fig, out_dir, name):
    fig.tight_layout()
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=130)
    _pyplot().close(fig)
    logger.info("wrote %s", path)


def write_csv(table, out_dir, name):
    path = os.path.join(out_dir, name)
    table.to_csv(path, index=False)
    logger.info("%d row(s) -> %s", len(table), path)


def summarize(traces, pixel, eras):
    """One line per era and chip; the question is whether Max stays under 1."""
    header = (
        f"{'era':>4s} {'chip':<6s} {'epochs':>7s} {'traces':>7s} "
        f"{'typ p50':>8s} {'typ p95':>8s} {'max p50':>8s} {'max p95':>8s} "
        f"{'worst':>8s} {'>1 pix':>7s}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for era in eras:
        for chip in CHIPS:
            subset = pixel[(pixel["Era"] == era) & (pixel["Chip"] == chip)]
            if subset.empty:
                continue
            epochs = traces[(traces["Era"] == era)]["Epoch"].nunique()
            over = int((subset["Max"] > 1.0).sum())
            logger.info(
                "%4s %-6s %7d %7d %8.3f %8.3f %8.3f %8.3f %8.3f %7d",
                era,
                chip,
                epochs,
                len(subset),
                subset["Typical"].quantile(0.50),
                subset["Typical"].quantile(0.95),
                subset["Max"].quantile(0.50),
                subset["Max"].quantile(0.95),
                subset["Max"].max(),
                over,
            )
    logger.info("typ = RMS row deviation from the era's median centerline; max = worst")
    logger.info("excursion. Both are per trace, over every epoch and sampled column.")

    worst = pixel.sort_values("Max", ascending=False).head(10)
    logger.info("worst 10 traces:")
    for row in worst.itertuples():
        logger.info(
            "  era %s %-6s %-5s order %2d  typ %7.3f  max %7.3f  (%d epochs)",
            row.Era,
            row.Chip,
            row.Fiber,
            row.Order,
            row.Typical,
            row.Max,
            row.Epochs,
        )


def report_jumps(jumps, eras):
    """The detected jumps, per era, with the chip that stepped."""
    logger.info("jumps (step = level change in pixels; sig = step / local scatter):")
    for era in eras:
        dates = jump_dates(jumps, era)
        logger.info("  era %s: %d jumps", era, len(dates))
        for row in jumps[jumps["Era"] == era].itertuples():
            logger.info(
                "    %s  %-6s step %.3f  sig %5.1f  first epoch after: %s",
                row.Date.date(),
                row.Chip,
                row.Step,
                row.Significance,
                row.Epoch,
            )


def main(argv=None):
    args = parse_args(argv)
    start, end = args.date_range

    _, log_path = setup_batch_logging(
        args.log_dir, "analyze_order_trace", args.run_id, level=args.log_level or "INFO"
    )

    logger.info("kpfpipe %s order trace analysis starting", kpfpipe.__version__)
    logger.info("argv: %s", " ".join(sys.argv))
    logger.info("date_range: %s..%s", start, end)
    logger.info("analysis log: %s", log_path)

    traces = load_traces(args.input_dir, start, end, load_science_eras())
    logger.info(
        "%d epochs over %d nights", traces["Epoch"].nunique(), traces["Night"].nunique()
    )
    coeff_cols = [c for c in traces.columns if re.fullmatch(r"Coeff\d+", c)]

    pixel, epochs = pixel_deviations(traces, coeff_cols)
    if pixel.empty:
        sys.exit(f"error: no era in {start}..{end} has {MIN_EPOCHS}+ trace epochs")
    coeffs = coefficient_deviations(traces, coeff_cols)
    jumps = find_jumps(epochs)

    eras = sorted(traces["Era"].unique(), key=float)
    colors = era_colors(eras)
    eras = [era for era in eras if era in set(pixel["Era"])]

    out_dir = os.path.join(args.analysis_dir, "order_trace")
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"{start}_{end}.csv"
    write_csv(pixel, out_dir, f"order_trace_pixel_{suffix}")
    write_csv(epochs, out_dir, f"order_trace_epochs_{suffix}")
    write_csv(coeffs, out_dir, f"order_trace_coefficients_{suffix}")
    write_csv(jumps, out_dir, f"order_trace_jumps_{suffix}")

    plot_pixel_by_era(pixel, eras, colors, out_dir)
    plot_pixel_by_fiber(pixel, eras, colors, out_dir)
    plot_pixel_by_order(pixel, eras, colors, out_dir)
    plot_coefficients(coeffs, coeff_cols, eras, colors, out_dir)
    plot_stability_over_time(epochs, jumps, eras, out_dir)
    plot_stability_over_time_by_chip(epochs, jumps, eras, out_dir)
    summarize(traces, pixel, eras)
    report_jumps(jumps, eras)


if __name__ == "__main__":
    main()
