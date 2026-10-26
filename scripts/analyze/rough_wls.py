#!/usr/bin/env python3
"""Measure how ThAr spectra move on the detector within and between instrument
eras, and propose one rough WLS per era (``kpfpipe analyze rough_wls``).

The rough WLS (``reference/rough_wls/rough_wls_{datecode}.csv``) only seeds the
ThAr line search, whose fit window is +/-5 px, so it must sit within ~1 px of
where each era's lines actually fall. The per-frame ThAr L2s the masters batch
writes under ``{input_dir}/masters/{datecode}/thar_L2/`` are the input; nothing
is reduced and the reference files are never touched.

For every science era (engineering eras are skipped):

1. each night's ThAr frames (the largest OBJECT group, at most 5) are
   median-stacked, and cross-correlated, order by order in 16 chunks, against
   the era's anchor -- the night nearest the middle of the era's nights whose
   lines sit cleanly in the fit window;
2. the anchor's lines are fit with the era's current reference rough WLS as
   the guess, tying the chunk shifts to the reference;
3. each night's offset from the reference is a per-chip quadratic in pixel,
   D(x) = a + b*xn + c*xn^2 (xn in [-1, 1]); the era candidate shifts the
   reference by the era-mean D and refits each order at degree 6.

Over an inclusive datecode range this writes, under ``{analysis_dir}/rough_wls/``:

- ``rough_wls_{datecode}.csv`` -- the candidate for each era, named for the
  reference it would replace;
- ``rough_wls_nights_{START}_{END}.csv`` -- the offset per night and chip;
- ``rough_wls_summary_{START}_{END}.md`` -- candidate quality, offsets between
  eras, and any night more than 1 px from its era's candidate;
- ``rough_wls_offsets_over_time.png`` -- each night's offset against date.

``--date_range`` defaults to the whole archive: first light through tomorrow.

    kpfpipe analyze rough_wls --date_range 20221109 20251231 \
        --input_dir /data/kpf-next --output_dir /data/kpf-next
"""

import argparse
import glob
import logging
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from astropy.io import fits
from numpy.polynomial import legendre, polynomial
from scipy.ndimage import median_filter

import kpfpipe
from kpfpipe import CHIPS, DETECTOR, INSTRUMENT_ERAS, REPO_ROOT
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.modules.masters import WLS
from kpfpipe.utils.io import datecode_dirs_in_range
from kpfpipe.utils.kpf import select_era_reference
from kpfpipe.utils.logger import setup_batch_logging
from scripts._argparse import analysis_parser, resolve_dates, resolve_dir_shortcuts
from scripts.analyze._eras import era_of, load_science_eras

logger = logging.getLogger(__name__)

OBS_ID_RE = re.compile(r"^KP\.(\d{8})\.(\d{5}\.\d+)")
NCOL = DETECTOR["ccd"]["ncol"]
FIBERS = ("SCI2", "CAL")  # cross-correlated and line-fit; all fibers share one guess
DEGREE = 6  # Legendre degree of the candidate rough WLS
MAX_FRAMES = 5  # frames stacked per night
ANCHOR_FRAMES = 3  # frames line-fit per era anchor
NCHUNK = 16  # cross-correlation chunks per order
MAX_LAG = 80  # pixels searched by the full-order cross-correlation
CHUNK_LAG = 8  # ... and by each chunk, around the full-order lag
MIN_PEAK = 0.5  # normalized correlation a chunk needs to count
FIT_WINDOW = 5  # the WLS line-fit window half-width, pixels
IN_WINDOW = 2.5  # a line this close to its guess is comfortably in the window
MAX_DEVIATION = 1.0  # pixels; a night further than this from its era is flagged
ANCHOR_TRIES = 5  # nights tried, outward from the era's middle, as its anchor
ANCHOR_MIN_IN_WINDOW = 0.9  # in-window fraction an anchor needs on every chip

_NIGHT_COLUMNS = [
    "Era", "Night", "Anchor", "Chip", "Frames", "Object",
    "s_a", "s_b", "s_c", "D_a", "D_b", "D_c", "RMS", "Chunks", "MaxDevFromEra",
]  # fmt: skip


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="kpfpipe analyze rough_wls",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[analysis_parser("analyzes every masters night in it")],
    )
    ap.add_argument(
        "--input_dir",
        required=True,
        help="root of the masters tree to read: "
        "{input_dir}/masters/{datecode}/thar_L2/",
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


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


def find_nights(input_dir, start, end, eras):
    """One row per science-era night with ThAr L2s: Night, Era, ObsTime, Object,
    Files (the largest OBJECT group, at most MAX_FRAMES, in time order), and
    HasMaster (a ``*_master_thar_L2.fits`` was written that night).

    Grouping by OBJECT keeps single-fiber ThAr stacks out of an all-fiber
    stack's median. Exits loudly when the tree is missing or holds no frames.
    """
    root = os.path.join(input_dir, "masters")
    if not os.path.isdir(root):
        sys.exit(f"error: masters directory not found: {root}")
    rows = []
    for night in datecode_dirs_in_range(root, start, end):
        paths = sorted(
            glob.glob(os.path.join(root, night, "thar_L2", "*_thar_L2.fits"))
        )
        if not paths:
            continue
        objects = {p: str(fits.getheader(p).get("OBJECT")) for p in paths}
        obj = Counter(objects.values()).most_common(1)[0][0]
        files = [p for p in paths if objects[p] == obj][:MAX_FRAMES]
        match = OBS_ID_RE.match(os.path.basename(files[0]))
        if not match:
            logger.warning("skipping unrecognised filename %s", files[0])
            continue
        date, seconds = match.groups()
        obs_time = pd.Timestamp(date) + pd.Timedelta(seconds=float(seconds))
        era = era_of(obs_time, eras)
        if era is None:  # engineering / service-mission night
            continue
        has_master = bool(glob.glob(os.path.join(root, night, "*_master_thar_L2.fits")))
        rows.append(
            dict(Night=night, Era=era, ObsTime=obs_time, Object=obj, Files=files,
                 HasMaster=has_master)
        )  # fmt: skip
    if not rows:
        sys.exit(f"error: no science-era ThAr L2s under {root} in {start}..{end}")
    return pd.DataFrame(rows)


def read_frame(path):
    """``{(chip, fiber): flux}`` for one ThAr L2, each normalized to unit median."""
    data = KPF2.from_fits(path).data
    out = {}
    for chip in CHIPS:
        for fiber in FIBERS:
            flux = np.array(data[f"{chip}_{fiber}_FLUX"], dtype=float)
            out[(chip, fiber)] = flux / np.nanmedian(np.abs(flux))
    return out


def stack_night(files):
    frames = [read_frame(f) for f in files]
    return {key: np.nanmedian([fr[key] for fr in frames], axis=0) for key in frames[0]}


# ---------------------------------------------------------------------------
# Shifts
# ---------------------------------------------------------------------------


def _prep(row):
    """Continuum-subtracted, clipped, sqrt-compressed, zero-mean order."""
    r = np.nan_to_num(row, nan=0.0)
    r = r - median_filter(r, 101, mode="nearest")
    r = np.sqrt(np.clip(r, 0, None))
    return r - r.mean() if r.std() > 0 else r


def xcorr(ref, tgt, max_lag=MAX_LAG, center=0):
    """Lag of ``tgt`` relative to ``ref`` (tgt(x) ~ ref(x - lag)), searched in
    center +/- max_lag; returns (lag, normalized peak)."""
    n = ref.size
    if ref.std() == 0 or tgt.std() == 0:
        return np.nan, 0.0
    lags = np.arange(-max_lag, max_lag + 1) + int(round(center))
    cc = np.array(
        [
            np.dot(ref[max(0, -k) : n - max(0, k)], tgt[max(0, k) : n - max(0, -k)])
            for k in lags
        ]
    ) / np.sqrt(np.dot(ref, ref) * np.dot(tgt, tgt))
    i = int(np.argmax(cc))
    frac = 0.0
    if 0 < i < cc.size - 1:
        y0, y1, y2 = cc[i - 1 : i + 2]
        if y0 - 2 * y1 + y2 != 0:
            frac = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return lags[i] + frac, cc[i]


def chunk_shifts(ref, tgt):
    """Per-chunk shifts of ``tgt`` against ``ref``: Chip, Fiber, Row, Xn, Shift,
    Peak."""
    edges = np.linspace(0, NCOL, NCHUNK + 1).astype(int)
    rows = []
    for (chip, fiber), ref_orders in ref.items():
        for j, (r_row, t_row) in enumerate(
            zip(ref_orders, tgt[(chip, fiber)], strict=True)
        ):
            r, t = _prep(r_row), _prep(t_row)
            if not t.any() or not r.any():
                continue
            lag_full, _ = xcorr(r, t)
            for lo, hi in zip(edges[:-1], edges[1:], strict=True):
                lag, peak = xcorr(
                    r[lo:hi], t[lo:hi], max_lag=CHUNK_LAG, center=lag_full
                )
                rows.append(
                    dict(Chip=chip, Fiber=fiber, Row=j, Xn=(lo + hi) / NCOL - 1,
                         Shift=lag, Peak=peak)
                )  # fmt: skip
    return pd.DataFrame(rows)


def robust_quadratic(xn, y, floor):
    """Iteratively 3-sigma-clipped quadratic fit; returns (coeffs, rms, kept).

    ``floor`` (pixels) keeps the clip from closing in on near-noiseless data."""
    xn, y = np.asarray(xn, float), np.asarray(y, float)
    keep = np.isfinite(y)
    if keep.sum() < 3:
        return np.full(3, np.nan), np.nan, 0
    for _ in range(10):
        coeffs = polynomial.polyfit(xn[keep], y[keep], 2)
        resid = y - polynomial.polyval(xn, coeffs)
        new = np.isfinite(y) & (
            np.abs(resid) <= max(3 * 1.4826 * np.nanmedian(np.abs(resid[keep])), floor)
        )
        if new.sum() < 3 or (new == keep).all():
            break
        keep = new
    return coeffs, float(resid[keep].std()), int(keep.sum())


# ---------------------------------------------------------------------------
# Line fits against a rough WLS
# ---------------------------------------------------------------------------


def fit_lines(wls, path, chip):
    """``(PIX_GUESS, FIT_MEAN - PIX_GUESS)`` of every converged line fit in one
    frame, against whichever rough WLS ``wls`` has loaded."""
    lines = wls._fit_line_positions_ffi(KPF2.from_fits(path), chip, list(FIBERS))
    ok = lines["FINITE"] & lines["CONVERGED"]
    return lines["PIX_GUESS"][ok], (lines["FIT_MEAN"] - lines["PIX_GUESS"])[ok]


def line_offsets(wls, rough_wls_file, files):
    """Per chip: the quadratic of fitted-line offsets from ``rough_wls_file``'s
    guesses, the line RMS about it, and the fraction within IN_WINDOW of the
    guess. Lines at the window edge (a fit locked onto the edge) are excluded
    from the quadratic but count against the in-window fraction."""
    wls._load_rough_wls(rough_wls_file)
    out = {}
    for chip in CHIPS:
        fits_ = [fit_lines(wls, f, chip) for f in files[:ANCHOR_FRAMES]]
        guess = np.concatenate([g for g, _ in fits_])
        delta = np.concatenate([d for _, d in fits_])
        usable = np.abs(delta) < FIT_WINDOW - 0.5
        coeffs, rms, _ = robust_quadratic(
            2 * guess[usable] / (NCOL - 1) - 1, delta[usable], floor=0.5
        )
        in_window = float(np.mean(np.abs(delta) <= IN_WINDOW)) if delta.size else np.nan
        out[chip] = dict(coeffs=coeffs, rms=rms, in_window=in_window, n=int(delta.size))
    return out


# ---------------------------------------------------------------------------
# Rough WLS tables
# ---------------------------------------------------------------------------


def _coeff_cols(table):
    return sorted(
        (c for c in table.columns if re.fullmatch(r"C\d+", c)), key=lambda c: int(c[1:])
    )


def shift_rough_wls(reference, shift):
    """``reference`` with each order moved by D(x) pixels (``shift[chip]`` =
    (a, b, c) in xn) and refit at DEGREE: a line at reference pixel p lands at
    p + D(p). WAVE_MIN/WAVE_MAX are the refit solution at xn = -1 / +1."""
    cols = _coeff_cols(reference)
    new_cols = [f"C{k}" for k in range(DEGREE + 1)]
    p = np.arange(-20, NCOL + 20, dtype=float)  # padded so the refit spans the detector
    xn = 2 * p / (NCOL - 1) - 1
    out = reference.drop(columns=cols)
    out[new_cols] = np.nan
    for i, row in reference.iterrows():
        lam = legendre.legval(xn, row[cols].to_numpy(float))
        moved = 2 * (p + polynomial.polyval(xn, shift[row.CHIP])) / (NCOL - 1) - 1
        coef = legendre.legfit(moved, lam, DEGREE)
        out.loc[i, new_cols] = coef
        out.loc[i, ["WAVE_MIN", "WAVE_MAX"]] = legendre.legval([-1.0, 1.0], coef)
    return out


def line_pixels(table, linelist):
    """Pixel of every line-list line under a rough WLS table: Chip, Echelle,
    Wave, Pixel."""
    cols = _coeff_cols(table)
    x = np.arange(NCOL, dtype=float)
    xn = 2 * x / (NCOL - 1) - 1
    rows = []
    for row in table.itertuples():
        lam = legendre.legval(xn, np.array([getattr(row, c) for c in cols], float))
        order = np.argsort(lam)
        waves = linelist.loc[
            (linelist.CHIP == row.CHIP) & (linelist.ECHELLE == row.ECHELLE), "WAVE"
        ].to_numpy(float)
        waves = waves[(waves >= lam.min()) & (waves <= lam.max())]
        pix = np.interp(waves, lam[order], x[order])
        rows += [
            (row.CHIP, row.ECHELLE, w, px) for w, px in zip(waves, pix, strict=True)
        ]
    return pd.DataFrame(rows, columns=["Chip", "Echelle", "Wave", "Pixel"])


def pixel_offsets(table_a, table_b, linelist):
    """Per chip, (median, min, max) pixel offset of the lines under ``table_b``
    relative to ``table_a``."""
    a, b = line_pixels(table_a, linelist), line_pixels(table_b, linelist)
    both = a.merge(b, on=["Chip", "Echelle", "Wave"], suffixes=("_a", "_b"))
    delta = both.Pixel_b - both.Pixel_a
    return {
        chip: (delta[both.Chip == chip].median(), delta[both.Chip == chip].min(),
               delta[both.Chip == chip].max())
        for chip in CHIPS
    }  # fmt: skip


def max_deviation(coeffs, era_coeffs):
    """Largest |D_night(x) - D_era(x)| over the detector, pixels."""
    xn = np.linspace(-1, 1, 101)
    return float(
        np.max(np.abs(polynomial.polyval(xn, np.subtract(coeffs, era_coeffs))))
    )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def era_reference(era, night):
    """The current reference rough WLS in effect for ``night`` of ``era``."""
    return select_era_reference(
        f"{REPO_ROOT}/reference/rough_wls/rough_wls_*.csv",
        "rough WLS",
        {"INSTERA": era, "JD_UTC": night.ObsTime.to_julian_date()},
        night.Night,
        INSTRUMENT_ERAS,
    )


def choose_anchor(era, nights, wls):
    """The era's anchor night, its reference, and its line offsets from it;
    ``None`` when no night of the era built a master WLS.

    Only nights whose master WLS was built are candidates, so a stretch where
    the WLS failed (a dying lamp, e.g. 2024-03) is never the anchor. Since the
    WLS stopped gating, a batch writes masters for such nights too, so the
    candidates are also tried outward from the middle of the era and the first
    with ANCHOR_MIN_IN_WINDOW of its lines in window on every chip is taken; if
    every tried night falls short, the best is used with a warning."""
    middle = len(nights) // 2
    candidates = [i for i in range(len(nights)) if nights.HasMaster.iloc[i]]
    if not candidates:
        return None
    tried = sorted(candidates, key=lambda i: abs(i - middle))[:ANCHOR_TRIES]
    best = None
    for i in tried:
        night = nights.iloc[i]
        reference_file = era_reference(era, night)
        lines = line_offsets(wls, reference_file, night.Files)
        score = min(lines[chip]["in_window"] for chip in CHIPS)
        if best is None or score > best[0]:
            best = (score, night, reference_file, lines)
        if score >= ANCHOR_MIN_IN_WINDOW:
            break
        logger.warning(
            "era %s: %s has %.0f%% of lines in window; not used as anchor",
            era, night.Night, 100 * score,
        )  # fmt: skip
    else:
        logger.warning(
            "era %s: no anchor candidate reached %.0f%% in window; using %s (%.0f%%)",
            era, 100 * ANCHOR_MIN_IN_WINDOW, best[1].Night, 100 * best[0],
        )  # fmt: skip
    return best[1], best[2], best[3]


def analyze_era(era, nights, wls, out_dir):
    """Night offsets and the candidate for one era; returns (night rows, era record)."""
    nights = nights.sort_values("ObsTime").reset_index(drop=True)
    chosen = choose_anchor(era, nights, wls)
    if chosen is None:
        logger.warning("era %s: no night built a master WLS; era not analyzed", era)
        return None, None
    anchor, reference_file, ref_lines = chosen
    logger.info(
        "era %s: %d nights, anchor %s, reference %s",
        era, len(nights), anchor.Night, os.path.basename(reference_file),
    )  # fmt: skip

    anchor_stack = stack_night(anchor.Files)
    rows = []
    for night in nights.itertuples():
        stack = (
            anchor_stack if night.Night == anchor.Night else stack_night(night.Files)
        )
        chunks = chunk_shifts(anchor_stack, stack)
        for chip in CHIPS:
            good = chunks[(chunks.Chip == chip) & (chunks.Peak > MIN_PEAK)]
            s, rms, n = robust_quadratic(good.Xn, good.Shift, floor=0.3)
            D = ref_lines[chip]["coeffs"] + s
            rows.append(
                dict(Era=era, Night=night.Night, Anchor=anchor.Night, Chip=chip,
                     Frames=len(night.Files), Object=night.Object, s_a=s[0], s_b=s[1],
                     s_c=s[2], D_a=D[0], D_b=D[1], D_c=D[2], RMS=rms, Chunks=n)
            )  # fmt: skip
    table = pd.DataFrame(rows)

    era_shift = {
        chip: table.loc[table.Chip == chip, ["D_a", "D_b", "D_c"]].mean().to_numpy()
        for chip in CHIPS
    }
    table["MaxDevFromEra"] = [
        max_deviation(row[["D_a", "D_b", "D_c"]].to_numpy(float), era_shift[row.Chip])
        for _, row in table.iterrows()
    ]

    reference = pd.read_csv(reference_file)
    candidate = shift_rough_wls(reference, era_shift)
    name = os.path.basename(reference_file)
    candidate_file = os.path.join(out_dir, name)
    candidate.to_csv(candidate_file, index=False, float_format="%.17g")
    logger.info("wrote %s", candidate_file)

    record = dict(
        era=era,
        reference=name,
        nights=len(nights),
        anchor=anchor.Night,
        shift=era_shift,
        ref_lines=ref_lines,
        cand_lines=line_offsets(wls, candidate_file, anchor.Files),
        cand_vs_ref=pixel_offsets(reference, candidate, wls._linelist_df),
        candidate=candidate,
    )
    return table[_NIGHT_COLUMNS], record


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _fmt_range(triple):
    med, lo, hi = triple
    return f"{med:+.2f} ({lo:+.2f}..{hi:+.2f})"


def summary_markdown(records, nights, skipped, start, end, linelist):
    """The reviewer's report: candidate quality per era, offsets between eras,
    every night further than MAX_DEVIATION from its era's candidate, and the
    ``skipped`` eras (ThAr L2s but no master WLS to anchor on)."""
    md = [
        f"# Rough WLS analysis, {start}..{end}",
        "",
        f"kpfpipe {kpfpipe.__version__}. Offsets in pixels; positive = lines at larger "
        "pixel. D(x) = a + b·xn + c·xn² is an era's mean offset from its current "
        "reference (xn = -1..+1 across the detector). Line fits use a "
        f"±{FIT_WINDOW} px window; 'in window' is the fraction of anchor lines within "
        f"±{IN_WINDOW} px of their guess.",
        "",
        "## Candidate quality per era",
        "",
        "| Era | Chip | Reference | Nights | Anchor | D_era a / b / c | "
        "Candidate − reference: median (min..max) | Reference: line rms / in window | "
        "Candidate: residual a / line rms / in window | Worst night dev | > 1 px |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rec in records:
        for chip in CHIPS:
            sel = nights[(nights.Era == rec["era"]) & (nights.Chip == chip)]
            worst = sel.MaxDevFromEra.max()
            ref, cand = rec["ref_lines"][chip], rec["cand_lines"][chip]
            a, b, c = rec["shift"][chip]
            md.append(
                f"| {rec['era']} | {chip} | {rec['reference']} | {rec['nights']} | "
                f"{rec['anchor']} | {a:+.2f} / {b:+.2f} / {c:+.2f} | "
                f"{_fmt_range(rec['cand_vs_ref'][chip])} | "
                f"{ref['rms']:.2f} / {ref['in_window']:.1%} | "
                f"{cand['coeffs'][0]:+.2f} / {cand['rms']:.2f} / "
                f"{cand['in_window']:.1%} | "
                f"{worst:.2f} | {'**YES**' if worst > MAX_DEVIATION else 'no'} |"
            )
    md += [
        "",
        "A reference whose in-window fraction is well below the candidate's is "
        "seeding lines off target. If the reference sits ≳4 px off, its line fits "
        "lock onto the window edge, so its D_era understates the shift; rerun after "
        "adopting the candidate.",
        "",
        "## Offsets between consecutive eras' candidates",
        "",
        "| From | To | Chip | Median (min..max) |",
        "|---|---|---|---|",
    ]
    for prev, rec in zip(records[:-1], records[1:], strict=True):
        offsets = pixel_offsets(prev["candidate"], rec["candidate"], linelist)
        for chip in CHIPS:
            md.append(
                f"| {prev['era']} | {rec['era']} | {chip} | "
                f"{_fmt_range(offsets[chip])} |"
            )
    if len(records) < 2:
        md.append("| – | – | – | fewer than two eras analyzed |")

    flagged = nights[nights.MaxDevFromEra > MAX_DEVIATION]
    md += [
        "",
        f"## Nights more than {MAX_DEVIATION:g} px from their era's candidate",
        "",
    ]
    if flagged.empty:
        md.append("None.")
    else:
        md += [
            "| Era | Night | Chip | Max deviation | D a / b / c |",
            "|---|---|---|---|---|",
        ]
        for row in flagged.itertuples():
            md.append(
                f"| {row.Era} | {row.Night} | {row.Chip} | {row.MaxDevFromEra:.2f} | "
                f"{row.D_a:+.2f} / {row.D_b:+.2f} / {row.D_c:+.2f} |"
            )
    failed = nights[nights.Chunks == 0]
    if not failed.empty:
        md += ["", "## Nights with no usable cross-correlation", ""]
        md += [f"- {r.Era} {r.Night} {r.Chip}" for r in failed.itertuples()]
    if skipped:
        md += ["", "## Eras not analyzed", ""]
        md += [f"- {era}: no night in range built a master WLS" for era in skipped]
    return "\n".join(md) + "\n"


def plot_offsets_over_time(nights, out_dir):
    """Each night's offset from its era's reference at the detector's blue edge,
    center and red edge, per chip; anchors circled."""
    plt = _pyplot()
    dates = pd.to_datetime(nights.Night)
    fig, axes = plt.subplots(len(CHIPS), 1, figsize=(10, 3.2 * len(CHIPS)), sharex=True)
    for axis, chip in zip(np.atleast_1d(axes), CHIPS, strict=True):
        sel = nights.Chip == chip
        coeffs = nights.loc[sel, ["D_a", "D_b", "D_c"]].to_numpy(float)
        for xn, label in ((-1, "xn = -1"), (0, "xn = 0"), (1, "xn = +1")):
            axis.plot(
                dates[sel], polynomial.polyval(xn, coeffs.T), ".", ms=4, label=label
            )
        for era in nights.Era.unique():
            first = dates[sel & (nights.Era == era)].min()
            axis.axvline(first, color="0.6", lw=0.8)
            axis.text(first, 1.0, f" {era}", transform=axis.get_xaxis_transform(),
                      va="top", fontsize=8)  # fmt: skip
        axis.set_ylabel(f"{chip} offset [px]")
        axis.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Per-night ThAr offset from the era's current rough WLS")
    fig.tight_layout()
    path = os.path.join(out_dir, "rough_wls_offsets_over_time.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info("wrote %s", path)


def main(argv=None):
    args = parse_args(argv)
    start, end = args.date_range

    _, log_path = setup_batch_logging(
        args.log_dir, "analyze_rough_wls", args.run_id, level=args.log_level or "INFO"
    )
    logger.info("kpfpipe %s rough WLS analysis starting", kpfpipe.__version__)
    logger.info("argv: %s", " ".join(sys.argv))
    logger.info("date_range: %s..%s", start, end)
    logger.info("analysis log: %s", log_path)

    nights = find_nights(args.input_dir, start, end, load_science_eras())
    out_dir = os.path.join(args.analysis_dir, "rough_wls")
    os.makedirs(out_dir, exist_ok=True)

    wls = WLS([])
    tables, records, skipped = [], [], []
    for era in sorted(nights.Era.unique(), key=float):
        table, record = analyze_era(era, nights[nights.Era == era], wls, out_dir)
        if record is None:
            skipped.append(era)
            continue
        tables.append(table)
        records.append(record)
    if not records:
        sys.exit(f"error: no era in {start}..{end} has a night with a master WLS")
    table = pd.concat(tables, ignore_index=True)

    path = os.path.join(out_dir, f"rough_wls_nights_{start}_{end}.csv")
    table.to_csv(path, index=False)
    logger.info("%d row(s) -> %s", len(table), path)

    summary = summary_markdown(records, table, skipped, start, end, wls._linelist_df)
    path = os.path.join(out_dir, f"rough_wls_summary_{start}_{end}.md")
    with open(path, "w") as f:
        f.write(summary)
    logger.info("wrote %s", path)
    for line in summary.splitlines():
        logger.info("%s", line)

    plot_offsets_over_time(table, out_dir)


if __name__ == "__main__":
    main()
