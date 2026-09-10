"""Render a target's RV timeseries plots from its L4 products.

Driven by ``scripts.processing.timeseries``, which hands over the frames it has
already discovered. Reads headers only.

Bursts of rapid-succession frames collapse to one RVERR-weighted point over a faint
underlay of the individual frames; observer-junk frames (``NOTJUNK == 0``) are
dropped. Writes ``{target}_rv_timeseries.png``, ``{target}_rv_nightly.png`` for
nights with several observations, and ``{target}_rv_timeseries_{datecode}.png`` for
each high-cadence night, which is held out of the main plot.
"""

import functools
import logging
import os
import subprocess
from collections import Counter
from datetime import UTC, datetime

import numpy as np
from astropy.io import fits

import kpfpipe
from kpfpipe.utils.io import kpf_filepath
from kpfpipe.utils.kpf import get_datecode
from kpfpipe.utils.stats import flag_outliers

logger = logging.getLogger(__name__)

_DPI = 150
_MINUTES_PER_DAY = 1440.0

# A bright-star burst is ~3-5 exposures at ~1-min readout cadence; revisits are tens
# of minutes apart, so 15 min sits cleanly between the two.
_BURST_GAP_MINUTES = 15.0

# Per-night observing modes, decided from how a night's frames are spaced.
# HIGH_CADENCE is tested first, since its frames all sit within the burst gap. The
# near-uniform ratio alone can't separate it from a short burst -- a 3-frame night
# gives ratio == 1 identically -- hence the frame-count floor.
_MODE_STANDARD = "standard"
_MODE_BURST = "burst"
_MODE_HIGH_CADENCE = "high cadence"

_HIGH_CADENCE_RATIO_MAX = 3.0
_HIGH_CADENCE_MIN_FRAMES = 10

# Marker styles: the per-night panels and single-night plot, the faint
# individual-frame underlay, the burst means over it, and a clipped-off outlier.
_NIGHT_MARKER = dict(fmt="o", ms=4, capsize=2, color="C3", mec="black", mew=0.5)
_UNDERLAY_MARKER = dict(fmt="o", ms=4, color="0.6", alpha=0.5, zorder=1)
_BURST_MARKER = dict(
    fmt="o", ms=8, capsize=3, color="C3", mec="black", mew=0.8, zorder=2
)
_OUTLIER_MARKER = dict(ls="none", color="black", ms=9, clip_on=False, zorder=4)


# Helpers below are module-level because none touch instance state. Anything reading
# the target, the directories, or the loaded arrays is a method instead.
def _pyplot():
    """``pyplot`` with the Agg backend pinned, imported on first use.

    Never at module scope: the wrapper sets ``MPLBACKEND`` only once its ``main()``
    runs, which is after this import, and nothing pins a backend for the tests.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


@functools.cache
def _git_commit():
    """The repo's short commit hash for the provenance footer, or 'unknown'."""
    try:
        out = subprocess.run(
            ["git", "-C", str(kpfpipe.REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def _sorted_by_time(times, rvs, errs):
    """The three arrays reordered into ascending time."""
    order = np.argsort(times)
    return times[order], rvs[order], errs[order]


def _group_bursts(times, rvs, errs, gap_minutes=_BURST_GAP_MINUTES):
    """Collapse each burst of rapid-succession frames to one RVERR-weighted point.

    Splits time-ordered frames wherever consecutive BJD_TDB values differ by more
    than ``gap_minutes``, then combines each burst with 1/RVERR**2 weights.
    """
    times, rvs, errs = _sorted_by_time(times, rvs, errs)
    breaks = np.nonzero(np.diff(times) * _MINUTES_PER_DAY > gap_minutes)[0] + 1
    g_times, g_rvs, g_errs = [], [], []
    for sel in np.split(np.arange(times.size), breaks):
        w = 1.0 / errs[sel] ** 2
        g_times.append(np.sum(w * times[sel]) / np.sum(w))
        g_rvs.append(np.sum(w * rvs[sel]) / np.sum(w))
        g_errs.append(1.0 / np.sqrt(np.sum(w)))
    return np.array(g_times), np.array(g_rvs), np.array(g_errs)


def _classify_observing_mode(times, nights, gap_minutes=_BURST_GAP_MINUTES):
    """Map each observing night to its mode: ``{datecode: one of _MODE_*}``."""
    modes = {}
    for night in np.unique(nights):
        t = np.sort(times[nights == night])
        gaps = np.diff(t)
        # High cadence first; med > 0 guards coincident timestamps (divide-by-zero).
        if t.size >= _HIGH_CADENCE_MIN_FRAMES:
            med = np.median(gaps)
            if med > 0 and gaps.mean() / med <= _HIGH_CADENCE_RATIO_MAX:
                modes[night] = _MODE_HIGH_CADENCE
                continue
        # Burst if any consecutive pair is within the burst gap, else isolated.
        if gaps.size and np.any(gaps * _MINUTES_PER_DAY <= gap_minutes):
            modes[night] = _MODE_BURST
        else:
            modes[night] = _MODE_STANDARD
    return modes


def _delta_rv_reference(g_times, g_rvs):
    """Zero-point and outlier mask ``(ref, outlier)`` for the grouped burst means.

    The zero-point is the median of the *retained* points, so an outlier skews
    neither it nor the RV_RMS about it. The trend method is shift- and
    scale-invariant, so the mask is well-defined before a reference exists; it is
    gated to >=10 points.
    """
    order = np.argsort(g_times)
    outlier = np.zeros(g_rvs.shape, dtype=bool)
    if g_rvs.size >= 10:
        outlier[order] = flag_outliers(g_rvs[order], 5.0, kernel_size=5, method="trend")
    ref = float(np.median(g_rvs[~outlier]) if np.any(~outlier) else np.median(g_rvs))
    return ref, outlier


def _delta_rv_stats(rvs, errs, ref, outlier):
    """Delta-RV [m/s] and error bars about ``ref``, with RV_RMS and median RV_ERR.

    Returns ``(drv, derr, rms, med_err)``. RV_RMS uses only the retained points, so a
    flagged outlier can't inflate it; an all-outlier input falls back to every point.
    """
    drv = (rvs - ref) * 1e3  # km/s -> m/s
    derr = errs * 1e3
    rms = float(np.std(drv[~outlier])) if np.any(~outlier) else float(np.std(drv))
    return drv, derr, rms, float(np.median(derr))


def _draw_night(ax, times, rvs, errs, ref):
    """Draw one night's frames as delta-RV vs. minutes from its first frame."""
    times, rvs, errs = _sorted_by_time(times, rvs, errs)
    minutes = (times - times.min()) * _MINUTES_PER_DAY
    ax.errorbar(minutes, (rvs - ref) * 1e3, yerr=errs * 1e3, **_NIGHT_MARKER)


def _symmetric_ylim(ax):
    """Set the y-limits symmetric about 0 (equal span above and below)."""
    ymax = max(abs(v) for v in ax.get_ylim())
    ax.set_ylim(-ymax, ymax)


def _new_axes():
    """A fresh ``(fig, ax)`` for a delta-RV [m/s] timeseries: zero line, y-label."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(0.0, color="0.6", lw=1, zorder=0)  # zero = median of retained points
    ax.set_ylabel(r"$\Delta$RV [m/s]")
    return fig, ax


def _finish_axes(ax, rms, med_err, ylim=None):
    """Add the RV_RMS/RV_ERR box, the grid, and the y-range shared by every plot."""
    ax.annotate(
        f"RV_RMS = {rms:.2f} m/s\nRV_ERR = {med_err:.2f} m/s",
        xy=(0.02, 0.96),
        xycoords="axes fraction",
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8),
    )
    ax.grid(True, alpha=0.3)
    if ylim is None:
        _symmetric_ylim(ax)
    else:
        ax.set_ylim(-ylim, ylim)


class PlotTimeseries:
    """The RV timeseries plots for one target, rendered from its L4 products.

    Parameters
    ----------
    target : str
        OBJECT name; titles and filenames are built from it.
    obs_ids : list of str
        The frames to plot.
    data_dir : str
        Science output root the L4 paths are built against.
    plot_dir : str
        Where the PNGs are written; created on first save.
    """

    def __init__(self, target, obs_ids, data_dir, plot_dir):
        self.target = target
        self.obs_ids = obs_ids
        self.data_dir = data_dir
        self.plot_dir = plot_dir
        # All four populated by _load().
        self.times = None
        self.rvs = None
        self.errs = None
        self.nights = None

    def run(self):
        """Read the L4 RVs and render this target's plots by observing mode.

        Each high-cadence night is drawn alone and held out of the rest; the
        remaining nights go to the main plot plus its per-night panels.

        Raises
        ------
        ValueError
            No frame carried a finite RV -- without this guard the plotter emits a
            blank PNG that looks like a successful run.
        """
        self._load()
        if self.times.size == 0:
            raise ValueError(f"no finite RV points to plot for target {self.target!r}")

        modes = _classify_observing_mode(self.times, self.nights)
        counts = Counter(modes.values())
        logger.info(
            "observing modes: %s",
            ", ".join(
                f"{counts[m]} {m}"
                for m in (_MODE_STANDARD, _MODE_BURST, _MODE_HIGH_CADENCE)
                if counts[m]
            ),
        )

        # Each high-cadence night is plotted on its own and dropped from the
        # multi-night plot, where it would collapse to a single burst point.
        hicad = sorted(n for n, m in modes.items() if m == _MODE_HIGH_CADENCE)
        times, rvs, errs, nights = self.times, self.rvs, self.errs, self.nights
        for night in hicad:
            sel = nights == night
            self._single_night(times[sel], rvs[sel], errs[sel], night)
        if hicad:
            keep = ~np.isin(nights, hicad)
            times, rvs, errs, nights = (a[keep] for a in (times, rvs, errs, nights))
            if times.size == 0:
                logger.info(
                    "all observed nights were high-cadence; no multi-night plot"
                )
                return

        self._multi_night(times, rvs, errs, nights)

    def _load(self):
        """Populate ``times``/``rvs``/``errs``/``nights`` from the L4 products.

        Each path is built with ``kpf_filepath`` -- no tree walk -- and opened once
        for PRIMARY (BJDTDB [d], RV/RVERR [km/s]) and QUALITY_CONTROL. A frame is
        dropped with a warning when its L4 is absent or unreadable, when ``NOTJUNK``
        marks it observer junk, or when its RV cards are missing or non-finite. An
        OBJECT disagreeing with the target is warned about but kept.

        Raises
        ------
        FileNotFoundError
            No L4 opened at all -- a missing-input failure, distinct from the empty
            result ``run`` rejects when products exist but carry no usable RV.
        """
        times, rvs, errs, nights = [], [], [], []
        n_read = 0
        for obs_id in sorted(set(self.obs_ids)):
            path = kpf_filepath(obs_id, "L4", data_root=self.data_dir)
            try:
                with fits.open(path) as hdul:
                    primary = hdul[0].header
                    junk = (
                        "QUALITY_CONTROL" in hdul
                        and hdul["QUALITY_CONTROL"].header.get("NOTJUNK", 1) == 0
                    )
            except OSError as e:
                logger.warning(
                    "no readable L4 for %s at %s (%s); skipping", obs_id, path, e
                )
                continue

            n_read += 1
            if junk:
                logger.warning(
                    "%s is an observer-junk frame (NOTJUNK=0); skipping", obs_id
                )
                continue
            if str(primary.get("OBJECT")).strip() != str(self.target):
                logger.warning(
                    "L4 for %s has OBJECT %r, not target %r; plotting anyway",
                    obs_id,
                    primary.get("OBJECT"),
                    self.target,
                )

            vals = (primary.get("BJDTDB"), primary.get("RV"), primary.get("RVERR"))
            # np.isfinite raises on a str, so require real finite numbers: a card may
            # be missing or a stringified 'nan', and one bad header must not abort.
            if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in vals):
                logger.warning("%s has no finite RV/RVERR/BJDTDB; skipping", obs_id)
                continue

            bjd, rv, err = vals
            times.append(bjd)
            rvs.append(rv)
            errs.append(err)
            nights.append(get_datecode(obs_id))

        if not n_read:
            raise FileNotFoundError(
                f"none of the {len(self.obs_ids)} supplied obs_id(s) have a "
                f"readable L4 product for target {self.target!r} under "
                f"{self.data_dir}"
            )
        self.times = np.array(times)
        self.rvs = np.array(rvs)
        self.errs = np.array(errs)
        self.nights = np.array(nights)

    def _multi_night(self, times, rvs, errs, nights):
        """Write the multi-night timeseries (bursts grouped) plus the per-night panels.

        Delta-RV [m/s] about the retained-burst-mean median vs. observation date, over
        a faint underlay of the individual frames. Flagged outliers are drawn as
        annotated edge triangles against a y-range clipped to the retained points, so
        one outlier can't rescale the plot. Returns the written path.
        """
        plt = _pyplot()
        from astropy.time import Time
        from matplotlib.ticker import FuncFormatter

        # One reference, shared by the underlay, the burst means and the panels.
        g_times, g_rvs, g_errs = _group_bursts(times, rvs, errs)
        ref, outlier = _delta_rv_reference(g_times, g_rvs)

        fig, ax = _new_axes()

        # Self-skips when no night has more than one observation.
        self._nightly_panels(times, rvs, errs, nights, ref)

        u_times, u_rvs, u_errs = _sorted_by_time(times, rvs, errs)
        ax.errorbar(
            u_times,
            (u_rvs - ref) * 1e3,
            yerr=u_errs * 1e3,
            label="individual frames",
            **_UNDERLAY_MARKER,
        )

        order = np.argsort(g_times)
        drv, derr, rms, med_err = _delta_rv_stats(g_rvs, g_errs, ref, outlier)

        # Clip to the retained points so a flagged outlier can't compress the plot.
        ylim = None
        if outlier.any():
            span = np.abs(drv[~outlier])
            ref_max = float(span.max()) if span.size else float(np.abs(drv).max())
            if ref_max > 0:
                ylim = 1.1 * ref_max

        # In-range points only; outliers are marked at the edge below instead.
        keep = order[~outlier[order]] if ylim is not None else order
        ax.errorbar(
            g_times[keep],
            drv[keep],
            yerr=derr[keep],
            label="burst mean",
            **_BURST_MARKER,
        )

        # Off-plot outliers, annotated with their delta-RV. clip_on=False so the
        # marker shows whole at the edge.
        if ylim is not None:
            label = r"5$\sigma$ outlier"
            for mask, marker in (
                (outlier & (drv > 0), "^"),
                (outlier & (drv < 0), "v"),
            ):
                if not mask.any():
                    continue
                ax.plot(
                    g_times[mask],
                    np.clip(drv[mask], -ylim, ylim),
                    marker=marker,
                    label=label,
                    **_OUTLIER_MARKER,
                )
                label = None
            for i in np.flatnonzero(outlier):
                ax.annotate(
                    f"{drv[i]:+.0f} m/s",
                    (g_times[i], float(np.clip(drv[i], -ylim, ylim))),
                    textcoords="offset points",
                    xytext=(0, -12 if drv[i] > 0 else 12),
                    ha="center",
                    va="top" if drv[i] > 0 else "bottom",
                    fontsize=7,
                )

        ax.legend(loc="upper right", fontsize=8)

        # Calendar dates for the BJD_TDB axis; the TDB-vs-UTC offset (~seconds) is
        # irrelevant at day granularity.
        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda jd, _p: Time(jd, format="jd", scale="tdb").strftime("%Y%m%d")
            )
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_xlabel("Date [UT]")
        ax.set_title(self.target)
        _finish_axes(ax, rms, med_err, ylim=ylim)

        out_path = self._save(fig, "timeseries")
        logger.info(
            "RV timeseries plot -> %s (RV_RMS %.2f m/s, RV_ERR %.2f m/s)",
            out_path,
            rms,
            med_err,
        )
        return out_path

    def _single_night(self, times, rvs, errs, night):
        """Write a standalone RV timeseries for one high-cadence night's frames.

        Such a night is held out of the multi-night plot, where its tens of
        near-uniform frames would collapse to one misleading burst point. Plotted
        *ungrouped*, so the within-night variation is visible. Returns the path.
        """
        times, rvs, errs = _sorted_by_time(times, rvs, errs)
        ref, outlier = _delta_rv_reference(times, rvs)
        _, _, rms, med_err = _delta_rv_stats(rvs, errs, ref, outlier)

        fig, ax = _new_axes()
        _draw_night(ax, times, rvs, errs, ref)
        ax.set_xlabel("Minutes from first frame of night")
        ax.set_title(f"{self.target}  {night}  (high cadence, n={times.size})")
        _finish_axes(ax, rms, med_err)

        out_path = self._save(fig, f"timeseries_{night}")
        logger.info(
            "high-cadence night plot -> %s (RV_RMS %.2f m/s, n=%d)",
            out_path,
            rms,
            times.size,
        )
        return out_path

    def _nightly_panels(self, times, rvs, errs, nights, ref):
        """Write a per-night multi-panel plot of the individual (ungrouped) frames.

        One panel per night with more than one observation; a single-frame night
        carries no within-night trend, so it is skipped, and a run with no such night
        writes nothing. ``ref`` is the main plot's zero-point, so the two agree.
        Panels share a y-axis to keep nights comparable. Returns the path, or None.
        """
        plt = _pyplot()
        counts = Counter(nights)
        unique = sorted(night for night, n in counts.items() if n > 1)
        if not unique:
            logger.info("no multi-observation nights; skipping nightly panels")
            return None

        ncols = min(4, len(unique))
        nrows = int(np.ceil(len(unique) / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.3 * ncols, 2.7 * nrows), sharey=True, squeeze=False
        )
        for ax, night in zip(axes.flat, unique, strict=False):
            sel = nights == night
            ax.axhline(0.0, color="0.6", lw=1, zorder=0)
            _draw_night(ax, times[sel], rvs[sel], errs[sel], ref)
            ax.set_title(night, fontsize=9)
            ax.grid(True, alpha=0.3)
        for ax in axes.flat[len(unique) :]:
            ax.set_visible(False)
        _symmetric_ylim(axes.flat[0])  # sharey: one range applies to every panel

        fig.supxlabel("Minutes from first frame of night")
        fig.supylabel(r"$\Delta$RV [m/s]")
        fig.suptitle(self.target)

        out_path = self._save(fig, "nightly")
        logger.info("nightly panels plot -> %s (%d night(s))", out_path, len(unique))
        return out_path

    def _save(self, fig, suffix):
        """Lay out, stamp, write and close one figure; returns its path.

        ``suffix`` names the plot within this target's set, e.g. 'nightly' gives
        '10700_rv_nightly.png'.
        """
        plt = _pyplot()
        out_path = os.path.join(self.plot_dir, f"{self.target}_rv_{suffix}.png")
        fig.tight_layout()
        fig.text(
            0.99,
            0.005,
            f"generated {datetime.now(UTC):%Y-%m-%d %H:%M:%S} UT · {_git_commit()}",
            fontsize=8,
            color="darkgray",
            ha="right",
            va="bottom",
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=_DPI)
        plt.close(fig)
        return out_path
