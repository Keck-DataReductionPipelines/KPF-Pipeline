"""Tests for scripts/plotting/timeseries.py: the RV-timeseries plotter.

PlotTimeseries reads a target's L4 products off disk -- the frames handed to it by
scripts.processing.timeseries -- and renders the RV-vs-date plot; bursts are always
grouped, and per-night panels are written only for nights with multiple observations.
These cover what the module owns: the L4 read and its per-frame filtering, the burst
grouping, the observing-mode split, and that the plot files actually get written.

Unit tests use synthetic bare-PRIMARY L4 frames in temp trees -- no real testdata.
"""

import os

import numpy as np
import pytest
from astropy.io import fits

from kpfpipe.utils.io import kpf_filepath
from scripts.plotting import timeseries as _pt

# Quicklook render suite: excluded from `make test-fast`, run by `make test-qlp`.
pytestmark = pytest.mark.quicklook

_TARGET = "10700"
_OID = "KP.20240101.03600.00"
_MIN = 1.0 / 1440.0  # one minute in BJD days


@pytest.fixture(scope="module")
def pt():
    return _pt


# ---------------------------------------------------------------------------
# Synthetic-frame helpers
# ---------------------------------------------------------------------------


def _oid(datecode, seconds, index=0):
    """An obs_id on `datecode` at `seconds` past UT midnight."""
    return f"KP.{datecode}.{seconds:05d}.{index:02d}"


def _write_l4(
    data_dir, obs_id, obj=_TARGET, bjd=2.4e6, rv=1.0, rverr=0.3, notjunk=None
):
    """Write one L4 at the exact kpf_filepath location for `obs_id`; return its path.

    Bare PRIMARY by default; pass `notjunk` (0/1) to add a QUALITY_CONTROL extension
    carrying that NOTJUNK card, the junk QC flag the plotter reads. Cards are dropped
    from PRIMARY by passing None for `bjd`/`rv`/`rverr`, which is how the
    missing-keyword and non-finite cases are built.
    """
    path = kpf_filepath(obs_id, "L4", data_root=data_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cards = {"OBJECT": obj, "BJDTDB": bjd, "RV": rv, "RVERR": rverr}
    hdus = [
        fits.PrimaryHDU(
            header=fits.Header({k: v for k, v in cards.items() if v is not None})
        )
    ]
    if notjunk is not None:
        hdus.append(
            fits.ImageHDU(
                name="QUALITY_CONTROL", header=fits.Header({"NOTJUNK": notjunk})
            )
        )
    fits.HDUList(hdus).writeto(path)
    return path


def _plot(tmp_path, obs_ids, target=_TARGET):
    """A PlotTimeseries over `obs_ids`, writing into tmp_path/plots."""
    plot_dir = tmp_path / "plots"
    return _pt.PlotTimeseries(target, obs_ids, str(tmp_path), str(plot_dir)), plot_dir


def _loaded(tmp_path, obs_ids, target=_TARGET):
    """Run _load over `obs_ids`; return the populated plotter."""
    plot, _ = _plot(tmp_path, obs_ids, target)
    plot._load()
    return plot


# ---------------------------------------------------------------------------
# _load: path building, per-frame filtering, junk exclusion
# ---------------------------------------------------------------------------


class TestLoad:
    def test_reads_values_and_night(self, pt, tmp_path):
        _write_l4(str(tmp_path), _OID, bjd=2.4e6, rv=1.5, rverr=0.25)
        plot = _loaded(tmp_path, [_OID])
        assert plot.times == pytest.approx([2.4e6])
        assert plot.rvs == pytest.approx([1.5])
        assert plot.errs == pytest.approx([0.25])
        assert list(plot.nights) == ["20240101"]

    def test_builds_paths_without_a_scan(self, pt, tmp_path):
        # Each obs_id resolves straight through kpf_filepath, so a frame sitting
        # anywhere else in the tree is simply not found.
        oids = [_OID, "KP.20240101.03660.00"]
        for oid in oids:
            _write_l4(str(tmp_path), oid)
        plot = _loaded(tmp_path, oids)
        assert plot.times.size == 2

    def test_skips_missing_l4(self, pt, tmp_path, caplog):
        _write_l4(str(tmp_path), _OID)
        plot = _loaded(tmp_path, [_OID, "KP.20240101.09999.00"])
        assert plot.times.size == 1
        assert "no readable L4" in caplog.text

    def test_mismatched_object_warns_but_is_kept(self, pt, tmp_path, caplog):
        # Surfacing the mismatch beats silently discarding data.
        _write_l4(str(tmp_path), _OID, obj="99999")
        plot = _loaded(tmp_path, [_OID])
        assert plot.times.size == 1
        assert "not target" in caplog.text

    def test_junk_frame_is_dropped(self, pt, tmp_path):
        good, junk = _OID, "KP.20240101.03660.00"
        _write_l4(str(tmp_path), good, rv=1.0, notjunk=1)
        _write_l4(str(tmp_path), junk, rv=2.0, notjunk=0)
        plot = _loaded(tmp_path, [good, junk])
        assert plot.rvs == pytest.approx([1.0])

    def test_missing_qc_extension_is_not_junk(self, pt, tmp_path):
        # Mirrors the observer-list no-op when the junk list itself is absent.
        _write_l4(str(tmp_path), _OID)
        plot = _loaded(tmp_path, [_OID])
        assert plot.times.size == 1

    @pytest.mark.parametrize("missing", ["bjd", "rv", "rverr"])
    def test_skips_missing_keyword(self, pt, tmp_path, missing):
        _write_l4(str(tmp_path), _OID, **{missing: None})
        plot = _loaded(tmp_path, [_OID])
        assert plot.times.size == 0

    def test_skips_string_valued_card_without_crashing(self, pt, tmp_path):
        # A real NaN cannot reach a FITS header (astropy rejects it on write), so the
        # non-finite guard's on-disk form is a stringified 'nan'. np.isfinite raises
        # on a str, so the type check must reject it rather than abort the read.
        _write_l4(str(tmp_path), _OID, rv="nan")
        plot = _loaded(tmp_path, [_OID])
        assert plot.times.size == 0

    def test_raises_when_no_l4_is_readable(self, pt, tmp_path):
        # A missing-input failure, distinct from "products exist but hold no RV".
        plot, _ = _plot(tmp_path, [_OID])
        with pytest.raises(FileNotFoundError, match="readable L4"):
            plot._load()


# ---------------------------------------------------------------------------
# _group_bursts
# ---------------------------------------------------------------------------


class TestGroupBursts:
    def test_single_burst_weighted_mean(self, pt):
        day = 1.0 / 1440.0  # one minute in days
        times = np.array([0.0, day, 2 * day])
        rvs = np.array([10.0, 20.0, 30.0])
        errs = np.array([1.0, 1.0, 2.0])
        gt, gr, ge = pt._group_bursts(times, rvs, errs)
        assert gt.size == 1
        w = 1.0 / errs**2
        assert gr[0] == pytest.approx(np.sum(w * rvs) / np.sum(w))
        assert ge[0] == pytest.approx(1.0 / np.sqrt(np.sum(w)))

    def test_gap_splits_bursts(self, pt):
        # A >15-min gap separates two bursts -> two grouped points.
        times = np.array([0.0, 1.0 / 1440.0, 0.5, 0.5 + 1.0 / 1440.0])  # ~12h apart
        rvs = np.array([1.0, 2.0, 3.0, 4.0])
        errs = np.ones(4)
        gt, _, _ = pt._group_bursts(times, rvs, errs)
        assert gt.size == 2


# ---------------------------------------------------------------------------
# _classify_observing_mode: standard / burst / high-cadence per night
# ---------------------------------------------------------------------------


class TestClassifyObservingMode:
    _MIN = 1.0 / 1440.0  # one minute in BJD days

    def _mode(self, pt, times):
        """The mode _classify_observing_mode assigns one night at the given epochs."""
        t = np.asarray(times, dtype=float)
        return pt._classify_observing_mode(t, np.array(["20240101"] * t.size))[
            "20240101"
        ]

    def test_uniform_many_frames_is_high_cadence(self, pt):
        # 40 frames at a steady ~1-min cadence: ratio ~1, above the frame floor.
        assert self._mode(pt, np.arange(40) * self._MIN) == pt._MODE_HIGH_CADENCE

    def test_three_frame_cluster_is_burst(self, pt):
        # A lone 3-frame burst is uniform but below the frame floor -> burst.
        assert self._mode(pt, [0.0, self._MIN, 2 * self._MIN]) == pt._MODE_BURST

    def test_isolated_singles_are_standard(self, pt):
        # Two frames ~6 h apart (> burst gap): each isolated -> standard.
        assert self._mode(pt, [0.0, 0.25]) == pt._MODE_STANDARD

    def test_single_frame_is_standard(self, pt):
        assert self._mode(pt, [0.0]) == pt._MODE_STANDARD

    def test_multi_burst_night_is_burst_not_high_cadence(self, pt):
        # Many frames, but in two clusters hours apart: ratio >> 3 -> burst.
        first = np.arange(6) * self._MIN
        second = 0.2 + np.arange(6) * self._MIN  # ~4.8 h later
        assert self._mode(pt, np.concatenate([first, second])) == pt._MODE_BURST

    def test_maps_every_night_to_its_mode(self, pt):
        hc = np.arange(30) * self._MIN
        burst = np.array([0.0, self._MIN, 0.5, 0.5 + self._MIN])
        std = np.array([0.1])
        times = np.concatenate([hc, burst, std])
        nights = np.array(
            ["20240926"] * hc.size + ["20240101"] * burst.size + ["20240102"]
        )
        assert pt._classify_observing_mode(times, nights) == {
            "20240926": pt._MODE_HIGH_CADENCE,
            "20240101": pt._MODE_BURST,
            "20240102": pt._MODE_STANDARD,
        }


# ---------------------------------------------------------------------------
# _delta_rv_reference: zero-point from the retained (non-outlier) points
# ---------------------------------------------------------------------------


class TestDeltaRvReference:
    def test_zero_point_excludes_outlier(self, pt):
        # The zero-point must be the median of the retained points (1.0), not the
        # all-points median that the 1000 outlier would tug upward.
        g_times = np.arange(13, dtype=float)
        g_rvs = np.full(13, 1.0)
        g_rvs[6] = 1000.0
        ref, outlier = pt._delta_rv_reference(g_times, g_rvs)
        assert ref == pytest.approx(1.0)
        assert outlier[6] and outlier.sum() == 1

    def test_reference_is_order_independent(self, pt):
        # The mask is found on the time-ordered series, so a shuffled input gives
        # the same reference and flags the same point (by identity, not index).
        g_times = np.arange(13, dtype=float)
        g_rvs = np.full(13, 5.0)
        g_rvs[3] = -900.0
        ref_a, out_a = pt._delta_rv_reference(g_times, g_rvs)
        perm = np.array([7, 0, 3, 11, 2, 9, 1, 12, 4, 8, 5, 10, 6])
        ref_b, out_b = pt._delta_rv_reference(g_times[perm], g_rvs[perm])
        assert ref_a == pytest.approx(ref_b) == pytest.approx(5.0)
        assert g_times[out_a][0] == g_times[perm][out_b][0] == 3.0

    def test_below_gate_flags_nothing(self, pt):
        # Below the 10-point gate: no trend fit, so nothing is flagged and the
        # zero-point is the plain median over all points.
        g_times = np.arange(5, dtype=float)
        g_rvs = np.array([1.0, 1.0, 1.0, 1.0, 50.0])
        ref, outlier = pt._delta_rv_reference(g_times, g_rvs)
        assert not outlier.any()
        assert ref == pytest.approx(np.median(g_rvs))


class TestDeltaRvStats:
    """``RV_RMS`` is the number an observer reads off the plot to judge RV
    stability, and nothing checked it -- the plot tests assert only that a PNG
    exists. Its sibling ``_delta_rv_reference`` is value-tested above; the
    asymmetry was unintentional.

    These are arithmetic checks on inputs the test supplies. They pin no
    pipeline RV.
    """

    def test_scaling_and_rms(self, pt):
        # km/s -> m/s on both the offsets and the error bars: rvs 2 mm/s either
        # side of ref give drv [-2, 0, 2] m/s, whose std is sqrt(8/3).
        rvs = np.array([1.000, 1.002, 1.004])
        errs = np.array([0.001, 0.002, 0.003])
        outlier = np.zeros(3, dtype=bool)
        drv, derr, rms, med_err = pt._delta_rv_stats(rvs, errs, 1.002, outlier)
        assert drv == pytest.approx([-2.0, 0.0, 2.0])
        assert derr == pytest.approx([1.0, 2.0, 3.0])
        assert rms == pytest.approx(np.sqrt(8.0 / 3.0))
        assert med_err == pytest.approx(2.0)

    def test_flagged_outlier_does_not_inflate_rms(self, pt):
        # A flagged point 1 km/s off would dominate the std if it leaked back in;
        # the retained three must give the same answer as if it were absent.
        rvs = np.array([1.000, 1.002, 1.004, 2.000])
        errs = np.full(4, 0.002)
        outlier = np.array([False, False, False, True])
        _, _, rms, _ = pt._delta_rv_stats(rvs, errs, 1.002, outlier)
        assert rms == pytest.approx(np.sqrt(8.0 / 3.0))
        # Guard the direction: including the outlier gives a vastly larger number.
        assert rms < np.std((rvs - 1.002) * 1e3) / 100

    def test_all_outliers_falls_back_to_every_point(self, pt):
        # With nothing retained there is no meaningful subset, so the std is taken
        # over everything rather than over an empty slice (which would be nan).
        rvs = np.array([1.000, 1.002, 1.004])
        errs = np.full(3, 0.002)
        outlier = np.ones(3, dtype=bool)
        _, _, rms, _ = pt._delta_rv_stats(rvs, errs, 1.002, outlier)
        assert rms == pytest.approx(np.sqrt(8.0 / 3.0))


# ---------------------------------------------------------------------------
# plot outputs
# ---------------------------------------------------------------------------


class TestPlot:
    def _run(self, tmp_path, spec):
        """Write an L4 per (datecode, seconds, bjd) in `spec`, then plot them."""
        obs_ids = []
        for i, (dc, sec, bjd) in enumerate(spec):
            oid = _oid(dc, sec)
            _write_l4(str(tmp_path), oid, bjd=bjd, rv=1.0 + 0.001 * i, rverr=0.3)
            obs_ids.append(oid)
        plot, plot_dir = _plot(tmp_path, obs_ids)
        plot.run()
        return plot_dir

    def test_no_finite_rv_points_raises(self, pt, tmp_path):
        # Without this guard the plotter runs on empty arrays and emits a blank PNG
        # that looks like a successful run. The one frame is readable -- so _load's
        # missing-input error does not fire -- but is dropped for a missing RVERR.
        _write_l4(str(tmp_path), _OID, rverr=None)
        plot, _ = _plot(tmp_path, [_OID])
        with pytest.raises(ValueError, match="no finite RV"):
            plot.run()

    def test_timeseries_png_written(self, pt, tmp_path):
        # Single-observation nights get no nightly panel.
        plot_dir = self._run(
            tmp_path, [("20240101", 100, 2.4e6), ("20240102", 100, 2.4e6 + 1)]
        )
        assert (plot_dir / "10700_rv_timeseries.png").exists()
        assert not (plot_dir / "10700_rv_nightly.png").exists()

    def test_nightly_png_only_for_multiobs_nights(self, pt, tmp_path):
        # One night with two frames (a burst) also gets the nightly panel.
        plot_dir = self._run(
            tmp_path, [("20240101", 100, 2.4e6), ("20240101", 160, 2.4e6 + _MIN)]
        )
        assert (plot_dir / "10700_rv_timeseries.png").exists()
        assert (plot_dir / "10700_rv_nightly.png").exists()

    def test_high_cadence_night_gets_own_plot_and_is_held_out(self, pt, tmp_path):
        # The high-cadence night gets its own PNG and is excluded from the main plot,
        # which is still written from the two remaining nights.
        spec = [("20240926", 100 + i, 2.4e6 + i * _MIN) for i in range(12)]
        spec += [("20240101", 100, 2.45e6), ("20240102", 100, 2.45e6 + 1)]
        plot_dir = self._run(tmp_path, spec)
        assert (plot_dir / "10700_rv_timeseries_20240926.png").exists()
        assert (plot_dir / "10700_rv_timeseries.png").exists()
        # The high-cadence night is one datecode with >1 obs, but it is held out of
        # the nightly panels, and the two survivors are single-obs -> no nightly PNG.
        assert not (plot_dir / "10700_rv_nightly.png").exists()

    def test_all_high_cadence_writes_only_per_night_plots(self, pt, tmp_path):
        # Every night high-cadence, so there is nothing left for a main plot.
        spec = [("20240926", 100 + i, 2.4e6 + i * _MIN) for i in range(12)]
        spec += [("20240927", 200 + i, 2.41e6 + i * _MIN) for i in range(12)]
        plot_dir = self._run(tmp_path, spec)
        assert (plot_dir / "10700_rv_timeseries_20240926.png").exists()
        assert (plot_dir / "10700_rv_timeseries_20240927.png").exists()
        assert not (plot_dir / "10700_rv_timeseries.png").exists()
        assert not (plot_dir / "10700_rv_nightly.png").exists()
