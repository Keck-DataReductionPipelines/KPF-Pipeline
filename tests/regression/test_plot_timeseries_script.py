"""Tests for scripts/plots/plot_timeseries.py: the standalone RV-timeseries plotter.

plot_timeseries reads a target's L4 products off disk and renders the RV-vs-date
plot; bursts are always grouped, and per-night panels are written only for nights
with multiple observations. These cover what the script owns: arg parsing, the
threaded L4 discovery, the RV-header read, the burst grouping, the main() wiring,
and that the plot files actually get written.

Unit tests use synthetic bare-PRIMARY L4 frames in temp trees -- no real testdata.
"""

import numpy as np
import pytest
from astropy.io import fits

from kpfpipe.utils.io import kpf_filepath
from scripts.plots import plot_timeseries as _pt

# scripts/CLI/tools-layer suite: excluded from `make test-fast`.
pytestmark = pytest.mark.cli

_BASE_ARGS = [
    "--target", "10700",
    "--date_range", "20240101", "20240131",
    "--data_dir", "/data", "--plot_dir", "/plots",
]  # fmt: skip
_OID = "KP.20240101.03600.00"


@pytest.fixture(scope="module")
def pt():
    return _pt


# ---------------------------------------------------------------------------
# Synthetic-frame helper
# ---------------------------------------------------------------------------


def _write_l4(data_dir, datecode, seconds, obj, bjd, rv, rverr, notjunk=None):
    """Write one L4 frame under {data_dir}/L4/{datecode}; return path.

    Bare PRIMARY by default; pass `notjunk` (0/1) to add a QUALITY_CONTROL
    extension carrying that NOTJUNK card, the junk QC flag the plotter reads.
    """
    from pathlib import Path

    l4_dir = Path(data_dir) / "L4" / datecode
    l4_dir.mkdir(parents=True, exist_ok=True)
    path = l4_dir / f"kpf_SL4_{datecode}T{seconds:06d}.fits"
    hdr = fits.Header({"OBJECT": obj, "BJDTDB": bjd, "RV": rv, "RVERR": rverr})
    hdus = [fits.PrimaryHDU(header=hdr)]
    if notjunk is not None:
        hdus.append(
            fits.ImageHDU(
                name="QUALITY_CONTROL", header=fits.Header({"NOTJUNK": notjunk})
            )
        )
    fits.HDUList(hdus).writeto(path)
    return str(path)


def _write_l4_for(data_dir, obs_id, obj, bjd=2.4e6, rv=1.0, rverr=0.3):
    """Write an L4 at the exact kpf_filepath location for `obs_id`; return the path."""
    import os

    path = kpf_filepath(obs_id, "L4", data_root=data_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdr = fits.Header({"OBJECT": obj, "BJDTDB": bjd, "RV": rv, "RVERR": rverr})
    fits.PrimaryHDU(header=hdr).writeto(path)
    return path


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_minimal_valid(self, pt):
        ns = pt.parse_args(_BASE_ARGS)
        assert ns.target == "10700"
        assert ns.date_range == ["20240101", "20240131"]
        assert ns.data_dir == "/data" and ns.plot_dir == "/plots"

    @pytest.mark.parametrize(
        "drop", ["--target", "--date_range", "--data_dir", "--plot_dir"]
    )
    def test_required_flags(self, pt, capsys, drop):
        # Removing any one required flag (and its value) is a parse error.
        argv, skip = [], 0
        for tok in _BASE_ARGS:
            if tok == drop:
                skip = 3 if drop == "--date_range" else 2
            if skip:
                skip -= 1
                continue
            argv.append(tok)
        with pytest.raises(SystemExit) as exc:
            pt.parse_args(argv)
        assert exc.value.code == 2
        assert drop in capsys.readouterr().err

    @pytest.mark.parametrize("rng", [["2024", "20240131"], ["20240101", "2024"]])
    def test_date_range_validated(self, pt, capsys, rng):
        argv = ["--target", "x", "--date_range", *rng, "--data_dir", "/d",
                "--plot_dir", "/p"]  # fmt: skip
        with pytest.raises(SystemExit) as exc:
            pt.parse_args(argv)
        assert exc.value.code == 2
        # Distinguishes the two validators: a malformed datecode vs start > end.
        assert "--date_range value is not a valid datecode" in capsys.readouterr().err

    def test_date_range_start_after_end(self, pt, capsys):
        argv = ["--target", "x", "--date_range", "20240201", "20240101",
                "--data_dir", "/d", "--plot_dir", "/p"]  # fmt: skip
        with pytest.raises(SystemExit) as exc:
            pt.parse_args(argv)
        assert exc.value.code == 2
        assert "--date_range START must be <= END" in capsys.readouterr().err

    def test_removed_flag_has_no_dest(self, pt):
        # Grouping is always on, so --group_bursts was dropped. Asserting the
        # *dest* is gone is the real contract; argparse rejects any unknown
        # string for free, so a bare "it exits" would pass on --nonsense too --
        # and would keep passing if the flag came back as a no-op alias.
        assert "group_bursts" not in vars(pt.parse_args(_BASE_ARGS))

    def test_obs_ids_source_valid(self, pt):
        ns = pt.parse_args(
            [
                "--target",
                "10700",
                "--obs_ids",
                _OID,
                "--data_dir",
                "/d",
                "--plot_dir",
                "/p",
            ]  # fmt: skip
        )
        assert ns.obs_ids == [_OID] and ns.date_range is None

    def test_date_range_and_obs_ids_mutually_exclusive(self, pt, capsys):
        with pytest.raises(SystemExit) as exc:
            pt.parse_args(_BASE_ARGS + ["--obs_ids", _OID])
        assert exc.value.code == 2
        assert "--obs_ids: not allowed with argument --date_range" in (
            capsys.readouterr().err
        )

    def test_neither_source_errors(self, pt, capsys):
        with pytest.raises(SystemExit) as exc:
            pt.parse_args(["--target", "x", "--data_dir", "/d", "--plot_dir", "/p"])
        assert exc.value.code == 2
        assert "one of the arguments --date_range --obs_ids is required" in (
            capsys.readouterr().err
        )

    def test_invalid_obs_id_errors(self, pt, capsys):
        with pytest.raises(SystemExit) as exc:
            pt.parse_args(
                [
                    "--target",
                    "10700",
                    "--obs_ids",
                    "not-an-obs-id",
                    "--data_dir",
                    "/d",
                    "--plot_dir",
                    "/p",
                ]  # fmt: skip
            )
        assert exc.value.code == 2
        assert "--obs_ids value is not a valid obs_id" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# discover_l4_files
# ---------------------------------------------------------------------------


class TestDiscoverL4Files:
    def _tree(self, data_dir, nights, per_night, target="10700"):
        expected = set()
        for i, dc in enumerate(nights):
            for j in range(per_night):
                expected.add(
                    _write_l4(data_dir, dc, 100 + j, target, 2.4e6 + j, 1.0, 0.5)
                )
            _write_l4(data_dir, dc, 900 + i, "99999", 2.4e6, 1.0, 0.5)  # decoy
        return expected

    def test_threaded_matches_expected(self, pt, tmp_path):
        nights = ["20240101", "20240102", "20240103", "20240104"]
        expected = self._tree(str(tmp_path), nights, per_night=3)
        got = pt.discover_l4_files(str(tmp_path), "10700", "20240101", "20240131", 8)
        assert got == sorted(expected)
        assert len(got) == len(set(got)) == 12

    def test_serial_equals_parallel(self, pt, tmp_path):
        self._tree(str(tmp_path), ["20240101", "20240102"], per_night=2)
        common = ("10700", "20240101", "20240131")
        assert pt.discover_l4_files(str(tmp_path), *common, 1) == pt.discover_l4_files(
            str(tmp_path), *common, 8
        )

    def test_filters_by_object(self, pt, tmp_path):
        good = _write_l4(str(tmp_path), "20240101", 100, "10700", 2.4e6, 1.0, 0.5)
        _write_l4(str(tmp_path), "20240101", 200, "99999", 2.4e6, 1.0, 0.5)
        got = pt.discover_l4_files(str(tmp_path), "10700", "20240101", "20240131", 4)
        assert got == [good]

    def test_no_matches_exits(self, pt, tmp_path):
        _write_l4(str(tmp_path), "20240101", 100, "99999", 2.4e6, 1.0, 0.5)
        with pytest.raises(SystemExit, match="no L4 products for target '10700'"):
            pt.discover_l4_files(str(tmp_path), "10700", "20240101", "20240131", 4)

    def test_missing_l4_root_exits(self, pt, tmp_path):
        # A distinct diagnostic from "no matches": without the match= the tmp
        # tree has no matches either way and this test duplicates its neighbour.
        with pytest.raises(SystemExit, match="L4 output directory not found"):
            pt.discover_l4_files(str(tmp_path), "10700", "20240101", "20240131", 4)


# ---------------------------------------------------------------------------
# l4_paths_for_obs_ids: explicit obs_ids, no scan
# ---------------------------------------------------------------------------


class TestL4PathsForObsIds:
    def test_builds_paths_from_obs_ids(self, pt, tmp_path):
        a = _write_l4_for(str(tmp_path), "KP.20240101.03600.00", "10700")
        b = _write_l4_for(str(tmp_path), "KP.20240102.03600.00", "10700")
        got = pt.l4_paths_for_obs_ids(
            str(tmp_path), ["KP.20240102.03600.00", "KP.20240101.03600.00"], "10700"
        )
        assert got == sorted([a, b])

    def test_skips_missing_l4(self, pt, tmp_path, capsys):
        a = _write_l4_for(str(tmp_path), "KP.20240101.03600.00", "10700")
        # The second obs_id has no L4 on disk: warn and skip, not fatal.
        got = pt.l4_paths_for_obs_ids(
            str(tmp_path), ["KP.20240101.03600.00", "KP.20240101.07200.00"], "10700"
        )
        assert got == [a]
        assert "no L4 product for KP.20240101.07200.00" in capsys.readouterr().out

    def test_mismatched_object_warns_but_kept(self, pt, tmp_path, capsys):
        a = _write_l4_for(str(tmp_path), "KP.20240101.03600.00", "99999")
        got = pt.l4_paths_for_obs_ids(str(tmp_path), ["KP.20240101.03600.00"], "10700")
        assert got == [a]
        out = capsys.readouterr().out
        assert "not target '10700'" in out and "plotting anyway" in out

    def test_exits_when_none_present(self, pt, tmp_path):
        with pytest.raises(SystemExit, match="none of the 1 supplied obs_id"):
            pt.l4_paths_for_obs_ids(str(tmp_path), ["KP.20240101.03600.00"], "10700")


# ---------------------------------------------------------------------------
# _read_l4_rv
# ---------------------------------------------------------------------------


class TestReadL4Rv:
    def test_reads_and_skips_missing_keyword(self, pt, tmp_path):
        good = _write_l4(str(tmp_path), "20240101", 100, "10700", 2.4e6, 1.5, 0.5)
        # A frame missing RVERR (None) is skipped; the good frame survives.
        no_err = tmp_path / "L4" / "20240101" / "kpf_SL4_missing.fits"
        fits.PrimaryHDU(
            header=fits.Header({"OBJECT": "10700", "BJDTDB": 2.4e6, "RV": 1.0})
        ).writeto(no_err)

        times, rvs, errs, nights = pt._read_l4_rv([good, str(no_err)])
        assert times.size == 1 and rvs[0] == 1.5 and errs[0] == 0.5
        assert nights[0] == "20240101"

    def test_skips_nonfinite(self, pt, monkeypatch):
        # astropy refuses to write NaN into a header, so exercise the finite guard
        # with a stubbed header: a non-finite RV is dropped.
        monkeypatch.setattr(
            pt.fits,
            "getheader",
            lambda path, ext: {"BJDTDB": 2.4e6, "RV": float("nan"), "RVERR": 0.5},
        )
        times, *_ = pt._read_l4_rv(["/some/L4/20240101/kpf_SL4_x.fits"])
        assert times.size == 0

    def test_skips_string_valued_card_without_crashing(self, pt, tmp_path):
        # These bare keywords aren't registry-validated, so a non-numeric card
        # (a stringified 'nan') must skip its frame rather than raise from
        # np.isfinite(str) and abort the whole plot.
        good = _write_l4(str(tmp_path), "20240101", 100, "10700", 2.4e6, 1.5, 0.5)
        bad = tmp_path / "L4" / "20240101" / "kpf_SL4_str.fits"
        # All three cards present so the None-guard passes and the string RV is
        # what reaches np.isfinite.
        fits.PrimaryHDU(
            header=fits.Header(
                {"OBJECT": "10700", "BJDTDB": 2.4e6, "RV": "nan", "RVERR": 0.5}
            )
        ).writeto(bad)

        times, rvs, *_ = pt._read_l4_rv([good, str(bad)])
        assert times.size == 1 and rvs[0] == 1.5


# ---------------------------------------------------------------------------
# _l4_is_junk / junk exclusion: QUALITY_CONTROL NOTJUNK == 0
# ---------------------------------------------------------------------------


class TestJunkExclusion:
    def test_is_junk_reads_notjunk_flag(self, pt, tmp_path):
        junk = _write_l4(str(tmp_path), "20240101", 100, "10700", 2.4e6, 1.0, 0.5, 0)
        good = _write_l4(str(tmp_path), "20240101", 200, "10700", 2.4e6, 1.0, 0.5, 1)
        assert pt._l4_is_junk(junk) is True
        assert pt._l4_is_junk(good) is False

    def test_missing_qc_extension_is_not_junk(self, pt, tmp_path):
        # A bare-PRIMARY L4 (no QUALITY_CONTROL) counts as not junk, not an error.
        bare = _write_l4(str(tmp_path), "20240101", 100, "10700", 2.4e6, 1.0, 0.5)
        assert pt._l4_is_junk(bare) is False

    def test_read_excludes_junk_keeps_good(self, pt, tmp_path, capsys):
        good = _write_l4(str(tmp_path), "20240101", 100, "10700", 2.4e6, 1.5, 0.5, 1)
        junk = _write_l4(
            str(tmp_path), "20240101", 200, "10700", 2.4e6 + 1, 9.9, 0.5, 0
        )
        times, rvs, *_ = pt._read_l4_rv([good, junk])
        assert times.size == 1 and rvs[0] == 1.5
        assert "NOTJUNK=0" in capsys.readouterr().out


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
    def _paths(self, data_dir, spec):
        """spec: list of (datecode, seconds, bjd) -> written L4 paths for '10700'."""
        return [
            _write_l4(data_dir, dc, sec, "10700", bjd, 1.0 + 0.001 * i, 0.3)
            for i, (dc, sec, bjd) in enumerate(spec)
        ]

    def test_no_finite_rv_points_exits(self, pt, tmp_path):
        # Without this guard the plotter runs on empty arrays and emits a blank
        # PNG that looks like a successful run; the wrapper only reads the exit
        # code, so the difference is "loud" vs "green run, no data".
        # The one frame is dropped by _read_l4_rv for a missing RVERR (the
        # builder idiom from TestReadL4Rv), leaving nothing to plot.
        l4_dir = tmp_path / "L4" / "20240101"
        l4_dir.mkdir(parents=True)
        path = l4_dir / "kpf_SL4_20240101T000100.fits"
        fits.PrimaryHDU(
            header=fits.Header({"OBJECT": "10700", "BJDTDB": 2.4e6, "RV": 1.0})
        ).writeto(path)

        with pytest.raises(SystemExit, match="no finite RV"):
            pt.plot_rv_timeseries("10700", [str(path)], str(tmp_path / "plots"))

    def test_timeseries_png_written(self, pt, tmp_path):
        # Single-observation nights get no nightly panel.
        paths = self._paths(
            str(tmp_path), [("20240101", 100, 2.4e6), ("20240102", 100, 2.4e6 + 1)]
        )
        plot_dir = tmp_path / "plots"
        pt.plot_rv_timeseries("10700", paths, str(plot_dir))
        assert (plot_dir / "10700_rv_timeseries.png").exists()
        assert not (plot_dir / "10700_rv_nightly.png").exists()

    def test_nightly_png_only_for_multiobs_nights(self, pt, tmp_path):
        # One night with two frames (a burst) also gets the nightly panel.
        day = 1.0 / 1440.0
        paths = self._paths(
            str(tmp_path),
            [("20240101", 100, 2.4e6), ("20240101", 160, 2.4e6 + day)],
        )
        plot_dir = tmp_path / "plots"
        pt.plot_rv_timeseries("10700", paths, str(plot_dir))
        assert (plot_dir / "10700_rv_timeseries.png").exists()
        assert (plot_dir / "10700_rv_nightly.png").exists()

    def test_high_cadence_night_gets_own_plot_and_is_held_out(self, pt, tmp_path):
        # The high-cadence night gets its own PNG and is excluded from the main
        # plot, which is still written from the two remaining nights.
        minute = 1.0 / 1440.0
        spec = [("20240926", 100 + i, 2.4e6 + i * minute) for i in range(12)]
        spec += [("20240101", 100, 2.45e6), ("20240102", 100, 2.45e6 + 1)]
        paths = self._paths(str(tmp_path), spec)
        plot_dir = tmp_path / "plots"
        pt.plot_rv_timeseries("10700", paths, str(plot_dir))
        assert (plot_dir / "10700_rv_timeseries_20240926.png").exists()
        assert (plot_dir / "10700_rv_timeseries.png").exists()
        # The high-cadence night is one datecode with >1 obs, but it is held out of
        # the nightly panels, and the two survivors are single-obs -> no nightly PNG.
        assert not (plot_dir / "10700_rv_nightly.png").exists()

    def test_all_high_cadence_writes_only_per_night_plots(self, pt, tmp_path):
        # Every night high-cadence, so there is nothing left for a main plot.
        minute = 1.0 / 1440.0
        spec = [("20240926", 100 + i, 2.4e6 + i * minute) for i in range(12)]
        spec += [("20240927", 200 + i, 2.41e6 + i * minute) for i in range(12)]
        paths = self._paths(str(tmp_path), spec)
        plot_dir = tmp_path / "plots"
        pt.plot_rv_timeseries("10700", paths, str(plot_dir))
        assert (plot_dir / "10700_rv_timeseries_20240926.png").exists()
        assert (plot_dir / "10700_rv_timeseries_20240927.png").exists()
        assert not (plot_dir / "10700_rv_timeseries.png").exists()
        assert not (plot_dir / "10700_rv_nightly.png").exists()


# ---------------------------------------------------------------------------
# main: discover then plot
# ---------------------------------------------------------------------------


class TestMain:
    def test_discovers_then_plots(self, pt, tmp_path, monkeypatch):
        a = _write_l4(str(tmp_path), "20240101", 100, "10700", 2.4e6, 1.0, 0.3)
        captured = {}

        def _fake_plot(target, l4_paths, plot_directory):
            captured.update(target=target, paths=l4_paths, plot_dir=plot_directory)

        monkeypatch.setattr(pt, "plot_rv_timeseries", _fake_plot)
        pt.main(
            [
                "--target",
                "10700",
                "--date_range",
                "20240101",
                "20240131",
                "--data_dir",
                str(tmp_path),
                "--plot_dir",
                str(tmp_path / "p"),
            ]  # fmt: skip
        )
        assert captured["target"] == "10700"
        assert captured["paths"] == [a]
        assert captured["plot_dir"] == str(tmp_path / "p")

    def test_obs_ids_branch_builds_paths_without_scan(self, pt, tmp_path, monkeypatch):
        # --obs_ids builds paths directly; a bogus scan would raise if reached.
        a = _write_l4_for(str(tmp_path), "KP.20240101.03600.00", "10700")
        captured = {}
        monkeypatch.setattr(
            pt, "plot_rv_timeseries", lambda t, p, d: captured.update(paths=p)
        )
        monkeypatch.setattr(
            pt,
            "discover_l4_files",
            lambda *a, **k: pytest.fail(
                "discover should not be called in obs_ids mode"
            ),
        )
        pt.main(
            [
                "--target",
                "10700",
                "--obs_ids",
                "KP.20240101.03600.00",
                "--data_dir",
                str(tmp_path),
                "--plot_dir",
                str(tmp_path / "p"),
            ]  # fmt: skip
        )
        assert captured["paths"] == [a]
