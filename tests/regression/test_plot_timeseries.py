"""Tests for scripts/plots/plot_timeseries.py: the standalone RV-timeseries plotter.

plot_timeseries reads a target's L4 products off disk (over a datecode range) and
renders the RV-vs-date plot; bursts are always grouped, and per-night panels are
written only for nights with multiple observations. These cover what the script
owns: arg parsing, the threaded L4 discovery, the RV-header read, the burst
grouping, the main() wiring, and that the plot files actually get written.

Unit tests use synthetic bare-PRIMARY L4 frames in temp trees -- no real testdata.
"""

import numpy as np
import pytest
from astropy.io import fits

from kpfpipe.utils.io import kpf_filepath
from scripts.plots import plot_timeseries as _pt

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


def _write_l4(data_dir, datecode, seconds, obj, bjd, rv, rverr):
    """Write one bare-PRIMARY L4 frame under {data_dir}/L4/{datecode}; return path."""
    from pathlib import Path

    l4_dir = Path(data_dir) / "L4" / datecode
    l4_dir.mkdir(parents=True, exist_ok=True)
    path = l4_dir / f"kpf_SL4_{datecode}T{seconds:06d}.fits"
    hdr = fits.Header({"OBJECT": obj, "BJDTDB": bjd, "RV": rv, "RVERR": rverr})
    fits.PrimaryHDU(header=hdr).writeto(path)
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
    def test_required_flags(self, pt, drop):
        # Removing any one required flag (and its value) is a parse error.
        argv, skip = [], 0
        for tok in _BASE_ARGS:
            if tok == drop:
                skip = 3 if drop == "--date_range" else 2
            if skip:
                skip -= 1
                continue
            argv.append(tok)
        with pytest.raises(SystemExit):
            pt.parse_args(argv)

    @pytest.mark.parametrize(
        "rng",
        [["2024", "20240131"], ["20240201", "20240101"]],  # malformed / start>end
    )
    def test_date_range_validated(self, pt, rng):
        argv = ["--target", "x", "--date_range", *rng, "--data_dir", "/d",
                "--plot_dir", "/p"]  # fmt: skip
        with pytest.raises(SystemExit):
            pt.parse_args(argv)

    def test_group_bursts_flag_removed(self, pt):
        # --group_bursts no longer exists (grouping is always on).
        with pytest.raises(SystemExit):
            pt.parse_args(_BASE_ARGS + ["--group_bursts"])

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

    def test_date_range_and_obs_ids_mutually_exclusive(self, pt):
        with pytest.raises(SystemExit):
            pt.parse_args(_BASE_ARGS + ["--obs_ids", _OID])

    def test_neither_source_errors(self, pt):
        # Exactly one of --date_range / --obs_ids is required.
        with pytest.raises(SystemExit):
            pt.parse_args(["--target", "x", "--data_dir", "/d", "--plot_dir", "/p"])

    def test_invalid_obs_id_errors(self, pt):
        with pytest.raises(SystemExit):
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
        with pytest.raises(SystemExit):
            pt.discover_l4_files(str(tmp_path), "10700", "20240101", "20240131", 4)

    def test_missing_l4_root_exits(self, pt, tmp_path):
        with pytest.raises(SystemExit):
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
        # The second obs_id has no L4 on disk -> warned and skipped, not fatal.
        got = pt.l4_paths_for_obs_ids(
            str(tmp_path), ["KP.20240101.03600.00", "KP.20240101.07200.00"], "10700"
        )
        assert got == [a]
        assert "no L4 product for KP.20240101.07200.00" in capsys.readouterr().out

    def test_mismatched_object_warns_but_kept(self, pt, tmp_path, capsys):
        # A frame whose L4 OBJECT != target is plotted anyway, with a warning.
        a = _write_l4_for(str(tmp_path), "KP.20240101.03600.00", "99999")
        got = pt.l4_paths_for_obs_ids(str(tmp_path), ["KP.20240101.03600.00"], "10700")
        assert got == [a]
        out = capsys.readouterr().out
        assert "not target '10700'" in out and "plotting anyway" in out

    def test_exits_when_none_present(self, pt, tmp_path):
        with pytest.raises(SystemExit):
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
        # A present-but-non-numeric card (e.g. a stringified 'nan' -- these bare
        # keywords aren't registry-validated) must skip the one frame, not raise
        # from np.isfinite(str) and abort the whole plot. The good frame survives.
        good = _write_l4(str(tmp_path), "20240101", 100, "10700", 2.4e6, 1.5, 0.5)
        bad = tmp_path / "L4" / "20240101" / "kpf_SL4_str.fits"
        # All three cards present so the None-guard passes; RV is a string, which
        # is exactly what would reach (and crash) np.isfinite before the fix.
        fits.PrimaryHDU(
            header=fits.Header(
                {"OBJECT": "10700", "BJDTDB": 2.4e6, "RV": "nan", "RVERR": 0.5}
            )
        ).writeto(bad)

        times, rvs, *_ = pt._read_l4_rv([good, str(bad)])
        assert times.size == 1 and rvs[0] == 1.5


# ---------------------------------------------------------------------------
# _group_bursts
# ---------------------------------------------------------------------------


class TestGroupBursts:
    def test_single_burst_weighted_mean(self, pt):
        # Three frames within a burst collapse to one 1/err^2-weighted point.
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
        # 40 frames at a steady ~1-min cadence: ratio ~ 1, well over the floor.
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
        # Many frames but in two clusters hours apart: ratio >> 3 -> burst, not hicad.
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
        # 12 points at RV=1.0 plus one gross outlier at 1000. The zero-point must be
        # the median of the retained points (1.0), NOT the all-points median that
        # the outlier would tug upward, and the outlier must be flagged.
        g_times = np.arange(13, dtype=float)
        g_rvs = np.full(13, 1.0)
        g_rvs[6] = 1000.0
        ref, outlier = pt._delta_rv_reference(g_times, g_rvs)
        assert ref == pytest.approx(1.0)
        assert outlier[6] and outlier.sum() == 1

    def test_reference_is_order_independent(self, pt):
        # The mask is found on the time-ordered series, so a shuffled input yields
        # the same reference and the same flagged point (by identity, not index).
        g_times = np.arange(13, dtype=float)
        g_rvs = np.full(13, 5.0)
        g_rvs[3] = -900.0
        ref_a, out_a = pt._delta_rv_reference(g_times, g_rvs)
        perm = np.array([7, 0, 3, 11, 2, 9, 1, 12, 4, 8, 5, 10, 6])
        ref_b, out_b = pt._delta_rv_reference(g_times[perm], g_rvs[perm])
        assert ref_a == pytest.approx(ref_b) == pytest.approx(5.0)
        assert g_times[out_a][0] == g_times[perm][out_b][0] == 3.0

    def test_below_gate_flags_nothing(self, pt):
        # Fewer than 10 points: no trend, nothing flagged, zero-point is the plain
        # median over all points (an outlier is not rejected here by design).
        g_times = np.arange(5, dtype=float)
        g_rvs = np.array([1.0, 1.0, 1.0, 1.0, 50.0])
        ref, outlier = pt._delta_rv_reference(g_times, g_rvs)
        assert not outlier.any()
        assert ref == pytest.approx(np.median(g_rvs))


# ---------------------------------------------------------------------------
# plot outputs (establishes the repo's first Agg/PNG test)
# ---------------------------------------------------------------------------


class TestPlot:
    def _paths(self, data_dir, spec):
        """spec: list of (datecode, seconds, bjd) -> written L4 paths for '10700'."""
        return [
            _write_l4(data_dir, dc, sec, "10700", bjd, 1.0 + 0.001 * i, 0.3)
            for i, (dc, sec, bjd) in enumerate(spec)
        ]

    def test_timeseries_png_written(self, pt, tmp_path):
        # Two single-observation nights: the main plot is written, no nightly panel.
        paths = self._paths(
            str(tmp_path), [("20240101", 100, 2.4e6), ("20240102", 100, 2.4e6 + 1)]
        )
        plot_dir = tmp_path / "plots"
        pt.plot_rv_timeseries("10700", paths, str(plot_dir))
        assert (plot_dir / "10700_rv_timeseries.png").exists()
        assert not (plot_dir / "10700_rv_nightly.png").exists()

    def test_nightly_png_only_for_multiobs_nights(self, pt, tmp_path):
        # One night with two frames (a burst) -> the nightly panel is written too.
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
        # A 12-frame uniform night (high cadence) plus two ordinary single-obs
        # nights: the high-cadence night gets its own PNG and is excluded from the
        # main plot, which is still written from the two remaining nights.
        minute = 1.0 / 1440.0
        spec = [("20240926", 100 + i, 2.4e6 + i * minute) for i in range(12)]
        spec += [("20240101", 100, 2.45e6), ("20240102", 100, 2.45e6 + 1)]
        paths = self._paths(str(tmp_path), spec)
        plot_dir = tmp_path / "plots"
        pt.plot_rv_timeseries("10700", paths, str(plot_dir))
        assert (plot_dir / "10700_20240926_rv_timeseries.png").exists()
        assert (plot_dir / "10700_rv_timeseries.png").exists()
        # The high-cadence night is one datecode with >1 obs, but it is held out of
        # the nightly panels, and the two survivors are single-obs -> no nightly PNG.
        assert not (plot_dir / "10700_rv_nightly.png").exists()

    def test_all_high_cadence_writes_only_per_night_plots(self, pt, tmp_path):
        # Every night high-cadence: only the per-night plots exist, no main plot.
        minute = 1.0 / 1440.0
        spec = [("20240926", 100 + i, 2.4e6 + i * minute) for i in range(12)]
        spec += [("20240927", 200 + i, 2.41e6 + i * minute) for i in range(12)]
        paths = self._paths(str(tmp_path), spec)
        plot_dir = tmp_path / "plots"
        pt.plot_rv_timeseries("10700", paths, str(plot_dir))
        assert (plot_dir / "10700_20240926_rv_timeseries.png").exists()
        assert (plot_dir / "10700_20240927_rv_timeseries.png").exists()
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
