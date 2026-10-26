"""Tests for scripts/analyze/rough_wls.py: the per-era rough-WLS analysis.

Cover what the script owns -- arg parsing, the chunked cross-correlation, the
shift-and-refit of a rough WLS table, the within-era deviation, the anchor
choice -- plus its ``kpfpipe analyze`` registration. Synthetic spectra only.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.polynomial import polynomial

import tools.cli as cli
from kpfpipe import REPO_ROOT
from scripts.analyze import rough_wls

# scripts/CLI/tools-layer suite: excluded from `make test-fast`.
pytestmark = pytest.mark.cli

_BASE_ARGS = ["--date_range", "20230101", "20230131", "--input_dir", "/in"]
_REFERENCE = f"{REPO_ROOT}/reference/rough_wls/rough_wls_20240224.csv"


def _spectrum(shift=lambda x: 0.0, norder=2, seed=0):
    """Synthetic ThAr orders: the same random lines in each order, every line
    moved by ``shift(xn)`` pixels."""
    rng = np.random.default_rng(seed)
    x = np.arange(rough_wls.NCOL, dtype=float)
    centers = np.sort(rng.uniform(20, rough_wls.NCOL - 20, 300))
    amps = rng.uniform(50, 500, centers.size)
    moved = centers + shift(2 * centers / (rough_wls.NCOL - 1) - 1)
    row = 10 + sum(
        a * np.exp(-0.5 * ((x - c) / 1.5) ** 2)
        for a, c in zip(amps, moved, strict=True)
    )
    return np.tile(row, (norder, 1))


class TestParseArgs:
    def test_output_dir_fans_out_to_analysis_and_log(self):
        args = rough_wls.parse_args([*_BASE_ARGS, "--output_dir", "/out"])
        assert args.analysis_dir == "/out/analysis"
        assert args.log_dir == "/out/logs"

    def test_no_output_dirs_exits(self, capsys):
        with pytest.raises(SystemExit):
            rough_wls.parse_args(_BASE_ARGS)
        assert "pass --output_dir" in capsys.readouterr().err


class TestChunkShifts:
    def test_recovers_an_offset_stretch_and_warp(self):
        truth = (2.0, 1.5, 0.3)
        ref = {("GREEN", "SCI2"): _spectrum()}
        tgt = {("GREEN", "SCI2"): _spectrum(lambda xn: polynomial.polyval(xn, truth))}

        chunks = rough_wls.chunk_shifts(ref, tgt)
        good = chunks[chunks.Peak > rough_wls.MIN_PEAK]
        coeffs, rms, _ = rough_wls.robust_quadratic(good.Xn, good.Shift, floor=0.3)

        np.testing.assert_allclose(coeffs, truth, atol=0.05)
        assert rms < 0.05

    def test_identical_spectra_have_no_shift(self):
        ref = {("RED", "CAL"): _spectrum()}
        chunks = rough_wls.chunk_shifts(ref, ref)
        assert np.abs(chunks.Shift).max() < 1e-6


class TestShiftRoughWls:
    def test_zero_shift_reproduces_the_reference(self):
        reference = pd.read_csv(_REFERENCE)
        linelist = pd.read_csv(f"{REPO_ROOT}/reference/thar_line_list.csv")
        zero = {chip: np.zeros(3) for chip in ("GREEN", "RED")}

        candidate = rough_wls.shift_rough_wls(reference, zero)
        offsets = rough_wls.pixel_offsets(reference, candidate, linelist)

        assert list(candidate.columns) == list(reference.columns)
        for _med, lo, hi in offsets.values():
            assert max(abs(lo), abs(hi)) < 0.01

    def test_lines_move_by_the_applied_shift(self):
        reference = pd.read_csv(_REFERENCE)
        linelist = pd.read_csv(f"{REPO_ROOT}/reference/thar_line_list.csv")
        shift = {"GREEN": np.array([2.0, 1.5, 0.3]), "RED": np.array([7.0, 0.0, 0.0])}

        candidate = rough_wls.shift_rough_wls(reference, shift)
        a = rough_wls.line_pixels(reference, linelist)
        b = rough_wls.line_pixels(candidate, linelist)
        both = a.merge(b, on=["Chip", "Echelle", "Wave"], suffixes=("_ref", "_cand"))
        xn = 2 * both.Pixel_ref / (rough_wls.NCOL - 1) - 1
        expected = [
            polynomial.polyval(x, shift[chip])
            for x, chip in zip(xn, both.Chip, strict=True)
        ]

        np.testing.assert_allclose(
            both.Pixel_cand - both.Pixel_ref, expected, atol=0.02
        )


class TestMaxDeviation:
    def test_flags_more_than_one_pixel(self):
        era = np.array([2.0, 1.5, 0.3])
        assert rough_wls.max_deviation(era + [0.2, 0.0, 0.0], era) <= 1.0
        assert rough_wls.max_deviation(era + [0.0, 1.2, 0.0], era) > 1.0


class TestChooseAnchor:
    def _nights(self, has_master):
        return pd.DataFrame(
            dict(
                Night=[f"2024030{i}" for i in range(len(has_master))],
                ObsTime=pd.date_range("2024-03-01", periods=len(has_master)),
                Files=[[f"f{i}"] for i in range(len(has_master))],
                HasMaster=has_master,
            )
        )

    def _stub(self, monkeypatch, in_window):
        monkeypatch.setattr(rough_wls, "era_reference", lambda era, night: "ref.csv")
        monkeypatch.setattr(
            rough_wls,
            "line_offsets",
            lambda wls, ref, files: {
                chip: dict(in_window=in_window[files[0]]) for chip in ("GREEN", "RED")
            },
        )

    def test_never_anchors_on_a_night_without_a_master_wls(self, monkeypatch):
        # The middle night (index 2) failed its WLS; its nearest neighbour is used.
        self._stub(monkeypatch, {f"f{i}": 0.99 for i in range(5)})
        nights = self._nights([True, True, False, True, True])
        anchor, _, _ = rough_wls.choose_anchor("2.0", nights, wls=None)
        assert anchor.Night in {"20240301", "20240303"}

    def test_passes_over_a_night_with_lines_out_of_window(self, monkeypatch):
        self._stub(monkeypatch, {"f0": 0.99, "f1": 0.5, "f2": 0.4})
        anchor, _, _ = rough_wls.choose_anchor(
            "2.0", self._nights([True, True, True]), wls=None
        )
        assert anchor.Night == "20240300"

    def test_no_master_wls_in_the_era_returns_none(self, monkeypatch):
        self._stub(monkeypatch, {})
        assert (
            rough_wls.choose_anchor("2.0", self._nights([False, False]), None) is None
        )


class TestAnalyzeDispatch:
    def test_rough_wls_is_registered(self):
        assert cli._ANALYSES["rough_wls"] is rough_wls.main
